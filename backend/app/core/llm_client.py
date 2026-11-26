"""Google GenAI client with .env-driven models, 503 backoff, fallback, embeddings, and usage."""

import os
import time
from typing import Any

from google import genai
from google.genai import errors, types
import httpx

from ..config import settings

PRIMARY_MODEL = settings.primary_model
FALLBACK_MODEL = settings.fallback_model

_client: genai.Client | None = None


def get_genai_client() -> genai.Client:
    """Lazily initialize the Google GenAI SDK client with explicit request timeout."""
    global _client
    if _client is None:
        api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        http_kwargs: dict[str, Any] = {
            "timeout": 30000,
            "retry_options": types.HttpRetryOptions(attempts=1),
        }
        if any(
            "preview" in model.lower()
            for model in (settings.primary_model, settings.fallback_model, settings.embedding_model)
        ):
            http_kwargs["api_version"] = "v1beta"
        _client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(**http_kwargs),
        )
    return _client


def reset_genai_client() -> None:
    """Clear the cached GenAI client instance (primarily for testing)."""
    global _client
    _client = None


def _is_overload_error(exc: Exception) -> bool:
    """Return True when exc is a transient HTTP 503/504/network error."""
    if isinstance(exc, httpx.HTTPError):
        return True
    if isinstance(exc, errors.ServerError):
        message = str(exc)
        return any(code in message for code in ("503", "504", "UNAVAILABLE", "DEADLINE_EXCEEDED"))
    return False


def _extract_usage_tokens(response: Any) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from GenAI usage_metadata if present."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    return input_tokens, output_tokens


def _extract_response_text(response: Any) -> str:
    """Return stripped response text or raise ValueError if the model returned nothing."""
    raw_text = getattr(response, "text", None)
    cleaned = raw_text.strip() if isinstance(raw_text, str) else ""
    if not cleaned:
        raise ValueError("Model returned an empty response.")
    return cleaned


def _invoke_fallback(
    active_client: genai.Client,
    fallback_model: str,
    prompt: str,
    reason: str,
) -> tuple[str, str, int, int]:
    """Call fallback_model cleanly and wrap any fallback outage in RuntimeError."""
    try:
        response = active_client.models.generate_content(
            model=fallback_model,
            contents=prompt,
        )
        in_tok, out_tok = _extract_usage_tokens(response)
        return _extract_response_text(response), fallback_model, in_tok, out_tok
    except Exception as fallback_error:
        raise RuntimeError(f"{reason}: {fallback_error}") from fallback_error


def generate_with_retry_telemetry(
    prompt: str,
    max_retries: int | None = None,
    client: genai.Client | None = None,
) -> tuple[str, str, int, int]:
    """Generate text with 503 exponential backoff and fallback, returning usage tokens."""
    retries = max_retries if max_retries is not None else settings.max_retries
    primary_model = settings.primary_model
    fallback_model = settings.fallback_model
    active_client = client or get_genai_client()

    for attempt in range(retries):
        try:
            response = active_client.models.generate_content(
                model=primary_model,
                contents=prompt,
            )
            in_tok, out_tok = _extract_usage_tokens(response)
            return _extract_response_text(response), primary_model, in_tok, out_tok

        except ValueError as exc:
            return _invoke_fallback(
                active_client,
                fallback_model,
                prompt,
                f"Primary model returned empty output ({exc}) and fallback failed",
            )

        except (errors.ServerError, errors.ClientError, httpx.HTTPError) as exc:
            if isinstance(exc, errors.ClientError):
                err_text = str(exc)
                if any(tok in err_text for tok in ("429", "RESOURCE_EXHAUSTED", "404", "NOT_FOUND")):
                    return _invoke_fallback(
                        active_client,
                        fallback_model,
                        prompt,
                        "Primary model unavailable/quota exhausted and fallback model failed",
                    )
                raise RuntimeError(f"LLM request rejected by provider: {exc}") from exc
            if _is_overload_error(exc):
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return _invoke_fallback(
                        active_client,
                        fallback_model,
                        prompt,
                        "Complete pipeline failure. Both primary and fallback models overloaded",
                    )
            else:
                raise RuntimeError(f"LLM server error: {exc}") from exc

    raise RuntimeError("Failed to generate response after retry loop.")


def generate_with_retry_with_model(
    prompt: str,
    max_retries: int | None = None,
    client: genai.Client | None = None,
) -> tuple[str, str]:
    """Generate text using primary model with 503 exponential backoff and fallback model."""
    text_out, model_used, _, _ = generate_with_retry_telemetry(
        prompt,
        max_retries=max_retries,
        client=client,
    )
    return text_out, model_used


def generate_with_retry(prompt: str, max_retries: int | None = None) -> str:
    """Return generated text while preserving the legacy string-only signature."""
    text_out, _ = generate_with_retry_with_model(prompt, max_retries=max_retries)
    return text_out


def generate_embedding(
    text_input: str,
    task_type: str = "RETRIEVAL_DOCUMENT",
    max_retries: int | None = None,
    client: genai.Client | None = None,
) -> list[float]:
    """Generate a dense embedding vector using EMBEDDING_MODEL with 503 retry handling."""
    retries = max_retries if max_retries is not None else settings.max_retries
    embedding_model = settings.embedding_model
    active_client = client or get_genai_client()

    for attempt in range(retries):
        try:
            response = active_client.models.embed_content(
                model=embedding_model,
                contents=text_input,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            if not getattr(response, "embeddings", None):
                raise RuntimeError("Embedding response did not contain any vectors.")
            return [float(val) for val in response.embeddings[0].values]

        except errors.ClientError as exc:
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc
        except (errors.ServerError, httpx.HTTPError) as exc:
            if _is_overload_error(exc) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc

    raise RuntimeError("Failed to generate embedding after retry loop.")
