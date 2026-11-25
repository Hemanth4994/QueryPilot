"""Google GenAI client with .env-driven models, 503 backoff, fallback, embeddings, and usage."""

import os
import time
from typing import Any

from google import genai
from google.genai import errors, types

from ..config import settings

PRIMARY_MODEL = settings.primary_model
FALLBACK_MODEL = settings.fallback_model

_client: genai.Client | None = None


def get_genai_client() -> genai.Client:
    """Lazily initialize the Google GenAI SDK client."""
    global _client
    if _client is None:
        api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        _client = genai.Client(api_key=api_key)
    return _client


def reset_genai_client() -> None:
    """Clear the cached GenAI client instance (primarily for testing)."""
    global _client
    _client = None


def _is_overload_error(exc: errors.ServerError) -> bool:
    """Return True when the GenAI ServerError indicates HTTP 503 / UNAVAILABLE."""
    message = str(exc)
    return "503" in message or "UNAVAILABLE" in message


def _extract_usage_tokens(response: Any) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from GenAI usage_metadata if present."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    return input_tokens, output_tokens


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
            return response.text.strip(), primary_model, in_tok, out_tok

        except errors.ServerError as exc:
            if _is_overload_error(exc):
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    try:
                        response = active_client.models.generate_content(
                            model=fallback_model,
                            contents=prompt,
                        )
                        in_tok, out_tok = _extract_usage_tokens(response)
                        return response.text.strip(), fallback_model, in_tok, out_tok
                    except Exception as fallback_error:
                        raise RuntimeError(
                            "Complete pipeline failure. Both primary and fallback models overloaded: "
                            f"{fallback_error}"
                        ) from fallback_error
            else:
                raise

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

        except errors.ServerError as exc:
            if _is_overload_error(exc) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc

    raise RuntimeError("Failed to generate embedding after retry loop.")
