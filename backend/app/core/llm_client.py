"""Google GenAI client with 3-attempt 503 exponential backoff and fallback routing."""

import os
import time

from google import genai
from google.genai import errors

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


def generate_with_retry_with_model(
    prompt: str,
    max_retries: int | None = None,
    client: genai.Client | None = None,
) -> tuple[str, str]:
    """Generate text using primary model with 503 exponential backoff and fallback model."""
    retries = max_retries if max_retries is not None else settings.max_retries
    active_client = client or get_genai_client()

    for attempt in range(retries):
        try:
            response = active_client.models.generate_content(
                model=PRIMARY_MODEL,
                contents=prompt,
            )
            return response.text.strip(), PRIMARY_MODEL

        except errors.ServerError as exc:
            if "503" in str(exc) or "UNAVAILABLE" in str(exc):
                if attempt < retries - 1:
                    sleep_time = 2 ** attempt
                    time.sleep(sleep_time)
                else:
                    try:
                        response = active_client.models.generate_content(
                            model=FALLBACK_MODEL,
                            contents=prompt,
                        )
                        return response.text.strip(), FALLBACK_MODEL
                    except Exception as fallback_error:
                        raise RuntimeError(
                            "Complete pipeline failure. Both primary and fallback models overloaded: "
                            f"{fallback_error}"
                        ) from fallback_error
            else:
                raise

    raise RuntimeError("Failed to generate response after retry loop.")


def generate_with_retry(prompt: str, max_retries: int = 3) -> str:
    """Return generated text while preserving the legacy string-only signature."""
    text, _ = generate_with_retry_with_model(prompt, max_retries=max_retries)
    return text
