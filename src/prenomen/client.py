"""Client for the OpenAI-compatible endpoint."""

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

FALLBACK_ENV = "OPENAI_API_KEY"


def resolve_api_key(api_key: str | None = None, api_key_env: str | None = None) -> str:
    """Return the API key, preferring the environment over a literal.

    ``api_key_env`` names an environment variable and is the way to pass a real
    credential, so keys stay out of files that get committed; naming one that is
    unset is an error rather than a silent fallback. A literal ``api_key`` is still
    accepted, which is what local servers that ignore authentication want -- as is
    passing neither, which falls back to ``$OPENAI_API_KEY`` and then to a dummy.
    """
    if api_key_env:
        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError(
                f"api_key_env names {api_key_env!r} but that environment variable "
                f"is unset. Export it, or pass a literal api_key for a local "
                f"endpoint that ignores authentication."
            )
        return key
    if api_key:
        return api_key
    return os.environ.get(FALLBACK_ENV) or "dummy"


def build_client(
    base_url: str,
    api_key: str,
    *,
    timeout: float = 120.0,
    max_retries: int = 2,
) -> OpenAI:
    """Build the OpenAI client for one endpoint.

    ``timeout`` and ``max_retries`` are the SDK's own, so a retry covers a timeout as
    well as an error status. Constructing this opens no connection.
    """
    if not base_url or not isinstance(base_url, str):
        raise ValueError(
            "base_url must be the endpoint to call, for example "
            '"https://api.openai.com/v1" or "http://localhost:8000/v1".'
        )
    base_url = base_url.rstrip("/")
    logger.info(f"Endpoint: {base_url}")
    return OpenAI(
        base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries
    )
