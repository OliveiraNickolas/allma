"""
Remote provider resolution: query OpenCode, OpenClaw, etc. for model metadata.
"""
import json
from typing import Optional, Dict, Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from core.config import logger, ENV_CREDENTIALS


class ProviderResolver:
    """Base class for remote provider integration."""

    def __init__(self, provider: str):
        self.provider = provider

    def validate_model(self, model_id: str) -> bool:
        """Check if model exists on provider."""
        raise NotImplementedError

    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Fetch model context window, max_tokens, etc."""
        raise NotImplementedError


class OpenCodeResolver(ProviderResolver):
    """Resolve models via OpenCode API."""

    def __init__(self):
        super().__init__("opencode")
        self.base_url = ENV_CREDENTIALS.get("OPENCODE_BASE_URL", "https://api.opencode.io/v1")
        self.api_key = ENV_CREDENTIALS.get("OPENCODE_API_KEY")

    def validate_model(self, model_id: str) -> bool:
        if not self.api_key:
            logger.warning("OPENCODE_API_KEY not set; skipping model validation")
            return True  # Assume valid if no key, fail at runtime
        try:
            # Quick HEAD request to check availability
            url = f"{self.base_url}/models/{model_id}"
            req = Request(url, headers={"Authorization": f"Bearer {self.api_key}"})
            req.get_method = lambda: "HEAD"
            with urlopen(req, timeout=5) as r:
                return r.status == 200
        except Exception as e:
            logger.debug(f"OpenCode validation failed for {model_id}: {e}")
            return False

    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata from OpenCode API."""
        if not self.api_key:
            logger.warning("OPENCODE_API_KEY not set")
            return None
        try:
            url = f"{self.base_url}/models/{model_id}"
            req = Request(url, headers={"Authorization": f"Bearer {self.api_key}"})
            with urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
                return {
                    "model_id": model_id,
                    "context_window": data.get("context_length", 4096),
                    "max_tokens": data.get("max_output_tokens", 2048),
                    "provider": "opencode",
                }
        except Exception as e:
            logger.debug(f"Failed to fetch OpenCode metadata for {model_id}: {e}")
            return None


class OpenClawResolver(ProviderResolver):
    """Resolve models via OpenClaw API."""

    def __init__(self):
        super().__init__("openclaw")
        self.base_url = ENV_CREDENTIALS.get("OPENCLAW_BASE_URL", "https://api.openclaw.ai/v1")
        self.api_key = ENV_CREDENTIALS.get("OPENCLAW_API_KEY")

    def validate_model(self, model_id: str) -> bool:
        """OpenClaw proxies to other providers; usually always valid if key exists."""
        return bool(self.api_key)

    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata from OpenClaw."""
        if not self.api_key:
            logger.warning("OPENCLAW_API_KEY not set")
            return None
        try:
            url = f"{self.base_url}/models"
            req = Request(url, headers={"Authorization": f"Bearer {self.api_key}"})
            with urlopen(req, timeout=10) as r:
                models = json.loads(r.read()).get("data", [])
                for m in models:
                    if m.get("id") == model_id:
                        return {
                            "model_id": model_id,
                            "context_window": m.get("context_window", 4096),
                            "max_tokens": m.get("max_tokens", 2048),
                            "provider": "openclaw",
                        }
            return None
        except Exception as e:
            logger.debug(f"Failed to fetch OpenClaw metadata for {model_id}: {e}")
            return None


def get_resolver(provider: str) -> Optional[ProviderResolver]:
    """Factory: get resolver for provider name."""
    if provider == "opencode":
        return OpenCodeResolver()
    elif provider == "openclaw":
        return OpenClawResolver()
    return None


def resolve_remote_model(provider: str, model_id: str, skip_validation: bool = False) -> Optional[Dict[str, Any]]:
    """
    Resolve a remote model: check existence + fetch metadata.
    Returns model config dict or None if not found.

    Args:
        provider: Provider name (opencode, openclaw)
        model_id: Model identifier
        skip_validation: If True, skip API validation (for testing/demo)
    """
    resolver = get_resolver(provider)
    if not resolver:
        logger.error(f"Unknown provider: {provider}")
        return None

    # Try validation unless skipped
    if not skip_validation:
        if not resolver.validate_model(model_id):
            logger.error(f"Model {model_id} not found on {provider}")
            return None
    else:
        logger.info(f"Skipping validation for {model_id} (demo mode)")

    # Try to fetch metadata
    metadata = resolver.get_model_metadata(model_id)
    if not metadata:
        # Build minimal metadata if fetch fails
        logger.warning(f"Could not fetch metadata for {model_id}; using defaults")
        metadata = {
            "model_id": model_id,
            "context_window": 4096,
            "max_tokens": 2048,
            "provider": provider,
        }

    return metadata
