"""Configuration for supported LiteLLM providers."""

from litellm.utils import ProviderConfigManager
import requests

from .const import LOGGER


class LiteLLMProvider:
    """Configuration for supported LiteLLM providers."""

    def __init__(
        self,
        key: str,
        provider_name: str,
        manual_default_base_url: str | None = None,
        model_list_path: str | None = None,
        supports_custom_base_url: bool = False,
    ) -> None:
        """Initialize the provider."""
        self.key = key
        self.provider_name = provider_name
        self.model_list_path = model_list_path
        self.default_base_url = None
        self.supports_custom_base_url = supports_custom_base_url
        provider_model_info = ProviderConfigManager.get_provider_model_info(
            model="", provider=self.key
        )
        if provider_model_info:
            self.default_base_url = provider_model_info.get_api_base()
        elif manual_default_base_url:
            self.default_base_url = manual_default_base_url

    def model_list_headers(self, api_key: str | None) -> dict[str, str]:
        """Get the headers for the model list request."""
        return {"Authorization": f"Bearer {api_key}"}

    def get_supported_models(
        self, base_url: str | None, api_key: str | None
    ) -> list[str]:
        """Get the supported models directly from the provider."""
        # Get the default base URL if not provided
        models = []
        if not base_url:
            base_url = self.default_base_url

        response = requests.get(
            f"{base_url}{self.model_list_path}",
            headers=self.model_list_headers(api_key),
            timeout=5,
        )
        if response.status_code == 200:
            try:
                models = response.json().get("data", [])
                return [model["id"] for model in models]
            except ValueError:
                LOGGER.error("Error parsing JSON response")
        else:
            LOGGER.error(
                "Error fetching models: %s - %s",
                response.status_code,
                response.text,
            )
        return models


class GeminiProvider(LiteLLMProvider):
    """Configuration for Gemini provider."""

    def __init__(self) -> None:
        """Initialize the provider."""
        super().__init__(
            key="gemini",
            provider_name="Gemini - Google AI Studio",
            model_list_path="/v1beta/models",
            supports_custom_base_url=False,  # See: https://github.com/BerriAI/litellm/issues/7830
        )

    def get_supported_models(
        self, base_url: str | None, api_key: str | None
    ) -> list[str]:
        """Get the supported models directly from the provider."""
        # Get the default base URL if not provided
        if not base_url:
            base_url = self.default_base_url

        response = requests.get(
            f"{base_url}{self.model_list_path}?key={api_key}",
            timeout=5,
        )
        if response.status_code == 200:
            try:
                models = response.json().get("models", [])
                # Gemini models are prepended with "models/" which we need to remove
                for model in models:
                    model["name"] = model["name"].replace("models/", "")
                return [model["name"] for model in models]
            except ValueError:
                LOGGER.error("Error parsing JSON response")
        else:
            LOGGER.error(
                "Error fetching models: %s - %s",
                response.status_code,
                response.text,
            )
        return []


class AnthropicProvider(LiteLLMProvider):
    """Configuration for Anthropic provider."""

    def __init__(self) -> None:
        """Initialize the provider."""
        super().__init__(
            key="anthropic",
            provider_name="Anthropic",
            model_list_path="/v1/models?limit=1000",
            supports_custom_base_url=True,
        )

    def model_list_headers(self, api_key: str | None) -> dict[str, str]:
        """Get the headers for the model list request."""
        return {"x-api-key": api_key or "", "anthropic-version": "2023-06-01"}


openai = LiteLLMProvider(
    key="openai",
    provider_name="OpenAI",
    model_list_path="/models",
    supports_custom_base_url=True,
)

gemini = GeminiProvider()

openrouter = LiteLLMProvider(
    key="openrouter",
    provider_name="OpenRouter",
    model_list_path="/models",
    supports_custom_base_url=True,
    manual_default_base_url="https://openrouter.ai/api/v1",
)

mistral = LiteLLMProvider(
    key="mistral",
    provider_name="Mistral",
    model_list_path="/models",
    supports_custom_base_url=True,
    manual_default_base_url="https://api.mistral.ai/v1",
)

anthropic = AnthropicProvider()

ollama = LiteLLMProvider(
    key="ollama",
    provider_name="Ollama",
    supports_custom_base_url=True,
)

ollama_chat = LiteLLMProvider(
    key="ollama_chat",
    provider_name="Ollama Chat",
    supports_custom_base_url=True,
)

SUPPORTED_PROVIDERS = [
    openai,
    gemini,
    openrouter,
    mistral,
    anthropic,
    # ollama, disabled pending litellm fixes for https://github.com/BerriAI/litellm/issues/6135 and https://github.com/BerriAI/litellm/issues/9602
    # ollama_chat
]


def get_provider(provider_key: str) -> LiteLLMProvider | None:
    """Get the provider by key."""
    for provider in SUPPORTED_PROVIDERS:
        if provider.key == provider_key:
            return provider
    return None
