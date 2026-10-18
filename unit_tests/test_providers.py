"""Unit tests for the providers module."""

from unittest.mock import MagicMock, patch

import pytest

from custom_components.custom_conversation.providers import (
    SUPPORTED_PROVIDERS,
    AnthropicProvider,
    GeminiProvider,
    anthropic,
    LiteLLMProvider,
    gemini,
    get_provider,
    mistral,
    openai,
    openrouter,
)


@pytest.fixture
def mock_requests_get():
    """Fixture to mock requests.get."""
    with patch("requests.get") as mock_get:
        yield mock_get

@pytest.fixture
def mock_provider_config_manager():
    """Fixture to mock ProviderConfigManager."""
    with patch("custom_components.custom_conversation.providers.ProviderConfigManager") as mock_pcm:
        yield mock_pcm



class TestLiteLLMProvider:
    """Tests for the LiteLLMProvider class."""

    def test_init_with_config_manager(self, mock_provider_config_manager):
        """Test LiteLLMProvider initialization using ProviderConfigManager."""
        mock_provider_config_manager.get_provider_model_info.return_value.get_api_base.return_value = "http://config.base.url"
        provider = LiteLLMProvider(key="test_provider", provider_name="Test Provider")
        assert provider.key == "test_provider"
        assert provider.provider_name == "Test Provider"
        assert provider.default_base_url == "http://config.base.url"
        assert provider.supports_custom_base_url is False
        mock_provider_config_manager.get_provider_model_info.assert_called_once_with(model="", provider="test_provider")

    def test_init_with_manual_base_url(self, mock_provider_config_manager):
        """Test LiteLLMProvider initialization with manual_default_base_url."""
        mock_provider_config_manager.get_provider_model_info.return_value = None
        provider = LiteLLMProvider(
            key="test_provider",
            provider_name="Test Provider",
            manual_default_base_url="http://manual.base.url",
        )
        assert provider.default_base_url == "http://manual.base.url"

    def test_get_supported_models_success(self, mock_requests_get):
        """Test successful retrieval of models."""
        provider = LiteLLMProvider(
            key="test",
            provider_name="Test",
            model_list_path="/models",
            manual_default_base_url="http://test.url"
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "model1"}, {"id": "model2"}]}
        mock_requests_get.return_value = mock_response

        models = provider.get_supported_models(base_url=None, api_key="test_key")
        assert models == ["model1", "model2"]
        mock_requests_get.assert_called_once_with(
            "http://test.url/models",
            headers={"Authorization": "Bearer test_key"},
            timeout=5,
        )

    def test_get_supported_models_success_with_base_url(self, mock_requests_get):
        """Test successful retrieval with explicit base_url."""
        provider = LiteLLMProvider(
            key="test",
            provider_name="Test",
            model_list_path="/models",
            manual_default_base_url="http://default.url"
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "model3"}]}
        mock_requests_get.return_value = mock_response

        models = provider.get_supported_models(base_url="http://custom.url", api_key="test_key")
        assert models == ["model3"]
        mock_requests_get.assert_called_once_with(
            "http://custom.url/models",
            headers={"Authorization": "Bearer test_key"},
            timeout=5,
        )

    def test_get_supported_models_api_error(self, mock_requests_get, caplog):
        """Test handling of API errors."""
        provider = LiteLLMProvider(
            key="test",
            provider_name="Test",
            model_list_path="/models",
            manual_default_base_url="http://test.url"
        )
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_requests_get.return_value = mock_response

        models = provider.get_supported_models(base_url=None, api_key="test_key")
        assert models == []
        assert "Error fetching models: 404 - Not Found" in caplog.text

    def test_get_supported_models_json_error(self, mock_requests_get, caplog):
        """Test handling of JSON parsing errors."""
        provider = LiteLLMProvider(
            key="test",
            provider_name="Test",
            model_list_path="/models",
            manual_default_base_url="http://test.url"
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_requests_get.return_value = mock_response

        models = provider.get_supported_models(base_url=None, api_key="test_key")
        assert models == []
        assert "Error parsing JSON response" in caplog.text



class TestGeminiProvider:
    """Tests for the GeminiProvider class."""

    @patch("custom_components.custom_conversation.providers.ProviderConfigManager")
    def test_init(self, mock_pcm):
        """Test GeminiProvider initialization."""
        mock_pcm.get_provider_model_info.return_value.get_api_base.return_value = "https://generativelanguage.googleapis.com"
        provider = GeminiProvider()
        assert provider.key == "gemini"
        assert provider.provider_name == "Gemini - Google AI Studio"
        assert provider.model_list_path == "/v1beta/models"
        assert provider.supports_custom_base_url is False
        assert provider.default_base_url == "https://generativelanguage.googleapis.com"

    def test_get_supported_models_success(self, mock_requests_get):
        """Test successful retrieval of Gemini models."""
        provider = GeminiProvider()
        provider.default_base_url = "http://gemini.test"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "models/gemini-pro"},
                {"name": "models/gemini-ultra"},
                {"name": "models/other-model"},
            ]
        }
        mock_requests_get.return_value = mock_response

        models = provider.get_supported_models(base_url=None, api_key="gemini_key")
        assert models == ["gemini-pro", "gemini-ultra", "other-model"]
        mock_requests_get.assert_called_once_with(
            "http://gemini.test/v1beta/models?key=gemini_key",
            timeout=5,
        )

    def test_get_supported_models_api_error(self, mock_requests_get, caplog):
        """Test handling of API errors for Gemini."""
        provider = GeminiProvider()
        provider.default_base_url = "http://gemini.test"

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server Error"
        mock_requests_get.return_value = mock_response

        models = provider.get_supported_models(base_url=None, api_key="gemini_key")
        assert models == []
        assert "Error fetching models: 500 - Server Error" in caplog.text

    def test_get_supported_models_json_error(self, mock_requests_get, caplog):
        """Test handling of JSON parsing errors for Gemini."""
        provider = GeminiProvider()
        provider.default_base_url = "http://gemini.test"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Bad JSON")
        mock_requests_get.return_value = mock_response

        models = provider.get_supported_models(base_url=None, api_key="gemini_key")
        assert models == []
        assert "Error parsing JSON response" in caplog.text


class TestAnthropicProvider:
    """Tests for the AnthropicProvider class."""

    @patch("custom_components.custom_conversation.providers.ProviderConfigManager")
    def test_init(self, mock_pcm):
        """Test AnthropicProvider initialization."""
        mock_pcm.get_provider_model_info.return_value.get_api_base.return_value = "https://api.anthropic.com"
        provider = AnthropicProvider()
        assert provider.key == "anthropic"
        assert provider.provider_name == "Anthropic"
        assert provider.supports_custom_base_url is True
        assert provider.default_base_url == "https://api.anthropic.com"

    def test_get_supported_models_success(self, mock_requests_get):
        """Test successful retrieval of Anthropic models."""
        provider = AnthropicProvider()
        provider.default_base_url = "http://anthropic.test"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"id": "claude-opus-5-5"}, {"id": "claude-haiku-5-5"}],
            "has_more": False,
        }
        mock_requests_get.return_value = mock_response

        models = provider.get_supported_models(base_url=None, api_key="anthropic_key")
        assert models == ["claude-opus-5-5", "claude-haiku-5-5"]
        mock_requests_get.assert_called_once_with(
            "http://anthropic.test/v1/models?limit=1000",
            headers={"x-api-key": "anthropic_key", "anthropic-version": "2023-06-01"},
            timeout=5,
        )


def test_get_provider_exists():
    """Test retrieving an existing provider."""
    assert get_provider("openai") is openai
    assert get_provider("gemini") is gemini
    assert get_provider("openrouter") is openrouter
    assert get_provider("anthropic") is anthropic

def test_get_provider_not_exists():
    """Test retrieving a non-existent provider."""
    assert get_provider("non_existent_provider") is None


def test_provider_instances():
    """Test basic attributes of the global provider instances."""
    assert openai.key == "openai"
    assert openai.provider_name == "OpenAI"
    assert openai.supports_custom_base_url is True

    assert gemini.key == "gemini"
    assert gemini.provider_name == "Gemini - Google AI Studio"
    assert gemini.supports_custom_base_url is False

    assert openrouter.key == "openrouter"
    assert openrouter.provider_name == "OpenRouter"
    assert openrouter.supports_custom_base_url is True
    assert openrouter.default_base_url == "https://openrouter.ai/api/v1"

    assert anthropic.key == "anthropic"
    assert anthropic.provider_name == "Anthropic"
    assert anthropic.supports_custom_base_url is True
    assert anthropic.default_base_url == "https://api.anthropic.com"

def test_supported_providers_list():
    """Check if the SUPPORTED_PROVIDERS list contains the expected instances."""
    expected_providers = [openai, gemini, openrouter, mistral, anthropic]
    assert expected_providers == SUPPORTED_PROVIDERS

