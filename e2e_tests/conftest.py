"""Fixtures for Custom Conversation E2E tests."""
import asyncio
import dataclasses
import os
from pathlib import Path

from dotenv import load_dotenv
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_socket import enable_socket, socket_allow_hosts

from custom_components.custom_conversation.const import (
    CONF_AGENTS_SECTION,
    CONF_ENABLE_HASS_AGENT,
    CONF_ENABLE_LANGFUSE,
    CONF_ENABLE_LLM_AGENT,
    CONF_LANGFUSE_API_PROMPT_ID,
    CONF_LANGFUSE_BASE_PROMPT_ID,
    CONF_LANGFUSE_HOST,
    CONF_LANGFUSE_PUBLIC_KEY,
    CONF_LANGFUSE_SECRET_KEY,
    CONF_LANGFUSE_SECTION,
    CONF_LANGFUSE_TRACING_ENABLED,
    CONF_PRIMARY_API_KEY,
    CONF_PRIMARY_BASE_URL,
    CONF_PRIMARY_CHAT_MODEL,
    CONF_PRIMARY_PROVIDER,
    CONFIG_VERSION,
    DOMAIN,
    LLM_API_ID,
)
from custom_components.custom_conversation.providers import (
    LiteLLMProvider,
    get_provider,
)
from homeassistant.components import switch
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component

env_path = Path(__file__).parent / ".env"
if env_path.is_file():
    load_dotenv(dotenv_path=env_path, verbose=True)

ALLOWED_HOSTS = ["api.openai.com", "generativelanguage.googleapis.com", "openrouter.ai", "mistral.ai", "api.anthropic.com"]
LANGFUSE_HOST_ENV = os.getenv("LANGFUSE_HOST")
if LANGFUSE_HOST_ENV:
    LANGFUSE_HOSTNAME = LANGFUSE_HOST_ENV.split("://")[-1].split("/")[0]
    ALLOWED_HOSTS.append(LANGFUSE_HOSTNAME)
else:
    LANGFUSE_HOSTNAME = None


@pytest.hookimpl(trylast=True)
def pytest_runtest_setup():
    """Enable socket access and allow specific hosts for E2E tests."""
    enable_socket()
    socket_allow_hosts([*ALLOWED_HOSTS, "localhost", "127.0.0.1", "::1", "localhost:11434"], allow_unix_socket=True)


@dataclasses.dataclass
class LLMProviderConfig:
    """Configuration for an LLM provider."""

    id: str
    model_env_var: str
    api_key_env_var: str
    base_url_env_var: str | None = None
    default_model: str | None = None
    default_base_url: str | None = None
    primary_provider: str | None = None

    def get_primary_provider(self) -> LiteLLMProvider:
        """Get the primary provider for this configuration."""
        return get_provider(self.primary_provider)

    def get_api_key(self) -> str:
        """Get API key, skipping test if not found."""
        key = os.getenv(self.api_key_env_var)
        if not key:
             pytest.skip(f"{self.api_key_env_var} environment variable not set. Skipping {self.id} E2E tests.")
        return key

    def get_model(self) -> str:
        """Get model name from env or use default."""
        return os.getenv(self.model_env_var, self.default_model)

    def get_base_url(self) -> str | None:
        """Get base URL from env or use default."""

        if self.base_url_env_var:
            return os.getenv(self.base_url_env_var, self.default_base_url)
        return self.default_base_url

    def get_mock_config_entry_data(self) -> dict:
        """Generate the data dict for MockConfigEntry (API Key, Base URL)."""
        data = {
            CONF_PRIMARY_API_KEY: self.get_api_key(),
            CONF_PRIMARY_CHAT_MODEL: self.get_model(),
            CONF_PRIMARY_PROVIDER: self.get_primary_provider().key,
        }
        base_url = self.get_base_url()
        if base_url:
            data[CONF_PRIMARY_BASE_URL] = base_url
        return data

    def get_mock_config_entry_options(self) -> dict:
        """Generate the options dict for MockConfigEntry (Model, etc.)."""
        return {}


SUPPORTED_PROVIDERS = [
    LLMProviderConfig(
        id="openai",
        api_key_env_var="OPENAI_API_KEY",
        model_env_var="OPENAI_MODEL",
        base_url_env_var="OPENAI_BASE_URL",
        default_base_url="https://api.openai.com/v1",
        default_model="gpt-4o",
        primary_provider="openai"
    ),
    LLMProviderConfig(
        id="google_gemini_openai_compat",
        api_key_env_var="GEMINI_API_KEY",
        model_env_var="GEMINI_OPENAI_MODEL",
        base_url_env_var="GEMINI_OPENAI_BASE_URL",
        default_base_url=None,
        default_model="gemini-1.5-flash-latest",
        primary_provider="openai"
    ),
    LLMProviderConfig(
        id="google_gemini",
        api_key_env_var="GEMINI_API_KEY",
        model_env_var="GEMINI_MODEL",
        base_url_env_var="GEMINI_BASE_URL",
        default_base_url=None,
        default_model="gemini-1.5-flash-latest",
        primary_provider="gemini"
    ),
    LLMProviderConfig(
        id="ollama",
        api_key_env_var="OLLAMA_API_KEY",
        model_env_var="OLLAMA_MODEL",
        base_url_env_var="OLLAMA_BASE_URL_PATH",
        default_base_url=None,
        default_model="llama2-7b-chat",
        primary_provider="ollama"
    ),
    LLMProviderConfig(
        id="ollama_chat",
        api_key_env_var="OLLAMA_API_KEY",
        model_env_var="OLLAMA_CHAT_MODEL",
        base_url_env_var="OLLAMA_BASE_URL_PATH",
        default_base_url=None,
        default_model="llama2-7b-chat",
        primary_provider="ollama_chat"
    ),
    LLMProviderConfig(
        id="ollama_openai_compat",
        api_key_env_var="OLLAMA_API_KEY",
        model_env_var="OLLAMA_MODEL",
        base_url_env_var="OLLAMA_BASE_URL_OPENAI_PATH",
        default_base_url=None,
        default_model="llama2-7b-chat",
        primary_provider="openai"
    ),
    LLMProviderConfig(
        id="openrouter",
        api_key_env_var="OPENROUTER_API_KEY",
        model_env_var="OPENROUTER_MODEL",
        base_url_env_var="OPENROUTER_BASE_URL_PATH",
        default_base_url=None,
        default_model="google/gemini-2.0-flash-exp:free",
        primary_provider="openrouter"
    ),
    LLMProviderConfig(
        id="mistral",
        api_key_env_var="MISTRAL_API_KEY",
        model_env_var="MISTRAL_MODEL",
        base_url_env_var=None,
        default_base_url=None,
        default_model="mistral-large-latest",
        primary_provider="mistral"
    ),
    LLMProviderConfig(
        id="anthropic",
        api_key_env_var="ANTHROPIC_API_KEY",
        model_env_var="ANTHROPIC_MODEL",
        base_url_env_var=None,
        default_base_url=None,
        default_model="claude-opus-5-5",
        primary_provider="anthropic"
    )
]

@pytest.fixture(params=SUPPORTED_PROVIDERS, ids=[p.id for p in SUPPORTED_PROVIDERS])
def llm_config(request) -> LLMProviderConfig:
    """Fixture to provide LLM provider configuration, skipping if keys/URLs are missing."""
    config: LLMProviderConfig = request.param
    config.get_api_key()
    # For Gemini compat, base URL is essential
    if config.id == "google_gemini_openai_compat" and not config.get_base_url():
         pytest.skip(f"{config.base_url_env_var} environment variable not set. Skipping {config.id} E2E tests.")

    return config


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable custom integrations for E2E testing."""
    return

@dataclasses.dataclass
class LangfuseTestConfig:
    """Dataclass for Langfuse test configuration."""

    public_key: str
    secret_key: str
    host: str
    base_prompt_id: str
    api_prompt_id: str

@pytest.fixture(scope="session")
def langfuse_config() -> LangfuseTestConfig:
    """Fixture to provide Langfuse config, skipping if env vars missing."""

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    base_prompt_id = os.getenv("LANGFUSE_BASE_PROMPT_ID")
    api_prompt_id = os.getenv("LANGFUSE_API_PROMPT_ID")

    langfuse_host = os.getenv("LANGFUSE_HOST")
    if not all([public_key, secret_key, base_prompt_id, api_prompt_id, langfuse_host]):
        pytest.skip("Required Langfuse environment variables (PUBLIC_KEY, SECRET_KEY, HOST, BASE_PROMPT_ID, API_PROMPT_ID) not set. Skipping Langfuse tests.")

    return LangfuseTestConfig(
        public_key=public_key,
        secret_key=secret_key,
        host=langfuse_host,
        base_prompt_id=base_prompt_id,
        api_prompt_id=api_prompt_id,
    )


@pytest.fixture
async def setup_config_entry(hass: HomeAssistant, llm_config: LLMProviderConfig):
    """Set up the component with the given LLM config and handle teardown.

    Yields the MockConfigEntry.
    """
    config_data = llm_config.get_mock_config_entry_data()
    config_options = llm_config.get_mock_config_entry_options()

    common_test_options = {
        CONF_LLM_HASS_API: LLM_API_ID,
        CONF_AGENTS_SECTION: {
            CONF_ENABLE_HASS_AGENT: False,
            CONF_ENABLE_LLM_AGENT: True,
        },
        CONF_LANGFUSE_SECTION: {
            CONF_ENABLE_LANGFUSE: False,
            CONF_LANGFUSE_TRACING_ENABLED: False,
        }
    }
    final_options = {**config_options, **common_test_options}

    entry = MockConfigEntry(
        domain=DOMAIN,
        version=CONFIG_VERSION,
        title=f"{llm_config.id} E2E Test",
        data=config_data,
        options=final_options,
        entry_id=f"{llm_config.id}_e2e_test_entry",
    )
    entry.add_to_hass(hass)

    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})
    assert await async_setup_component(hass, switch.DOMAIN, {})
    assert await async_setup_component(hass, DOMAIN, {})
    await hass.async_block_till_done()

    yield entry

    await hass.config_entries.async_unload(entry.entry_id)
    await asyncio.sleep(0.1)
    await hass.async_block_till_done()

@pytest.fixture
async def langfuse_enabled_entry(
    hass: HomeAssistant,
    setup_config_entry: MockConfigEntry,
    langfuse_config: LangfuseTestConfig,
):
    """Set up the component with Langfuse enabled and yields the entry."""

    entry = setup_config_entry
    updated_options = {
        **entry.options,
        CONF_LANGFUSE_SECTION: {
            CONF_ENABLE_LANGFUSE: True,
            CONF_LANGFUSE_TRACING_ENABLED: True,
            CONF_LANGFUSE_PUBLIC_KEY: langfuse_config.public_key,
            CONF_LANGFUSE_SECRET_KEY: langfuse_config.secret_key,
            CONF_LANGFUSE_HOST: langfuse_config.host,
            CONF_LANGFUSE_BASE_PROMPT_ID: langfuse_config.base_prompt_id,
            CONF_LANGFUSE_API_PROMPT_ID: langfuse_config.api_prompt_id,
        }
    }

    hass.config_entries.async_update_entry(entry, options=updated_options)
    await hass.async_block_till_done()
    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    return entry

@pytest.fixture
async def mock_test_switch(hass: HomeAssistant):
    """Set up a mock switch.test_switch entity."""
    hass.states.async_set("switch.test_switch", "off", {"friendly_name": "Test Switch"})
    await hass.async_block_till_done()
