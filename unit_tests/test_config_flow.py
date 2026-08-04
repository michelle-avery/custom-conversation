"""Test the Custom Conversation config flow."""
from unittest.mock import patch

from litellm.exceptions import APIConnectionError, AuthenticationError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.custom_conversation.config_flow import DEFAULT_OPTIONS
from custom_components.custom_conversation.const import (
    CONF_AGENTS_SECTION,
    CONF_API_PROMPT_BASE,
    CONF_CUSTOM_PROMPTS_SECTION,
    CONF_ENABLE_HASS_AGENT,
    CONF_ENABLE_LANGFUSE,
    CONF_ENABLE_LLM_AGENT,
    CONF_IGNORED_INTENTS,
    CONF_IGNORED_INTENTS_SECTION,
    CONF_INSTRUCTIONS_PROMPT,
    CONF_LANGFUSE_API_PROMPT_ID,
    CONF_LANGFUSE_BASE_PROMPT_ID,
    CONF_LANGFUSE_HOST,
    CONF_LANGFUSE_PUBLIC_KEY,
    CONF_LANGFUSE_SECRET_KEY,
    CONF_LANGFUSE_SECTION,
    CONF_LANGFUSE_TRACING_ENABLED,
    CONF_MAX_TOKENS,
    CONF_PRIMARY_API_KEY,
    CONF_PRIMARY_BASE_URL,
    CONF_PRIMARY_CHAT_MODEL,
    CONF_PRIMARY_PROVIDER,
    CONF_PROMPT_BASE,
    CONF_PROMPT_DEVICE_KNOWN_LOCATION,
    CONF_PROMPT_DEVICE_UNKNOWN_LOCATION,
    CONF_PROMPT_EXPOSED_ENTITIES,
    CONF_PROMPT_NO_ENABLED_ENTITIES,
    CONF_PROMPT_TIMERS_UNSUPPORTED,
    CONF_SECONDARY_PROVIDER_ENABLED,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONFIGURING_SECONDARY_PROVIDER,
    DEFAULT_API_PROMPT_BASE,
    DEFAULT_API_PROMPT_DEVICE_KNOWN_LOCATION,
    DEFAULT_API_PROMPT_DEVICE_UNKNOWN_LOCATION,
    DEFAULT_API_PROMPT_EXPOSED_ENTITIES,
    DEFAULT_API_PROMPT_TIMERS_UNSUPPORTED,
    DEFAULT_BASE_PROMPT,
    DEFAULT_INSTRUCTIONS_PROMPT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT_NO_ENABLED_ENTITIES,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DOMAIN,
)
from homeassistant import config_entries
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import llm


async def test_config_flow_user_steps_openai(hass: HomeAssistant):
    """Test the full user configuration flow for openai provider."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"
    assert not result["errors"]

    with patch(
        "custom_components.custom_conversation.config_flow.CustomConversationConfigFlow._validate_credentials_and_get_models",
        return_value=["gpt-4o-mini", "gpt-4-turbo"],
    ), patch(
        "custom_components.custom_conversation.async_setup_entry",
        return_value=True,
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_PRIMARY_PROVIDER: "openai"}
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "credentials"
        assert not result["errors"]
        assert CONF_PRIMARY_API_KEY in result["data_schema"].schema
        assert CONF_PRIMARY_BASE_URL in result["data_schema"].schema

        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_PRIMARY_API_KEY: "sk-testkey",
                CONF_PRIMARY_BASE_URL: "https://api.openai.com/v1",
            },
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "model"
        assert not result["errors"]
        assert CONF_PRIMARY_CHAT_MODEL in result["data_schema"].schema

        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_PRIMARY_CHAT_MODEL: "gpt-4o-mini"}
        )
        await hass.async_block_till_done()

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "Custom Conversation" # Default title
    assert result["data"] == {
        CONF_PRIMARY_PROVIDER: "openai",
        CONF_PRIMARY_API_KEY: "sk-testkey",
        CONF_PRIMARY_BASE_URL: "https://api.openai.com/v1",
        CONF_PRIMARY_CHAT_MODEL: "gpt-4o-mini",
        CONFIGURING_SECONDARY_PROVIDER: False,
        CONF_SECONDARY_PROVIDER_ENABLED: False,
    }
    assert result["options"] == DEFAULT_OPTIONS


async def test_config_flow_user_steps_gemini(hass: HomeAssistant):
    """Test the full user configuration flow for gemini provider."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"

    with patch(
        "custom_components.custom_conversation.config_flow.CustomConversationConfigFlow._validate_credentials_and_get_models",
        return_value=["gemini-pro", "gemini-1.5-pro-latest"],
    ), patch(
        "custom_components.custom_conversation.async_setup_entry",
        return_value=True,
    ):
        # Step 2: Configure provider (gemini) -> credentials step
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_PRIMARY_PROVIDER: "gemini"}
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "credentials"
        assert CONF_PRIMARY_API_KEY in result["data_schema"].schema
        # See https://github.com/BerriAI/litellm/issues/7830
        #assert CONF_PRIMARY_BASE_URL in result["data_schema"].schema

        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_PRIMARY_API_KEY: "gemini-key"}
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "model"

        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_PRIMARY_CHAT_MODEL: "gemini-1.5-pro-latest"}
        )
        await hass.async_block_till_done()

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"].get(CONF_PRIMARY_PROVIDER) == "gemini"
    assert result["data"].get(CONF_PRIMARY_API_KEY) == "gemini-key"
    # See https://github.com/BerriAI/litellm/issues/7830
    #assert result["data"].get(CONF_PRIMARY_BASE_URL) == "https://generativelanguage.googleapis.com",
    assert result["data"].get(CONF_PRIMARY_CHAT_MODEL) == "gemini-1.5-pro-latest"
    assert result["options"] == DEFAULT_OPTIONS


async def test_config_flow_validation_errors(hass: HomeAssistant):
    """Test credential validation errors during config flow."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_PRIMARY_PROVIDER: "openai"}
    )

    with patch(
        "custom_components.custom_conversation.config_flow.CustomConversationConfigFlow._validate_credentials_and_get_models",
        side_effect=AuthenticationError(message="Invalid API Key", llm_provider="openai", model="model"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_PRIMARY_API_KEY: "bad-key"}
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "credentials"
        assert result["errors"]["base"] == "invalid_auth"

    with patch(
        "custom_components.custom_conversation.config_flow.CustomConversationConfigFlow._validate_credentials_and_get_models",
        side_effect=APIConnectionError(message="Cannot connect", llm_provider="openai", model="model"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_PRIMARY_API_KEY: "good-key"} # Use same flow ID
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "credentials"
        assert result["errors"]["base"] == "cannot_connect"


async def test_options_flow(hass: HomeAssistant, config_entry: MockConfigEntry):
    """Test config flow options with sections."""

    config_entry.add_to_hass(hass)

    with patch(
        "custom_components.custom_conversation.async_setup", return_value=True
    ), patch(
        "custom_components.custom_conversation.async_setup_entry",
        return_value=True,
    ):
        result = await hass.config_entries.options.async_init(config_entry.entry_id)
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "init"

        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={
                CONF_LLM_HASS_API: [],
                CONF_IGNORED_INTENTS_SECTION: {
                    CONF_IGNORED_INTENTS: ["HassGetState"],
                },
                CONF_AGENTS_SECTION: {
                    CONF_ENABLE_HASS_AGENT: True,
                    CONF_ENABLE_LLM_AGENT: False,
                },
                CONF_TEMPERATURE: 0.5,
                CONF_TOP_P: 0.5,
                CONF_MAX_TOKENS: 50,
                CONF_CUSTOM_PROMPTS_SECTION: {
                    CONF_INSTRUCTIONS_PROMPT: "test-prompt",
                    CONF_PROMPT_BASE: "test-prompt-base",
                    CONF_PROMPT_NO_ENABLED_ENTITIES: "test-prompt-no-entities",
                    CONF_API_PROMPT_BASE: "test-api-prompt-base",
                    CONF_PROMPT_DEVICE_KNOWN_LOCATION: "test-prompt-known-location",
                    CONF_PROMPT_DEVICE_UNKNOWN_LOCATION: "test-prompt-unknown-location",
                    CONF_PROMPT_TIMERS_UNSUPPORTED: "test-prompt-timers-unsupported",
                    CONF_PROMPT_EXPOSED_ENTITIES: "test-prompt-exposed-entities",
                },
                CONF_LANGFUSE_SECTION: {
                    CONF_ENABLE_LANGFUSE: True,
                    CONF_LANGFUSE_SECRET_KEY: "sk-test-secret-key",
                    CONF_LANGFUSE_PUBLIC_KEY: "pk-test-public-key",
                    CONF_LANGFUSE_HOST: "http://langfuse.test",
                    CONF_LANGFUSE_BASE_PROMPT_ID: "test-base-prompt-id",
                    CONF_LANGFUSE_API_PROMPT_ID: "test-api-prompt-id",
                    CONF_LANGFUSE_TRACING_ENABLED: False
                }
            },
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert CONF_LLM_HASS_API not in result["data"]
        assert "agents" in result["data"]
        assert result["data"][CONF_IGNORED_INTENTS_SECTION][CONF_IGNORED_INTENTS] == ["HassGetState"]
        assert result["data"][CONF_AGENTS_SECTION][CONF_ENABLE_HASS_AGENT] is True
        assert result["data"][CONF_AGENTS_SECTION][CONF_ENABLE_LLM_AGENT] is False
        # Assert LLM params are direct options
        assert result["data"][CONF_TEMPERATURE] == 0.5
        assert result["data"][CONF_TOP_P] == 0.5
        assert result["data"][CONF_MAX_TOKENS] == 50
        # Assert other sections
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_INSTRUCTIONS_PROMPT] == "test-prompt"
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_BASE] == "test-prompt-base"
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_NO_ENABLED_ENTITIES] == "test-prompt-no-entities"
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_API_PROMPT_BASE] == "test-api-prompt-base"
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_DEVICE_KNOWN_LOCATION] == "test-prompt-known-location"
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_DEVICE_UNKNOWN_LOCATION] == "test-prompt-unknown-location"
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_TIMERS_UNSUPPORTED] == "test-prompt-timers-unsupported"
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_EXPOSED_ENTITIES] == "test-prompt-exposed-entities"
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_ENABLE_LANGFUSE] is True
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_SECRET_KEY] == "sk-test-secret-key"
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_PUBLIC_KEY] == "pk-test-public-key"
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_HOST] == "http://langfuse.test"
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_BASE_PROMPT_ID] == "test-base-prompt-id"
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_API_PROMPT_ID] == "test-api-prompt-id"

async def test_options_flow_empty_fields_reset(hass: HomeAssistant, config_entry):
    """Test config flow options with empty fields reset to recommended."""

    config_entry.add_to_hass(hass)

    with patch(
        "custom_components.custom_conversation.async_setup", return_value=True
    ), patch(
        "custom_components.custom_conversation.async_setup_entry",
        return_value=True,
    ):
        result = await hass.config_entries.options.async_init(config_entry.entry_id)
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "init"

        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={
                CONF_LLM_HASS_API: [],
                CONF_IGNORED_INTENTS_SECTION: {
                    CONF_IGNORED_INTENTS: [],
                },
                CONF_AGENTS_SECTION: {
                    CONF_ENABLE_HASS_AGENT: True,
                    CONF_ENABLE_LLM_AGENT: False,
                },
                CONF_TEMPERATURE: 0.5,
                CONF_TOP_P: 0.5,
                CONF_MAX_TOKENS: 50,
                CONF_CUSTOM_PROMPTS_SECTION: {
                    CONF_INSTRUCTIONS_PROMPT: "",
                    CONF_PROMPT_BASE: "",
                    CONF_PROMPT_NO_ENABLED_ENTITIES: "",
                    CONF_API_PROMPT_BASE: "",
                    CONF_PROMPT_DEVICE_KNOWN_LOCATION: "",
                    CONF_PROMPT_DEVICE_UNKNOWN_LOCATION: "",
                    CONF_PROMPT_TIMERS_UNSUPPORTED: "",
                    CONF_PROMPT_EXPOSED_ENTITIES: "",
                },
                CONF_LANGFUSE_SECTION: {
                    CONF_ENABLE_LANGFUSE: False,
                    CONF_LANGFUSE_SECRET_KEY: "",
                    CONF_LANGFUSE_PUBLIC_KEY: "",
                    CONF_LANGFUSE_HOST: "",
                    CONF_LANGFUSE_BASE_PROMPT_ID: "",
                    CONF_LANGFUSE_API_PROMPT_ID: "",
                    CONF_LANGFUSE_TRACING_ENABLED: False
                }

            },
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert CONF_LLM_HASS_API not in result["data"]
        assert "agents" in result["data"]
        assert result["data"][CONF_IGNORED_INTENTS_SECTION][CONF_IGNORED_INTENTS] == llm.AssistAPI.IGNORE_INTENTS
        assert result["data"][CONF_AGENTS_SECTION][CONF_ENABLE_HASS_AGENT] is True
        assert result["data"][CONF_AGENTS_SECTION][CONF_ENABLE_LLM_AGENT] is False
        # Assert LLM params are direct options
        assert result["data"][CONF_TEMPERATURE] == 0.5
        assert result["data"][CONF_TOP_P] == 0.5
        assert result["data"][CONF_MAX_TOKENS] == 50
        # Assert other sections
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_INSTRUCTIONS_PROMPT].strip() == DEFAULT_INSTRUCTIONS_PROMPT.strip()
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_BASE] == DEFAULT_BASE_PROMPT
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_NO_ENABLED_ENTITIES] == DEFAULT_PROMPT_NO_ENABLED_ENTITIES
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_API_PROMPT_BASE] == DEFAULT_API_PROMPT_BASE
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_DEVICE_KNOWN_LOCATION] == DEFAULT_API_PROMPT_DEVICE_KNOWN_LOCATION
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_DEVICE_UNKNOWN_LOCATION] == DEFAULT_API_PROMPT_DEVICE_UNKNOWN_LOCATION
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_TIMERS_UNSUPPORTED] == DEFAULT_API_PROMPT_TIMERS_UNSUPPORTED
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_EXPOSED_ENTITIES] == DEFAULT_API_PROMPT_EXPOSED_ENTITIES
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_ENABLE_LANGFUSE] is False
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_SECRET_KEY] == ""
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_PUBLIC_KEY] == ""
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_HOST] == ""
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_BASE_PROMPT_ID] == ""
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_API_PROMPT_ID] == ""
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_TRACING_ENABLED] is False


async def test_options_flow_ignored_intents(hass: HomeAssistant, config_entry):
    """Test config flow options with ignored intents."""

    config_entry.add_to_hass(hass)
    with patch(
        "custom_components.custom_conversation.async_setup", return_value=True
    ), patch(
        "custom_components.custom_conversation.async_setup_entry",
        return_value=True,
    ), patch(
        "homeassistant.helpers.intent.async_get",
        return_value=[
            type("Intent", (), {"intent_type": "HassTurnOn"}),
            type("Intent", (), {"intent_type": "HassTurnOff"}),
        ],
    ):
        result = await hass.config_entries.options.async_init(config_entry.entry_id)
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "init"

        test_intents = ["HassTurnOn", "HassTurnOff"]
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={
                CONF_IGNORED_INTENTS_SECTION: {
                    CONF_IGNORED_INTENTS: test_intents,
                },
                CONF_AGENTS_SECTION: {
                    CONF_ENABLE_HASS_AGENT: True,
                    CONF_ENABLE_LLM_AGENT: True,
                },
                CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
                CONF_TOP_P: DEFAULT_TOP_P,
                CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
                CONF_CUSTOM_PROMPTS_SECTION: {},
                CONF_LANGFUSE_SECTION: {}
            },
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_IGNORED_INTENTS_SECTION][CONF_IGNORED_INTENTS] == test_intents



async def test_reconfigure_flow_steps(hass: HomeAssistant, config_entry: MockConfigEntry):
    """Test the full multi-step reconfigure flow."""
    config_entry.add_to_hass(hass)

    with patch(
        "custom_components.custom_conversation.config_flow.CustomConversationConfigFlow._validate_credentials_and_get_models",
        return_value=["new-model-1", "new-model-2"],
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={
                "source": config_entries.SOURCE_RECONFIGURE,
                "entry_id": config_entry.entry_id,
            },
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "reconfigure_provider"

        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_PRIMARY_PROVIDER: "gemini"}
        )
        await hass.async_block_till_done()
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "reconfigure_credentials"
        assert result["description_placeholders"]["provider"] == "Gemini - Google AI Studio"


        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_PRIMARY_API_KEY: "new-gemini-key"}
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "reconfigure_model"

        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_PRIMARY_CHAT_MODEL: "new-model-1"}
        )
        await hass.async_block_till_done()

    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"