"""Config flow for Custom Conversation integration."""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any

import openai
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import section
from homeassistant.helpers import intent, llm
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
)
from homeassistant.helpers.typing import VolDictType

from .const import (
    CONF_AGENTS_SECTION,
    CONF_API_PROMPT_BASE,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_CUSTOM_PROMPTS_SECTION,
    CONF_ENABLE_HASS_AGENT,
    CONF_ENABLE_LANGFUSE,
    CONF_ENABLE_LLM_AGENT,
    CONF_IGNORED_INTENTS,
    CONF_IGNORED_INTENTS_SECTION,
    CONF_INSTRUCTIONS_PROMPT,
    CONF_LANGFUSE_API_PROMPT_ID,
    CONF_LANGFUSE_API_PROMPT_LABEL,
    CONF_LANGFUSE_BASE_PROMPT_ID,
    CONF_LANGFUSE_BASE_PROMPT_LABEL,
    CONF_LANGFUSE_HOST,
    CONF_LANGFUSE_PUBLIC_KEY,
    CONF_LANGFUSE_SCORE_ENABLED,
    CONF_LANGFUSE_SECRET_KEY,
    CONF_LANGFUSE_SECTION,
    CONF_LANGFUSE_TAGS,
    CONF_LANGFUSE_TRACING_ENABLED,
    CONF_LLM_PARAMETERS_SECTION,
    CONF_MAX_TOKENS,
    CONF_PROMPT_BASE,
    CONF_PROMPT_DEVICE_KNOWN_LOCATION,
    CONF_PROMPT_DEVICE_UNKNOWN_LOCATION,
    CONF_PROMPT_EXPOSED_ENTITIES,
    CONF_PROMPT_NO_ENABLED_ENTITIES,
    CONF_PROMPT_TIMERS_UNSUPPORTED,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_API_PROMPT_BASE,
    DEFAULT_API_PROMPT_DEVICE_KNOWN_LOCATION,
    DEFAULT_API_PROMPT_DEVICE_UNKNOWN_LOCATION,
    DEFAULT_API_PROMPT_EXPOSED_ENTITIES,
    DEFAULT_API_PROMPT_TIMERS_UNSUPPORTED,
    DEFAULT_BASE_PROMPT,
    DEFAULT_INSTRUCTIONS_PROMPT,
    DEFAULT_PROMPT_NO_ENABLED_ENTITIES,
    DOMAIN,
    LLM_API_ID,
    RECOMMENDED_BASE_URL,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
        vol.Required(CONF_BASE_URL, default=RECOMMENDED_BASE_URL): str,
    }
)
DEFAULT_IGNORED_INTENTS = llm.AssistAPI.IGNORE_INTENTS

RECOMMENDED_OPTIONS = {
    CONF_LLM_HASS_API: LLM_API_ID,
    CONF_INSTRUCTIONS_PROMPT: DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_AGENTS_SECTION: {
        CONF_ENABLE_HASS_AGENT: True,
        CONF_ENABLE_LLM_AGENT: True,
    },
    CONF_LLM_PARAMETERS_SECTION: {
        CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
        CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
        CONF_TOP_P: RECOMMENDED_TOP_P,
        CONF_TEMPERATURE: RECOMMENDED_TEMPERATURE,
    },
    CONF_IGNORED_INTENTS: DEFAULT_IGNORED_INTENTS,
    CONF_CUSTOM_PROMPTS_SECTION: {
        CONF_PROMPT_BASE: DEFAULT_BASE_PROMPT,
        CONF_INSTRUCTIONS_PROMPT: DEFAULT_INSTRUCTIONS_PROMPT,
        CONF_PROMPT_NO_ENABLED_ENTITIES: DEFAULT_PROMPT_NO_ENABLED_ENTITIES,
        CONF_API_PROMPT_BASE: DEFAULT_API_PROMPT_BASE,
        CONF_PROMPT_DEVICE_KNOWN_LOCATION: DEFAULT_API_PROMPT_DEVICE_KNOWN_LOCATION,
        CONF_PROMPT_DEVICE_UNKNOWN_LOCATION: DEFAULT_API_PROMPT_DEVICE_UNKNOWN_LOCATION,
        CONF_PROMPT_TIMERS_UNSUPPORTED: DEFAULT_API_PROMPT_TIMERS_UNSUPPORTED,
        CONF_PROMPT_EXPOSED_ENTITIES: DEFAULT_API_PROMPT_EXPOSED_ENTITIES,
    },
    CONF_LANGFUSE_SECTION: {
        CONF_ENABLE_LANGFUSE: False,
        CONF_LANGFUSE_HOST: "",
        CONF_LANGFUSE_PUBLIC_KEY: "",
        CONF_LANGFUSE_SECRET_KEY: "",
        CONF_LANGFUSE_BASE_PROMPT_ID: "",
        CONF_LANGFUSE_BASE_PROMPT_LABEL: "production",
        CONF_LANGFUSE_API_PROMPT_ID: "",
        CONF_LANGFUSE_API_PROMPT_LABEL: "production",
        CONF_LANGFUSE_TRACING_ENABLED: False,
        CONF_LANGFUSE_TAGS: [],
        CONF_LANGFUSE_SCORE_ENABLED: False,
    },
}


def custom_conversation_config_option_schema(
    hass: HomeAssistant,
    options: dict[str, Any] | MappingProxyType[str, Any],
) -> vol.Schema:
    """Return a Schema for Custom Conversation Options."""
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(
            label="No control",
            value="none",
        )
    ]
    hass_apis.extend(
        SelectOptionDict(
            label=api.name,
            value=api.id,
        )
        for api in llm.async_get_apis(hass)
    )

    # Get the recommended list of ignored intents from the default API
    hass_recommended_ignored = llm.AssistAPI.IGNORE_INTENTS

    intents: list[SelectOptionDict] = [
        {
            "value": intent.intent_type,
            "label": f"{intent.intent_type} (Hass Recommended)"
            if intent.intent_type in hass_recommended_ignored
            else intent.intent_type,
        }
        for intent in intent.async_get(hass)
    ]

    # Basic options that are always shown
    schema: VolDictType = {
        vol.Optional(
            CONF_INSTRUCTIONS_PROMPT,
            description={
                "suggested_value": options.get(
                    CONF_INSTRUCTIONS_PROMPT, DEFAULT_INSTRUCTIONS_PROMPT
                ),
            },
            default=DEFAULT_INSTRUCTIONS_PROMPT,
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
            default="none",
        ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
        # Add ignored intents selector
        vol.Required(CONF_IGNORED_INTENTS_SECTION): section(
            vol.Schema(
                {
                    vol.Required(
                        CONF_IGNORED_INTENTS,
                        default=options.get(CONF_IGNORED_INTENTS_SECTION, {}).get(
                            CONF_IGNORED_INTENTS, DEFAULT_IGNORED_INTENTS
                        ),
                    ): SelectSelector(
                        SelectSelectorConfig(options=intents, multiple=True)
                    ),
                }
            )
        ),
        # Agent section
        vol.Required(CONF_AGENTS_SECTION): section(
            vol.Schema(
                {
                    vol.Required(
                        CONF_ENABLE_HASS_AGENT,
                        default=options.get("agents", {}).get(
                            CONF_ENABLE_HASS_AGENT, True
                        ),
                    ): bool,
                    vol.Required(
                        CONF_ENABLE_LLM_AGENT,
                        default=options.get("agents", {}).get(
                            CONF_ENABLE_LLM_AGENT, True
                        ),
                    ): bool,
                }
            )
        ),
        vol.Required(CONF_LLM_PARAMETERS_SECTION): section(
            vol.Schema(
                {
                    vol.Optional(
                        CONF_CHAT_MODEL,
                        description={
                            "suggested_value": options.get(
                                CONF_LLM_PARAMETERS_SECTION, {}
                            ).get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
                        },
                        default=RECOMMENDED_CHAT_MODEL,
                    ): str,
                    vol.Optional(
                        CONF_MAX_TOKENS,
                        description={
                            "suggested_value": options.get(
                                CONF_LLM_PARAMETERS_SECTION, {}
                            ).get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
                        },
                        default=RECOMMENDED_MAX_TOKENS,
                    ): int,
                    vol.Optional(
                        CONF_TOP_P,
                        description={
                            "suggested_value": options.get(
                                CONF_LLM_PARAMETERS_SECTION, {}
                            ).get(CONF_TOP_P, RECOMMENDED_TOP_P)
                        },
                        default=RECOMMENDED_TOP_P,
                    ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
                    vol.Optional(
                        CONF_TEMPERATURE,
                        description={
                            "suggested_value": options.get(
                                CONF_LLM_PARAMETERS_SECTION, {}
                            ).get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)
                        },
                        default=RECOMMENDED_TEMPERATURE,
                    ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
                }
            )
        ),
        vol.Required(CONF_CUSTOM_PROMPTS_SECTION): section(
            vol.Schema(
                {
                    vol.Optional(
                        CONF_PROMPT_BASE,
                        description={
                            "suggested_value": options.get(
                                CONF_CUSTOM_PROMPTS_SECTION, {}
                            ).get(CONF_PROMPT_BASE, DEFAULT_BASE_PROMPT)
                        },
                        default=DEFAULT_BASE_PROMPT,
                    ): TemplateSelector(),
                    vol.Optional(
                        CONF_PROMPT_NO_ENABLED_ENTITIES,
                        default=options.get(CONF_CUSTOM_PROMPTS_SECTION, {}).get(
                            CONF_PROMPT_NO_ENABLED_ENTITIES,
                            DEFAULT_PROMPT_NO_ENABLED_ENTITIES,
                        ),
                    ): TextSelector(TextSelectorConfig(multiline=True)),
                    vol.Optional(
                        CONF_API_PROMPT_BASE,
                        default=options.get(CONF_CUSTOM_PROMPTS_SECTION, {}).get(
                            CONF_API_PROMPT_BASE, DEFAULT_API_PROMPT_BASE
                        ),
                    ): TextSelector(TextSelectorConfig(multiline=True)),
                    vol.Optional(
                        CONF_PROMPT_DEVICE_KNOWN_LOCATION,
                        default=options.get(CONF_CUSTOM_PROMPTS_SECTION, {}).get(
                            CONF_PROMPT_DEVICE_KNOWN_LOCATION,
                            DEFAULT_API_PROMPT_DEVICE_KNOWN_LOCATION,
                        ),
                    ): TextSelector(TextSelectorConfig(multiline=True)),
                    vol.Optional(
                        CONF_PROMPT_DEVICE_UNKNOWN_LOCATION,
                        default=options.get(CONF_CUSTOM_PROMPTS_SECTION, {}).get(
                            CONF_PROMPT_DEVICE_UNKNOWN_LOCATION,
                            DEFAULT_API_PROMPT_DEVICE_UNKNOWN_LOCATION,
                        ),
                    ): TextSelector(TextSelectorConfig(multiline=True)),
                    vol.Optional(
                        CONF_PROMPT_TIMERS_UNSUPPORTED,
                        default=options.get(CONF_CUSTOM_PROMPTS_SECTION, {}).get(
                            CONF_PROMPT_TIMERS_UNSUPPORTED,
                            DEFAULT_API_PROMPT_TIMERS_UNSUPPORTED,
                        ),
                    ): TextSelector(TextSelectorConfig(multiline=True)),
                    vol.Optional(
                        CONF_PROMPT_EXPOSED_ENTITIES,
                        default=options.get(CONF_CUSTOM_PROMPTS_SECTION, {}).get(
                            CONF_PROMPT_EXPOSED_ENTITIES,
                            DEFAULT_API_PROMPT_EXPOSED_ENTITIES,
                        ),
                    ): TextSelector(TextSelectorConfig(multiline=True)),
                }
            )
        ),
        # Langfuse Section
        vol.Required(CONF_LANGFUSE_SECTION): section(
            vol.Schema(
                {
                    vol.Required(
                        CONF_ENABLE_LANGFUSE,
                        default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                            CONF_ENABLE_LANGFUSE, False
                        ),
                    ): bool,
                    vol.Optional(
                        CONF_LANGFUSE_HOST,
                        default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                            CONF_LANGFUSE_HOST, ""
                        ),
                    ): str,
                    vol.Optional(
                        CONF_LANGFUSE_PUBLIC_KEY,
                        default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                            CONF_LANGFUSE_PUBLIC_KEY, ""
                        ),
                    ): str,
                    vol.Optional(
                        CONF_LANGFUSE_SECRET_KEY,
                        default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                            CONF_LANGFUSE_SECRET_KEY, ""
                        ),
                    ): str,
                    vol.Optional(
                        CONF_LANGFUSE_BASE_PROMPT_ID,
                        default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                            CONF_LANGFUSE_BASE_PROMPT_ID, ""
                        ),
                    ): str,
                    vol.Optional(
                        CONF_LANGFUSE_BASE_PROMPT_LABEL,
                        default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                            CONF_LANGFUSE_BASE_PROMPT_LABEL, "production"
                        ),
                    ): str,
                    vol.Optional(
                        CONF_LANGFUSE_API_PROMPT_ID,
                        default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                            CONF_LANGFUSE_API_PROMPT_ID, ""
                        ),
                    ): str,
                    vol.Optional(
                        CONF_LANGFUSE_API_PROMPT_LABEL,
                        default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                            CONF_LANGFUSE_API_PROMPT_LABEL, "production"
                        ),
                    ): str,
                    vol.Optional(
                        CONF_LANGFUSE_TRACING_ENABLED,
                        default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                            CONF_LANGFUSE_TRACING_ENABLED, False
                        ),
                    ): bool,
                    vol.Optional(
                        CONF_LANGFUSE_TAGS,
                        default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                            CONF_LANGFUSE_TAGS, []
                        ),
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[], multiple=True, custom_value=True
                        )
                    ),
                    vol.Optional(
                        CONF_LANGFUSE_SCORE_ENABLED,
                        default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                            CONF_LANGFUSE_SCORE_ENABLED, False
                        ),
                    ): bool,
                }
            )
        ),
    }

    return schema


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """
    client = await hass.async_add_executor_job(
        lambda: openai.AsyncOpenAI(
            api_key=data[CONF_API_KEY], base_url=data[CONF_BASE_URL]
        )
    )
    await hass.async_add_executor_job(client.with_options(timeout=10.0).models.list)


class CustomConversationConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Custom Conversation."""

    VERSION = 1

    async def _validate_entry(
        self, user_input: dict[str, Any], step_id: str
    ) -> ConfigFlowResult:
        """Validate input and create entry if valid."""
        errors: dict[str, str] = {}

        try:
            await validate_input(self.hass, user_input)
        except openai.APIConnectionError:
            errors["base"] = "cannot_connect"
        except openai.AuthenticationError:
            errors["base"] = "invalid_auth"
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        return errors

    def _get_schema_with_defaults(self, step_id: str) -> vol.Schema:
        """Get schema with defaults filled in."""
        if step_id == "reconfigure":
            config_entry = self._get_reconfigure_entry()
            return vol.Schema(
                {
                    vol.Required(
                        CONF_API_KEY, default=config_entry.data.get(CONF_API_KEY, "")
                    ): str,
                    vol.Required(
                        CONF_BASE_URL,
                        default=config_entry.data.get(
                            CONF_BASE_URL, RECOMMENDED_BASE_URL
                        ),
                    ): str,
                }
            )
        return STEP_USER_DATA_SCHEMA

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=self._get_schema_with_defaults("user")
            )

        errors = await self._validate_entry(user_input, "user")

        if errors:
            return self.async_show_form(
                step_id="user",
                data_schema=self._get_schema_with_defaults("user"),
                errors=errors,
            )

        return self.async_create_entry(
            title="CustomConversation",
            data=user_input,
            options=RECOMMENDED_OPTIONS,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle reconfiguration of the integration."""
        if user_input is None:
            return self.async_show_form(
                step_id="reconfigure",
                data_schema=self._get_schema_with_defaults("reconfigure"),
            )

        errors = await self._validate_entry(user_input, "reconfigure")

        if errors:
            return self.async_show_form(
                step_id="reconfigure",
                data_schema=self._get_schema_with_defaults("reconfigure"),
                errors=errors,
            )

        return self.async_update_reload_and_abort(
            self._get_reconfigure_entry(), data_updates=user_input
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return CustomConversationOptionsFlow()


class CustomConversationOptionsFlow(OptionsFlow):
    """Custom Conversation config flow options handler."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        options: dict[str, Any] | MappingProxyType[str, Any] = self.config_entry.options

        if user_input is not None:
            if user_input[CONF_LLM_HASS_API] == "none":
                user_input.pop(CONF_LLM_HASS_API)
            # If the ignored intents are an empty list, use the defaults
            if not user_input.get(CONF_IGNORED_INTENTS_SECTION, {}).get(
                CONF_IGNORED_INTENTS
            ):
                user_input[CONF_IGNORED_INTENTS_SECTION][CONF_IGNORED_INTENTS] = (
                    DEFAULT_IGNORED_INTENTS
                )
            # If any of the custom prompts are an empty string, use the default for that prompt
            for prompt in user_input.get(CONF_CUSTOM_PROMPTS_SECTION, {}):
                if not user_input[CONF_CUSTOM_PROMPTS_SECTION][prompt]:
                    user_input[CONF_CUSTOM_PROMPTS_SECTION][prompt] = (
                        RECOMMENDED_OPTIONS[CONF_CUSTOM_PROMPTS_SECTION][prompt]
                    )

            return self.async_create_entry(title="", data=user_input)

        schema = custom_conversation_config_option_schema(self.hass, options)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )
