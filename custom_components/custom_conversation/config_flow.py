"""Config flow for Custom Conversation integration."""

from __future__ import annotations

from typing import Any

from litellm.exceptions import APIConnectionError, AuthenticationError
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry, ConfigFlow, OptionsFlow
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult, section
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

from .const import (
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
    CONF_SECONDARY_API_KEY,
    CONF_SECONDARY_BASE_URL,
    CONF_SECONDARY_CHAT_MODEL,
    CONF_SECONDARY_PROVIDER,
    CONF_SECONDARY_PROVIDER_ENABLED,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONFIG_VERSION,
    CONFIGURING_SECONDARY_PROVIDER,
    coerce_llm_hass_api_ids,
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
    LOGGER,
)
from .providers import SUPPORTED_PROVIDERS, LiteLLMProvider, get_provider

_LOGGER = LOGGER

DEFAULT_OPTIONS = {
    CONF_AGENTS_SECTION: {
        CONF_ENABLE_HASS_AGENT: True,
        CONF_ENABLE_LLM_AGENT: True,
    },
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


class CustomConversationConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Custom Conversation."""

    VERSION = CONFIG_VERSION
    _flow_data: dict[str, Any] = {}

    async def _validate_credentials_and_get_models(
        self, user_input: dict[str, Any], is_secondary: bool = False
    ) -> list[str]:
        """Validate credentials and fetch valid models."""
        if is_secondary:
            conf_provider = CONF_SECONDARY_PROVIDER
            conf_api_key = CONF_SECONDARY_API_KEY
            conf_base_url = CONF_SECONDARY_BASE_URL
        else:
            conf_provider = CONF_PRIMARY_PROVIDER
            conf_api_key = CONF_PRIMARY_API_KEY
            conf_base_url = CONF_PRIMARY_BASE_URL
        provider: LiteLLMProvider = self._flow_data[conf_provider]
        api_key = user_input.get(conf_api_key)
        base_url = user_input.get(conf_base_url)

        return await self.hass.async_add_executor_job(
            lambda: provider.get_supported_models(base_url=base_url, api_key=api_key),
        )

    def _build_credentials_schema(
        self, existing_data: dict[str, Any] | None = None, is_secondary: bool = False
    ) -> vol.Schema:
        """Build the schema for the credentials step."""
        if existing_data is None:
            existing_data = {}
        if is_secondary:
            conf_provider = CONF_SECONDARY_PROVIDER
            conf_api_key = CONF_SECONDARY_API_KEY
            conf_base_url = CONF_SECONDARY_BASE_URL
        else:
            conf_provider = CONF_PRIMARY_PROVIDER
            conf_api_key = CONF_PRIMARY_API_KEY
            conf_base_url = CONF_PRIMARY_BASE_URL
        provider: LiteLLMProvider = self._flow_data[conf_provider]

        schema_dict = {
            vol.Required(
                conf_api_key,
                default=existing_data.get(conf_api_key),
            ): TextSelector(TextSelectorConfig(type="password")),
        }

        default_base_url = existing_data.get(conf_base_url)
        if default_base_url is None or self._flow_data.get("changed_provider"):
            # If the base URL is not set or the provider has changed, fetch the default base URL
            if provider.supports_custom_base_url:
                default_base_url = provider.default_base_url
            else:  # If the provider does not support custom base URL, set it to None
                default_base_url = None
                self._flow_data.pop(conf_base_url, None)

        if provider.supports_custom_base_url:
            # If the provider supports custom base URL, add it to the schema
            schema_dict[
                vol.Optional(
                    conf_base_url,
                    default=default_base_url,
                )
            ] = str

        return vol.Schema(schema_dict)

    def _build_model_schema(
        self,
        valid_models: list[str] | None,
        current_model: str | None = None,
        is_secondary: bool = False,
        secondary_enabled: bool = False,
    ) -> vol.Schema:
        """Build the schema for the model selection step."""

        schema_dict = {}
        if is_secondary:
            conf_model = CONF_SECONDARY_CHAT_MODEL
        else:
            conf_model = CONF_PRIMARY_CHAT_MODEL
        if valid_models:
            schema_dict[vol.Required(conf_model, default=current_model)] = (
                SelectSelector(
                    SelectSelectorConfig(
                        options=valid_models, custom_value=True, sort=True
                    )
                )
            )
        else:
            schema_dict[vol.Required(conf_model, default=current_model)] = str

        if not is_secondary:
            schema_dict[
                vol.Optional(CONF_SECONDARY_PROVIDER_ENABLED, default=secondary_enabled)
            ] = bool

        return vol.Schema(schema_dict)

    def _get_reconfigure_entry(self) -> ConfigEntry:
        """Get the config entry to reconfigure."""
        if not self.context or "entry_id" not in self.context:
            _LOGGER.error(
                "Config entry ID not found in context for reconfiguration. Context: %s",
                self.context,
            )
            raise ValueError("Config entry ID not found in context for reconfiguration")
        entry_id = self.context["entry_id"]
        entry = self.hass.config_entries.async_get_entry(entry_id)
        if not entry:
            _LOGGER.error("Config entry %s not found during reconfiguration.", entry_id)
            raise ValueError(f"Config entry {entry_id} not found")
        return entry

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the provider selection step."""
        if self._flow_data.get(CONFIGURING_SECONDARY_PROVIDER):
            conf_provider = CONF_SECONDARY_PROVIDER
        else:
            conf_provider = CONF_PRIMARY_PROVIDER
        if user_input is not None:
            self._flow_data[conf_provider] = get_provider(user_input[conf_provider])
            return await self.async_step_credentials()

        schema = vol.Schema(
            {
                vol.Required(
                    conf_provider, default=SUPPORTED_PROVIDERS[0].key
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(label=p.provider_name, value=p.key)
                            for p in SUPPORTED_PROVIDERS
                        ]
                    )
                )
            }
        )
        return self.async_show_form(step_id="user", data_schema=schema)

    async def async_step_credentials(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the credentials step."""
        errors: dict[str, str] = {}
        is_secondary = self._flow_data.get(CONFIGURING_SECONDARY_PROVIDER)
        if is_secondary:
            conf_provider = CONF_SECONDARY_PROVIDER
        else:
            conf_provider = CONF_PRIMARY_PROVIDER
        provider: LiteLLMProvider = self._flow_data[conf_provider]

        if user_input is not None:
            self._flow_data.update(user_input)
            try:
                valid_models = await self._validate_credentials_and_get_models(
                    self._flow_data, is_secondary=is_secondary
                )
                return await self.async_step_model(valid_models=valid_models)
            except AuthenticationError:
                errors["base"] = "invalid_auth"
            except APIConnectionError:
                errors["base"] = "cannot_connect"

        schema = self._build_credentials_schema(is_secondary=is_secondary)

        return self.async_show_form(
            step_id="credentials",
            data_schema=schema,
            errors=errors,
            description_placeholders={"provider": provider.provider_name},
        )

    async def async_step_model(
        self,
        user_input: dict[str, Any] | None = None,
        valid_models: list[str] | None = None,
    ) -> FlowResult:
        """Handle the model selection step."""
        errors: dict[str, str] = {}
        is_secondary = self._flow_data.get(CONFIGURING_SECONDARY_PROVIDER)
        if is_secondary:
            conf_provider = CONF_SECONDARY_PROVIDER
        else:
            conf_provider = CONF_PRIMARY_PROVIDER
        provider: LiteLLMProvider = self._flow_data[conf_provider]

        if user_input is not None:
            self._flow_data.update(user_input)
            if user_input.get(CONF_SECONDARY_PROVIDER_ENABLED) and not is_secondary:
                self._flow_data[CONFIGURING_SECONDARY_PROVIDER] = True
                return await self.async_step_user()

            self._flow_data[CONFIGURING_SECONDARY_PROVIDER] = False

            final_data = self._flow_data.copy()
            final_data[conf_provider] = provider.key
            # If this is the secondary provider, we'll need to set the primary provider key too
            if is_secondary:
                final_data[CONF_PRIMARY_PROVIDER] = self._flow_data[
                    CONF_PRIMARY_PROVIDER
                ].key

            return self.async_create_entry(
                title="Custom Conversation",
                data=final_data,
                options=DEFAULT_OPTIONS,
            )

        schema = self._build_model_schema(
            valid_models,
            is_secondary=self._flow_data.get(CONFIGURING_SECONDARY_PROVIDER, False),
        )

        return self.async_show_form(
            step_id="model",
            data_schema=schema,
            errors=errors,
            description_placeholders={"provider": provider.provider_name},
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle reconfiguration of the integration."""
        entry = self._get_reconfigure_entry()
        initial_data = {**entry.data}
        primary_provider = initial_data.get(CONF_PRIMARY_PROVIDER)
        initial_data[CONF_PRIMARY_PROVIDER] = get_provider(primary_provider)
        if initial_data.get(CONF_SECONDARY_PROVIDER):
            secondary_provider = initial_data.get(CONF_SECONDARY_PROVIDER)
            initial_data[CONF_SECONDARY_PROVIDER] = get_provider(secondary_provider)

        self._flow_data = initial_data
        self.context["entry_id"] = entry.entry_id

        return await self.async_step_reconfigure_provider()

    async def async_step_reconfigure_provider(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle provider selection during reconfiguration."""
        errors: dict[str, str] = {}
        if self._flow_data.get(CONFIGURING_SECONDARY_PROVIDER):
            conf_provider = CONF_SECONDARY_PROVIDER
        else:
            conf_provider = CONF_PRIMARY_PROVIDER

        if user_input is not None:
            provider_key = user_input[conf_provider]
            # If the provider_key is different from the current one, we've changed the provider and need to track it to change the default base_url
            if provider_key != self._flow_data[conf_provider].key:
                self._flow_data["changed_provider"] = True
            self._flow_data[conf_provider] = get_provider(provider_key)
            return await self.async_step_reconfigure_credentials()

        current_provider: LiteLLMProvider = self._flow_data.get(conf_provider)

        schema = vol.Schema(
            {
                vol.Required(
                    conf_provider, default=current_provider.key
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(label=p.provider_name, value=p.key)
                            for p in SUPPORTED_PROVIDERS
                        ]
                    )
                )
            }
        )
        return self.async_show_form(
            step_id="reconfigure_provider", data_schema=schema, errors=errors
        )

    async def async_step_reconfigure_credentials(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle credentials during reconfiguration."""
        errors: dict[str, str] = {}
        is_secondary = self._flow_data.get(CONFIGURING_SECONDARY_PROVIDER)
        if is_secondary:
            conf_provider = CONF_SECONDARY_PROVIDER
        else:
            conf_provider = CONF_PRIMARY_PROVIDER
        provider: LiteLLMProvider = self._flow_data[conf_provider]

        if user_input is not None:
            self._flow_data.update(user_input)
            try:
                valid_models = await self._validate_credentials_and_get_models(
                    self._flow_data, is_secondary=is_secondary
                )
                return await self.async_step_reconfigure_model(
                    valid_models=valid_models
                )
            except AuthenticationError:
                errors["base"] = "invalid_auth"
            except APIConnectionError:
                errors["base"] = "cannot_connect"

        schema = self._build_credentials_schema(
            existing_data=self._flow_data, is_secondary=is_secondary
        )

        return self.async_show_form(
            step_id="reconfigure_credentials",
            data_schema=schema,
            errors=errors,
            description_placeholders={"provider": provider.provider_name},
        )

    async def async_step_reconfigure_model(
        self,
        user_input: dict[str, Any] | None = None,
        valid_models: list[str] | None = None,
    ) -> FlowResult:
        """Handle model selection during reconfiguration."""
        errors: dict[str, str] = {}
        is_secondary = self._flow_data.get(CONFIGURING_SECONDARY_PROVIDER)
        if is_secondary:
            conf_provider = CONF_SECONDARY_PROVIDER
            conf_model = CONF_SECONDARY_CHAT_MODEL
        else:
            conf_provider = CONF_PRIMARY_PROVIDER
            conf_model = CONF_PRIMARY_CHAT_MODEL
        provider: LiteLLMProvider = self._flow_data[conf_provider]

        if user_input is not None:
            self._flow_data.update(user_input)
            if user_input.get(CONF_SECONDARY_PROVIDER_ENABLED) and not is_secondary:
                self._flow_data[CONFIGURING_SECONDARY_PROVIDER] = True
                return await self.async_step_reconfigure_provider()
            self._flow_data[CONFIGURING_SECONDARY_PROVIDER] = False
            entry = self._get_reconfigure_entry()

            final_data = self._flow_data.copy()
            final_data[conf_provider] = provider.key
            if is_secondary:
                final_data[CONF_PRIMARY_PROVIDER] = self._flow_data[
                    CONF_PRIMARY_PROVIDER
                ].key

            return self.async_update_reload_and_abort(
                entry, data=final_data, reason="reconfigure_successful"
            )

        current_model = self._flow_data.get(conf_model)
        schema = self._build_model_schema(
            valid_models,
            current_model,
            is_secondary=is_secondary,
            secondary_enabled=self._flow_data.get(
                CONF_SECONDARY_PROVIDER_ENABLED, False
            ),
        )

        return self.async_show_form(
            step_id="reconfigure_model",
            data_schema=schema,
            errors=errors,
            description_placeholders={"provider": provider.provider_name},
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return CustomConversationOptionsFlow(config_entry)


class CustomConversationOptionsFlow(OptionsFlow):
    """Custom Conversation config flow options handler."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            # Process user input before saving
            processed_input = {**user_input}  # Start with a copy

            # MCP server ids (CONF_LLM_HASS_API): store list or omit when none selected
            llm_apis = processed_input.get(CONF_LLM_HASS_API)
            normalized = coerce_llm_hass_api_ids(llm_apis)
            if normalized:
                processed_input[CONF_LLM_HASS_API] = normalized
            else:
                processed_input.pop(CONF_LLM_HASS_API, None)

            # Handle empty ignored intents - use default
            ignored_intents_section = processed_input.get(
                CONF_IGNORED_INTENTS_SECTION, {}
            )
            if not ignored_intents_section.get(CONF_IGNORED_INTENTS):
                ignored_intents_section[CONF_IGNORED_INTENTS] = (
                    llm.AssistAPI.IGNORE_INTENTS
                )
                processed_input[CONF_IGNORED_INTENTS_SECTION] = ignored_intents_section

            # If any of the custom prompts are an empty string, use the defaults
            prompts = processed_input.get(CONF_CUSTOM_PROMPTS_SECTION, {})
            for prompt in prompts:
                if not prompts.get(prompt):
                    prompts[prompt] = DEFAULT_OPTIONS[CONF_CUSTOM_PROMPTS_SECTION].get(
                        prompt
                    )
            processed_input[CONF_CUSTOM_PROMPTS_SECTION] = prompts
            return self.async_create_entry(title="", data=processed_input)

        # Build schema for options
        options = self.config_entry.options
        hass = self.hass
        hass_apis = self._get_hass_apis(hass)
        intents = await self._get_intents(hass)
        default_ignored = llm.AssistAPI.IGNORE_INTENTS

        # Define the schema for options, using existing options as defaults
        schema = vol.Schema(
            {
                # LLM Parameters (now direct options)
                vol.Optional(
                    CONF_TEMPERATURE,
                    default=options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
                vol.Optional(
                    CONF_TOP_P,
                    default=options.get(CONF_TOP_P, DEFAULT_TOP_P),
                ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
                vol.Optional(
                    CONF_MAX_TOKENS,
                    default=options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
                ): vol.All(vol.Coerce(int), vol.Range(min=1)),
                # MCP servers (multiple merged like core conversation)
                vol.Optional(
                    CONF_LLM_HASS_API,
                    description={
                        "suggested_value": coerce_llm_hass_api_ids(
                            options.get(CONF_LLM_HASS_API)
                        )
                        or []
                    },
                    default=[],
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=hass_apis, multiple=True, sort=True
                    )
                ),
                # Agent Section
                vol.Required(CONF_AGENTS_SECTION): section(
                    vol.Schema(
                        {
                            vol.Required(
                                CONF_ENABLE_HASS_AGENT,
                                default=options.get(CONF_AGENTS_SECTION, {}).get(
                                    CONF_ENABLE_HASS_AGENT, True
                                ),
                            ): bool,
                            vol.Required(
                                CONF_ENABLE_LLM_AGENT,
                                default=options.get(CONF_AGENTS_SECTION, {}).get(
                                    CONF_ENABLE_LLM_AGENT, True
                                ),
                            ): bool,
                        }
                    )
                ),
                # Ignored Intents Section
                vol.Required(CONF_IGNORED_INTENTS_SECTION): section(
                    vol.Schema(
                        {
                            vol.Required(
                                CONF_IGNORED_INTENTS,
                                default=options.get(
                                    CONF_IGNORED_INTENTS_SECTION, {}
                                ).get(CONF_IGNORED_INTENTS, default_ignored),
                            ): SelectSelector(
                                SelectSelectorConfig(
                                    options=intents, multiple=True, sort=True
                                )
                            ),
                        }
                    )
                ),
                # Custom Prompts Section
                vol.Required(CONF_CUSTOM_PROMPTS_SECTION): section(
                    vol.Schema(
                        {
                            vol.Optional(
                                CONF_INSTRUCTIONS_PROMPT,
                                default=options.get(
                                    CONF_CUSTOM_PROMPTS_SECTION, {}
                                ).get(
                                    CONF_INSTRUCTIONS_PROMPT,
                                    DEFAULT_INSTRUCTIONS_PROMPT,
                                ),
                            ): TemplateSelector(),
                            vol.Optional(
                                CONF_PROMPT_BASE,
                                default=options.get(
                                    CONF_CUSTOM_PROMPTS_SECTION, {}
                                ).get(CONF_PROMPT_BASE, DEFAULT_BASE_PROMPT),
                            ): TemplateSelector(),
                            vol.Optional(
                                CONF_PROMPT_NO_ENABLED_ENTITIES,
                                default=options.get(
                                    CONF_CUSTOM_PROMPTS_SECTION, {}
                                ).get(
                                    CONF_PROMPT_NO_ENABLED_ENTITIES,
                                    DEFAULT_PROMPT_NO_ENABLED_ENTITIES,
                                ),
                            ): TextSelector(TextSelectorConfig(multiline=True)),
                            vol.Optional(
                                CONF_API_PROMPT_BASE,
                                default=options.get(
                                    CONF_CUSTOM_PROMPTS_SECTION, {}
                                ).get(CONF_API_PROMPT_BASE, DEFAULT_API_PROMPT_BASE),
                            ): TextSelector(TextSelectorConfig(multiline=True)),
                            vol.Optional(
                                CONF_PROMPT_DEVICE_KNOWN_LOCATION,
                                default=options.get(
                                    CONF_CUSTOM_PROMPTS_SECTION, {}
                                ).get(
                                    CONF_PROMPT_DEVICE_KNOWN_LOCATION,
                                    DEFAULT_API_PROMPT_DEVICE_KNOWN_LOCATION,
                                ),
                            ): TextSelector(TextSelectorConfig(multiline=True)),
                            vol.Optional(
                                CONF_PROMPT_DEVICE_UNKNOWN_LOCATION,
                                default=options.get(
                                    CONF_CUSTOM_PROMPTS_SECTION, {}
                                ).get(
                                    CONF_PROMPT_DEVICE_UNKNOWN_LOCATION,
                                    DEFAULT_API_PROMPT_DEVICE_UNKNOWN_LOCATION,
                                ),
                            ): TextSelector(TextSelectorConfig(multiline=True)),
                            vol.Optional(
                                CONF_PROMPT_TIMERS_UNSUPPORTED,
                                default=options.get(
                                    CONF_CUSTOM_PROMPTS_SECTION, {}
                                ).get(
                                    CONF_PROMPT_TIMERS_UNSUPPORTED,
                                    DEFAULT_API_PROMPT_TIMERS_UNSUPPORTED,
                                ),
                            ): TextSelector(TextSelectorConfig(multiline=True)),
                            vol.Optional(
                                CONF_PROMPT_EXPOSED_ENTITIES,
                                default=options.get(
                                    CONF_CUSTOM_PROMPTS_SECTION, {}
                                ).get(
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
                            ): TextSelector(TextSelectorConfig(type="password")),
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
                                    options=[],
                                    multiple=True,
                                    custom_value=True,
                                    sort=True,
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
        )

        return self.async_show_form(
            step_id="init",
            data_schema=schema,
        )

    def _get_hass_apis(self, hass: HomeAssistant) -> list[SelectOptionDict]:
        """Get registered MCP servers / LLM tool APIs for multi-select."""
        return [
            SelectOptionDict(label=api.name, value=api.id)
            for api in llm.async_get_apis(hass)
        ]

    async def _get_intents(self, hass: HomeAssistant) -> list[SelectOptionDict]:
        """Get available intents."""
        hass_recommended_ignored = llm.AssistAPI.IGNORE_INTENTS
        return [
            {
                "value": intent_obj.intent_type,
                "label": f"{intent_obj.intent_type} (Hass Recommended)"
                if intent_obj.intent_type in hass_recommended_ignored
                else intent_obj.intent_type,
            }
            for intent_obj in intent.async_get(hass)
        ]
