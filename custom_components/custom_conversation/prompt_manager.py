"""Prompt management for  Custom Conversation component."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from langfuse import Langfuse
from langfuse.api import CreateScoreConfigRequest
from langfuse.api.resources.commons.types import ScoreDataType
from langfuse.decorators import observe
from langfuse.model import Prompt

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import template
from homeassistant.util import yaml as yaml_util, dt as dt_util

from .const import (
    CONF_API_PROMPT_BASE,
    CONF_CUSTOM_PROMPTS_SECTION,
    CONF_ENABLE_LANGFUSE,
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
    CONF_LANGFUSE_TRACING_ENABLED,
    CONF_PROMPT_BASE,
    CONF_PROMPT_DEVICE_KNOWN_LOCATION,
    CONF_PROMPT_DEVICE_UNKNOWN_LOCATION,
    CONF_PROMPT_EXPOSED_ENTITIES,
    CONF_PROMPT_NO_ENABLED_ENTITIES,
    CONF_PROMPT_TIMERS_UNSUPPORTED,
    DEFAULT_API_PROMPT_BASE,
    DEFAULT_API_PROMPT_DEVICE_KNOWN_LOCATION,
    DEFAULT_API_PROMPT_DEVICE_UNKNOWN_LOCATION,
    DEFAULT_API_PROMPT_EXPOSED_ENTITIES,
    DEFAULT_API_PROMPT_TIMERS_UNSUPPORTED,
    DEFAULT_BASE_PROMPT,
    DEFAULT_INSTRUCTIONS_PROMPT,
    DEFAULT_PROMPT_NO_ENABLED_ENTITIES,
    LANGFUSE_SCORE_NAME,
    LANGFUSE_SCORE_NEGATIVE,
    LANGFUSE_SCORE_POSITIVE,
    LOGGER,
)


class LangfuseError(Exception):
    """Base class for Langfuse errors."""


class LangfuseInitError(LangfuseError):
    """Error initializing Langfuse client."""


class LangfusePromptError(LangfuseError):
    """Error getting or compiling Langfuse prompt."""


@dataclass
class PromptContext:
    """Context for prompt generation."""

    hass: HomeAssistant
    ha_name: str
    user_name: str | None = None
    llm_context: Any | None = None
    location: str | None = None
    exposed_entities: dict | None = None
    supports_timers: bool = True


class PromptManager:
    """Manager for Custom Conversation prompts."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the prompt manager."""
        self.hass = hass
        self._langfuse_client = None

    def _get_prompt_config(
        self, config_entry: ConfigEntry | None, key: str, default: str
    ) -> str:
        """Get prompt configuration with fallback to defaults."""
        if not config_entry:
            return default

        return config_entry.options.get(CONF_CUSTOM_PROMPTS_SECTION, {}).get(
            key, default
        )

    @observe(capture_input=False)
    async def _get_langfuse_prompt(
        self, prompt_id: str, variables: dict[str, Any]
    ) -> tuple[Prompt, str] | None:
        """Get a prompt from Langfuse if enabled."""
        if not self._langfuse_client:
            return None

        try:
            return await self._langfuse_client.get_prompt(prompt_id, variables)
        except Exception as err:
            LOGGER.error("Error getting Langfuse prompt: %s", err)
            return None

    @observe(capture_input=False)
    async def async_get_base_prompt(
        self, context: PromptContext, config_entry: ConfigEntry | None = None
    ) -> tuple[Prompt, str] | str:
        """Get the base prompt with rendered template."""
        if config_entry and config_entry.options.get(CONF_LANGFUSE_SECTION, {}).get(
            CONF_ENABLE_LANGFUSE
        ):
            prompt_object, langfuse_prompt = await self._get_langfuse_prompt(
                config_entry.options.get(CONF_LANGFUSE_SECTION, {}).get(
                    CONF_LANGFUSE_BASE_PROMPT_ID
                ),
                {
                    "current_time": dt_util.now().strftime("%H:%M"),
                    "current_date": dt_util.now().strftime("%Y-%m-%d"),
                    "ha_name": context.ha_name,
                    "user_name": context.user_name,
                },
            )
            if langfuse_prompt:
                return prompt_object, langfuse_prompt

        try:
            base_prompt = self._get_prompt_config(
                config_entry, CONF_PROMPT_BASE, DEFAULT_BASE_PROMPT
            )
            instructions_prompt = self._get_prompt_config(
                config_entry, CONF_INSTRUCTIONS_PROMPT, DEFAULT_INSTRUCTIONS_PROMPT
            )

            return template.Template(
                base_prompt + "\n" + instructions_prompt,
                context.hass,
            ).async_render(
                {
                    "ha_name": context.ha_name,
                    "user_name": context.user_name,
                    "llm_context": context.llm_context,
                },
                parse_result=False,
            )
        except TemplateError as err:
            LOGGER.error("Error rendering base prompt: %s", err)
            raise

    @observe(capture_input=False)
    async def get_api_prompt(
        self, context: PromptContext, config_entry: ConfigEntry | None = None
    ) -> tuple[Prompt, str] | str:
        """Get the API prompt based on context."""
        if config_entry and config_entry.options.get(CONF_LANGFUSE_SECTION, {}).get(
            CONF_ENABLE_LANGFUSE
        ):
            prompt_object, langfuse_prompt = await self._get_langfuse_prompt(
                config_entry.options.get(CONF_LANGFUSE_SECTION, {}).get(
                    CONF_LANGFUSE_API_PROMPT_ID
                ),
                {
                    "current_time": dt_util.now().strftime("%H:%M"),
                    "current_date": dt_util.now().strftime("%Y-%m-%d"),
                    "ha_name": context.ha_name,
                    "user_name": (
                        context.user_name if context.user_name else "unknown"
                    ),
                    "location": (context.location if context.location else "unknown"),
                    "exposed_entities": (
                        yaml_util.dump(list(context.exposed_entities.values()))
                        if context.exposed_entities
                        else None
                    ),
                    "supports_timers": (
                        "This device is not able to start timers."
                        if not context.supports_timers
                        else ""
                    ),
                },
            )
            if langfuse_prompt:
                return prompt_object, langfuse_prompt
        prompt_parts = []

        if not context.exposed_entities:
            return self._get_prompt_config(
                config_entry,
                CONF_PROMPT_NO_ENABLED_ENTITIES,
                DEFAULT_PROMPT_NO_ENABLED_ENTITIES,
            )

        # Add base API prompt
        prompt_parts.append(
            self._get_prompt_config(
                config_entry, CONF_API_PROMPT_BASE, DEFAULT_API_PROMPT_BASE
            )
        )

        # Add location-specific prompt
        if context.location:
            location_prompt = self._get_prompt_config(
                config_entry,
                CONF_PROMPT_DEVICE_KNOWN_LOCATION,
                DEFAULT_API_PROMPT_DEVICE_KNOWN_LOCATION,
            )
            prompt_parts.append(
                template.Template(location_prompt, context.hass).async_render(
                    {"location": context.location}, parse_result=False
                )
            )
        else:
            prompt_parts.append(
                self._get_prompt_config(
                    config_entry,
                    CONF_PROMPT_DEVICE_UNKNOWN_LOCATION,
                    DEFAULT_API_PROMPT_DEVICE_UNKNOWN_LOCATION,
                )
            )

        # Add timer capability prompt if needed
        if not context.supports_timers:
            prompt_parts.append(
                self._get_prompt_config(
                    config_entry,
                    CONF_PROMPT_TIMERS_UNSUPPORTED,
                    DEFAULT_API_PROMPT_TIMERS_UNSUPPORTED,
                )
            )

        # Add exposed entities prompt and data
        if context.exposed_entities:
            prompt_parts.append(
                self._get_prompt_config(
                    config_entry,
                    CONF_PROMPT_EXPOSED_ENTITIES,
                    DEFAULT_API_PROMPT_EXPOSED_ENTITIES,
                )
            )
            prompt_parts.append(yaml_util.dump(list(context.exposed_entities.values())))

        return "\n".join(prompt_parts)

    def set_langfuse_client(self, langfuse_client: Any) -> None:
        """Set the Langfuse client."""
        self._langfuse_client = langfuse_client


class LangfuseClient:
    """Client for Langfuse prompt management."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: Langfuse,
        prompts: dict,
        score_config_id: str | None = None,
    ) -> None:
        """Initialize the client."""
        self._client = client
        self.hass = hass
        self.prompts = prompts
        self.score_config_id = score_config_id
        self.score_config_id = score_config_id

    @classmethod
    async def create(
        cls, hass: HomeAssistant, config_entry: ConfigEntry
    ) -> LangfuseClient | None:
        """Create a Langfuse client instance."""
        if not config_entry.options.get(CONF_LANGFUSE_SECTION, {}).get(
            CONF_ENABLE_LANGFUSE
        ):
            return None
        # Set up prompt dictionary from config entry
        prompts = {
            config_entry.options.get(CONF_LANGFUSE_SECTION, {}).get(
                CONF_LANGFUSE_BASE_PROMPT_ID
            ): config_entry.options.get(CONF_LANGFUSE_SECTION, {}).get(
                CONF_LANGFUSE_BASE_PROMPT_LABEL, "production"
            ),
            config_entry.options.get(CONF_LANGFUSE_SECTION, {}).get(
                CONF_LANGFUSE_API_PROMPT_ID
            ): config_entry.options.get(CONF_LANGFUSE_SECTION, {}).get(
                CONF_LANGFUSE_API_PROMPT_LABEL, "production"
            ),
        }
        try:

            def create_client() -> Langfuse:
                return Langfuse(
                    public_key=config_entry.options[CONF_LANGFUSE_SECTION][
                        CONF_LANGFUSE_PUBLIC_KEY
                    ],
                    secret_key=config_entry.options[CONF_LANGFUSE_SECTION][
                        CONF_LANGFUSE_SECRET_KEY
                    ],
                    host=config_entry.options[CONF_LANGFUSE_SECTION].get(
                        CONF_LANGFUSE_HOST
                    ),
                    enabled=config_entry.options[CONF_LANGFUSE_SECTION][
                        CONF_LANGFUSE_TRACING_ENABLED
                    ],
                    max_retries=0,
                )

            client = await hass.async_add_executor_job(create_client)
            score_config = None
            # Ensure the score config is created if it's enabled
            if config_entry.options.get(CONF_LANGFUSE_SECTION, {}).get(
                CONF_LANGFUSE_SCORE_ENABLED
            ):
                score_configs = await hass.async_add_executor_job(
                    client.api.score_configs.get
                )
                score_config = next(
                    (
                        score
                        for score in score_configs.data
                        if score.name == LANGFUSE_SCORE_NAME
                    ),
                    None,
                )
                if not score_config:
                    score_config_request = CreateScoreConfigRequest(
                        name=LANGFUSE_SCORE_NAME,
                        data_type=ScoreDataType.CATEGORICAL,
                        categories=[
                            {
                                "label": LANGFUSE_SCORE_POSITIVE,
                                "value": 1,
                            },
                            {
                                "label": LANGFUSE_SCORE_NEGATIVE,
                                "value": 0,
                            },
                        ],
                        description="Score for Custom Conversation Home Assistant integration",
                    )
                    score_config = await hass.async_add_executor_job(
                        lambda: client.api.score_configs.create(
                            request=score_config_request
                        )
                    )
            return cls(hass, client, prompts, score_config.id if score_config else None)
        except Exception as err:
            LOGGER.error("Error initializing Langfuse client: %s", err)
            raise LangfuseInitError("Failed to initialize Langfuse client") from err

    @observe(capture_input=False)
    async def get_prompt(
        self, prompt_id: str, variables: dict[str, Any]
    ) -> tuple[Prompt, str]:
        """Get and compile a prompt from Langfuse."""
        try:
            # Get the prompt object in an executor
            prompt_object = await self.hass.async_add_executor_job(
                lambda: self._client.get_prompt(
                    prompt_id, label=self.prompts[prompt_id], type="chat"
                )
            )
            # Compile the prompt in an executor
            compiled_prompt = prompt_object.compile(**variables)[0]["content"]
        except Exception as err:
            LOGGER.error("Error getting Langfuse prompt: %s", err)
            raise LangfusePromptError(f"Failed to get Langfuse prompt: {err}") from err
        return prompt_object, compiled_prompt

    async def score(self, score: str, device_id: str) -> None:
        """Score a conversation using Langfuse."""
        if not self.score_config_id:
            LOGGER.warning("Score config ID not set, skipping scoring")
            return

        try:
            # Get the latest trace that matches this device
            traces = await self.hass.async_add_executor_job(
                lambda: self._client.get_traces(
                    name="cc_process",
                    tags=f"device_id:{device_id}",
                    from_timestamp=(datetime.now() - timedelta(minutes=10)),
                )
            )
            LOGGER.debug("Traces found for device %s: %s", device_id, traces.data)
            if not traces.data:
                LOGGER.warning("No traces found for device %s", device_id)
                return
            # Score the latest trace
            latest_trace = traces.data[0]
            LOGGER.debug("Scoring trace %s with score %s", latest_trace.id, score)

            await self.hass.async_add_executor_job(
                lambda: self._client.score(
                    name=LANGFUSE_SCORE_NAME,
                    value=score,
                    comment="Score based on Home Assistant Service Call",
                    trace_id=latest_trace.id,
                    config_id=self.score_config_id,
                )
            )
        except Exception as err:
            LOGGER.error("Error scoring conversation: %s", err)

    async def cleanup(self) -> None:
        """Clean up Langfuse client resources."""
        if self._client:
            try:
                # Flush any pending data and stop the consumer thread
                await self.hass.async_add_executor_job(self._client.flush)
            except Exception as err:
                LOGGER.warning("Error cleaning up Langfuse client: %s", err)
