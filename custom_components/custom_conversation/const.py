"""Constants for the Custom Conversation integration."""

from __future__ import annotations

import logging
from typing import Any

DOMAIN = "custom_conversation"
LOGGER = logging.getLogger(__package__)

CONFIG_VERSION = 2

CONF_PRIMARY_PROVIDER = "primary_provider"
CONF_PRIMARY_API_KEY = "primary_api_key"
CONF_PRIMARY_BASE_URL = "primary_base_url"
CONF_PRIMARY_CHAT_MODEL = "primary_chat_model"

CONF_SECONDARY_PROVIDER_ENABLED = "secondary_provider_enabled"

CONF_SECONDARY_PROVIDER = "secondary_provider"
CONF_SECONDARY_API_KEY = "secondary_api_key"
CONF_SECONDARY_BASE_URL = "secondary_base_url"
CONF_SECONDARY_CHAT_MODEL = "secondary_chat_model"

CONFIGURING_SECONDARY_PROVIDER = "configuring_secondary_provider"

DEFAULT_PROVIDER = "openai"

SERVICE_GENERATE_IMAGE = "generate_image"
CONF_ENABLE_HASS_AGENT = "enable_home_assistant_agent"
CONF_ENABLE_LLM_AGENT = "enable_llm_agent"
CONF_AGENTS_SECTION = "agents"
CONF_LLM_PARAMETERS_SECTION = "llm_parameters"
CONF_IGNORED_INTENTS_SECTION = "ignored_intents_section"
CONF_IGNORED_INTENTS = "ignored_intents"

CONF_MAX_TOKENS = "max_tokens"
DEFAULT_MAX_TOKENS = 150
CONF_TOP_P = "top_p"
DEFAULT_TOP_P = 1.0
CONF_TEMPERATURE = "temperature"
DEFAULT_TEMPERATURE = 1.0

LLM_API_ID = "custom-conversation"
HOME_ASSISTANT_AGENT = "conversation.home_assistant"
CONVERSATION_STARTED_EVENT = f"{DOMAIN}_conversation_started"
CONVERSATION_ENDED_EVENT = f"{DOMAIN}_conversation_ended"
CONVERSATION_ERROR_EVENT = f"{DOMAIN}_conversation_error"

CONF_CUSTOM_PROMPTS_SECTION = "custom_prompts"
CONF_PROMPT_BASE = "prompt_base"
DEFAULT_BASE_PROMPT = (
    'Current time is {{ now().strftime("%H:%M:%S") }}. '
    'Today\'s date is {{ now().strftime("%Y-%m-%d") }}.\n'
)

CONF_INSTRUCTIONS_PROMPT = "instructions_prompt"
DEFAULT_INSTRUCTIONS_PROMPT = """You are a voice assistant for Home Assistant.
Answer questions about the world truthfully.
Answer in plain text. Keep it simple and to the point.
"""

CONF_PROMPT_NO_ENABLED_ENTITIES = "prompt_no_enabled_entities"
DEFAULT_PROMPT_NO_ENABLED_ENTITIES = (
    "Only if the user wants to control a device, tell them to expose entities "
    "to their voice assistant in Home Assistant."
)

CONF_API_PROMPT_BASE = "prompt_api_base"
DEFAULT_API_PROMPT_BASE = (
    "When controlling Home Assistant always call the intent tools. "
    "Use HassTurnOn to lock and HassTurnOff to unlock a lock. "
    "When controlling a device, prefer passing just name and domain. "
    "When controlling an area, prefer passing just area name and domain."
)

# If the user request came from a device, it may have a known location
CONF_PROMPT_DEVICE_KNOWN_LOCATION = "prompt_device_known_location"
DEFAULT_API_PROMPT_DEVICE_KNOWN_LOCATION = (
    "You are in area {{ location }} and all generic commands "
    "like 'turn on the lights' should target this area."
)
CONF_PROMPT_DEVICE_UNKNOWN_LOCATION = "prompt_device_unknown_location"
DEFAULT_API_PROMPT_DEVICE_UNKNOWN_LOCATION = (
    "When a user asks to turn on all devices of a specific type, "
    "ask user to specify an area, unless there is only one device of that type."
)

CONF_PROMPT_TIMERS_UNSUPPORTED = "prompt_timers_unsupported"
DEFAULT_API_PROMPT_TIMERS_UNSUPPORTED = "This device is not able to start timers."

CONF_PROMPT_EXPOSED_ENTITIES = "prompt_exposed_entities"
DEFAULT_API_PROMPT_EXPOSED_ENTITIES = (
    "An overview of the areas and the devices in this smart home:"
)

# Langfuse Constants
CONF_LANGFUSE_SECTION = "langfuse"
CONF_ENABLE_LANGFUSE = "enable_langfuse"
CONF_LANGFUSE_BASE_PROMPT_ID = "base_prompt_id"
CONF_LANGFUSE_BASE_PROMPT_LABEL = "base_prompt_label"
CONF_LANGFUSE_API_PROMPT_ID = "api_prompt_id"
CONF_LANGFUSE_API_PROMPT_LABEL = "api_prompt_label"
CONF_LANGFUSE_SECRET_KEY = "langfuse_secret_key"
CONF_LANGFUSE_PUBLIC_KEY = "langfuse_public_key"
CONF_LANGFUSE_HOST = "langfuse_host"
CONF_LANGFUSE_TRACING_ENABLED = "langfuse_tracing_enabled"
CONF_LANGFUSE_TAGS = "langfuse_tags"
CONF_LANGFUSE_SCORE_ENABLED = "langfuse_score_enabled"
LANGFUSE_SCORE_NAME = "cc_score"
LANGFUSE_SCORE_POSITIVE = "positive"
LANGFUSE_SCORE_NEGATIVE = "negative"

# These intents are deprecated, but also in the IGNORE_INTENTS list
HASS_DEPRECATED_INTENTS = [
    "HassOpenCover",
    "HassCloseCover",
]

# Deprecated constants for migrations only
CONF_CHAT_MODEL = "chat_model"
CONF_BASE_URL = "base_url"


def coerce_llm_hass_api_ids(raw: Any) -> list[str] | None:
    """Normalize CONF_LLM_HASS_API (MCP server / LLM API ids) from options."""
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw in ("", "none"):
            return None
        return [raw]
    if isinstance(raw, list):
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            if not item or item == "none":
                continue
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out or None
    return None
