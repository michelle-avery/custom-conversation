"""Custom Version of LLM API for Custom Conversation."""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from functools import cache, partial
from typing import Any

from langfuse.model import Prompt
import slugify as unicode_slug
import voluptuous as vol

from homeassistant.components.homeassistant import async_should_expose
from homeassistant.components.intent import async_device_supports_timers
from homeassistant.components.script import DOMAIN as SCRIPT_DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_DOMAIN,
    ATTR_SERVICE,
    EVENT_HOMEASSISTANT_CLOSE,
    EVENT_SERVICE_REMOVED,
)
from homeassistant.core import Event, HomeAssistant, callback, split_entity_id
from homeassistant.helpers import (
    area_registry as ar,
    config_validation as cv,
    device_registry as dr,
    entity_registry as er,
    floor_registry as fr,
    intent,
    llm,
    selector,
    service,
)
from homeassistant.util.hass_dict import HassKey
from homeassistant.util.json import JsonObjectType

from .const import CONF_IGNORED_INTENTS, CONF_IGNORED_INTENTS_SECTION, LLM_API_ID
from .prompt_manager import PromptContext, PromptManager


class CustomLLMAPI(llm.API):
    """An API for the Custom Conversation integration to use to call Home Assistant services."""

    def __init__(
        self,
        hass: HomeAssistant,
        user_name: str | None = None,
        conversation_config_entry: ConfigEntry | None = None,
    ) -> None:
        """Initialize the API."""
        super().__init__(hass=hass, id=LLM_API_ID, name="Custom Conversation LLM API")
        self.cached_slugify = cache(
            partial(unicode_slug.slugify, separator="_", lowercase=False)
        )
        self._hass = hass
        self._request_user_name = user_name
        self._prompt_manager = PromptManager(hass)
        self.prompt_object = None
        self.conversation_config_entry = conversation_config_entry

    def set_langfuse_client(self, langfuse_client: Any) -> None:
        """Set the Langfuse client."""
        self._prompt_manager.set_langfuse_client(langfuse_client)

    async def async_get_api_instance(
        self, llm_context: llm.LLMContext
    ) -> llm.APIInstance:
        """Return an instance of the Custom Conversation LLM API."""
        if llm_context.assistant:
            exposed_entities: dict | None = _get_exposed_entities(
                self.hass, llm_context.assistant
            )
        else:
            exposed_entities = None

        return llm.APIInstance(
            api=self,
            api_prompt=self._async_get_api_prompt(llm_context, exposed_entities),
            llm_context=llm_context,
            tools=self._async_get_tools(llm_context, exposed_entities),
            custom_serializer=llm.selector_serializer,
        )

    @callback
    def _async_get_api_prompt(
        self, llm_context: llm.LLMContext, exposed_entities: dict | None
    ) -> tuple[Prompt, str] | str:
        """Return the prompt for the API."""

        area_name = None
        floor_name = None
        supports_timers = False

        area: ar.AreaEntry | None = None
        floor: fr.FloorEntry | None = None
        if llm_context.device_id:
            device_reg = dr.async_get(self.hass)
            device = device_reg.async_get(llm_context.device_id)

            if device:
                area_reg = ar.async_get(self.hass)
                if device.area_id and (area := area_reg.async_get_area(device.area_id)):
                    area_name = area.name
                    floor_reg = fr.async_get(self.hass)
                    if area.floor_id and (
                        floor := floor_reg.async_get_floor(area.floor_id)
                    ):
                        floor_name = floor.name

            supports_timers = async_device_supports_timers(
                self.hass, llm_context.device_id
            )

        location = f"{area_name} (floor: {floor_name})" if floor_name else area_name
        context = PromptContext(
            hass=self.hass,
            ha_name=self.hass.config.location_name,
            user_name=self._request_user_name,
            llm_context=llm_context,
            location=location,
            exposed_entities=exposed_entities,
            supports_timers=supports_timers,
        )

        return self._prompt_manager.get_api_prompt(
            context, self.conversation_config_entry
        )

    @callback
    def _async_get_tools(
        self, llm_context: llm.LLMContext, exposed_entities: dict | None
    ) -> list[llm.Tool]:
        """Return a list of LLM tools."""
        config_entry = self.conversation_config_entry
        # Get ignored intents from options, fallback to defaults
        if config_entry:
            ignored_intents_section = config_entry.options.get(
                CONF_IGNORED_INTENTS_SECTION, {}
            )
            ignore_intents = set(
                ignored_intents_section.get(
                    CONF_IGNORED_INTENTS, llm.AssistAPI.IGNORE_INTENTS
                )
            )
        else:
            ignore_intents = llm.AssistAPI.IGNORE_INTENTS

        if not llm_context.device_id or not async_device_supports_timers(
            self.hass, llm_context.device_id
        ):
            ignore_intents = ignore_intents | {
                intent.INTENT_START_TIMER,
                intent.INTENT_CANCEL_TIMER,
                intent.INTENT_INCREASE_TIMER,
                intent.INTENT_DECREASE_TIMER,
                intent.INTENT_PAUSE_TIMER,
                intent.INTENT_UNPAUSE_TIMER,
                intent.INTENT_TIMER_STATUS,
            }

        intent_handlers = [
            intent_handler
            for intent_handler in intent.async_get(self.hass)
            if intent_handler.intent_type not in ignore_intents
        ]

        exposed_domains: set[str] | None = None
        if exposed_entities is not None:
            exposed_domains = {
                split_entity_id(entity_id)[0] for entity_id in exposed_entities
            }
            intent_handlers = [
                intent_handler
                for intent_handler in intent_handlers
                if intent_handler.platforms is None
                or intent_handler.platforms & exposed_domains
            ]

        tools: list[llm.Tool] = [
            IntentTool(self.cached_slugify(intent_handler.intent_type), intent_handler)
            for intent_handler in intent_handlers
        ]

        if llm_context.assistant is not None:
            for state in self.hass.states.async_all(SCRIPT_DOMAIN):
                if not async_should_expose(
                    self.hass, llm_context.assistant, state.entity_id
                ):
                    continue

                tools.append(llm.ScriptTool(self.hass, state.entity_id))

        return tools


class IntentTool(llm.Tool):
    """LLM Tool representing an Intent."""

    def __init__(
        self,
        name: str,
        intent_handler: intent.IntentHandler,
    ) -> None:
        """Init the class."""
        self.name = name
        self.description = (
            intent_handler.description or f"Execute Home Assistant {self.name} intent"
        )
        self.extra_slots = None
        if not (slot_schema := intent_handler.slot_schema):
            return

        slot_schema = {**slot_schema}
        extra_slots = set()

        for field in ("preferred_area_id", "preferred_floor_id"):
            if field in slot_schema:
                extra_slots.add(field)
                del slot_schema[field]

        self.parameters = vol.Schema(slot_schema)
        if extra_slots:
            self.extra_slots = extra_slots

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        """Handle the intent."""
        slots = {key: {"value": val} for key, val in tool_input.tool_args.items()}

        if self.extra_slots and llm_context.device_id:
            device_reg = dr.async_get(hass)
            device = device_reg.async_get(llm_context.device_id)

            area: ar.AreaEntry | None = None
            floor: fr.FloorEntry | None = None
            if device:
                area_reg = ar.async_get(hass)
                if device.area_id and (area := area_reg.async_get_area(device.area_id)):
                    if area.floor_id:
                        floor_reg = fr.async_get(hass)
                        floor = floor_reg.async_get_floor(area.floor_id)

            for slot_name, slot_value in (
                ("preferred_area_id", area.id if area else None),
                ("preferred_floor_id", floor.floor_id if floor else None),
            ):
                if slot_value and slot_name in self.extra_slots:
                    slots[slot_name] = {"value": slot_value}

        intent_response = await intent.async_handle(
            hass=hass,
            platform=llm_context.platform,
            intent_type=self.name,
            slots=slots,
            text_input=None,
            context=llm_context.context,
            language=llm_context.language,
            assistant=llm_context.assistant,
            device_id=llm_context.device_id,
        )
        response = intent_response.as_dict()
        del response["language"]
        return response


def _get_exposed_entities(
    hass: HomeAssistant, assistant: str
) -> dict[str, dict[str, Any]]:
    """Get exposed entities."""
    area_registry = ar.async_get(hass)
    entity_registry = er.async_get(hass)
    device_registry = dr.async_get(hass)
    interesting_attributes = {
        "temperature",
        "current_temperature",
        "temperature_unit",
        "brightness",
        "humidity",
        "unit_of_measurement",
        "device_class",
        "current_position",
        "percentage",
        "volume_level",
        "media_title",
        "media_artist",
        "media_album_name",
    }

    entities = {}

    for state in hass.states.async_all():
        if not async_should_expose(hass, assistant, state.entity_id):
            continue

        description: str | None = None
        if state.domain == SCRIPT_DOMAIN:
            description, parameters = _get_cached_script_parameters(
                hass, state.entity_id
            )
            if parameters.schema:  # Only list scripts without input fields here
                continue

        entity_entry = entity_registry.async_get(state.entity_id)
        names = [state.name]
        area_names = []

        if entity_entry is not None:
            names.extend(entity_entry.aliases)
            if entity_entry.area_id and (
                area := area_registry.async_get_area(entity_entry.area_id)
            ):
                # Entity is in area
                area_names.append(area.name)
                area_names.extend(area.aliases)
            elif entity_entry.device_id and (
                device := device_registry.async_get(entity_entry.device_id)
            ):
                # Check device area
                if device.area_id and (
                    area := area_registry.async_get_area(device.area_id)
                ):
                    area_names.append(area.name)
                    area_names.extend(area.aliases)

        info: dict[str, Any] = {
            "names": ", ".join(str(n) for n in names),
            "domain": state.domain,
            "state": state.state,
        }

        if description:
            info["description"] = description

        if area_names:
            info["areas"] = ", ".join(str(n) for n in area_names)

        if attributes := {
            attr_name: str(attr_value)
            if isinstance(attr_value, (Enum, Decimal, int))
            else attr_value
            for attr_name, attr_value in state.attributes.items()
            if attr_name in interesting_attributes
        }:
            info["attributes"] = attributes

        entities[state.entity_id] = info

    return entities


def _get_cached_script_parameters(
    hass: HomeAssistant, entity_id: str
) -> tuple[str | None, vol.Schema]:
    """Get script description and schema."""
    entity_registry = er.async_get(hass)

    description = None
    parameters = vol.Schema({})
    entity_entry = entity_registry.async_get(entity_id)
    if entity_entry and entity_entry.unique_id:
        parameters_cache = hass.data.get(SCRIPT_PARAMETERS_CACHE)

        if parameters_cache is None:
            parameters_cache = hass.data[SCRIPT_PARAMETERS_CACHE] = {}

            @callback
            def clear_cache(event: Event) -> None:
                """Clear script parameter cache on script reload or delete."""
                if (
                    event.data[ATTR_DOMAIN] == SCRIPT_DOMAIN
                    and event.data[ATTR_SERVICE] in parameters_cache
                ):
                    parameters_cache.pop(event.data[ATTR_SERVICE])

            cancel = hass.bus.async_listen(EVENT_SERVICE_REMOVED, clear_cache)

            @callback
            def on_homeassistant_close(event: Event) -> None:
                """Cleanup."""
                cancel()

            hass.bus.async_listen_once(
                EVENT_HOMEASSISTANT_CLOSE, on_homeassistant_close
            )

        if entity_entry.unique_id in parameters_cache:
            return parameters_cache[entity_entry.unique_id]

        if service_desc := service.async_get_cached_service_description(
            hass, SCRIPT_DOMAIN, entity_entry.unique_id
        ):
            description = service_desc.get("description")
            schema: dict[vol.Marker, Any] = {}
            fields = service_desc.get("fields", {})

            for field, config in fields.items():
                field_description = config.get("description")
                if not field_description:
                    field_description = config.get("name")
                key: vol.Marker
                if config.get("required"):
                    key = vol.Required(field, description=field_description)
                else:
                    key = vol.Optional(field, description=field_description)
                if "selector" in config:
                    schema[key] = selector.selector(config["selector"])
                else:
                    schema[key] = cv.string

            parameters = vol.Schema(schema)

            aliases: list[str] = []
            if entity_entry.name:
                aliases.append(entity_entry.name)
            if entity_entry.aliases:
                aliases.extend(entity_entry.aliases)
            if aliases:
                if description:
                    description = description + ". Aliases: " + str(list(aliases))
                else:
                    description = "Aliases: " + str(list(aliases))

            parameters_cache[entity_entry.unique_id] = (description, parameters)

    return description, parameters


SCRIPT_PARAMETERS_CACHE: HassKey[dict[str, tuple[str | None, vol.Schema]]] = HassKey(
    "llm_script_parameters_cache"
)
