"""The Custom Conversation integration."""

from __future__ import annotations

from functools import cache, partial

import openai
import slugify as unicode_slug
import voluptuous as vol

from homeassistant.components.climate import INTENT_GET_TEMPERATURE
from homeassistant.components.cover import INTENT_CLOSE_COVER, INTENT_OPEN_COVER
from homeassistant.components.homeassistant import async_should_expose
from homeassistant.components.intent import async_device_supports_timers
from homeassistant.components.script import DOMAIN as SCRIPT_DOMAIN
from homeassistant.components.weather import INTENT_GET_WEATHER
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
    callback,
    split_entity_id,
)
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    HomeAssistantError,
    ServiceValidationError,
)
from homeassistant.helpers import (
    area_registry as ar,
    config_validation as cv,
    device_registry as dr,
    floor_registry as fr,
    intent,
    llm,
    selector,
)
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import yaml
from homeassistant.util.json import JsonObjectType

from .const import CONF_BASE_URL, DOMAIN, LLM_API_ID, LOGGER

SERVICE_GENERATE_IMAGE = "generate_image"
PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type CustomConversationConfigEntry = ConfigEntry[openai.AsyncClient]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Custom Conversation."""

    # Make sure the API is registered
    if not any(x.id == LLM_API_ID for x in llm.async_get_apis(hass)):
        llm.async_register_api(hass, CustomLLMAPI(hass))

    async def render_image(call: ServiceCall) -> ServiceResponse:
        """Render an image with dall-e."""
        entry_id = call.data["config_entry"]
        entry = hass.config_entries.async_get_entry(entry_id)

        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry",
                translation_placeholders={"config_entry": entry_id},
            )

        client: openai.AsyncClient = entry.runtime_data

        try:
            response = await client.images.generate(
                model="dall-e-3",
                prompt=call.data["prompt"],
                size=call.data["size"],
                quality=call.data["quality"],
                style=call.data["style"],
                response_format="url",
                n=1,
            )
        except openai.OpenAIError as err:
            raise HomeAssistantError(f"Error generating image: {err}") from err

        return response.data[0].model_dump(exclude={"b64_json"})

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_IMAGE,
        render_image,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): selector.ConfigEntrySelector(
                    {
                        "integration": DOMAIN,
                    }
                ),
                vol.Required("prompt"): cv.string,
                vol.Optional("size", default="1024x1024"): vol.In(
                    ("1024x1024", "1024x1792", "1792x1024")
                ),
                vol.Optional("quality", default="standard"): vol.In(("standard", "hd")),
                vol.Optional("style", default="vivid"): vol.In(("vivid", "natural")),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )
    return True


async def async_setup_entry(
    hass: HomeAssistant, entry: CustomConversationConfigEntry
) -> bool:
    """Set up a  Custom Conversation from a config entry."""
    client = openai.AsyncOpenAI(
        api_key=entry.data[CONF_API_KEY],
        http_client=get_async_client(hass),
        base_url=entry.data[CONF_BASE_URL],
    )

    # Cache current platform data which gets added to each request (caching done by library)
    _ = await hass.async_add_executor_job(client.platform_headers)

    try:
        await hass.async_add_executor_job(client.with_options(timeout=10.0).models.list)
    except openai.AuthenticationError as err:
        LOGGER.error("Invalid API key: %s", err)
        return False
    except openai.OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload OpenAI."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


class CustomLLMAPI(llm.API):
    """An API for the Custom Conversation integration to use to call Home Assistant services."""

    IGNORE_INTENTS = {
        INTENT_GET_TEMPERATURE,
        INTENT_GET_WEATHER,
        INTENT_OPEN_COVER,  # deprecated
        INTENT_CLOSE_COVER,  # deprecated
        intent.INTENT_GET_STATE,
        intent.INTENT_NEVERMIND,
        intent.INTENT_TOGGLE,
        intent.INTENT_GET_CURRENT_DATE,
        intent.INTENT_GET_CURRENT_TIME,
    }

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the API."""
        super().__init__(hass=hass, id=LLM_API_ID, name="Custom Conversation LLM API")
        self.cached_slugify = cache(
            partial(unicode_slug.slugify, separator="_", lowercase=False)
        )

    async def async_get_api_instance(
        self, llm_context: llm.LLMContext
    ) -> llm.APIInstance:
        """Return an instance of the Custom Conversation LLM API."""
        if llm_context.assistant:
            exposed_entities: dict | None = llm._get_exposed_entities(
                self.hass, llm_context.assistant
            )
        else:
            exposed_entities = None

        return llm.APIInstance(
            api=self,
            api_prompt=self._async_get_api_prompt(llm_context, exposed_entities),
            llm_context=llm_context,
            tools=self._async_get_tools(llm_context, exposed_entities),
        )

    @callback
    def _async_get_api_prompt(
        self, llm_context: llm.LLMContext, exposed_entities: dict | None
    ) -> str:
        """Return the prompt for the API."""
        if not exposed_entities:
            return (
                "Only if the user wants to control a device, tell them to expose entities "
                "to their voice assistant in Home Assistant."
            )

        prompt = [
            (
                "When controlling Home Assistant always call the intent tools. "
                "Use HassTurnOn to lock and HassTurnOff to unlock a lock. "
                "When controlling a device, prefer passing just name and domain. "
                "When controlling an area, prefer passing just area name and domain."
            )
        ]
        area: ar.AreaEntry | None = None
        floor: fr.FloorEntry | None = None
        if llm_context.device_id:
            device_reg = dr.async_get(self.hass)
            device = device_reg.async_get(llm_context.device_id)

            if device:
                area_reg = ar.async_get(self.hass)
                if device.area_id and (area := area_reg.async_get_area(device.area_id)):
                    floor_reg = fr.async_get(self.hass)
                    if area.floor_id:
                        floor = floor_reg.async_get_floor(area.floor_id)

            extra = "and all generic commands like 'turn on the lights' should target this area."

        if floor and area:
            prompt.append(f"You are in area {area.name} (floor {floor.name}) {extra}")
        elif area:
            prompt.append(f"You are in area {area.name} {extra}")
        else:
            prompt.append(
                "When a user asks to turn on all devices of a specific type, "
                "ask user to specify an area, unless there is only one device of that type."
            )

        if not llm_context.device_id or not async_device_supports_timers(
            self.hass, llm_context.device_id
        ):
            prompt.append("This device is not able to start timers.")

        if exposed_entities:
            prompt.append(
                "An overview of the areas and the devices in this smart home:"
            )
            prompt.append(yaml.dump(list(exposed_entities.values())))

        return "\n".join(prompt)

    @callback
    def _async_get_tools(
        self, llm_context: llm.LLMContext, exposed_entities: dict | None
    ) -> list[llm.Tool]:
        """Return a list of LLM tools."""
        ignore_intents = self.IGNORE_INTENTS
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
            text_input=llm_context.user_prompt,
            context=llm_context.context,
            language=llm_context.language,
            assistant=llm_context.assistant,
            device_id=llm_context.device_id,
        )
        response = intent_response.as_dict()
        del response["language"]
        return response
