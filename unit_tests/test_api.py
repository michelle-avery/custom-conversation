"""Unit tests for the Custom Conversation API module."""

from unittest.mock import MagicMock, patch

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
import voluptuous as vol

from custom_components.custom_conversation.api import (
    CustomLLMAPI,
    IntentTool,
    _get_exposed_entities,
)
from custom_components.custom_conversation.const import (
    CONF_IGNORED_INTENTS,
    CONF_IGNORED_INTENTS_SECTION,
    LLM_API_ID,
)
from custom_components.custom_conversation.prompt_manager import (
    PromptContext,
    PromptManager,
)
from homeassistant.core import Context
from homeassistant.helpers import (
    device_registry as dr,
    entity_registry as er,
    intent,
    llm,
)


@pytest.fixture
def mock_prompt_manager() -> MagicMock:
    """Fixture for a mocked PromptManager."""
    return MagicMock(spec=PromptManager)

@pytest.fixture
def custom_llm_api(hass, config_entry, mock_prompt_manager) -> CustomLLMAPI:
    """Fixture for a CustomLLMAPI instance with mocked PromptManager."""
    with patch("custom_components.custom_conversation.api.PromptManager", return_value=mock_prompt_manager):
        return CustomLLMAPI(
            hass=hass,
            user_name="Test User",
            conversation_config_entry=config_entry,
        )

@pytest.fixture
def mock_exposed_entities_data() -> dict:
    """Fixture for a default mock_exposed_entities dictionary."""
    return {"light.test": {"names": "Test Light", "domain": "light", "state": "on"}}

@pytest.fixture
def mock_floor(floor_registry):
    """Fixture for a mocked floor."""
    return floor_registry.async_create("Test Floor")

@pytest.fixture
def mock_area(area_registry, mock_floor):
    """Fixture for a mocked area."""
    return area_registry.async_create("Test Area", floor_id=mock_floor.floor_id)

@pytest.fixture
def mock_assist_device(device_registry, mock_area, hass):
    """Fixture for a mocked assist device."""
    config_entry = MockConfigEntry(domain="assist.satellite", data={})
    config_entry.add_to_hass(hass)
    device_registry = dr.async_get(hass)
    device = device_registry.async_get_or_create(config_entry_id=config_entry.entry_id, identifiers={("assist", "test_device")}, name="Test Device")
    device_registry.async_update_device(device_id=device.id, area_id=mock_area.id)
    return device

@pytest.fixture
def mock_target_device(device_registry, mock_area, hass):
    """Fixture for a mocked light."""
    config_entry = MockConfigEntry(domain="light")
    config_entry.add_to_hass(hass)
    device_registry = dr.async_get(hass)
    device = device_registry.async_get_or_create(config_entry_id=config_entry.entry_id, identifiers={("light", "test_light")}, name="Test Light")
    device_registry.async_update_device(device_id=device.id, area_id=mock_area.id)
    return device

@pytest.fixture
def mock_target_entity(entity_registry, mock_target_device, hass):
    """Fixture for a mocked light entity."""
    entity_registry = er.async_get(hass)
    entity = entity_registry.async_get_or_create(
        domain="light",
        platform="test_platform",
        unique_id="test_light_unique_id",
        device_id=mock_target_device.id,
    )
    hass.states.async_set(
        entity.entity_id,
        "on",
        {
            "brightness": 100,
            "device_class": "light",
            "friendly_name": "Test Light",
        }
    )
    return entity

@pytest.fixture
def mock_script(entity_registry, hass):
    """Fixture for a mocked script entity."""
    entity_registry = er.async_get(hass)
    entity = entity_registry.async_get_or_create(
        domain="script",
        platform="test_platform",
        unique_id="my_script",
    )
    hass.states.async_set(
        entity.entity_id,
        "unknown",
        {
            "friendly_name": "My Script",
        }
    )
    return entity


@pytest.fixture
def mock_llm_context(mock_assist_device) -> MagicMock:
    """Fixture for a mocked LLMContext."""
    context = MagicMock(spec=llm.LLMContext)
    context.assistant = "conversation.home_assistant"
    context.language = "en"
    context.device_id = mock_assist_device.id
    context.platform = "test_platform"
    context.context = Context()
    return context

@pytest.mark.asyncio
async def test_custom_llm_api_init(hass, config_entry):
    """Test CustomLLMAPI initialization."""
    api = CustomLLMAPI(
        hass=hass,
        user_name="Test User",
        conversation_config_entry=config_entry,
    )
    assert api._hass is hass
    assert api._request_user_name == "Test User"
    assert api.conversation_config_entry is config_entry
    assert api.id == LLM_API_ID
    assert api.name == "Custom Conversation LLM API"

def test_custom_llm_api_set_langfuse_client(custom_llm_api, mock_prompt_manager):
    """Test setting the Langfuse client."""
    mock_client = MagicMock()
    custom_llm_api.set_langfuse_client(mock_client)
    mock_prompt_manager.set_langfuse_client.assert_called_once_with(mock_client)

@pytest.mark.asyncio
async def test_custom_llm_api_get_api_instance(custom_llm_api, mock_llm_context, mock_exposed_entities_data, mock_assist_device):
    """Test getting an API instance."""
    mock_api_prompt = "Test API Prompt"
    mock_tools = [MagicMock(spec=llm.Tool)]

    with patch("custom_components.custom_conversation.api._get_exposed_entities", return_value=mock_exposed_entities_data) as mock_get_exposed, \
         patch.object(custom_llm_api, "_async_get_api_prompt", return_value=mock_api_prompt) as mock_get_prompt, \
         patch.object(custom_llm_api, "_async_get_tools", return_value=mock_tools) as mock_get_tools, \
         patch("homeassistant.helpers.llm.APIInstance") as mock_api_instance_cls, \
         patch("homeassistant.helpers.llm.selector_serializer") as mock_serializer:

        instance = await custom_llm_api.async_get_api_instance(mock_llm_context)

        mock_get_exposed.assert_called_once_with(custom_llm_api.hass, mock_llm_context.assistant)
        mock_get_prompt.assert_called_once_with(mock_llm_context, mock_exposed_entities_data)
        mock_get_tools.assert_called_once_with(mock_llm_context, mock_exposed_entities_data)
        mock_api_instance_cls.assert_called_once_with(
            api=custom_llm_api,
            api_prompt=mock_api_prompt,
            llm_context=mock_llm_context,
            tools=mock_tools,
            custom_serializer=mock_serializer,
        )
        assert instance is mock_api_instance_cls.return_value

@pytest.mark.asyncio
async def test_custom_llm_api_get_api_prompt(custom_llm_api, hass, mock_llm_context, mock_prompt_manager, config_entry, mock_exposed_entities_data, device_registry, area_registry, floor_registry):
    """Test generating the API prompt."""

    with patch("custom_components.custom_conversation.api.async_device_supports_timers", return_value=True) as mock_supports_timers:

        expected_prompt = "Generated Prompt"
        mock_prompt_manager.get_api_prompt.return_value = expected_prompt

        prompt = await custom_llm_api._async_get_api_prompt(mock_llm_context, mock_exposed_entities_data)

        mock_supports_timers.assert_called_once_with(hass, mock_llm_context.device_id)

        mock_prompt_manager.get_api_prompt.assert_called_once()
        context_arg = mock_prompt_manager.get_api_prompt.call_args[0][0]
        config_entry_arg = mock_prompt_manager.get_api_prompt.call_args[0][1]

        assert isinstance(context_arg, PromptContext)
        assert context_arg.hass is hass
        assert context_arg.ha_name == "test home"
        assert context_arg.user_name == "Test User"
        assert context_arg.llm_context is mock_llm_context
        assert context_arg.location == "Test Area (floor: Test Floor)"
        assert context_arg.exposed_entities is mock_exposed_entities_data
        assert context_arg.supports_timers is True
        assert config_entry_arg is config_entry

        assert prompt == expected_prompt

@pytest.mark.asyncio
async def test_custom_llm_api_get_tools(custom_llm_api, hass, mock_llm_context, config_entry, mock_exposed_entities_data):
    """Test generating the list of tools."""
    mock_intent_handler_1 = MagicMock(spec=intent.IntentHandler, intent_type="HassTurnOn", description="Turn something on", slot_schema={"name": {"type": "string"}}, platforms={"light"})
    mock_intent_handler_2 = MagicMock(spec=intent.IntentHandler, intent_type="HassTurnOff", description="Turn something off", slot_schema=None, platforms=None)
    mock_intent_handler_timer = MagicMock(spec=intent.IntentHandler, intent_type=intent.INTENT_START_TIMER, description="Start timer", slot_schema=None)

    mock_intent_handlers = [mock_intent_handler_1, mock_intent_handler_2, mock_intent_handler_timer]
    mock_exposed_entities = mock_exposed_entities_data

    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_IGNORED_INTENTS_SECTION: {CONF_IGNORED_INTENTS: ["HassTurnOff"]}},
    )
    await hass.async_block_till_done()

    with patch("custom_components.custom_conversation.api.intent.async_get", return_value=mock_intent_handlers) as mock_intent_get, \
         patch("custom_components.custom_conversation.api.async_device_supports_timers", return_value=False) as mock_supports_timers, \
         patch("custom_components.custom_conversation.api.IntentTool") as mock_intent_tool_cls:

        tools = custom_llm_api._async_get_tools(mock_llm_context, mock_exposed_entities)

        mock_intent_get.assert_called_once_with(hass)
        mock_supports_timers.assert_called_once_with(hass, mock_llm_context.device_id)

        mock_intent_tool_cls.assert_called_once_with("HassTurnOn", mock_intent_handler_1)
        assert len(tools) == 1
        assert tools[0] is mock_intent_tool_cls.return_value


def test_intent_tool_init():
    """Test IntentTool initialization."""
    mock_handler = MagicMock(
        spec=intent.IntentHandler,
        intent_type="MyIntent",
        description="Does my intent",
        slot_schema={
            "name": {"type": "string", "required": True},
            "area": {"type": "string"},
            "preferred_area_id": {"type": "string"},
        }
    )

    tool = IntentTool("myintent", mock_handler)

    assert tool.name == "myintent"
    assert tool.description == "Does my intent"
    assert isinstance(tool.parameters, vol.Schema)
    assert "preferred_area_id" not in tool.parameters.schema
    assert "name" in tool.parameters.schema
    assert "area" in tool.parameters.schema
    assert tool.extra_slots == {"preferred_area_id"}

def test_intent_tool_init_no_schema():
    """Test IntentTool initialization with no slot schema."""
    mock_handler = MagicMock(
        spec=intent.IntentHandler,
        intent_type="SimpleIntent",
        description=None,
        slot_schema=None
    )
    tool = IntentTool("simpleintent", mock_handler)

    assert tool.name == "simpleintent"
    assert tool.description == "Execute Home Assistant simpleintent intent"
    assert tool.extra_slots is None

@pytest.mark.asyncio
async def test_intent_tool_async_call(hass, mock_llm_context, mock_target_device, mock_target_entity):
    """Test calling the IntentTool."""
    mock_handler = MagicMock(
        spec=intent.IntentHandler,
        intent_type="HassTurnOn",
        description="Turn on",
        slot_schema={
            "name": {"type": "string"},
            "area": {"type": "string"},
            "preferred_area_id": {"type": "string"},
        }
    )
    tool_input = llm.ToolInput(tool_name="HassTurnOn", tool_args={"name": "kitchen light", "area": "Test Area"})
    mock_intent_response = MagicMock(spec=intent.IntentResponse)
    mock_intent_response.as_dict.return_value = {"speech": {"plain": {"speech": "Turned on kitchen light"}}, "language": "en"}


    with patch("homeassistant.helpers.intent.async_handle", return_value=mock_intent_response) as mock_async_handle:

        tool = IntentTool("HassTurnOn", mock_handler)
        response = await tool.async_call(hass, tool_input, mock_llm_context)

        expected_slots = {
            "name": {"value": "kitchen light"},
            "area": {"value": "Test Area"},
            "preferred_area_id": {"value": "test_area"},
        }

        mock_async_handle.assert_called_once_with(
            hass=hass,
            platform=mock_llm_context.platform,
            intent_type="HassTurnOn",
            slots=expected_slots,
            text_input=None,
            context=mock_llm_context.context,
            language=mock_llm_context.language,
            assistant=mock_llm_context.assistant,
            device_id=mock_llm_context.device_id,
        )

        assert response == {"speech": {"plain": {"speech": "Turned on kitchen light"}}}



@patch("custom_components.custom_conversation.api._get_cached_script_parameters")
@patch("custom_components.custom_conversation.api.async_should_expose", return_value=True)
@patch("homeassistant.core.StateMachine.async_all")
def test_get_exposed_entities_with_area(mock_async_all, mock_should_expose, mock_get_script_params, hass, mock_area, mock_target_entity):
    """Test _get_exposed_entities for an entity with a direct area."""
    entity_registry = er.async_get(hass)
    entity_registry.async_update_entity(mock_target_entity.entity_id, area_id=mock_area.id)
    entity_id = mock_target_entity.entity_id
    mock_async_all.return_value = [hass.states.get(entity_id)]

    assistant_id = "conversation.test_assistant"
    entities = _get_exposed_entities(hass, assistant_id)

    mock_async_all.assert_called_once()
    mock_should_expose.assert_called_once_with(hass, assistant_id, entity_id)

    assert len(entities) == 1
    assert entity_id in entities
    light_info = entities[entity_id]
    assert light_info["names"] == "Test Light"
    assert light_info["domain"] == "light"
    assert light_info["state"] == "on"
    assert light_info["areas"] == "Test Area"
    assert light_info["attributes"] == {"brightness": "100", "device_class": "light"}
    assert "description" not in light_info
    mock_get_script_params.assert_not_called()


@patch("custom_components.custom_conversation.api._get_cached_script_parameters")
@patch("custom_components.custom_conversation.api.async_should_expose", return_value=True)
@patch("homeassistant.core.StateMachine.async_all")
def test_get_exposed_entities_with_device_area(mock_async_all, mock_should_expose, mock_get_script_params, hass, mock_target_entity):
    """Test _get_exposed_entities for an entity using device area fallback."""
    entity_id = mock_target_entity.entity_id
    mock_async_all.return_value = [hass.states.get(entity_id)]

    assistant_id = "conversation.test_assistant"
    entities = _get_exposed_entities(hass, assistant_id)

    mock_async_all.assert_called_once()
    mock_should_expose.assert_called_once_with(hass, assistant_id, entity_id)

    assert len(entities) == 1
    assert entity_id in entities
    light_info = entities[entity_id]
    assert light_info["names"] == "Test Light"
    assert light_info["domain"] == "light"
    assert light_info["state"] == "on"
    assert light_info["areas"] == "Test Area"
    assert light_info["attributes"] == {"brightness": "100", "device_class": "light"}
    assert "description" not in light_info
    mock_get_script_params.assert_not_called()


@patch("custom_components.custom_conversation.api._get_cached_script_parameters", return_value=("Script Desc", vol.Schema({})))
@patch("custom_components.custom_conversation.api.async_should_expose", return_value=True)
@patch("homeassistant.core.StateMachine.async_all")
def test_get_exposed_entities_script(mock_async_all, mock_should_expose, mock_get_script_params, hass, mock_script):
    """Test _get_exposed_entities for a script entity."""

    mock_async_all.return_value = [hass.states.get(mock_script.entity_id)]

    assistant_id = "conversation.test_assistant"
    entities = _get_exposed_entities(hass, assistant_id)

    mock_async_all.assert_called_once()
    mock_should_expose.assert_called_once_with(hass, assistant_id, "script.test_platform_my_script")
    mock_get_script_params.assert_called_once_with(hass, "script.test_platform_my_script")

    assert len(entities) == 1
    assert "script.test_platform_my_script" in entities
    script_info = entities["script.test_platform_my_script"]
    assert script_info["names"] == "My Script"
    assert script_info["domain"] == "script"
    assert script_info["state"] == "unknown"
    assert script_info["description"] == "Script Desc"
    assert "areas" not in script_info
    assert "attributes" not in script_info


@patch("custom_components.custom_conversation.api._get_cached_script_parameters")
@patch("custom_components.custom_conversation.api.async_should_expose", return_value=False)
@patch("homeassistant.core.StateMachine.async_all")
def test_get_exposed_entities_unexposed(mock_async_all, mock_should_expose, mock_get_script_params, hass, mock_target_entity):
    """Test _get_exposed_entities for an entity that should not be exposed."""

    mock_async_all.return_value = [hass.states.get(mock_target_entity.entity_id)]

    assistant_id = "conversation.test_assistant"
    entities = _get_exposed_entities(hass, assistant_id)

    mock_async_all.assert_called_once()
    mock_should_expose.assert_called_once_with(hass, assistant_id, mock_target_entity.entity_id)
    mock_get_script_params.assert_not_called()

    assert len(entities) == 0
    assert mock_target_entity.entity_id not in entities
