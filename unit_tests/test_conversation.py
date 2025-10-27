"""Unit tests for the Custom Conversation component."""
from unittest.mock import AsyncMock, Mock, patch

from litellm import RateLimitError
import pytest

from custom_components.custom_conversation import CustomConversationConfigEntry
from custom_components.custom_conversation.const import (
    CONF_AGENTS_SECTION,
    CONF_ENABLE_HASS_AGENT,
    CONF_ENABLE_LLM_AGENT,
    CONVERSATION_ERROR_EVENT,
    LLM_API_ID,
)
from custom_components.custom_conversation.conversation import CustomConversationEntity
from homeassistant.components import conversation
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent
from homeassistant.setup import async_setup_component


async def test_custom_conversation_entity_initialization(hass: HomeAssistant, config_entry: CustomConversationConfigEntry):
    """Test the initialization of CustomConversationEntity."""
    assert await async_setup_component(hass, "custom_conversation", {})
    await hass.async_block_till_done()
    state = hass.states.get("conversation.test")

    assert state
    assert state.attributes["supported_features"] == 0

    hass.config_entries.async_update_entry(
        config_entry,
        options={
            **config_entry.options,
            CONF_LLM_HASS_API: "assist",
        },
    )
    await hass.config_entries.async_reload(config_entry.entry_id)

    state = hass.states.get("conversation.test")
    assert state
    assert (
        state.attributes["supported_features"]
        == conversation.ConversationEntityFeature.CONTROL
    )

async def test_custom_conversation_tries_hass_agent_first(hass: HomeAssistant, config_entry: CustomConversationConfigEntry):
    """Test that CustomConversationEntity tries the Home Assistant agent first when both are enabled."""
    assert await async_setup_component(hass, "custom_conversation", {})
    await hass.async_block_till_done()
    mock_response = intent.IntentResponse(language="en", intent=Mock())
    mock_response.error_code = None
    mock_result = conversation.ConversationResult(mock_response, "test-conversation-id")
    with patch(
        "custom_components.custom_conversation.conversation.CustomConversationEntity._async_handle_message_with_hass", return_value=mock_result
    ) as mock_process_hass:

        hass.config_entries.async_update_entry(
            config_entry,
            options={
                **config_entry.options,
                CONF_LLM_HASS_API: LLM_API_ID,
                CONF_AGENTS_SECTION: {
                    CONF_ENABLE_HASS_AGENT: True,
                    CONF_ENABLE_LLM_AGENT: True,
                },
            },
        )
        await hass.config_entries.async_reload(config_entry.entry_id)

        result = await conversation.async_converse(hass, "hello", "test-conversation-id", Context(), agent_id=config_entry.entry_id)
    assert result.conversation_id == "test-conversation-id"
    assert mock_process_hass.called

async def test_custom_conversation_rate_limit_error(hass: HomeAssistant, config_entry: CustomConversationConfigEntry):
    """Test that rate limit errors are properly handled and event is fired."""
    assert await async_setup_component(hass, "custom_conversation", {})
    await hass.async_block_till_done()

    # Mock the process_hass method to return an error to force using LLM agent
    mock_response = intent.IntentResponse(language="en", intent=Mock())
    mock_response.error_code = intent.IntentResponseErrorCode.UNKNOWN
    mock_result = conversation.ConversationResult(mock_response, "test-conversation-id")

    class MockRateLimitError(RateLimitError):
        """Mock for RateLimitError that works with try/except."""

        def __init__(self, message="Rate limited - out of quota", response=None, llm_provider="test_provider", model="test_model"):
            super().__init__(message=message, response=response, llm_provider=llm_provider, model=model)
            self.body = message

        def __str__(self):
            return "Rate limited - out of quota"

    rate_limit_error = MockRateLimitError()

    events = []
    hass.bus.async_listen(CONVERSATION_ERROR_EVENT, lambda e: events.append(e))

    with patch(
        "custom_components.custom_conversation.conversation.CustomConversationEntity._async_handle_message_with_hass", 
        return_value=mock_result
    ), patch(
        "custom_components.custom_conversation.conversation.CustomConversationEntity._async_handle_message_with_llm", 
        side_effect=rate_limit_error
    ), patch(
        "custom_components.custom_conversation.conversation.CustomConversationEntity._async_fire_conversation_error",
        AsyncMock()
    ) as mock_fire_error:

        hass.config_entries.async_update_entry(
            config_entry,
            options={
                **config_entry.options,
                CONF_LLM_HASS_API: LLM_API_ID,
                CONF_AGENTS_SECTION: {
                    CONF_ENABLE_HASS_AGENT: True,
                    CONF_ENABLE_LLM_AGENT: True,
                },
            },
        )
        await hass.config_entries.async_reload(config_entry.entry_id)
        response = await conversation.async_converse(
                hass, "hello", "test-conversation-id", Context(), agent_id=config_entry.entry_id
            )

        assert response.response.speech["plain"]["speech"] == "Rate limited or insufficient funds"

        assert mock_fire_error.called
        call_args = mock_fire_error.call_args[0]
        assert call_args[0] == str(rate_limit_error)  # Now checking the string value
        assert call_args[1] == "LLM"

async def test_custom_conversation_openai_error(hass: HomeAssistant, config_entry: CustomConversationConfigEntry):
    """Test that general OpenAI errors are properly handled and event is fired."""
    assert await async_setup_component(hass, "custom_conversation", {})
    await hass.async_block_till_done()

    # Set up a custom async_process to raise a HomeAssistantError directly
    # This simulates the process failing with OpenAIError and converting it to HomeAssistantError
    async def mock_process(user_input):
        await entity._async_fire_conversation_error(
            "API connection error", "LLM", user_input, None
        )
        raise HomeAssistantError("Error talking to OpenAI API")

    entity = CustomConversationEntity(config_entry, Mock(), hass)
    entity.hass = hass

    with patch.object(
        CustomConversationEntity,
        "async_process",
        side_effect=mock_process
    ):

        hass.config_entries.async_update_entry(
            config_entry,
            options={
                **config_entry.options,
                CONF_LLM_HASS_API: LLM_API_ID,
                CONF_AGENTS_SECTION: {
                    CONF_ENABLE_HASS_AGENT: True,
                    CONF_ENABLE_LLM_AGENT: True,
                },
            },
        )
        await hass.config_entries.async_reload(config_entry.entry_id)

        events = []
        hass.bus.async_listen(CONVERSATION_ERROR_EVENT, lambda e: events.append(e))

        user_input = conversation.ConversationInput(
            text="Turn on the lights",
            context=Context(),
            conversation_id="test-conversation-id",
            device_id="test-device-id",
            satellite_id="test-device-satellite-id",
            language="en",
            agent_id=config_entry.entry_id
        )

        # The converse function should raise HomeAssistantError with appropriate message
        with pytest.raises(HomeAssistantError, match="Error talking to OpenAI API"):
            await entity.async_process(user_input)

        # Wait for event to be processed
        await hass.async_block_till_done()

        # Check that the error event was fired
        assert len(events) == 1
        event_data = events[0].data
        assert event_data["agent_id"] == config_entry.entry_id
        assert event_data["handling_agent"] == "LLM"
        assert event_data["request"] == "Turn on the lights"
        assert event_data["error"] == "API connection error"

async def test_async_fire_conversation_error(hass: HomeAssistant, config_entry: CustomConversationConfigEntry):
    """Test that _async_fire_conversation_error fires the expected event."""
    assert await async_setup_component(hass, "custom_conversation", {})
    await hass.async_block_till_done()

    entity = CustomConversationEntity(config_entry, Mock(), hass)
    entity.hass = hass

    events = []
    hass.bus.async_listen(CONVERSATION_ERROR_EVENT, lambda e: events.append(e))

    user_input = conversation.ConversationInput(
        text="Turn on the lights",
        context=Context(),
        conversation_id="test-conversation-id",
        device_id="test-device-id",
        satellite_id="test-satellite-id",
        language="en",
        agent_id=config_entry.entry_id
    )

    device_data = {
        "device_name": "Test Device",
        "device_area": "Living Room"
    }


    error = "Test error message"
    await entity._async_fire_conversation_error(error, "LLM", user_input, device_data)

    await hass.async_block_till_done()

    assert len(events) == 1
    event_data = events[0].data
    assert event_data["agent_id"] == config_entry.entry_id
    assert event_data["handling_agent"] == "LLM"
    assert event_data["device_id"] == "test-device-id"
    assert event_data["device_name"] == "Test Device"
    assert event_data["device_area"] == "Living Room"
    assert event_data["request"] == "Turn on the lights"
    assert event_data["error"] == "Test error message"
