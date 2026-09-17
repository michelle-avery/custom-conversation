"""Tests for deterministic conversation escalation."""

from unittest.mock import AsyncMock, Mock, patch

from custom_components.custom_conversation.const import (
    CONF_AGENTS_SECTION,
    CONF_ENABLE_HASS_AGENT,
    CONF_ENABLE_LLM_AGENT,
    CONF_ESCALATION_ACKNOWLEDGEMENT,
    CONF_ESCALATION_MODE,
    ESCALATION_EVENT,
    ESCALATION_MODE_ASYNC,
    ESCALATION_MODE_OFF,
)
from custom_components.custom_conversation.conversation import CustomConversationEntity
from custom_components.custom_conversation.escalation import (
    classify_transcript,
    post_check_reason,
)
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import intent
from homeassistant.setup import async_setup_component


def test_word_boundary_does_not_match_substrings() -> None:
    """A doctrine phrase must not match inside another word."""
    assert classify_transcript("remember this").escalate
    assert not classify_transcript("September is here").escalate


def test_denylist_wins_over_doctrine() -> None:
    """A security phrase suppresses escalation even when doctrine also matches."""
    result = classify_transcript("Remember to open the gate")
    assert not result.escalate
    assert result.security_denied


def test_primary_post_check_sentinel() -> None:
    """The configured sentinel escalates a primary response."""
    assert post_check_reason("I cannot answer [ESCALATE: memory]", [], set()) == (
        "primary-sentinel"
    )


def test_primary_post_check_unexposed_entity() -> None:
    """A tool call naming an entity outside the exposed set escalates."""
    reason = post_check_reason(
        "Done",
        [{"role": "assistant", "tool_calls": [{"entity_id": "light.hidden"}]}],
        {"light.visible"},
    )
    assert reason == "primary-unexposed-entity:light.hidden"


async def test_async_doctrine_route_fires_event_and_returns_acknowledgement(
    hass: HomeAssistant, config_entry
) -> None:
    """Async doctrine routing acknowledges immediately and publishes its payload."""
    assert await async_setup_component(hass, "custom_conversation", {})
    await hass.async_block_till_done()
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            **config_entry.options,
            CONF_ESCALATION_MODE: ESCALATION_MODE_ASYNC,
            CONF_ESCALATION_ACKNOWLEDGEMENT: "Working on it.",
            CONF_AGENTS_SECTION: {
                CONF_ENABLE_HASS_AGENT: False,
                CONF_ENABLE_LLM_AGENT: True,
            },
        },
    )
    entity = CustomConversationEntity(config_entry, Mock(), hass)
    events = []
    hass.bus.async_listen(ESCALATION_EVENT, events.append)
    user_input = conversation.ConversationInput(
        text="Please remember this",
        context=Context(),
        conversation_id="conversation-1",
        device_id="device-1",
        satellite_id=None,
        language="en",
        agent_id=config_entry.entry_id,
    )
    with patch.object(
        entity, "_async_handle_message_with_llm", new_callable=AsyncMock
    ) as llm_handler:
        result = await entity.async_process(user_input)

    assert result.response.speech["plain"]["speech"] == "Working on it."
    assert result.continue_conversation is False
    llm_handler.assert_not_called()
    await hass.async_block_till_done()
    assert len(events) == 1
    assert events[0].data == {
        "text": "Please remember this",
        "conversation_id": "conversation-1",
        "device_id": "device-1",
        "language": "en",
        "reason": "doctrine",
        "correlation_id": user_input.context.id,
    }


async def test_off_mode_preserves_primary_response(
    hass: HomeAssistant, config_entry
) -> None:
    """The off mode does not alter the response path."""
    assert await async_setup_component(hass, "custom_conversation", {})
    await hass.async_block_till_done()
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            **config_entry.options,
            CONF_ESCALATION_MODE: ESCALATION_MODE_OFF,
            CONF_AGENTS_SECTION: {
                CONF_ENABLE_HASS_AGENT: False,
                CONF_ENABLE_LLM_AGENT: True,
            },
        },
    )
    entity = CustomConversationEntity(config_entry, Mock(), hass)
    response = conversation.ConversationResult(
        response=intent.IntentResponse(language="en"),
        conversation_id="conversation-2",
    )
    response.response.async_set_speech("Primary response")
    with patch.object(
        entity,
        "_async_handle_message_with_llm",
        new_callable=AsyncMock,
        return_value=(response, {}),
    ) as llm_handler:
        user_input = conversation.ConversationInput(
            text="Please remember this",
            context=Context(),
            conversation_id="conversation-2",
            device_id=None,
            satellite_id=None,
            language="en",
            agent_id=config_entry.entry_id,
        )
        result = await entity.async_process(user_input)

    assert result.response.speech["plain"]["speech"] == "Primary response"
    llm_handler.assert_awaited_once()
