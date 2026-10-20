"""Tests for bounded asynchronous escalation state."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.custom_conversation import escalation
from custom_components.custom_conversation.const import (
    CONF_AGENTS_SECTION,
    CONF_CUSTOM_PROMPTS_SECTION,
    CONF_ENABLE_HASS_AGENT,
    CONF_ENABLE_LLM_AGENT,
    CONF_ESCALATION_ACKNOWLEDGEMENT,
    CONF_ESCALATION_DENYLIST,
    CONF_ESCALATION_DOCTRINE_ENTITY,
    CONF_ESCALATION_MODE,
    CONF_IGNORED_INTENTS,
    CONF_IGNORED_INTENTS_SECTION,
    CONF_LANGFUSE_SECTION,
    ESCALATION_EVENT,
    ESCALATION_MODE_ASYNC,
    ESCALATION_MODE_OFF,
)
from custom_components.custom_conversation.conversation import CustomConversationEntity
from homeassistant.components import conversation
from homeassistant.components.conversation.chat_log import (
    AssistantContent,
    UserContent,
    async_get_chat_log,
)
from homeassistant.core import Context, HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import chat_session
from homeassistant.setup import async_setup_component


@pytest.mark.parametrize(
    ("text", "doctrine"),
    [
        ("Show my reminders", "reminder"),
        ("Find documents", "document"),
        ("Check invoices", "invoice"),
        ("Find recipes", "recipe"),
        ("Find a recipe", "recipes"),
        ("Add this to the calendar", "add to the calendar"),
        ("Help me plan meals", "meal plan"),
    ],
)
def test_doctrine_normalizes_common_phrases(text: str, doctrine: str) -> None:
    """Singular and plural forms select the same route."""
    assert escalation.classify_transcript(text, doctrine).escalate


def test_plural_denylist_still_wins() -> None:
    """Plural security terms cannot bypass the denylist."""
    result = escalation.classify_transcript("Remember the alarms", "remember", "alarm")
    assert result.security_denied
    assert not result.escalate


def test_ledger_outcomes_are_isolated_and_terminal() -> None:
    """A result needs both IDs and cannot replace a completed answer."""
    ledger = escalation.HandoffLedger()
    ledger.start("one", "request", "Find a document")
    assert "pending" in ledger.system_block("one")
    assert ledger.system_block("two") == ""
    assert not ledger.finish("two", "request", "done", "Wrong conversation")
    assert not ledger.finish("one", "wrong", "done", "Wrong request")
    assert not ledger.finish("one", "request", "unknown", "Wrong status")
    assert ledger.finish("one", "request", "timed_out", "")
    assert "timed_out" in ledger.system_block("one")
    assert ledger.finish("one", "request", "done", "The document is ready")
    assert not ledger.finish("one", "request", "timed_out", "")
    assert not ledger.finish("one", "request", "done", "A duplicate")
    assert "The document is ready" in ledger.system_block("one")
    assert "A duplicate" not in ledger.system_block("one")


def test_ledger_ttl_and_duplicate_start() -> None:
    """Duplicate starts do not reset state or extend the retention period."""
    clock = SimpleNamespace(value=0)
    ledger = escalation.HandoffLedger(clock=lambda: clock.value)
    ledger.start("one", "request", "Original request")
    clock.value = 1799
    ledger.start("one", "request", "Duplicate request")
    assert "Original request" in ledger.system_block("one")
    assert "Duplicate request" not in ledger.system_block("one")
    clock.value = 1800
    assert ledger.system_block("one") == ""
    assert not ledger.finish("one", "request", "done", "Too late")


def test_ledger_caps_entries_and_conversations() -> None:
    """Both the per-conversation list and the number of conversations are bounded."""
    ledger = escalation.HandoffLedger()
    for number in range(11):
        ledger.start("one", f"request-{number}", f"task-{number:02d}")
    block = ledger.system_block("one")
    assert "task-00" not in block
    assert "task-10" in block
    assert not ledger.finish("one", "request-0", "done", "Evicted")
    for number in range(128):
        ledger.start(f"conversation-{number}", "request", "Recent request")
    assert ledger.system_block("one") == ""
    assert ledger.system_block("conversation-127")


def test_ledger_context_and_prompt_bound_untrusted_text() -> None:
    """Only recent user and assistant text enters the bounded context."""
    ledger = escalation.HandoffLedger()
    for number in range(4):
        ledger.start("one", str(number), f"task-{number}")
        ledger.finish("one", str(number), "done", "answer" * 1000)
    turns = [SimpleNamespace(role="system", content="secret-system-prompt")]
    turns.extend(UserContent(content=f"turn-{number}") for number in range(6))
    turns.append(SimpleNamespace(role="tool_result", content="secret-tool-result"))
    turns.append(SimpleNamespace(role="assistant", content=None))
    context = ledger.context("one", turns)
    assert "task-0" not in context
    assert "task-3" in context
    assert "turn-0" not in context
    assert "turn-2" in context
    assert "turn-5" in context
    assert "secret" not in context
    assert len(context) <= 1500
    assert len(ledger.system_block("one")) < 15000
    assert ledger.context("empty", []) == ""


def test_person_lookup_from_authenticated_context_only(hass: HomeAssistant) -> None:
    """Person attribution does not come from text or an unlinked person."""
    hass.states.async_set("person.account", "home", {"user_id": "user-1"})
    hass.states.async_set("person.other", "home", {"user_id": "user-2"})
    persons = hass.states.async_all("person")
    assert escalation.person_for_user("user-1", persons) == "person.account"
    assert escalation.person_for_user("missing", persons) is None
    assert escalation.person_for_user(None, persons) is None


@pytest.fixture
async def async_agent(hass: HomeAssistant, config_entry) -> CustomConversationEntity:
    """Use the real entity and chat log without an external model."""
    assert await async_setup_component(hass, "custom_conversation", {})
    await hass.async_block_till_done()
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            **config_entry.options,
            CONF_ESCALATION_MODE: ESCALATION_MODE_ASYNC,
            CONF_AGENTS_SECTION: {
                CONF_ENABLE_HASS_AGENT: False,
                CONF_ENABLE_LLM_AGENT: True,
            },
        },
    )
    await hass.async_block_till_done()
    entity = conversation.async_get_agent(hass, config_entry.entry_id)
    assert isinstance(entity, CustomConversationEntity)
    return entity


def make_input(entity, text: str, conversation_id: str | None = None, user_id=None):
    """Make a request with a fresh correlation ID."""
    return conversation.ConversationInput(
        text=text,
        context=Context(user_id=user_id),
        conversation_id=conversation_id,
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id=entity.entry.entry_id,
    )


async def model_turn(
    entity, text, conversation_id, answer="Model answer", user_id=None
):
    """Capture a real turn at the external model boundary."""
    captured = []

    async def completion(**kwargs):
        captured.extend(kwargs["messages"])

        async def stream():
            yield {"role": "assistant", "content": answer}

        return stream()

    with (
        patch(
            "custom_components.custom_conversation.conversation.async_update_llm_data",
            new=AsyncMock(return_value=None),
        ),
        patch.object(entity, "_async_generate_completion", new=completion),
    ):
        result = await entity.async_process(
            make_input(entity, text, conversation_id, user_id)
        )
    return result, captured


async def test_first_escalation_has_persistent_id_and_history(
    hass: HomeAssistant, async_agent
) -> None:
    """A new asynchronous turn can receive a result and continue the same chat."""
    events = []
    hass.bus.async_listen(ESCALATION_EVENT, events.append)
    hass.states.async_set("person.account", "home", {"user_id": "user-1"})
    user_input = make_input(async_agent, "Remember the document", user_id="user-1")
    result = await async_agent.async_process(user_input)
    await hass.async_block_till_done()
    assert result.conversation_id
    assert result.continue_conversation is False
    assert len(events) == 1
    data = events[0].data
    assert data["conversation_id"] == result.conversation_id
    assert data["text"] == user_input.text
    assert data["person_entity_id"] == "person.account"
    assert data["context"] == ""
    with (
        chat_session.async_get_chat_session(hass, result.conversation_id) as session,
        async_get_chat_log(hass, session) as log,
    ):
        assert isinstance(log.content[-2], UserContent)
        assert log.content[-1].content == result.response.speech["plain"]["speech"]
    _, messages = await model_turn(async_agent, "Status please", result.conversation_id)
    assert "pending" in str(messages)


async def test_context_excludes_current_request_and_primary_marker(
    hass: HomeAssistant, async_agent
) -> None:
    """Post-check escalation preserves prior turns, not the rejected response."""
    events = []
    hass.bus.async_listen(ESCALATION_EVENT, events.append)
    await model_turn(
        async_agent, "Earlier question", "conversation-1", "Earlier answer"
    )
    result, _ = await model_turn(
        async_agent, "Current request", "conversation-1", "[ESCALATE: memory]"
    )
    with (
        chat_session.async_get_chat_session(hass, "conversation-1") as session,
        async_get_chat_log(hass, session) as log,
    ):
        assert isinstance(log.content[-1], AssistantContent)
        assert log.content[-1].content == result.response.speech["plain"]["speech"]
        assert all(content.content != "[ESCALATE: memory]" for content in log.content)
    await hass.async_block_till_done()
    assert (
        result.response.speech["plain"]["speech"]
        == "Let me look into that and get back to you."
    )
    assert "Earlier question" in events[0].data["context"]
    assert "Earlier answer" in events[0].data["context"]
    assert "Current request" not in events[0].data["context"]
    assert "ESCALATE" not in events[0].data["context"]


async def test_result_listener_obeys_entity_lifecycle(
    hass: HomeAssistant, async_agent
) -> None:
    """Unloaded entities do not accept result events."""
    user_input = make_input(async_agent, "Find a document", "one")
    await async_agent.async_process(user_input)
    hass.bus.async_fire(
        "custom_conversation_escalation_result",
        {
            "conversation_id": "one",
            "correlation_id": user_input.context.id,
            "status": "done",
            "answer": "Found it",
        },
    )
    await hass.async_block_till_done()
    _, messages = await model_turn(async_agent, "Status please", "one")
    assert "Found it" in str(messages)
    assert await hass.config_entries.async_unload(async_agent.entry.entry_id)
    next_input = make_input(async_agent, "Find another document", "two")
    await async_agent.async_process(next_input)
    hass.bus.async_fire(
        "custom_conversation_escalation_result",
        {
            "conversation_id": "two",
            "correlation_id": next_input.context.id,
            "status": "done",
            "answer": "After unload",
        },
    )
    await hass.async_block_till_done()
    _, messages = await model_turn(async_agent, "Status please", "two")
    assert "pending" in str(messages)
    assert "After unload" not in str(messages)


@pytest.mark.parametrize(
    "bad_field", ["conversation_id", "correlation_id", "status", "answer"]
)
async def test_malformed_result_is_ignored(
    hass: HomeAssistant, async_agent, bad_field: str
) -> None:
    """Non-string event fields cannot corrupt ledger state."""
    user_input = make_input(async_agent, "Find a document", "one")
    await async_agent.async_process(user_input)
    data = {
        "conversation_id": "one",
        "correlation_id": user_input.context.id,
        "status": "done",
        "answer": "Found it",
    }
    data[bad_field] = {"not": "a string"}
    hass.bus.async_fire("custom_conversation_escalation_result", data)
    await hass.async_block_till_done()
    _, messages = await model_turn(async_agent, "Status please", "one")
    assert "pending" in str(messages)
    assert "Found it" not in str(messages)


@pytest.mark.parametrize("mode", [ESCALATION_MODE_ASYNC, ESCALATION_MODE_OFF])
async def test_handoff_status_reaches_model_only_when_enabled(
    hass: HomeAssistant, async_agent, mode: str
) -> None:
    """Completed hand-offs enter the next prompt without changing the off path."""
    hass.states.async_set("person.account", "home", {"user_id": "user-1"})
    user_input = make_input(async_agent, "Find a document", "one")
    await async_agent.async_process(user_input)
    hass.bus.async_fire(
        "custom_conversation_escalation_result",
        {
            "conversation_id": "one",
            "correlation_id": user_input.context.id,
            "status": "done",
            "answer": "The document is ready",
        },
    )
    await hass.async_block_till_done()
    if mode == ESCALATION_MODE_OFF:
        hass.config_entries.async_update_entry(
            async_agent.entry,
            options={**async_agent.entry.options, CONF_ESCALATION_MODE: mode},
        )
        await hass.async_block_till_done()
    result, captured = await model_turn(
        async_agent, "Is it ready?", "one", user_id="user-1"
    )
    assert result.response.speech["plain"]["speech"] == "Model answer"
    prompt_text = str(captured)
    assert ("The document is ready" in prompt_text) is (mode == ESCALATION_MODE_ASYNC)
    assert ("person.account" in prompt_text) is (mode == ESCALATION_MODE_ASYNC)


@pytest.mark.parametrize("role", ["user", "system", "tool"])
def test_unexposed_check_ignores_non_tool_call_text(role: str) -> None:
    """An entity ID in ordinary content is not a tool call."""
    assert (
        escalation.post_check_reason(
            "No tool call", [{"role": role, "content": "light.hidden"}], set()
        )
        is None
    )


def test_unexposed_check_accepts_exposed_tool_call() -> None:
    """An exposed entity does not need escalation."""
    message = {
        "role": "assistant",
        "tool_calls": [{"function": {"arguments": '{"entity_id": "light.visible"}'}}],
    }
    assert escalation.post_check_reason("Done", [message], {"light.visible"}) is None


async def test_denylist_blocks_primary_post_check(
    hass: HomeAssistant, async_agent
) -> None:
    """A primary sentinel cannot bypass the denylist."""
    events = []
    hass.bus.async_listen(ESCALATION_EVENT, events.append)
    result, _ = await model_turn(
        async_agent, "Remember the alarm", "one", "[ESCALATE: memory]"
    )
    await hass.async_block_till_done()
    assert result.response.speech["plain"]["speech"] == "[ESCALATE: memory]"
    assert events == []


@pytest.mark.parametrize("mode", ["off", "sync", "async"])
async def test_escalation_options_round_trip(
    hass: HomeAssistant, async_agent, mode: str
) -> None:
    """The options flow stores all four escalation options without loss."""
    flow = await hass.config_entries.options.async_init(async_agent.entry.entry_id)
    assert flow["type"] == FlowResultType.FORM
    options = {
        CONF_ESCALATION_MODE: mode,
        CONF_ESCALATION_DOCTRINE_ENTITY: "input_text.routing",
        CONF_ESCALATION_ACKNOWLEDGEMENT: "Checking.",
        CONF_ESCALATION_DENYLIST: "alarm; lock",
    }
    submitted = flow["data_schema"](
        {
            **options,
            CONF_AGENTS_SECTION: {},
            CONF_IGNORED_INTENTS_SECTION: {CONF_IGNORED_INTENTS: []},
            CONF_CUSTOM_PROMPTS_SECTION: {},
            CONF_LANGFUSE_SECTION: {},
        }
    )
    result = await hass.config_entries.options.async_configure(
        flow["flow_id"], submitted
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    for key, value in options.items():
        assert result["data"][key] == value
