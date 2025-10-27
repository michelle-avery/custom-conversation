"""Unit tests for the cc_llm module."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.custom_conversation import CustomConversationConfigEntry
from custom_components.custom_conversation.cc_llm import async_update_llm_data
from custom_components.custom_conversation.prompt_manager import PromptManager
from homeassistant.auth.models import User
from homeassistant.components.conversation import (
    ChatLog,
    ConversationInput,
    SystemContent,
)
from homeassistant.core import Context, HomeAssistant


@pytest.fixture
def mock_user():
    """Mock user object."""
    user = MagicMock(spec=User)
    user.id = "test_user_id"
    user.name = "Test User"
    return user

@pytest.fixture
def mock_context(mock_user):
    """Mock context object."""
    context = MagicMock(spec=Context)
    context.user_id = mock_user.id
    return context

@pytest.fixture
def mock_user_input(mock_context):
    """Mock ConversationInput object."""
    return ConversationInput(
        text="Hello",
        context=mock_context,
        conversation_id="test-convo-id",
        device_id="test-device-id",
        satellite_id="test-satellite-id",
        language="en",
        agent_id="test-agent-id",
        extra_system_prompt=None,
    )

@pytest.fixture
def mock_chat_log(hass):
    """Mock ChatLog object."""
    log = ChatLog(hass, conversation_id="test-convo-id", content=[SystemContent(content="initial")])
    log.llm_api = None
    log.extra_system_prompt = None
    return log

@pytest.fixture
def mock_prompt_manager():
    """Mock PromptManager object."""
    manager = MagicMock(spec=PromptManager)
    manager.async_get_base_prompt = AsyncMock(return_value="Base Prompt")
    return manager

async def test_async_update_llm_data_no_api(
    hass: HomeAssistant,
    mock_user_input: ConversationInput,
    config_entry: CustomConversationConfigEntry,
    mock_chat_log: ChatLog,
    mock_prompt_manager: PromptManager,
    mock_user: User,
):
    """Test async_update_llm_data when llm_api_name is None."""
    with patch("homeassistant.auth.AuthManager.async_get_user", return_value=mock_user):
        await async_update_llm_data(
            hass,
            mock_user_input,
            config_entry,
            mock_chat_log,
            mock_prompt_manager,
            llm_api_name=None,
        )

    mock_prompt_manager.async_get_base_prompt.assert_awaited_once()
    prompt_context = mock_prompt_manager.async_get_base_prompt.call_args[0][0]
    assert prompt_context.ha_name == 'test home'
    assert prompt_context.user_name == 'Test User'
    assert mock_chat_log.content[0].content == "Base Prompt"
    assert mock_chat_log.content[0].role == "system"
    assert mock_chat_log.llm_api is None
    assert mock_chat_log.extra_system_prompt is None
