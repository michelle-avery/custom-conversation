"""Test helpers that don't import the component."""
import sys
from unittest.mock import AsyncMock, MagicMock, Mock


def setup_mocks():
    """Set up all mocks before any component imports."""
    # Create mock modules
    mock_openai = MagicMock()
    mock_openai.AsyncOpenAI = MagicMock()
    client_instance = MagicMock()
    client_instance.platform_headers = MagicMock()
    client_instance.with_options = MagicMock(return_value=client_instance)
    client_instance.chat.completions.create = AsyncMock()
    mock_openai.AsyncOpenAI.return_value = client_instance

    # Create a mock langfuse client instance that get_client() returns
    mock_langfuse_client = MagicMock()
    mock_langfuse_client.update_current_span = MagicMock()
    mock_langfuse_client.get_current_observation_id = MagicMock(return_value="mock-observation-id")
    mock_langfuse_client.get_current_trace_id = MagicMock(return_value="mock-trace-id")

    # Create full langfuse structure
    mock_langfuse = Mock()
    mock_langfuse.openai = Mock(openai=mock_openai)
    mock_langfuse.model = Mock()
    mock_langfuse.model.Prompt = Mock
    mock_langfuse.model.PromptClient = Mock
    mock_langfuse.api = Mock()
    # Mock the observe decorator to simply return the function
    mock_langfuse.observe = lambda *args, **kwargs: lambda f: f
    # Mock get_client to return a mock langfuse client
    mock_langfuse.get_client = MagicMock(return_value=mock_langfuse_client)
    # Mock the Langfuse constructor (used for configuration)
    mock_langfuse.Langfuse = MagicMock(return_value=mock_langfuse_client)

    # Mock all required langfuse modules
    sys.modules['langfuse'] = mock_langfuse
    sys.modules['langfuse.openai'] = mock_langfuse.openai
    sys.modules['langfuse.openai.openai'] = mock_openai
    sys.modules['langfuse.model'] = mock_langfuse.model
    sys.modules['langfuse.api'] = mock_langfuse.api
    sys.modules['langfuse.api.resources'] = mock_langfuse.api.resources
    sys.modules['langfuse.api.resources.commons'] = mock_langfuse.api.resources.commons
    sys.modules['langfuse.api.resources.commons.types'] = mock_langfuse.api.resources.commons.types
    sys.modules['langfuse.types'] = mock_langfuse.types
