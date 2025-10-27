"""Conversation support for Custom Conversation APIs."""

import ast
from collections.abc import AsyncGenerator, Callable
import json
from typing import TYPE_CHECKING, Any, Literal, Union, cast

from langfuse.decorators import langfuse_context, observe

if TYPE_CHECKING:
    from langfuse.types import PromptClient
from litellm import OpenAIError, RateLimitError, Router
from litellm.types.completion import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)
from litellm.types.llms.openai import ChatCompletionToolParam, Function
from litellm.types.utils import StreamingChatCompletionChunk
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.components.conversation.chat_log import (
    AssistantContent,
    AssistantContentDeltaDict,
    UserContent,
    async_get_chat_log,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import chat_session, device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import CustomConversationConfigEntry
from .api import IntentTool
from .cc_llm import async_update_llm_data
from .const import (
    CONF_AGENTS_SECTION,
    CONF_ENABLE_HASS_AGENT,
    CONF_ENABLE_LLM_AGENT,
    CONF_LANGFUSE_HOST,
    CONF_LANGFUSE_PUBLIC_KEY,
    CONF_LANGFUSE_SECRET_KEY,
    CONF_LANGFUSE_SECTION,
    CONF_LANGFUSE_TAGS,
    CONF_LANGFUSE_TRACING_ENABLED,
    CONF_MAX_TOKENS,
    CONF_PRIMARY_API_KEY,
    CONF_PRIMARY_BASE_URL,
    CONF_PRIMARY_CHAT_MODEL,
    CONF_PRIMARY_PROVIDER,
    CONF_SECONDARY_API_KEY,
    CONF_SECONDARY_BASE_URL,
    CONF_SECONDARY_CHAT_MODEL,
    CONF_SECONDARY_PROVIDER,
    CONF_SECONDARY_PROVIDER_ENABLED,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONVERSATION_ENDED_EVENT,
    CONVERSATION_ERROR_EVENT,
    CONVERSATION_STARTED_EVENT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DOMAIN,
    HOME_ASSISTANT_AGENT,
    LOGGER,
)
from .prompt_manager import PromptManager

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


def _fix_invalid_arguments(value: Any) -> Any:
    """Attempt to repair incorrectly formatted json function arguments.

    Small models (for example llama3.1 8B) may produce invalid argument values
    which we attempt to repair here.
    """
    if not isinstance(value, str):
        return value
    if (value.startswith("[") and value.endswith("]")) or (
        value.startswith("{") and value.endswith("}")
    ):
        try:
            return json.loads(value)
        except json.decoder.JSONDecodeError:
            pass
    return value


def _parse_tool_args(arguments: dict[str, Any]) -> dict[str, Any]:
    """Rewrite tool arguments.

    This function improves tool use quality by fixing common mistakes made by
    small local tool use models. This will repair invalid json arguments and
    omit unnecessary arguments with empty values that will fail intent parsing.
    """
    if not isinstance(arguments, dict):
        try:
            arguments = arguments.replace("null", "None")
            arguments = ast.literal_eval(arguments)
        except ValueError as err:
            LOGGER.error("Failed to parse tool arguments: %s", arguments)
            raise HomeAssistantError("Failed to parse tool arguments") from err
    return {k: _fix_invalid_arguments(v) for k, v in arguments.items() if v}


def _get_llm_details(messages: list[ChatCompletionMessageParam]) -> dict:
    """Get the LLM details from the messages."""
    llm_details = {}
    new_tags = []
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            if "tool_calls" not in llm_details:
                llm_details["tool_calls"] = []
            for tool_call in message["tool_calls"]:
                tool_call_data = {
                    "tool_name": tool_call["function"]["name"],
                    "tool_args": tool_call["function"]["arguments"],
                    "tool_call_id": tool_call["id"],
                }
                llm_details["tool_calls"].append(tool_call_data)
                new_tags.append(f"intent:{tool_call['function']['name']}")
        if message.get("role") == "tool":
            message_tool_call_id = message.get("tool_call_id")
            for tool_call_request in llm_details.get("tool_calls", []):
                if tool_call_request.get("tool_call_id") == message_tool_call_id:
                    tool_call_request["tool_response"] = json.loads(message["content"])
                    new_tags.extend(
                        [
                            f"affected_entity:{entity['id']}"
                            for entity in tool_call_request["tool_response"]
                            .get("data", {})
                            .get("success", [])
                        ]
                    )
                    new_tags.extend(
                        [
                            f"failed_entity:{entity['id']}"
                            for entity in tool_call_request["tool_response"]
                            .get("data", {})
                            .get("failure", [])
                        ]
                    )
    return llm_details, new_tags


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: CustomConversationConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    langfuse_client = None
    if DOMAIN in hass.data and config_entry.entry_id in hass.data[DOMAIN]:
        langfuse_client = hass.data[DOMAIN][config_entry.entry_id].get(
            "langfuse_client"
        )
    prompt_manager = PromptManager(hass)
    if langfuse_client:
        prompt_manager.set_langfuse_client(langfuse_client)
    agent = CustomConversationEntity(config_entry, prompt_manager, hass)
    async_add_entities([agent])


def _format_tool(
    tool: IntentTool, custom_serializer: Callable[[Any], Any] | None
) -> ChatCompletionToolParam:
    """Format tool specification."""
    tool_spec = {
        "name": tool.name,
        "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
    }
    if tool.description:
        tool_spec["description"] = tool.description
    return ChatCompletionToolParam(type="function", function=tool_spec)


def _convert_content_to_param(
    content: conversation.Content,
) -> ChatCompletionMessageParam:
    """Convert any native chat message for this agent to the native format."""
    if content.role == "tool_result":
        assert type(content) is conversation.ToolResultContent
        return ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=content.tool_call_id,
            content=json.dumps(content.tool_result),
        )
    if content.role != "assistant" or not content.tool_calls:
        role = content.role
        if role == "system":
            role = "developer"
        return cast(
            ChatCompletionMessageParam,
            {
                "role": content.role,
                "content": content.content,
            },
        )

    assert type(content) is conversation.AssistantContent
    return ChatCompletionAssistantMessageParam(
        role="assistant",
        content=content.content,
        tool_calls=[
            ChatCompletionMessageToolCallParam(
                id=tool_call.id,
                function=Function(
                    arguments=json.dumps(tool_call.tool_args),
                    name=tool_call.tool_name,
                ),
                type="function",
            )
            for tool_call in content.tool_calls
        ],
    )


async def _transform_litellm_stream(
    result: AsyncGenerator[StreamingChatCompletionChunk, None],
) -> AsyncGenerator[AssistantContentDeltaDict, None]:
    """Transform a LiteLLM delta stream into HA format."""
    current_tool_call: dict | None = None

    async for chunk in result:
        LOGGER.debug("Received chunk: %s", chunk)
        if not chunk.choices:
            if chunk.usage:
                LOGGER.debug("Received usage chunk: %s", chunk.usage)
            continue

        choice = chunk.choices[0]

        if choice.finish_reason:
            if current_tool_call:
                # Gemini's OpenAI interface doesn't generate ids for tool calls, so we'll create one from the index
                if not current_tool_call.get("id"):
                    current_tool_call["id"] = f"call_{current_tool_call['index']}"
                yield {
                    "tool_calls": [
                        llm.ToolInput(
                            id=current_tool_call.get("id"),
                            tool_name=current_tool_call.get("name"),
                            tool_args=_parse_tool_args(
                                current_tool_call.get("tool_args", "{}")
                            ),
                        )
                    ]
                }
            current_tool_call = None
            continue

        delta = choice.delta

        # Yield messages that don't involve tool calls
        if current_tool_call is None and not delta.tool_calls:
            yield {
                key: value
                for key in ("role", "content")
                if (value := getattr(delta, key)) is not None
            }
            continue

        # When doing tool calls, we should always have a tool call
        # object or we have gotten stopped above with a finish_reason set
        if (
            not delta.tool_calls
            or not (delta_tool_call := delta.tool_calls[0])
            or not delta_tool_call.function
        ):
            raise ValueError("Expected delta with tool call")

        if current_tool_call and delta_tool_call.index == current_tool_call["index"]:
            current_tool_call["tool_args"] += delta_tool_call.function.arguments or ""
            continue

        # We got a tool call with new index, so we need to yield the previous
        if current_tool_call:
            yield {
                "tool_calls": [
                    llm.ToolInput(
                        id=current_tool_call["id"],
                        tool_name=current_tool_call["name"],
                        tool_args=json.loads(current_tool_call["tool_args"]),
                    )
                ]
            }

        # Gemini's OpenAI interface doesn't generate ids for tool calls, so we'll create one from the index
        if not delta_tool_call.id:
            delta_tool_call.id = f"call_{delta_tool_call.index}"
        current_tool_call = {
            "index": delta_tool_call.index,
            "id": delta_tool_call.id,
            "name": delta_tool_call.function.name,
            "tool_args": delta_tool_call.function.arguments or "",
        }


async def _remove_failed_hass_agent_messages(
    content: list[conversation.Content],
) -> list[conversation.Content]:
    """Remove failed messages from the HASS agent."""
    # If the last two messages are AssistantContent followed by UserContent, remove them
    if len(content) >= 2 and isinstance(content[-1], AssistantContent) and isinstance(
        content[-2], UserContent
    ):
        content = content[:-2]
    return content


class CustomConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Custom conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True

    def __init__(
        self,
        entry: CustomConversationConfigEntry,
        prompt_manager: PromptManager,
        hass: HomeAssistant,
    ) -> None:
        """Initialize the agent."""
        self.entry = entry
        self.hass = hass
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Custom",
            model="Custom Conversation",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )
        self.prompt_manager = prompt_manager
        if entry.options.get(CONF_LANGFUSE_SECTION, {}).get(
            CONF_LANGFUSE_TRACING_ENABLED, False
        ):
            try:
                hass.async_add_executor_job(
                    lambda: langfuse_context.configure(
                        host=entry.options[CONF_LANGFUSE_SECTION][CONF_LANGFUSE_HOST],
                        public_key=entry.options[CONF_LANGFUSE_SECTION][
                            CONF_LANGFUSE_PUBLIC_KEY
                        ],
                        secret_key=entry.options[CONF_LANGFUSE_SECTION][
                            CONF_LANGFUSE_SECRET_KEY
                        ],
                    )
                )
            except ValueError as e:
                LOGGER.error("Error configuring langfuse: %s", e)

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @observe(name="cc_async_process")
    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        return await self._async_handle_message(user_input)

    @observe(name="cc_handle_message")
    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
    ) -> conversation.ConversationResult:
        """Process enabled agents started with the built in agent."""
        LOGGER.debug("Processing user input: %s", user_input)
        assert user_input.agent_id
        options = self.entry.options
        device_registry = dr.async_get(self.hass)
        device = device_registry.async_get(user_input.device_id)
        device_data = {
            "device_id": user_input.device_id,
            "device_name": device.name if device else "Unknown",
            "device_area": device.area_id if device else "Unknown",
        }
        device_tags = [
            f"device_id:{device_data['device_id']}",
            f"device_name:{device_data['device_name']}",
            f"device_area:{device_data['device_area']}",
        ]
        user_configured_tags = options.get(CONF_LANGFUSE_SECTION, {}).get(
            CONF_LANGFUSE_TAGS, []
        )
        new_tags = user_configured_tags + device_tags

        langfuse_context.update_current_trace(tags=new_tags)
        event_data = {
            "agent_id": user_input.agent_id,
            "conversation_id": user_input.conversation_id,
            "language": user_input.language,
            "device_id": user_input.device_id,
            "device_name": device_data["device_name"],
            "device_area": device_data["device_area"],
            "text": user_input.text,
            "user_id": user_input.context.user_id,
        }

        self.hass.bus.async_fire(CONVERSATION_STARTED_EVENT, event_data)
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_error(
            intent.IntentResponseErrorCode.UNKNOWN,
            "Sorry, there are no enabled Agents",
        )
        result = conversation.ConversationResult(
            response=intent_response, conversation_id=user_input.conversation_id
        )

        if options.get(CONF_AGENTS_SECTION, {}).get(CONF_ENABLE_HASS_AGENT):
            LOGGER.debug("Processing with Home Assistant agent")
            with (
                chat_session.async_get_chat_session(
                    self.hass, user_input.conversation_id
                ) as session,
                async_get_chat_log(self.hass, session, user_input) as chat_log,
            ):
                result = await self._async_handle_message_with_hass(user_input)
                LOGGER.debug("Received response: %s", result.response.speech)
                if result.response.error_code is None:
                    await self._async_fire_conversation_ended(
                        result,
                        HOME_ASSISTANT_AGENT,
                        user_input,
                        device_data=device_data,
                    )
                    new_tags = ["handling_agent:home_assistant"]
                    if result.response.intent.intent_type is not None:
                        new_tags.append(f"intent:{result.response.intent.intent_type}")
                    if len(result.response.success_results) > 0:
                        for success_result in result.response.success_results:
                            new_tags.append(f"affected_entity:{success_result.id}")
                    langfuse_context.update_current_observation(output=result.as_dict())
                    langfuse_context.update_current_trace(tags=new_tags)
                    return conversation.ConversationResult(
                        response=result.response,
                        conversation_id=session.conversation_id,
                    )
                # If we're about to call the LLM Agent next, we want to delete the last two messages
                if options.get(CONF_AGENTS_SECTION, {}).get(CONF_ENABLE_LLM_AGENT):
                    chat_log.content = await _remove_failed_hass_agent_messages(
                        chat_log.content
                    )

        if options.get(CONF_AGENTS_SECTION, {}).get(CONF_ENABLE_LLM_AGENT):
            LOGGER.debug("Processing with LLM agent")
            try:
                with (
                    chat_session.async_get_chat_session(
                        self.hass, user_input.conversation_id
                    ) as session,
                    async_get_chat_log(self.hass, session, user_input) as chat_log,
                ):
                    LOGGER.debug("Trying to handle the message with LLM")
                    result, llm_data = await self._async_handle_message_with_llm(
                        user_input, chat_log
                    )
                    LOGGER.debug("Received response: %s", result.response.speech)
                    if result.response.error_code is None:
                        await self._async_fire_conversation_ended(
                            result,
                            "LLM",
                            user_input,
                            llm_data=llm_data,
                            device_data=device_data,
                        )
                        langfuse_context.update_current_trace(
                            tags=["handling_agent:llm"]
                        )
                    else:
                        await self._async_fire_conversation_error(
                            result.response.error_code,
                            "LLM",
                            user_input,
                            device_data=device_data,
                        )
            except RateLimitError as err:
                error_message = getattr(err, "body", str(err))
                await self._async_fire_conversation_error(
                    error_message,
                    "LLM",
                    user_input,
                    device_data=device_data,
                )
                raise HomeAssistantError("Rate limited or insufficient funds") from err
            except OpenAIError as err:
                error_message = getattr(err, "body", str(err))
                await self._async_fire_conversation_error(
                    error_message,
                    "LLM",
                    user_input,
                    device_data=device_data,
                )
                raise HomeAssistantError("Error talking to OpenAI API") from err
        return result

    @observe(name="cc_handle_message_with_hass")
    async def _async_handle_message_with_hass(
        self,
        user_input: conversation.ConversationInput,
    ) -> conversation.ConversationResult:
        """Process a sentence with the Home Assistant agent."""
        hass_agent = conversation.async_get_agent(self.hass, HOME_ASSISTANT_AGENT)
        if hass_agent is None:
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "Sorry, I had a problem talking to Home Assistant",
            )
            langfuse_context.update_current_observation(
                output=intent_response.as_dict()
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=user_input.conversation_id
            )
        response = await hass_agent.async_process(user_input)
        if response.response.intent:
            if response.response.intent.intent_type is not None:
                LOGGER.debug(
                    "Hass agent handled intent_type: %s",
                    response.response.intent.intent_type,
                )
            if response.response.intent.slots is not None:
                LOGGER.debug(
                    "Hass agent handled intent with slots: %s",
                    response.response.intent.slots,
                )
        if response.response.response_type is not None:
            LOGGER.debug(
                "Hass agent returned response_type: %s",
                response.response.response_type,
            )
        if response.response.error_code is not None:
            LOGGER.debug(
                "Hass agent responded with error_code: %s", response.response.error_code
            )
        langfuse_context.update_current_observation(output=response.as_dict())
        return response

    @observe(name="cc_handle_message_with_llm")
    async def _async_handle_message_with_llm(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> tuple[conversation.ConversationResult, dict]:
        """Process a sentence with the llm."""

        try:
            LOGGER.debug("Updating LLM Data")
            llm_api = self.entry.options.get(CONF_LLM_HASS_API)
            if llm_api == "none":
                llm_api = None
            prompt_object = await async_update_llm_data(
                self.hass,
                user_input,
                self.entry,
                chat_log,
                self.prompt_manager,
                llm_api,
            )
            if prompt_object:
                LOGGER.debug(
                    "Prompt name: %s, version: %s",
                    prompt_object.name,
                    prompt_object.version,
                )
            else:
                LOGGER.debug("No prompt object found")

        except conversation.ConverseError as err:
            return (
                err.as_conversation_result(),
                {},
            )

        tools: list[ChatCompletionToolParam] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]
        messages: list[ChatCompletionMessageParam] = [
            _convert_content_to_param(content) for content in chat_log.content
        ]
        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            LOGGER.debug("Iteration %s, messages: %s", _iteration, messages)
            transformed_stream = await self._async_generate_completion(
                entry=self.entry,
                messages=messages,
                tools=tools,
                conversation_id=chat_log.conversation_id,
                prompt=prompt_object,
            )

            try:
                messages.extend(
                    [
                        _convert_content_to_param(content)
                        async for content in chat_log.async_add_delta_content_stream(
                            user_input.agent_id, transformed_stream
                        )
                    ]
                )
            except HomeAssistantError as err:
                LOGGER.error("Error processing LLM stream: %s", err)
                raise
            except Exception as err:
                LOGGER.error("Unexpected error processing LLM stream: %s", err)
                raise HomeAssistantError("Error processing LLM response") from err

            if not chat_log.unresponded_tool_results:
                break

        final_assistant_message = chat_log.content[-1]
        if not isinstance(final_assistant_message, AssistantContent):
            # This should not happen if the stream processed correctly
            LOGGER.error(
                "Last message in chat log is not AssistantContent: %s",
                final_assistant_message,
            )
            raise HomeAssistantError("LLM response processing failed")

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(final_assistant_message.content or "")

        llm_details, new_tags = _get_llm_details(messages)
        langfuse_context.update_current_trace(tags=new_tags)

        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        ), llm_details

    @observe(
        name="cc_generate_completion",
        as_type="generation",
        capture_input=False,
        capture_output=False,
    )
    async def _async_generate_completion(
        self,
        entry: CustomConversationConfigEntry,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam] | None,
        conversation_id: str,
        prompt: Union["PromptClient", None] = None,
    ) -> AsyncGenerator[AssistantContentDeltaDict, None]:
        """Generate a completion stream from the LLM."""
        cleaned_input = {
            "messages": messages,
            "tools": tools,
            "conversation_id": conversation_id,
            "prompt": prompt.__dict__ if prompt else None,
            "config_entry": {
                "entry_id": entry.entry_id,
                "title": entry.title,
                "options": {**entry.options},
            },
        }
        langfuse_context.update_current_observation(
            input=cleaned_input,
        )
        generation_id = langfuse_context.get_current_observation_id()
        existing_trace_id = langfuse_context.get_current_trace_id()
        primary_model = f"{entry.data.get(CONF_PRIMARY_PROVIDER)}/{entry.data.get(CONF_PRIMARY_CHAT_MODEL)}"
        secondary_model = (
            f"{entry.data.get(CONF_SECONDARY_PROVIDER)}/{entry.data.get(CONF_SECONDARY_CHAT_MODEL)}"
            if entry.data.get(CONF_SECONDARY_PROVIDER_ENABLED)
            else ""
        )
        fallbacks = []
        model_list = [
            {
                "model_name": primary_model,
                "litellm_params": {
                    "model": primary_model,
                    "api_base": entry.data.get(CONF_PRIMARY_BASE_URL),
                    "api_key": entry.data.get(CONF_PRIMARY_API_KEY),
                },
            },
        ]
        if entry.data.get(CONF_SECONDARY_PROVIDER_ENABLED):
            model_list.append(
                {
                    "model_name": secondary_model,
                    "litellm_params": {
                        "model": secondary_model,
                        "api_base": entry.data.get(CONF_SECONDARY_BASE_URL),
                        "api_key": entry.data.get(CONF_SECONDARY_API_KEY),
                    },
                }
            )
            fallbacks = [{primary_model: [secondary_model]}]

        router = Router(model_list=model_list, fallbacks=fallbacks)

        temperature = entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_p = entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        max_tokens = entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)

        langfuse_params = entry.options.get(CONF_LANGFUSE_SECTION, {})

        completion_kwargs = {
            "model": primary_model,
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "temperature": temperature,
            "user": conversation_id,
            "stream": True,
            "metadata": {
                "generation_id": generation_id,
                "existing_trace_id": existing_trace_id,
                "generation_name": "cc_generate_completion",
                "prompt": prompt.__dict__ if prompt else None,
            },
            "langfuse_secret_key": langfuse_params.get(CONF_LANGFUSE_SECRET_KEY),
            "langfuse_public_key": langfuse_params.get(CONF_LANGFUSE_PUBLIC_KEY),
            "langfuse_host": langfuse_params.get(CONF_LANGFUSE_HOST),
            "stream_options": {"include_usage": True},
            "callbacks": ["langfuse"]
            if langfuse_params.get(CONF_LANGFUSE_TRACING_ENABLED)
            else None,
        }

        try:
            raw_stream: AsyncGenerator[
                StreamingChatCompletionChunk
            ] = await router.acompletion(**completion_kwargs)
            langfuse_context.update_current_observation(prompt=prompt)

            return _transform_litellm_stream(raw_stream)

        except RateLimitError as err:
            LOGGER.error("Rate limit error during acompletion: %s", err)
            raise
        except OpenAIError as err:
            LOGGER.error("API error during acompletion: %s", err)
            raise
        except Exception as err:
            LOGGER.error(
                "Unexpected error generating completion with acompletion: %s", err
            )
            debug_kwargs = {
                k: v
                for k, v in completion_kwargs.items()
                if k not in ["api_key", "langfuse_secret_key"]
            }
            LOGGER.debug(
                "acompletion kwargs (sensitive info removed): %s", debug_kwargs
            )
            raise HomeAssistantError("Error generating LLM completion stream") from err

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features
        await hass.config_entries.async_reload(entry.entry_id)

    async def _async_fire_card_requested_event(
        self, conversation_id: str, device_id: str, card: dict
    ) -> None:
        """Fire an event to request a card be displayed."""
        self.hass.bus.async_fire(
            "assistant_card_requested",
            {
                "conversation_id": conversation_id,
                "device_id": device_id,
                "card": card,
            },
        )

    async def _async_fire_conversation_error(
        self,
        error: str,
        agent: str,
        user_input: conversation.ConversationInput,
        device_data: dict | None = None,
    ) -> None:
        """Fire an event to notify that an error occurred."""
        event_data = {
            "agent_id": user_input.agent_id,
            "handling_agent": agent,
            "device_id": user_input.device_id,
            "device_name": device_data.get("device_name") if device_data else "Unknown",
            "device_area": device_data.get("device_area") if device_data else "Unknown",
            "request": user_input.text,
            "error": error,
        }
        self.hass.bus.async_fire(CONVERSATION_ERROR_EVENT, event_data)

    async def _async_fire_conversation_ended(
        self,
        result: conversation.ConversationResult,
        agent: str,
        user_input: conversation.ConversationInput,
        llm_data: dict | None = None,
        device_data: dict | None = None,
    ) -> None:
        """Fire an event to notify that a conversation has completed."""
        event_data = {
            "agent_id": user_input.agent_id,
            "handling_agent": agent,
            "device_id": user_input.device_id,
            "device_name": device_data.get("device_name") if device_data else "Unknown",
            "device_area": device_data.get("device_area") if device_data else "Unknown",
            "request": user_input.text,
            "result": result.as_dict(),
        }
        if llm_data:
            # If there's any card in the llm_data, we attach one to the response
            if any(
                "card" in tool_call.get("tool_response", {})
                for tool_call in llm_data.get("tool_calls", [])
            ):
                event_data["result"]["response"]["card"] = choose_card(
                    llm_data["tool_calls"]
                )
            event_data["llm_data"] = llm_data
            # If any of the tool calls has data matching intent entities, we attach it to the response
            data_dict = {"targets": [], "success": [], "failed": []}
            for tool_call in llm_data.get("tool_calls", []):
                tool_response = tool_call.get("tool_response", {}).get("data", {})
                for field in ("targets", "success", "failed"):
                    if values := tool_response.get(field, False):
                        data_dict[field].extend(values)
            event_data["result"]["response"]["data"].update(data_dict)
        self.hass.bus.async_fire(CONVERSATION_ENDED_EVENT, event_data)


def choose_card(tool_calls):
    """Choose the most likely card from the tool calls."""
    # It's possible that multiple tools have requested cards, but we only want to show one. For now, we'll choose the last tool call that has a card response.
    filtered_tool_calls = [
        tool_call
        for tool_call in tool_calls
        if isinstance(tool_call.get("tool_response"), dict)
        and "card" in tool_call["tool_response"]
    ]
    if filtered_tool_calls:
        return filtered_tool_calls[-1]["tool_response"]["card"]
    return None
