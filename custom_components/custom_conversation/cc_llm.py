"""Replaces Some of Home Assistant's helpers/llm.py code to allow us to choose the correct prompt."""
from langfuse import get_client as get_langfuse_client, observe

from homeassistant.components.conversation import (
    ChatLog,
    ConversationInput,
    ConverseError,
    SystemContent,
    trace,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import intent, llm

from . import CustomConversationConfigEntry
from .api import CustomLLMAPI
from .const import DOMAIN, LLM_API_ID, LOGGER, coerce_llm_hass_api_ids
from .prompt_manager import PromptContext, PromptManager


@observe(name="cc_update_llm_data", capture_input=False)
async def async_update_llm_data(
    hass: HomeAssistant,
    user_input: ConversationInput,
    config_entry: CustomConversationConfigEntry,
    chat_log: ChatLog,
    prompt_manager: PromptManager,
    llm_hass_api: str | list[str] | None = None,
):
    """Process the incoming message for the LLM.

    Overrides the session's async_process_llm_message method
    to allow us to implement prompt management
    """

    llm_context = llm.LLMContext(
        platform=DOMAIN,
        context=user_input.context,
        language=user_input.language,
        assistant="conversation", # Todo: Confirm
        device_id=user_input.device_id,
    )

    user_name: str | None = None

    if (
        user_input.context
        and user_input.context.user_id
        and (
            user := await hass.auth.async_get_user(user_input.context.user_id)
            )
    ):
        user_name = user.name

    llm_api: llm.APIInstance | None = None
    api_ids = coerce_llm_hass_api_ids(llm_hass_api)

    if api_ids:
        try:
            registered = {api.id: api for api in llm.async_get_apis(hass)}
            apis_to_merge: list[llm.API] = []
            for api_id in api_ids:
                if api_id == LLM_API_ID:
                    custom_api = CustomLLMAPI(
                        hass,
                        user_name,
                        conversation_config_entry=config_entry,
                    )
                    if (
                        langfuse_client := hass.data.get(DOMAIN, {})
                        .get(config_entry.entry_id, {})
                        .get("langfuse_client")
                    ):
                        custom_api.set_langfuse_client(langfuse_client)
                    apis_to_merge.append(custom_api)
                    LOGGER.debug("Including Custom LLM API in merge")
                elif api_id in registered:
                    apis_to_merge.append(registered[api_id])
                    LOGGER.debug("Including LLM API %s in merge", api_id)
                else:
                    raise HomeAssistantError(f"API {api_id} not found")

            if len(apis_to_merge) == 1:
                llm_api = await apis_to_merge[0].async_get_api_instance(llm_context)
            else:
                llm_api = await llm.MergedAPI(apis_to_merge).async_get_api_instance(
                    llm_context
                )

        except HomeAssistantError as err:
            LOGGER.error(
                "Error getting LLM APIs %s for %s: %s",
                api_ids,
                DOMAIN,
                err,
            )
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "Error preparing LLM API",
            )
            raise ConverseError(
                f"Error getting LLM APIs {api_ids}",
                conversation_id=chat_log.conversation_id,
                response=intent_response,
            ) from err
    prompt_object = None
    try:
        prompt_context = PromptContext(
            hass=hass,
            ha_name=hass.config.location_name,
            user_name=user_name,
        )
        if llm_api and isinstance(llm_api.api, CustomLLMAPI):
            # The LLM API is the CustomLLMAPI, so use its prompt. The prompt manager
            # will pull in the base prompt if langfuse is disabled.
            prompt = await llm_api.api_prompt
            # If langfuse is successfully used, we'll get back a tuple that contains a
            # prompt object as well
            if isinstance(prompt, tuple):
                LOGGER.debug("Retrieved Langfuse Prompt")
                prompt_object, prompt = prompt
            LOGGER.debug("LLM API prompt: %s", prompt)
        elif not llm_api:
            # No API is enabled - just get the base prompt
            prompt = await prompt_manager.async_get_base_prompt(
                prompt_context,
                config_entry,
            )
            # If langfuse is successfully used, we'll get back a tuple that contains a
            # prompt object as well
            if isinstance(prompt, tuple):
                LOGGER.debug("Retrieved Basic Langfuse Prompt")
                prompt_object, prompt = prompt
            LOGGER.debug("Base prompt: %s", prompt)
        else:
            # Non-custom API (or merged APIs): base prompt plus API / merged API text
            base_prompt = await prompt_manager.async_get_base_prompt(
                prompt_context,
                config_entry,
            )
            if isinstance(base_prompt, tuple):
                prompt_object, base_text = base_prompt
            else:
                base_text = base_prompt
            prompt = "\n".join([base_text, llm_api.api_prompt])
            LOGGER.debug("Combined prompt: %s", prompt)

    except TemplateError as err:
        LOGGER.error("Error rendering prompt: %s", err)
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_error(
            intent.IntentResponseErrorCode.UNKNOWN,
            "Sorry, I had a problem with my template",
        )
        raise ConverseError(
            "Error rendering prompt",
            conversation_id=chat_log.conversation_id,
            response=intent_response,
        ) from err

    extra_system_prompt = (
        # Take new system prompt if one was given
        user_input.extra_system_prompt or chat_log.extra_system_prompt
    )

    if extra_system_prompt:
        LOGGER.debug("Using extra system prompt: %s", extra_system_prompt)
        prompt += "\n" + extra_system_prompt
        get_langfuse_client().update_current_span(metadata={"tags": ["extra_system_prompt"]})

    chat_log.llm_api = llm_api
    chat_log.extra_system_prompt = extra_system_prompt
    chat_log.content[0] = SystemContent(content=prompt)
    trace.async_conversation_trace_append(
        trace.ConversationTraceEventType.AGENT_DETAIL,
        {
            "messages": chat_log.content,
            "tools": chat_log.llm_api.tools if chat_log.llm_api else None,
        }
    )
    return prompt_object
