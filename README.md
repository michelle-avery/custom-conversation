# Custom Conversation for Home Assistant

A highly configurable conversation agent for Home Assistant, designed for users who want fine-grained control over their LLM interactions. 
While Home Assistant's built-in conversation component works well for most users, this project provides additional customization options 
and experimentation features for those who want to dive deeper into LLM integration.

## Features

### Core Features
- Compatible with any OpenAI API-compatible LLM through configurable base URL
- Dynamic reconfiguration of API settings (change your base URL without having to create a new instance)
- Enhanced event system for conversation tracking and debugging
- Configurable intent handling and fallback behavior
- Fine-grained prompt customization
- Support for Langfuse prompt management and tracing (experimental)

### Enhanced Response Handling
- Preservation of intent script "card" parameters
- Detailed conversation events published to Home Assistant's event bus
- Consistent event format regardless of handling agent (local or LLM)

## Installation

### With HACS
* Install [HACS](https://hacs.xyz/docs/use/) if you have not already
* Click 
[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.
](https://my.home-assistant.io/badges/hacs_repository.svg)
](https://my.home-assistant.io/redirect/hacs_repository/?owner=michelle-avery&category=integration&repository=custom-conversation)  
to add this as a custom repository, or add it manually.
* Click "Add" to confirm, and then click "Download" to download and install the integration
* Restart Home Assistant

### Manual Installation
* Copy the contents of [custom_components/custom_conversation](/custom_components/custom_conversation) to 
`config/custom_components/custom_conversation` in your Home Assistant instance
* Restart Home Assistant

### Completing the Installation
* Go to your Devices and Services page
* Click "Add Integration"
* Search for Custom Conversation and click on it
* You'll be asked for your api key, and the base URL for your OpenAI-compatible LLM API. Enter those, and click "Submit".

That's it for the basic installation, but if you're using this component, you probably want to jump right to the Configuration section to
learn how to tweak all of the settings.


## Configuration

The component can be configured through Home Assistant's UI. Configuration is divided into several sections:

### Basic Settings
- **Instructions Prompt**: This is the same prompt you can customize in Home Assistant's built-in OpenAI Conversation component. 
It's sent with each conversation sent to the LLM, along with any prompt parts added by the API, and a basic prompt Home Assistant adds, which contains the current
date and time.
- **API Choice**: Choose which API to expose to the LLM.
  - **No control**: The LLM can answer basic questions about the world, but will have no ability to control your home
  - **Assist**: Uses Home Assistant's built-in LLM API support to expose Intents, Intent Scripts, and Custom Sentences
  - **Custom Conversation LLM API**: Use this component's own LLM API, allowing more control over the API's prompts, exposed intents, etc.


### Ignored Intents
Home Assistant maintains a list of "ignored intents" which will not be exposed via the LLM API. These are generally for a good reason, so those intents are ignored by default, and they're marked as
"Hass Recommended." You may, however, have certain reasons for wanting to not ignore one, and if you choose to do so, you can remove it from the list.  You can also select additional intents and intent 
scripts to be removed from the list exposed to the LLM.

### Custom LLM API Agents
Here you can choose which agents you want to enable:
* Enabling the Home Assistant Agent allows Home Assistant's built-in intent matching to work
* Enabling the LLM Agent enables interaction with Large Language models
Enabling both will first send the user's request to the built-in agent (which is signficantly faster, and essentially "free"), then if it doesn't return a successful response, 
will send the request to the LLM Agent (which is much more flexible, but slower, may include a cost, and may be unpredictable).  Disabling the LLM Agent will effectively disable all LLM-based
functionality of this component.
 

### LLM Parameters
- **Chat Model**: Choose the model to use
- **Max Tokens**: Maximum response length (default: 150)
- **Temperature**: Controls response randomness (0-2, default: 1.0)
- **Top P**: Controls response diversity (0-1, default: 1.0)

### Prompt Customization
Custom Conversation breaks down prompts into configurable components:

- **Base Prompt**: This is sent to the LLM with every message. By default, it contains current time and date (default: "Current time is {{time}}. Today's date is {{date}}."), but any valid Home Assistant template
can be used.
- **API-Enabled Prompts**: These are only used if you have the Custom Conversation LLM API used, and the defaults are the same as the default Home Assistant LLM API
  - No Entities Prompt: Used when no entities are exposed, indicates to the user that they may have forgotten to expose entities to the assistant.
  - Base API Prompt: Core instructions for entity control
  - Device Location Prompts: Context-specific prompts for known/unknown device locations
  - Timer Support Prompt: Instructions regarding timer capabilities
  - Entities List Prompt: Indicates that the following text describes the entities present in the smart home (exposed entities will automatically be appended after this prompt segment)

### Langfuse Integration (Experimental)
Langfuse support enables:
- Remote prompt management
- Conversation tracing and analytics
- A/B testing of different prompts

Configuration options:
- **Enable Langfuse Prompt Management**: Toggles Langfuse prompt management integration.
- **Host**: Custom Langfuse instance URL - you can use Langfuse's cloud instance or host your own
- **Langfuse Public/Secret Keys**: The public and secret keys from your langfuse project
- **Base Prompt ID**: This is the prompt that is used if the LLM API isn't enabled. It's effectively a combination of the Base Prompt and Instruction Prompt above.
- **Base Prompt Label**: The label to use to select the version of your Prompt ID to use. For example, in a production environment, you may want to use the automatic "production" label, whereas in a
dev environment, you may want to select a specific prompt version you're testing, or set it to the automatically created "latest" label.
- **API Prompt ID**: This is the ID of the prompt that will be used if you have the LLM API enabled. Because Langfuse does not yet support composable prompts, this will likely have some redundant content with the Base Prompt (unless you don't bother with the base prompt, because you're always going ot have the LLM API enabled)

Creating Langfuse prompts:
When using Langfuse for prompt management, the content of your prompts is stored in Langfuse itself, and you only configure the corresponding IDs in the Custom Conversation integration. Home Assistant templates aren't
supported with Langfuse integration, but the following variables can be used in Langfuse and will be substituted by the Custom Conversation integration for either the Base Prompt or the API Prompt. Make sure to note the lack of space between the braces and the variable:
- `{{current_time}}` - The current time of your Home Assistant system, in `%H:%M` format.
- `{{current_date}}` - The current date of your Home Assistant system, in `%Y-%m-%d` format.
- `{{ha_name}}`- The name of your Home Assistant instance.
- `{{user_name}}` - The name of the user making the request. Note that currently this is only available if the request is made from a user logged in to the Home Assistant website - ie, it's not generally available
when the request comes from a Voice Assistant.

Additionally, the API-enabled prompt can make use of the following:
- `{{location}}` - This will be the location (and potentially floor) of the device from which the request came (available if the request came from a device and that device's location is set) or "Unknown" if there's no 
location (this is the case with requests that come from the chat interface of the website).
- `{{exposed_entities}}` - This is the yaml-formatted list of your entities, their names, locations, and states. If you want the LLM to be able to do anything useful with your home, this MUST be included.
- `{{supports_timers}}` - If the device is capable of setting a timer, this will be an empty string. If the device is NOT capable of setting a timer, this will be "This device is not able to start timers."

The following Langfuse prompt will result in roughly equivalent functionality to the built-in Home Assistant prompt, with the exception that in the event there are no devices exposed, the user will not be warned:

```
Current time is {{current_time}}.
Today's date is {{current_date}}.
You are a voice assistant for Home Assistant.
Answer questions about the world truthfully.
Answer in plain text. Keep it simple and to the point.
When controlling Home Assistant always call the intent tools.
Use HassTurnOn to lock and HassTurnOff to unlock a lock.
When controlling a device, prefer passing just name and domain.
When controlling an area, prefer passing just area name and domain.
If your location is known, all generic commands like 'turn on the lights' target this area. Otherwise, when a user asks to turn on all devices of a specific type, ask user to specify an area, unless there is only one device of that type.
Your location is {{location}}.
{{supports_timers}}
An overview of the areas and the devices in this smart home:
{{exposed_entities}}
```


## Events

The component publishes detailed events for conversation tracking:

### Conversation Started
The `custom_conversation_conversation_started` event is triggered when a conversation begins. This event includes:
- Agent ID
- Conversation ID
- User ID (if available)
- Device ID, name, and area (if available)
- Input text

Example:
```
event_type: custom_conversation_conversation_started
data:
  agent_id: conversation.customconversation
  conversation_id: 01JMNCEGCMB36JGKGXXEN28SSZ
  language: en
  device_id: f98e98256bd2936ba3736a5078c51e16
  device_name: Thinksmart 1
  device_area: office
  text: Can you turn off the switch?
  user_id: null
origin: LOCAL
time_fired: "2025-02-21T23:11:00.143444+00:00"
context:
  id: 01JMNCEM1FKS9PS3590F1XG726
  parent_id: null
  user_id: null
```

### Conversation Ended
The `custom_conversation_conversation_ended` event is triggered when a conversation completes, This event includes:
- Agent ID
- Handling Agent (whether the request was ultimately handled by the LLM or the Home Assistant Assist agent)
- The device id, name, and area (if available)
- The initial request
- The result of the request, which includes the response text, the card (if an intent script with a card was matched), and data about 
any entities affected. Note that if the handling agent was an LLM, this information is duplicated here and under the `llm_data` section. This 
is because the Custom Conversation integration tries to ensure that, if the user is asking to turn off a switch, the response structure under `result` is the same
regardless of whether it was handled by the Assist agent or the LLM agent.
- Details on any tools called by the LLM and the parameters.

Example:
```
vent_type: custom_conversation_conversation_ended
data:
  agent_id: conversation.customconversation
  handling_agent: LLM
  device_id: f98e98256bd2936ba3736a5078c51e16
  device_name: Thinksmart 1
  device_area: office
  request: Can you turn off the switch?
  result:
    response:
      speech:
        plain:
          speech: The switch has been turned off.
          extra_data: null
      card: {}
      language: en
      response_type: action_done
      data:
        targets: []
        success:
          - name: Test Switch
            type: entity
            id: input_boolean.test_switch
        failed: []
    conversation_id: 01JMNCEN0QWBCMBJ49MN77WRJV
  llm_data:
    tool_calls:
      - tool_name: HassTurnOff
        tool_args: "{\"name\":\"Test Switch\",\"domain\":\"input_boolean\"}"
        tool_response:
          speech: {}
          card: {}
          response_type: action_done
          data:
            targets: []
            success:
              - name: Test Switch
                type: entity
                id: input_boolean.test_switch
            failed: []
origin: LOCAL
time_fired: "2025-02-21T23:11:03.776721+00:00"
context:
  id: 01JMNCEQK0Y8GQKCXBG0M40PFV
  parent_id: null
  user_id: null
```

## Use Cases

This component is particularly useful for:
- Testing different LLM providers
- Experimenting with prompt engineering
- Debugging conversation flows
- Developing custom conversation patterns
- A/B testing different approaches to home automation interactions
- Gathering real-world interaction data for further experimentation and development
- Developing prompts in a test environment and promoting them into production
- Works together with [ViewAssist](https://dinki.github.io/View-Assist/) and [Remote Assist Display](https://github.com/michelle-avery/remote-assist-display/) to enable visual feedback for smart home
assistants