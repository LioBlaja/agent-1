from dotenv import load_dotenv
from openai import OpenAI
from tools.function_tools import TOOLS, AVAILABLE_FUNCTIONS
import os
import json 

load_dotenv()


def load_session_history():
    # make sure session.json exists in data foler
    if not os.path.exists("data/session.json"):
        with open("data/session.json", "w") as f:
            return json.dump([], f)
    try:
        with open("data/session.json", "r") as f:
            return json.load(f)
    except OSError:
        return []

client = OpenAI(
    api_key = os.getenv("OPENAI_LLM_KEY"),
    base_url = "https://api.openai.com/v1"
)

SESSION_FILE="data/session.json"
session_history = load_session_history()


while True:
    input_text = input("You: ")
    if input_text.lower() == '--exit':
        break
    if input_text.lower() == '--clear':
        session_history = []
        with open("data/session.json", "w") as f:
            json.dump(session_history, f, indent=4)
        print("Session history cleared.")
        continue

    user_message = {
        "role": "user", 
        "content": [
            { 
                "type": "input_text", 
                "text": input_text
            },
        ],
    }


    try:
        events = client.responses.create(
            model="gpt-5.2",
            input=session_history + [user_message],
            store=False,
            stream=True,
            reasoning={
                "effort": "high",
                "summary": "detailed",
            },
            include=["reasoning.encrypted_content"],
            tools=TOOLS
        )

        assistant_messages = []

        for event in events:
            # print(event)
            # print("#####################################################")
            event_dict = event.model_dump()
            event_type = event_dict.get("type")

            if event_type == "response.content_part.added":
                print("\nAgent: ", end="", flush=True)
            elif event_type == "response.output_text.delta":
                delta = event_dict.get("delta", "")
                print(delta, end="", flush=True)
            elif event_type == "response.reasoning_summary_part.added":
                print("\n[Reasoning Summary]: ", end="", flush=True)
            elif event_type == "response.reasoning_summary_text.delta":
                delta = event_dict.get("delta", "")
                print(delta, end="", flush=True)
            elif event_type == "response.completed":
                response = event_dict.get("response", {})
                if response:
                    output = response.get("output", [])
                    if output:
                        for item in output:
                            if item.get("type") == "message":
                                assistant_messages.append(item)
                            elif item.get("type") == "reasoning":
                                assistant_messages.append(item)
                            elif item.get("type") == "function_call":
                                name = item.get("name")
                                arguments = json.loads(item.get("arguments", "{}"))

                                result = AVAILABLE_FUNCTIONS[name](**arguments)
                                tool_result = {
                                    "type": "function_call_output",
                                    "call_id": item["call_id"],
                                    "output": json.dumps(result),
                                }
                                assistant_messages.append(item)
                                assistant_messages.append(tool_result)

                                tool_events = client.responses.create(
                                    model="gpt-5.2",
                                    input=session_history + [user_message, item, tool_result],
                                    tools=TOOLS,
                                    stream=True,
                                )
                                for tool_event in tool_events:
                                    tool_event_dict = tool_event.model_dump()
                                    tool_event_type = tool_event_dict.get("type")

                                    if tool_event_type == "response.content_part.added":
                                        print("\nAgent: ", end="", flush=True)
                                    elif tool_event_type == "response.output_text.delta":
                                        delta = tool_event_dict.get("delta", "")
                                        print(delta, end="", flush=True)
                                    elif tool_event_type == "response.completed":
                                        tool_response = tool_event_dict.get("response", {})
                                        tool_output = tool_response.get("output", [])
                                        for tool_item in tool_output:
                                            if tool_item.get("type") == "message":
                                                assistant_messages.append(tool_item)
                                            elif tool_item.get("type") == "reasoning":
                                                assistant_messages.append(tool_item)
                                    elif tool_event_type in {
                                        "response.created",
                                        "response.in_progress",
                                        "response.output_item.added",
                                        "response.output_item.done",
                                        "response.content_part.done",
                                        "response.output_text.done",
                                    }:
                                        pass
                                    else:
                                        print(f"Unknown event type: {tool_event_type}")

            elif event_type == "response.function_call_arguments.delta":
                print()
            elif event_type == "response.function_call_arguments.done":
                print()
            else:
                print(f"Unknown event type: {event_type}")

        print()  # for new line after assistant response
            

        session_history.append(user_message)
        session_history.extend(assistant_messages)

        # save session history to session.json
        with open("data/session.json", "w") as f:
            json.dump(session_history, f, indent=4)


    except Exception as e:
        print(f"An error occurred: {e}")
