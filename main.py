from dotenv import load_dotenv
from openai import OpenAI
from tools.function_tools import TOOLS, AVAILABLE_FUNCTIONS
import os
import json

load_dotenv()

SESSION_FILE = "data/session.json"
MODEL = "gpt-5.2"

IGNORED_EVENTS = {
    "response.created",
    "response.in_progress",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.done",
    "response.output_text.done",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "response.reasoning_summary_part.done",
    "response.reasoning_summary_text.done",
}


client = OpenAI(
    api_key=os.getenv("OPENAI_LLM_KEY"),
    base_url="https://api.openai.com/v1"
)


def load_session_history():
    if not os.path.exists(SESSION_FILE) or os.path.getsize(SESSION_FILE) == 0:
        save_session_history([])
        return []

    try:
        with open(SESSION_FILE, "r") as f:
            session_history = json.load(f)
    except (OSError, json.JSONDecodeError):
        save_session_history([])
        return []

    if not isinstance(session_history, list):
        save_session_history([])
        return []

    return session_history


def save_session_history(session_history):
    os.makedirs(os.path.dirname(SESSION_FILE), exist_ok=True)
    with open(SESSION_FILE, "w") as f:
        json.dump(session_history, f, indent=4)


def create_response(input_items):
    return client.responses.create(
        model=MODEL,
        input=input_items,
        store=False,
        stream=True,
        reasoning={
            "effort": "high",
            "summary": "detailed",
        },
        include=["reasoning.encrypted_content"],
        tools=TOOLS,
    )


def process_events(events):
    output_items = []

    for event in events:
        event_dict = event.model_dump()
        event_type = event_dict.get("type")

        if event_type == "response.content_part.added":
            print("\nAgent: ", end="", flush=True)
        elif event_type == "response.output_text.delta":
            print(event_dict.get("delta", ""), end="", flush=True)
        elif event_type == "response.reasoning_summary_part.added":
            print("\n[Reasoning Summary]: ", end="", flush=True)
        elif event_type == "response.reasoning_summary_text.delta":
            print(event_dict.get("delta", ""), end="", flush=True)
        elif event_type == "response.completed":
            response = event_dict.get("response", {})
            output_items = response.get("output", [])
        elif event_type in IGNORED_EVENTS:
            pass
        else:
            print(f"\nUnknown event type: {event_type}")

    return output_items


def run_function_call(item):
    name = item.get("name")
    arguments = json.loads(item.get("arguments") or "{}")
    function = AVAILABLE_FUNCTIONS.get(name)

    if function is None:
        result = {
            "error": f"Unknown function: {name}",
        }
    else:
        try:
            result = function(**arguments)
        except Exception as error:
            result = {
                "error": str(error),
            }

    return {
        "type": "function_call_output",
        "call_id": item["call_id"],
        "output": json.dumps(result),
    }


def run_turn(session_history, user_message):
    turn_items = [user_message]
    output_items = process_events(create_response(session_history + turn_items))

    for item in output_items:
        if item.get("type") == "message":
            turn_items.append({
                "type": "message",
                "role": item["role"],
                "content": item["content"],
            })
            continue

        if item.get("type") != "function_call":
            continue

        function_call = {
            "type": "function_call",
            "call_id": item["call_id"],
            "name": item["name"],
            "arguments": item.get("arguments", "{}"),
        }
        tool_result = run_function_call(item)

        turn_items.append(function_call)
        turn_items.append(tool_result)

        tool_input = session_history + turn_items
        tool_output_items = process_events(create_response(tool_input))

        for tool_item in tool_output_items:
            if tool_item.get("type") == "message":
                turn_items.append({
                    "type": "message",
                    "role": tool_item["role"],
                    "content": tool_item["content"],
                })

    print()
    return turn_items


session_history = load_session_history()

while True:
    input_text = input("You: ")
    if input_text.lower() == "--exit":
        break
    if input_text.lower() == "--clear":
        session_history = []
        save_session_history(session_history)
        print("Session history cleared.")
        continue

    user_message = {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": input_text,
            },
        ],
    }

    try:
        session_history.extend(run_turn(session_history, user_message))
        save_session_history(session_history)
    except Exception as e:
        print(f"An error occurred: {e}")
