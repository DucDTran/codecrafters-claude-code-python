import argparse
import json
from json import tool
import os
import sys

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    messages = [{"role": "user", "content": args.p}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read and return the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to read",
                        }
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Write",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "required": ["file_path", "content"],
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path of the file to write to"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute"
                        }
                    }
                }
            }
        }
    ]

    print("Logs from your program will appear here!", file=sys.stderr)

    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=tools,
        )

        if not chat.choices:
            raise RuntimeError("no choices in response")

        message = chat.choices[0].message
        assistant_message = {"role": "assistant", "content": message.content}

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            assistant_message["tool_calls"] = []
            for tool_call in tool_calls:
                assistant_message["tool_calls"].append(
                    {
                        "id": tool_call.id,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                        "type": "function",
                    }
                )
            messages.append(assistant_message)

            for tool_call in tool_calls:
                if tool_call.function.name != "Read" and tool_call.function.name != "Write" and tool_call.function.name != "Bash":
                    continue
                elif tool_call.function.name == "Read":
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("invalid tool arguments") from exc

                    file_path = arguments.get("file_path")
                    if not file_path:
                        raise RuntimeError("missing file_path in tool arguments")

                    with open(file_path, "r") as f:
                        file_contents = f.read()

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": file_contents,
                        }
                    )
                elif tool_call.function.name == "Write":
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("invalid tool arguments") from exc

                    file_path = arguments.get("file_path")
                    content = arguments.get("content")
                    if not file_path or content is None:
                        raise RuntimeError("missing file_path or content in tool arguments")

                    with open(file_path, "w") as f:
                        f.write(content)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Wrote to {file_path}",
                        }
                    )
                elif tool_call.function.name == "Bash":
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("invalid tool arguments") from exc

                    command = arguments.get("command")
                    if not command:
                        raise RuntimeError("missing command in tool arguments")

                    stream = os.popen(command)
                    output = stream.read()

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": output,
                        }
                    )
            continue

        messages.append(assistant_message)
        if message.content:
            print(message.content)
        break

if __name__ == "__main__":
    main()
