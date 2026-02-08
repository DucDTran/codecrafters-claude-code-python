import argparse
import json
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

    chat = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": args.p}],
        tools=[
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
            }
        ],
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")
    
    print("Logs from your program will appear here!", file=sys.stderr)

    message = chat.choices[0].message
    if getattr(message, "tool_calls", None):
        for tool_call in message.tool_calls:
            if tool_call.function.name != "Read":
                continue
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as exc:
                raise RuntimeError("invalid tool arguments") from exc

            file_path = arguments.get("file_path")
            if not file_path:
                raise RuntimeError("missing file_path in tool arguments")

            with open(file_path, "r") as f:
                file_contents = f.read()
            print(file_contents)
            break
    
    print(chat.choices[0].message.content)

if __name__ == "__main__":
    main()
