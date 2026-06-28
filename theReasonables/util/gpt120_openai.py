#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

import requests


CONFIG_FILE = Path(__file__).resolve().parent / "config.txt"
API_URL = CONFIG_FILE.read_text(encoding="utf-8").strip()
MODEL_NAME = "openai/gpt-oss-120b"

def call_model(prompt: str, seed: int, timeout: int = 120):
    response = requests.post(
        API_URL,
        headers={
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "response": {"effort":"high","summary":"concise"},
            "temperature": 0.0,
            "seed": seed,
            "max_tokens": 4096,
        },
        timeout=timeout,
    )

    response.raise_for_status()

    result = response.json()
    # Save the complete API response
    with open("response.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    output = result["choices"][0]["message"]["content"]

    return output, result


def main():

    if len(sys.argv) != 4:
        print("Usage:")
        print("python gpt120.py prompt.txt response.txt seed")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    seed = int(sys.argv[3])

    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        sys.exit(1)

    prompt = input_file.read_text(encoding="utf-8")

    try:
        output, result = call_model(prompt, seed)

    except requests.HTTPError as e:
        print(f"HTTP Error {e.response.status_code}")
        print(e.response.text)
        sys.exit(1)

    except Exception as e:
        print(e)
        sys.exit(1)

    # Save model output
    output_file.write_text(output, encoding="utf-8")

    # Print model output
    print("\n===== MODEL OUTPUT =====\n")
    print(output)

    # Save the complete API response
    log_file = output_file.with_suffix(".log.json")

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved response to: {output_file}")
    print(f"Saved API log to:  {log_file}")

    if "usage" in result:
        print("\nToken usage:")
        print(json.dumps(result["usage"], indent=2))

    # If reasoning exists, print where it is
    message = result["choices"][0]["message"]

    if "reasoning" in message:
        print("\n===== REASONING =====\n")
        print(message["reasoning"])


if __name__ == "__main__":
    main()