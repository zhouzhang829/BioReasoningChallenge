#!/usr/bin/env python3

import sys
import requests

API_KEY ='sk-lit-93cea5ba-7b93-474c-82e9-109b4095d708'  # or load from an environment variable


def main():

    if len(sys.argv) != 3:
        print("Usage:")
        print("python gpt120.py prompt.txt response.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Read prompt
    with open(input_file, "r", encoding="utf-8") as f:
        prompt = f.read()

    # Call model
    response = requests.post(
        "https://lightning.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "lightning-ai/gpt-oss-120b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 10
        },

        
        timeout=120,
    )

    if response.status_code != 200:
        print(f"HTTP Error {response.status_code}")
        print(response.text)
        sys.exit(1)

    try:
        result = response.json()
    except Exception:
        print("Could not decode JSON response")
        print(response.text)
        sys.exit(1)

    try:
        output = result["choices"][0]["message"]["content"]
    except Exception:
        print("Unexpected API response:")
        print(result)
        sys.exit(1)

    # Save model output
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"Saved response to {output_file}")

    if "usage" in result:
        print("Token usage:")
        print(result["usage"])


if __name__ == "__main__":
    main()