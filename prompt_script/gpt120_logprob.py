#!/usr/bin/env python3

import json
import math
import re
import sys
from pathlib import Path
import requests


API_URL = "https://vurr6lx60hp06e-8000.proxy.runpod.net/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-120b"


def call_model(prompt: str, seed: int, timeout: int = 120):
    response = requests.post(
        API_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
           # "response": {"summary": "concise"},#"effort": "high", 

            "temperature": 0.0,
            "seed": seed,
            "max_tokens": 4690,
            "logprobs": True,
            "top_logprobs": 20,
            "chat_template_kwargs": {
            "enable_thinking": False
            },
          # Disable GPT-OSS thinking if the backend supports it.  
         },  
        timeout=timeout,
    )

    response.raise_for_status()
    result = response.json()

    with open("response.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    choice = result["choices"][0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "\n".join(
            str(x.get("text", x.get("content", "")))
            for x in content
            if isinstance(x, dict)
        )
    content = str(content).strip()

    return content, result


def extract_abcs_from_logprobs(logprobs_content):
    """
    Find the <answer>X</answer> tag in the token stream, then read top_logprobs
    at the token position that emits X. Returns dict with probabilities for A/B/C.
    """
    if not logprobs_content:
        return None

    tokens = [t.get("token", "") for t in logprobs_content]
    reconstructed = "".join(tokens)

    m = re.search(r"<answer>\s*([ABCabc])\s*</answer>", reconstructed)
    if not m:
        return None

    answer_char_start = m.start(1)

    char_pos = 0
    answer_token_idx = None
    for i, tok_text in enumerate(tokens):
        tok_end = char_pos + len(tok_text)
        if char_pos <= answer_char_start < tok_end:
            answer_token_idx = i
            break
        char_pos = tok_end

    if answer_token_idx is None:
        return None

    top_lps = logprobs_content[answer_token_idx].get("top_logprobs") or []
    scores = {}

    for entry in top_lps:
        tok = str(entry.get("token", "")).strip().upper()
        lp = entry.get("logprob")
        if lp is None:
            continue

        if tok == "A" or tok.endswith("A"):
            scores["A"] = float(lp)
        elif tok == "B" or tok.endswith("B"):
            scores["B"] = float(lp)
        elif tok == "C" or tok.endswith("C"):
            scores["C"] = float(lp)

    if not scores:
        return None

    # Fill missing labels with a low floor so probabilities still normalize.
    # normalize -> sum to 1 while ignoring others
    floor = min(scores.values()) - 20.0
    la = scores.get("A", floor)
    lb = scores.get("B", floor)
    lc = scores.get("C", floor)

    mx = max(la, lb, lc)
    ea = math.exp(la - mx)
    eb = math.exp(lb - mx)
    ec = math.exp(lc - mx)
    s = ea + eb + ec
    if s <= 0:
        return None

    pa = ea / s
    pb = eb / s
    pc = ec / s

    return {
        "prediction_up": pa,
        "prediction_down": pb,
        "prediction_none": pc,
        "answer_token": tokens[answer_token_idx],
    }


def main():
    if len(sys.argv) != 4:
        print("Usage:")
        print("python gpt120_openai.py prompt.txt response.txt seed")
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

    # Save model output safely, even if content is empty
    output_file.write_text(output or "", encoding="utf-8")

    print("\n===== MODEL OUTPUT =====\n")
    print(output)

    log_file = output_file.with_suffix(".json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved response to: {output_file}")
    print(f"Saved API log to:  {log_file}")

    if "usage" in result:
        print("\nToken usage:")
        print(json.dumps(result["usage"], indent=2))

    choice = result["choices"][0]
    message = choice.get("message", {})
    logprobs_content = (choice.get("logprobs") or {}).get("content") or []

    probs = extract_abcs_from_logprobs(logprobs_content)
    if probs is not None:
        probs_file = output_file.with_suffix(".probs.json")
        with open(probs_file, "w", encoding="utf-8") as f:
            json.dump(probs, f, indent=2, ensure_ascii=False)

        print("\n===== A/B/C PROBABILITIES =====\n")
        print(json.dumps(probs, indent=2, ensure_ascii=False))
        print(f"\nSaved probabilities to: {probs_file}")

    if message.get("reasoning_content"):
        print("\n===== REASONING =====\n")
        print(message["reasoning_content"])


if __name__ == "__main__":
    main()