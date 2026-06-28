#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV containing id, pert, gene columns.",
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help="Path to prompt template.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file.",
    )

    parser.add_argument(
        "--gpt-script",
        default="gpt120_openai.py",
        help="Path to gpt120_openai.py",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    train = pd.read_csv(args.input)
    prompt_template = Path(args.prompt).read_text()

    results = []

    for _, row in train.iterrows():

        prompt = prompt_template.format(
            model_id=row["id"],
            pert=row["pert"],
            gene=row["gene"],
        )

        tmp_prompt = Path("prompt_tmp.txt")
        tmp_prompt.write_text(prompt)

        subprocess.run(
            [
                "python",
                args.gpt_script,
                str(tmp_prompt),
                "output.txt",
                str(args.seed),
            ],
            check=True,
        )

        with open("response.json", "r") as f:
            response = json.load(f)

        message = response["choices"][0]["message"]

        results.append(
            {
                "model_id": row["id"],
                "pert": row["pert"],
                "gene": row["gene"],
                "content": message.get("content", ""),
                "reasoning_content": message.get("reasoning_content", ""),
            }
        )

    pd.DataFrame(results).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()