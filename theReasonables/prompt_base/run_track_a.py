import csv
import json
import random
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PROMPT_PATH = ROOT / "prompt.txt"
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test.csv"
SAMPLE_PATH = ROOT / "sample_submission_track_a.csv"


SEEDS = [42, 43, 44]


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_train(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id", "pert", "gene", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv missing columns: {missing}")
    return df


def load_test(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id", "pert", "gene"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"test.csv missing columns: {missing}")
    return df


def build_fewshot(train_df: pd.DataFrame, seed: int, k_per_class: int = 2) -> str:
    rng = random.Random(seed)
    chunks = []
    for label in ["up", "down", "none"]:
        subset = train_df[train_df["label"] == label].sample(
            n=min(k_per_class, (train_df["label"] == label).sum()),
            random_state=rng.randint(0, 10**9),
        )
        for _, row in subset.iterrows():
            chunks.append(
                f'{{"pert": "{row["pert"]}", "gene": "{row["gene"]}", "label": "{row["label"]}"}}'
            )
    rng.shuffle(chunks)
    return "\n".join(chunks)


def make_prompt(template: str, pert: str, gene: str, fewshot: str) -> str:
    return (
        template
        .replace("{{FEWSHOT}}", fewshot)
        .replace("{{PERT}}", str(pert))
        .replace("{{GENE}}", str(gene))
    )


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_response(raw: str) -> dict:
    raw = strip_code_fences(raw)

    # First try whole-text JSON.
    try:
        obj = json.loads(raw)
        return obj
    except json.JSONDecodeError:
        pass

    # Then try extracting the first JSON object.
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "prediction_up": 0.5,
        "prediction_down": 0.5,
        "prediction_none": 0.0,
        "reasoning": "none",
    }


def normalize_probs(obj: dict) -> dict:
    up = float(obj.get("prediction_up", 0.5))
    down = float(obj.get("prediction_down", 0.5))
    none = float(obj.get("prediction_none", max(0.0, 1.0 - up - down)))

    # Clip negative values.
    up = max(0.0, up)
    down = max(0.0, down)
    none = max(0.0, none)

    s = up + down + none
    if s <= 0:
        up, down, none = 0.5, 0.5, 0.0
        s = 1.0

    return {
        "prediction_up": up / s,
        "prediction_down": down / s,
        "prediction_none": none / s,
        "reasoning": str(obj.get("reasoning", "none")) if obj.get("reasoning") else "none",
    }


def call_model(prompt: str, seed: int) -> str:
    """
    Replace this with your actual model call.
    Return raw text from the model.
    """
    raise NotImplementedError("Connect your model endpoint here.")


def run_one(test_row, train_df, template, seed):
    fewshot = build_fewshot(train_df, seed=seed, k_per_class=2)
    prompt = make_prompt(template, test_row["pert"], test_row["gene"], fewshot)
    raw = call_model(prompt, seed=seed)
    parsed = normalize_probs(parse_response(raw))
    return raw, parsed


def main():
    train_df = load_train(TRAIN_PATH)
    test_df = load_test(TEST_PATH)
    template = load_text(PROMPT_PATH)

    sample = pd.read_csv(SAMPLE_PATH)
    out = sample.copy()

    # Fill required metadata columns.
    out["model_name"] = "your-model-name-here"

    # Collect per-seed outputs.
    for seed in SEEDS:
        up_col = f"prediction_up_seed{seed}"
        down_col = f"prediction_down_seed{seed}"
        trace_col = f"reasoning_trace_seed{seed}"

        out[up_col] = 0.5
        out[down_col] = 0.5
        out[trace_col] = "none"

    total_tokens = 0

    for i, row in test_df.iterrows():
        per_seed = []

        for seed in SEEDS:
            raw, parsed = run_one(row, train_df, template, seed)
            per_seed.append(parsed)

            out.loc[out["id"] == row["id"], f"prediction_up_seed{seed}"] = parsed["prediction_up"]
            out.loc[out["id"] == row["id"], f"prediction_down_seed{seed}"] = parsed["prediction_down"]
            out.loc[out["id"] == row["id"], f"reasoning_trace_seed{seed}"] = raw if raw.strip() else "none"

            # Replace with actual token accounting from your model client.
            total_tokens += 0

        avg_up = sum(x["prediction_up"] for x in per_seed) / len(per_seed)
        avg_down = sum(x["prediction_down"] for x in per_seed) / len(per_seed)

        out.loc[out["id"] == row["id"], "prediction_up"] = avg_up
        out.loc[out["id"] == row["id"], "prediction_down"] = avg_down
        out.loc[out["id"] == row["id"], "tokens_used"] = total_tokens
        if "reasoning_trace_seed42" in out.columns:
            pass

    # Enforce non-null cells.
    out = out.fillna("none")

    out.to_csv(ROOT / "submission.csv", index=False)
    print(f"Wrote {ROOT / 'submission.csv'}")


if __name__ == "__main__":
    main()