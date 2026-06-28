"""Stratified subsampling of the train dataset.

Reduces a large labelled dataset (e.g. ``data/train.csv`` with ~7700 rows)
down to a smaller sample while preserving the original label distribution.

The class allocation uses the largest-remainder method so that the returned
sample size matches ``n`` exactly and each label's share stays as close as
possible to its proportion in the full dataset.

FUNCTION:
from mlgenx import subsample_csv, subsample_stratified

# from a CSV path
sample = subsample_csv("data/train.csv", n=150, output_csv="data/train_150.csv")

# or from an in-memory dataframe
import pandas as pd
df = pd.read_csv("data/train.csv")
sample = subsample_stratified(df, n=150)

CLI:
    python -m mlgenx.subsample data/train.csv -n 150 -o data/train_150.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def subsample_stratified(
    df: pd.DataFrame,
    n: int = 150,
    *,
    label_col: str = "label",
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Return a stratified subsample of ``df`` of size ``n``.

    Each label keeps (approximately) the same proportion it has in ``df``.
    The largest-remainder method distributes rounding so the result has
    exactly ``min(n, len(df))`` rows.

    Args:
        df: Source dataframe.
        n: Target number of rows. Capped at ``len(df)``.
        label_col: Column holding the class labels.
        random_state: Seed for reproducible row selection.

    Returns:
        A new dataframe (rows shuffled) with the same columns as ``df``.
    """
    if label_col not in df.columns:
        raise ValueError(
            f"label column {label_col!r} not found; available: {list(df.columns)}"
        )

    total = len(df)
    n = min(n, total)
    if n <= 0:
        return df.iloc[0:0].copy()

    counts = df[label_col].value_counts()

    # Ideal (fractional) number of rows per label.
    ideal = {lbl: cnt / total * n for lbl, cnt in counts.items()}

    # Floor each, then hand out leftover slots by largest fractional remainder.
    alloc = {lbl: int(val) for lbl, val in ideal.items()}
    remainder = n - sum(alloc.values())
    if remainder > 0:
        leftovers = sorted(
            ideal,
            key=lambda lbl: (ideal[lbl] - alloc[lbl], counts[lbl]),
            reverse=True,
        )
        for lbl in leftovers[:remainder]:
            alloc[lbl] += 1

    parts = [
        df[df[label_col] == lbl].sample(
            n=min(k, int(counts[lbl])), random_state=random_state
        )
        for lbl, k in alloc.items()
        if k > 0
    ]

    sample = pd.concat(parts)
    return sample.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def subsample_csv(
    input_csv: str | Path,
    n: int = 150,
    output_csv: str | Path | None = None,
    *,
    label_col: str = "label",
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Load ``input_csv``, stratified-subsample to ``n`` rows, optionally write.

    Returns the subsampled dataframe. If ``output_csv`` is given, also writes
    it there (without the index).
    """
    df = pd.read_csv(input_csv)
    sample = subsample_stratified(
        df, n=n, label_col=label_col, random_state=random_state
    )
    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(output_csv, index=False)
    return sample


def _format_dist(df: pd.DataFrame, label_col: str) -> str:
    total = len(df)
    counts = df[label_col].value_counts()
    return ", ".join(
        f"{lbl}={cnt} ({cnt / total:.1%})" for lbl, cnt in counts.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified subsample of a labelled train CSV."
    )
    parser.add_argument("input_csv", type=Path, help="Path to train.csv")
    parser.add_argument(
        "-n", "--num-samples", type=int, default=150,
        help="Target number of rows (default: 150).",
    )
    parser.add_argument(
        "-o", "--output-csv", type=Path, default=None,
        help="Where to write the subsample. Defaults to "
             "<input_stem>_subsampled<N>.csv next to the input.",
    )
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output = args.output_csv
    if output is None:
        output = args.input_csv.with_name(
            f"{args.input_csv.stem}_subsampled{args.num_samples}.csv"
        )

    src = pd.read_csv(args.input_csv)
    sample = subsample_stratified(
        src, n=args.num_samples,
        label_col=args.label_col, random_state=args.random_state,
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output, index=False)

    print(f"Input : {len(src)} rows -> {_format_dist(src, args.label_col)}")
    print(f"Output: {len(sample)} rows -> {_format_dist(sample, args.label_col)}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
