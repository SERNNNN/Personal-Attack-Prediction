
"""
Combine Traditional and RoBERTa test results.

Input:
    results/traditional/traditional_results.csv
    results/roberta/roberta_test_results.csv

Output:
    results/combined_summary.csv
"""

import argparse
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]

TRADITIONAL_FILE = "traditional/traditional_results.csv"
ROBERTA_FILE = "roberta/roberta_test_results.csv"


def load_results(path: Path, model_name: str) -> pd.DataFrame:
    """Load and validate one model's test results."""

    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} results not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {"stage", *METRIC_COLUMNS}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{model_name} results are missing columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()
    df["stage"] = pd.to_numeric(df["stage"], errors="raise").astype(int)

    df.insert(0, "model", model_name)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Combine Traditional and RoBERTa test results."
    )

    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root directory containing model results."
    )

    parser.add_argument(
        "--output",
        default="results/combined_summary.csv",
        help="Output CSV path."
    )

    args = parser.parse_args()

    root = Path(args.results_dir)

    # Load both models
    traditional = load_results(
        root / TRADITIONAL_FILE,
        "Traditional"
    )

    roberta = load_results(
        root / ROBERTA_FILE,
        "RoBERTa"
    )

    # Combine results
    summary = pd.concat(
        [traditional, roberta],
        ignore_index=True,
        sort=False
    )

    # Sort by prediction stage and model
    model_order = {
        "Traditional": 0,
        "RoBERTa": 1
    }

    summary["_model_order"] = summary["model"].map(model_order)

    summary = (
        summary
        .sort_values(["stage", "_model_order"])
        .drop(columns="_model_order")
        .reset_index(drop=True)
    )

    # Arrange columns
    preferred_columns = [
        "model",
        "stage",
        "train_size",
        "validation_size",
        "test_size",
        *METRIC_COLUMNS,
    ]

    columns = [
        col for col in preferred_columns
        if col in summary.columns
    ]

    summary = summary[columns]

    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(output, index=False)

    # Display
    print("\n========== COMBINED TEST RESULTS ==========\n")
    print(summary.to_string(index=False))

    print(f"\nTotal rows: {len(summary)}")
    print(f"Output saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
