"""Validate the conversation-level CSV before training."""
import argparse
from pathlib import Path
import pandas as pd

from src.utils.common import validate_dataframe, SEPARATOR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/conversations_clean.csv")
    args = parser.parse_args()
    path = Path(args.data)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    errors = validate_dataframe(df, SEPARATOR)
    print(f"Rows: {len(df)}")
    print("Split counts:")
    print(df["split"].value_counts(dropna=False).to_string())
    print("Label counts:")
    print(df["label"].value_counts(dropna=False).to_string())
    if errors:
        print("\nVALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("\nVALIDATION PASSED")


if __name__ == "__main__":
    main()
