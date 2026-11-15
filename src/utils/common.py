"""Shared data, prefix, metric, and reproducibility utilities."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

SEPARATOR = "▪◦▪"
REQUIRED_COLUMNS = {"conversation_id", "text", "label", "split"}


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def split_utterances(text: str, separator: str = SEPARATOR) -> list[str]:
    if not isinstance(text, str):
        return []
    return [part.strip() for part in text.split(separator) if part.strip()]


def make_prefix(text: str, percent: int, separator: str = SEPARATOR) -> str:
    utterances = split_utterances(text, separator)
    if not utterances:
        return ""
    count = max(1, math.ceil(len(utterances) * percent / 100))
    return f" {separator} ".join(utterances[:count])


def validate_dataframe(df: pd.DataFrame, separator: str = SEPARATOR) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
        return errors
    if df["conversation_id"].isna().any():
        errors.append("conversation_id contains null values")
    if df["conversation_id"].duplicated().any():
        errors.append("conversation_id contains duplicates")
    labels = set(pd.to_numeric(df["label"], errors="coerce").dropna().unique())
    if labels != {0, 1}:
        errors.append(f"Expected both binary labels 0 and 1; found {labels}")
    if df["label"].isna().any() or pd.to_numeric(df["label"], errors="coerce").isna().any():
        errors.append("label contains missing or non-numeric values")
    allowed_splits = {"train", "val", "test"}
    found_splits = set(df["split"].dropna().astype(str).str.lower().unique())
    if not found_splits.issubset(allowed_splits):
        errors.append(f"Unexpected split names: {sorted(found_splits - allowed_splits)}")
    if df["split"].isna().any():
        errors.append("split contains null values")
    if "pair_id" in df.columns:
        pairs = df.dropna(subset=["pair_id"]).groupby("pair_id")
        for pair_id, group in pairs:
            if group["split"].nunique() != 1:
                errors.append(f"pair_id {pair_id!r} crosses dataset splits")
                break
            if len(group) == 2 and set(group["label"].astype(int)) != {0, 1}:
                errors.append(f"pair_id {pair_id!r} does not contain one example per class")
                break
    if "n_utterances" in df.columns:
        for idx, row in df.iterrows():
            actual = len(split_utterances(row["text"], separator))
            try:
                expected = int(row["n_utterances"])
            except (TypeError, ValueError):
                errors.append(f"Invalid n_utterances at row index {idx}")
                break
            if actual != expected:
                errors.append(
                    f"n_utterances mismatch at row index {idx}: expected {expected}, found {actual}"
                )
                break
    return errors


def compute_metrics(y_true, y_pred, y_score=None) -> dict[str, Any]:
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_score is not None and len(set(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, y_score))
        result["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None
    return result
