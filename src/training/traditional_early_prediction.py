import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# Project root: .../Personal-Attack-Prediction
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value):
    """Resolve relative paths from the repository root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_config(config_path):
    config_path = resolve_project_path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f), config_path


def get_config_value(config, *keys, default=None):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_data(path, columns, separator):
    path = resolve_project_path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Check dataset.path in configs/traditional_config.json."
        )

    df = pd.read_csv(path)
    required = [
        columns["id"],
        columns["text"],
        columns["label"],
        columns["split"],
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.dropna(subset=required).copy()
    df[columns["text"]] = df[columns["text"]].astype(str)
    df[columns["label"]] = pd.to_numeric(
        df[columns["label"]], errors="raise"
    ).astype(int)
    df[columns["split"]] = (
        df[columns["split"]].astype(str).str.strip().str.lower()
    )

    if not set(df[columns["label"]].unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary values 0 and 1.")
    if df[columns["id"]].duplicated().any():
        raise ValueError(
            f"Duplicate conversation IDs found in column {columns['id']!r}."
        )

    if "n_utterances" in df.columns:
        actual_counts = df[columns["text"]].str.count(separator) + 1
        declared_counts = pd.to_numeric(df["n_utterances"], errors="coerce")
        if declared_counts.isna().any() or not (
            actual_counts.to_numpy() == declared_counts.to_numpy()
        ).all():
            raise ValueError(
                "n_utterances does not match the utterance separator count."
            )

    print("\nDataset information")
    print("-------------------")
    print(f"Dataset: {path}")
    print(f"Total conversations: {len(df)}")
    print("\nLabel distribution:")
    print(df[columns["label"]].value_counts().sort_index())
    print("\nSplit distribution:")
    print(df[columns["split"]].value_counts())

    return df


def build_prefix_data(df, stage, columns, separator, minimum_prefix=1):
    records = []
    for _, row in df.iterrows():
        utterances = [
            part.strip()
            for part in str(row[columns["text"]]).split(separator)
        ]
        utterances = [u for u in utterances if u]
        total = len(utterances)
        if total == 0:
            continue

        n_keep = max(minimum_prefix, int(np.ceil(total * stage / 100)))
        n_keep = min(n_keep, total)
        records.append(
            {
                "conversation_id": row[columns["id"]],
                "text": " ".join(utterances[:n_keep]),
                "label": int(row[columns["label"]]),
                "split": str(row[columns["split"]]).lower(),
                "total_utterances": total,
                "observed_utterances": n_keep,
            }
        )
    return pd.DataFrame(records)


def train_one_stage(
    data,
    stage,
    output_dir,
    tfidf_config,
    classifier_config,
    random_seed,
):
    train_df = data[data["split"] == "train"].copy()
    val_df = data[data["split"] == "val"].copy()
    test_df = data[data["split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"Stage {stage}% has an empty split. "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
    if train_df["label"].nunique() < 2:
        raise ValueError(
            f"Training split at {stage}% contains fewer than two classes."
        )
    if test_df["label"].nunique() < 2:
        raise ValueError(
            f"Test split at {stage}% contains fewer than two classes; "
            "ROC-AUC cannot be computed."
        )

    vectorizer = TfidfVectorizer(
        ngram_range=tuple(tfidf_config.get("ngram_range", [1, 2])),
        max_features=int(tfidf_config.get("max_features", 50000)),
        min_df=int(tfidf_config.get("min_df", 2)),
        sublinear_tf=bool(tfidf_config.get("sublinear_tf", True)),
    )
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    # max_iter=20 with tol=None runs the configured 20 training epochs.
    classifier_params = {
        "loss": classifier_config.get("loss", "log_loss"),
        "penalty": classifier_config.get("penalty", "l2"),
        "alpha": float(classifier_config.get("alpha", 0.0001)),
        "max_iter": int(classifier_config.get("max_iter", 20)),
        "tol": classifier_config.get("tol", None),
        "shuffle": bool(classifier_config.get("shuffle", True)),
        "learning_rate": classifier_config.get("learning_rate", "optimal"),
        "random_state": int(random_seed),
    }
    model = SGDClassifier(**classifier_params)
    model.fit(X_train, train_df["label"])

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    y_test = test_df["label"]

    metrics = {
        "stage": stage,
        "train_size": len(train_df),
        "validation_size": len(val_df),
        "test_size": len(test_df),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
    }

    prediction_df = pd.DataFrame(
        {
            "conversation_id": test_df["conversation_id"].to_numpy(),
            "stage": stage,
            "true_label": y_test.to_numpy(),
            "predicted_label": predictions,
            "predicted_probability": probabilities,
        }
    )
    prediction_df.to_csv(
        output_dir / f"predictions_{stage}pct.csv", index=False
    )

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": model,
            "stage": stage,
            "classifier_params": classifier_params,
            "tfidf_config": tfidf_config,
        },
        output_dir / f"model_{stage}pct.joblib",
    )

    print(f"\n========== {stage}% prefix ==========")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, (float, np.floating))
              else f"{key}: {value}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    return metrics, prediction_df


def main():
    parser = argparse.ArgumentParser(
        description="Traditional ML early prediction experiment."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/traditional_config.json",
        help="Path to the JSON experiment configuration.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional dataset path; overrides dataset.path in the config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory; overrides the config.",
    )
    args = parser.parse_args()

    config, config_path = load_config(args.config)

    columns = {
        "id": get_config_value(
            config, "dataset", "conversation_id_column",
            default="conversation_id"
        ),
        "text": get_config_value(
            config, "dataset", "text_column", default="text"
        ),
        "label": get_config_value(
            config, "dataset", "label_column", default="label"
        ),
        "split": get_config_value(
            config, "dataset", "split_column", default="split"
        ),
    }
    data_path = args.data or get_config_value(
        config, "dataset", "path", default=None
    )
    output_path = (
        args.output_dir
        or get_config_value(config, "output", "directory", default=None)
        or get_config_value(config, "dataset", "output_dir", default=None)
        or "results/traditional"
    )
    if not data_path:
        raise ValueError(
            "Dataset path is not configured. Set dataset.path in the JSON "
            "file or pass --data."
        )

    output_dir = resolve_project_path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    separator = get_config_value(
        config, "dataset", "utterance_separator", default="▪◦▪"
    )
    stages = get_config_value(
        config, "early_prediction", "stages_percent",
        default=[20, 40, 60, 80, 100]
    )
    minimum_prefix = int(
        get_config_value(
            config, "early_prediction",
            "minimum_prefix_utterances", default=1
        )
    )
    random_seed = int(
        get_config_value(config, "training", "random_seed", default=42)
    )
    tfidf_config = get_config_value(
        config, "model", "tfidf", default={}
    )
    classifier_config = get_config_value(
        config, "model", "classifier", default={}
    )
    model_type = get_config_value(
        config, "model", "type", default="sgd_log_loss"
    )
    if model_type != "sgd_log_loss":
        raise ValueError(
            f"Unsupported model.type {model_type!r}. "
            "This script implements model.type='sgd_log_loss'."
        )

    print(f"Config: {config_path}")
    print(f"Dataset: {resolve_project_path(data_path)}")
    print(f"Output directory: {output_dir}")
    print(f"Model: TF-IDF + SGDClassifier ({model_type})")
    print(f"Training epochs (max_iter): {classifier_config.get('max_iter', 20)}")
    print(f"Stages: {stages}")

    df = load_data(data_path, columns, separator)

    all_metrics = []
    all_predictions = []
    for stage in stages:
        stage = int(stage)
        if stage < 1 or stage > 100:
            raise ValueError(f"Invalid prediction stage: {stage}")
        print(f"\nPreparing {stage}% prefix data...")
        stage_data = build_prefix_data(
            df, stage, columns, separator, minimum_prefix
        )
        metrics, predictions = train_one_stage(
            stage_data,
            stage,
            output_dir,
            tfidf_config,
            classifier_config,
            random_seed,
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(output_dir / "traditional_results.csv", index=False)

    pd.concat(all_predictions, ignore_index=True).to_csv(
        output_dir / "traditional_test_predictions.csv", index=False
    )

    # Save the exact config used, along with resolved paths.
    run_config = dict(config)
    run_config["_run_metadata"] = {
        "config_file": str(config_path),
        "dataset_path": str(resolve_project_path(data_path)),
        "output_directory": str(output_dir),
        "model_implementation": "TF-IDF + SGDClassifier",
        "random_seed": random_seed,
    }
    with (output_dir / "experiment_config.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(run_config, f, indent=4, ensure_ascii=False)

    print("\n===================================")
    print("Traditional ML experiment completed.")
    print("\nFinal results:")
    print(results_df.to_string(index=False))
    print(f"\nOutput directory: {output_dir}")
    print("===================================")


if __name__ == "__main__":
    main()
