#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import math
import os
import random
import re
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
print("========== ALL IMPORTS COMPLETED ==========", flush=True)
MODEL_NAME = "FacebookAI/roberta-base"
DEFAULT_STAGES = [20, 40, 60, 80, 100]
DEFAULT_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_text_to_units(text: str, expected_units: int):
    """Recover ordered text units from the CSV text column.

    First use the explicit CGA delimiter if it yields the expected count.
    Otherwise split by lines and, if there are extra lines, merge adjacent lines
    into expected_units contiguous groups using a greedy length-aware strategy.
    """
    text = "" if pd.isna(text) else str(text)
    text = text.strip()
    if not text:
        return []

    expected_units = max(1, int(expected_units))

    delimiter_parts = [
        p.strip() for p in re.split(r"\s*▪◦▪\s*", text) if p.strip()
    ]
    if len(delimiter_parts) == expected_units:
        return delimiter_parts

    lines = [p.strip() for p in text.splitlines() if p.strip()]
    if not lines:
        return [text]
    if len(lines) == expected_units:
        return lines
    if len(lines) < expected_units:
        # Do not fabricate missing utterances; retain the available ordered text.
        return lines

    # There are more non-empty lines than expected utterances. Group adjacent
    # lines into expected_units chunks, preserving their original order.
    # The target is approximately equal character length per utterance while
    # leaving at least one line for every remaining group.
    total_chars = sum(len(line) for line in lines)
    target = total_chars / expected_units
    groups = []
    current = []
    current_chars = 0
    remaining_groups = expected_units

    for idx, line in enumerate(lines):
        remaining_lines_after = len(lines) - idx - 1
        if current and remaining_groups > 1:
            # Close a group when adding another line would overshoot the target,
            # or when enough lines remain to start the remaining groups.
            can_close = remaining_lines_after >= remaining_groups - 1
            if can_close and abs(current_chars - target) <= abs(
                (current_chars + len(line)) - target
            ):
                groups.append(" ".join(current))
                current = []
                current_chars = 0
                remaining_groups -= 1
        current.append(line)
        current_chars += len(line)

        # Ensure exactly one line minimum for each future group.
        lines_left = len(lines) - idx - 1
        if remaining_groups > 1 and lines_left == remaining_groups - 1:
            groups.append(" ".join(current))
            current = []
            current_chars = 0
            remaining_groups -= 1

    if current:
        groups.append(" ".join(current))

    # Defensive reconciliation if a corner case produced a different count.
    while len(groups) > expected_units:
        groups[-2] = groups[-2] + " " + groups[-1]
        groups.pop()
    while len(groups) < expected_units:
        # If not enough groups can be recovered, do not invent content.
        break

    return groups


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    required = {"conversation_id", "text", "label", "split", "n_utterances"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"CSV missing required columns: {sorted(missing_cols)}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["conversation_id", "text", "label", "split"]).copy()
    df["conversation_id"] = df["conversation_id"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    df["n_utterances"] = pd.to_numeric(
        df["n_utterances"], errors="coerce"
    ).fillna(1).astype(int)

    if df["conversation_id"].duplicated().any():
        duplicates = df.loc[df["conversation_id"].duplicated(), "conversation_id"]
        raise ValueError(f"Duplicate conversation_id values, e.g. {duplicates.iloc[0]}")

    invalid_labels = sorted(set(df["label"].unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(f"Expected binary labels 0/1; found {invalid_labels}")

    allowed_splits = {"train", "val", "validation", "dev", "test"}
    bad_splits = sorted(set(df["split"].unique()) - allowed_splits)
    if bad_splits:
        raise ValueError(
            f"Unexpected split values: {bad_splits}. "
            "Expected train/val/test (validation/dev are also accepted)."
        )
    df["split"] = df["split"].replace({"validation": "val", "dev": "val"})

    records = []
    approximated = 0
    skipped = 0
    for row in df.itertuples(index=False):
        turns = split_text_to_units(row.text, row.n_utterances)
        if not turns:
            skipped += 1
            continue
        if len(turns) != int(row.n_utterances):
            approximated += 1
        record = {
            "conversation_id": str(row.conversation_id),
            "text": str(row.text),
            "turns": turns,
            "label": int(row.label),
            "split": str(row.split),
            "pair_id": str(getattr(row, "pair_id", "")),
        }
        records.append(record)

    print(f"Loaded rows: {len(df)}")
    print(f"Usable conversations: {len(records)}")
    print(f"Skipped empty conversations: {skipped}")
    print(
        "Note: conversations whose reconstructed turn count differs from "
        f"n_utterances: {approximated}"
    )

    for split in ["train", "val", "test"]:
        subset = [r for r in records if r["split"] == split]
        if not subset:
            raise ValueError(f"No records found for split='{split}'")
        positives = sum(r["label"] for r in subset)
        print(
            f"{split}: n={len(subset)}, positive={positives}, "
            f"negative={len(subset)-positives}"
        )

    # Check pair IDs do not cross the official splits, where pair_id is available.
    pair_splits = {}
    for r in records:
        pair = r["pair_id"]
        if pair and pair.lower() != "nan":
            pair_splits.setdefault(pair, set()).add(r["split"])
    leaked_pairs = [p for p, splits in pair_splits.items() if len(splits) > 1]
    if leaked_pairs:
        print(
            f"WARNING: {len(leaked_pairs)} pair_id values occur across multiple "
            "splits. Check for paired-conversation leakage before reporting results."
        )

    return records


def prefix_text(record, stage):
    turns = record["turns"]
    count = max(1, math.ceil(len(turns) * stage / 100.0))
    return "\n".join(turns[:count])


class ConversationDataset(Dataset):
    def __init__(self, records, stage, tokenizer, max_length):
        self.records = records
        self.stage = stage
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        encoded = self.tokenizer(
            prefix_text(record, self.stage),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(record["label"], dtype=torch.long)
        return item


def safe_auc_metrics(labels, probabilities):
    if len(set(labels)) < 2:
        return float("nan"), float("nan")
    return (
        roc_auc_score(labels, probabilities),
        average_precision_score(labels, probabilities),
    )


def evaluate(model, loader, device, verbose=False):
    model.eval()
    labels_all, preds_all, probs_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()
            labels_all.extend(labels.cpu().numpy().tolist())
            preds_all.extend(preds.cpu().numpy().tolist())
            probs_all.extend(probs.cpu().numpy().tolist())

    if not labels_all:
        raise ValueError("Evaluation split is empty.")
    roc_auc, pr_auc = safe_auc_metrics(labels_all, probs_all)
    if verbose:
        print(
            f"Evaluation labels: positives={sum(labels_all)}, "
            f"negatives={len(labels_all)-sum(labels_all)}"
        )
        print(
            f"Predicted positives={sum(preds_all)}, "
            f"predicted negatives={len(preds_all)-sum(preds_all)}"
        )

    return {
        "accuracy": accuracy_score(labels_all, preds_all),
        "precision": precision_score(labels_all, preds_all, zero_division=0),
        "recall": recall_score(labels_all, preds_all, zero_division=0),
        "f1": f1_score(labels_all, preds_all, zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def train_one_stage(records, stage, tokenizer, args, device):
    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "val"]
    test_records = [r for r in records if r["split"] == "test"]

    if args.smoke_test:
        # Small sample to verify that data/model/training run end-to-end.
        train_records = train_records[: min(32, len(train_records))]
        val_records = val_records[: min(32, len(val_records))]
        test_records = test_records[: min(32, len(test_records))]
        epochs = 1
    else:
        epochs = args.epochs

    train_ds = ConversationDataset(train_records, stage, tokenizer, args.max_length)
    val_ds = ConversationDataset(val_records, stage, tokenizer, args.max_length)
    test_ds = ConversationDataset(test_records, stage, tokenizer, args.max_length)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, generator=generator
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"\n{'='*16} Stage {stage}% {'='*16}")
    print(
        f"Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}, "
        f"epochs={epochs}, batch_size={args.batch_size}, device={device}"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    best_val_auc = -float("inf")
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())

        val_metrics = evaluate(model, val_loader, device)
        mean_loss = total_loss / max(1, len(train_loader))
        print(
            f"Epoch {epoch + 1}/{epochs} | loss={mean_loss:.4f} | "
            f"val_F1={val_metrics['f1']:.4f} | "
            f"val_ROC-AUC={val_metrics['roc_auc']:.4f}"
        )
        history.append({
            "stage": stage,
            "epoch": epoch + 1,
            "train_loss": mean_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        # Use validation ROC-AUC for model selection, never the test set.
        score = val_metrics["roc_auc"]
        if not np.isnan(score) and score > best_val_auc:
            best_val_auc = score
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        elif best_state is None:
            # Fallback if ROC-AUC is undefined (e.g. one-class validation sample).
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Test set is evaluated only after selecting the best validation checkpoint.
    test_metrics = evaluate(model, test_loader, device, verbose=True)
    print("Test:", {k: round(v, 4) if not np.isnan(v) else None
                     for k, v in test_metrics.items()})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(
        output_dir / f"roberta_{stage}pct_training_history.csv", index=False
    )

    if not args.smoke_test:
        model_dir = output_dir / f"roberta_{stage}pct"
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

    result = {"stage": stage, **test_metrics}
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result



def resolve_project_path(value, project_root):
    """Resolve relative paths from the project root."""
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


def main():
    print("========== MAIN FUNCTION ENTERED ==========", flush=True)
    parser = argparse.ArgumentParser(
        description="Train RoBERTa for early personal-attack prediction."
    )

    parser.add_argument(
        "--config",
        default="configs/roberta_config.json"
    )

    parser.add_argument("--data", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)

    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        default=None
    )

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--smoke-test",
        action="store_true"
    )

    args_cli = parser.parse_args()
    print("========== ARGUMENTS PARSED ==========", flush=True)
    # --------------------------------------------------
    # 1. Load JSON configuration
    # --------------------------------------------------

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]

    config_path = resolve_project_path(
        args_cli.config,
        project_root
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    dataset_cfg = config["dataset"]
    early_cfg = config["early_prediction"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    output_cfg = config["output"]

    # --------------------------------------------------
    # 2. Read configuration values
    # --------------------------------------------------

    data_path = args_cli.data or dataset_cfg["path"]

    output_path = (
        args_cli.output_dir
        or output_cfg["directory"]
    )

    stages = (
        args_cli.stages
        if args_cli.stages is not None
        else early_cfg["stages_percent"]
    )

    epochs = (
        args_cli.epochs
        if args_cli.epochs is not None
        else training_cfg["epochs"]
    )

    batch_size = (
        args_cli.batch_size
        if args_cli.batch_size is not None
        else training_cfg["batch_size"]
    )

    eval_batch_size = (
        args_cli.eval_batch_size
        if args_cli.eval_batch_size is not None
        else training_cfg["eval_batch_size"]
    )

    max_length = (
        args_cli.max_length
        if args_cli.max_length is not None
        else model_cfg["max_sequence_length"]
    )

    learning_rate = (
        args_cli.learning_rate
        if args_cli.learning_rate is not None
        else training_cfg["learning_rate"]
    )

    weight_decay = (
        args_cli.weight_decay
        if args_cli.weight_decay is not None
        else training_cfg["weight_decay"]
    )

    seed = (
        args_cli.seed
        if args_cli.seed is not None
        else training_cfg["random_seed"]
    )

    model_name = model_cfg["checkpoint"]

    # --------------------------------------------------
    # 3. Construct training arguments
    # --------------------------------------------------

    args = argparse.Namespace(
        data=str(resolve_project_path(data_path, project_root)),
        output_dir=str(resolve_project_path(output_path, project_root)),
        stages=[int(x) for x in stages],
        epochs=int(epochs),
        batch_size=int(batch_size),
        eval_batch_size=int(eval_batch_size),
        max_length=int(max_length),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        seed=int(seed),
        smoke_test=args_cli.smoke_test,
        model_name=model_name,
        fp16=bool(training_cfg.get("fp16", False))
    )

    # --------------------------------------------------
    # 4. Validate parameters
    # --------------------------------------------------

    invalid_stages = sorted(
        set(args.stages) - set(DEFAULT_STAGES)
    )

    if invalid_stages:
        parser.error(
            f"Invalid stages: {invalid_stages}. "
            f"Choose from {DEFAULT_STAGES}."
        )

    if not args.stages:
        parser.error("At least one stage must be selected.")

    if args.epochs < 1:
        parser.error("Epochs must be positive.")

    if args.batch_size < 1 or args.eval_batch_size < 1:
        parser.error("Batch sizes must be positive.")

    if args.max_length < 8:
        parser.error("Max length must be at least 8.")

    if args.learning_rate <= 0:
        parser.error("Learning rate must be positive.")

    # --------------------------------------------------
    # 5. Initialize environment
    # --------------------------------------------------

    set_seed(args.seed)

    torch.set_num_threads(
        max(1, min(4, os.cpu_count() or 1))
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n========== EXPERIMENT CONFIGURATION ==========")
    print("Script:", script_path)
    print("Config:", config_path)
    print("Dataset:", args.data)
    print("Output:", args.output_dir)
    print("Model:", args.model_name)
    print("Stages:", args.stages)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Evaluation batch size:", args.eval_batch_size)
    print("Max length:", args.max_length)
    print("Learning rate:", args.learning_rate)
    print("Weight decay:", args.weight_decay)
    print("Seed:", args.seed)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", device)
    print("==============================================\n")

    # --------------------------------------------------
    # 6. Check files and output directory
    # --------------------------------------------------

    if not Path(args.data).exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.data}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 7. Load dataset and tokenizer
    # --------------------------------------------------

    print("Loading CSV and reconstructing turns...")

    records = load_data(args.data)

    print("Loading RoBERTa tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name
    )

    # --------------------------------------------------
    # 8. Train each early prediction stage
    # --------------------------------------------------

    results = []

    for stage in args.stages:

        result = train_one_stage(
            records,
            stage,
            tokenizer,
            args,
            device
        )

        results.append(result)

        # Save results immediately after each completed stage.
        result_path = output_dir / (
            "roberta_smoke_test_results.csv"
            if args.smoke_test
            else "roberta_test_results.csv"
        )

        pd.DataFrame(results).to_csv(
            result_path,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"Saved results: {result_path}")

    # --------------------------------------------------
    # 9. Save experiment metadata
    # --------------------------------------------------

    metadata = {
        "experiment_name": config.get(
            "experiment_name",
            "roberta_early_prediction"
        ),
        "model_name": args.model_name,
        "dataset": args.data,
        "stages": args.stages,
        "epochs": 1 if args.smoke_test else args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "fp16_requested": args.fp16,
        "smoke_test": args.smoke_test
    }

    metadata_path = output_dir / "run_metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4
        )

    # --------------------------------------------------
    # 10. Final results
    # --------------------------------------------------

    print("\n========== FINAL TEST RESULTS ==========")

    print(
        pd.DataFrame(results).to_string(index=False)
    )

    print("\n========== OUTPUT FILES ==========")

    for stage in args.stages:

        history_file = (
            output_dir
            / f"roberta_{stage}pct_training_history.csv"
        )

        if history_file.exists():
            print("[OK]", history_file)

        else:
            print("[MISSING]", history_file)

        if not args.smoke_test:
            model_dir = output_dir / f"roberta_{stage}pct"

            print(
                "[OK]" if model_dir.exists() else "[MISSING]",
                model_dir
            )

    print("[OK]", result_path)
    print("[OK]", metadata_path)

    print("\nAll experiments completed.")
    print("Output directory:", output_dir.resolve())
if __name__ == "__main__":
    main()
