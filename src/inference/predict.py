
"""Run a saved traditional or RoBERTa model on one conversation text file."""

import argparse
from pathlib import Path

import joblib
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from src.utils.common import make_prefix


def load_traditional_model(model_path):
    """Load a traditional model bundle."""
    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise TypeError(
            f"Expected a dictionary model bundle, "
            f"but got {type(bundle).__name__}."
        )

    print("[INFO] Model bundle keys:", list(bundle.keys()))

    # Actual saved model key is "model", not "classifier".
    if "vectorizer" not in bundle:
        raise KeyError("Missing 'vectorizer' in model bundle.")

    if "model" not in bundle:
        raise KeyError("Missing 'model' in model bundle.")

    vectorizer = bundle["vectorizer"]
    classifier = bundle["model"]

    saved_stage = bundle.get("stage")

    return vectorizer, classifier, saved_stage


def predict_traditional(model_path, text, requested_stage):
    """Predict using TF-IDF + SGDClassifier."""

    vectorizer, classifier, saved_stage = load_traditional_model(
        model_path
    )

    # Use the stage stored in the model if the user did not specify one.
    if requested_stage is None:
        stage = saved_stage if saved_stage is not None else 100
    else:
        stage = requested_stage

    # Prevent using a model trained for a different prefix stage.
    if saved_stage is not None and stage != saved_stage:
        raise ValueError(
            f"This model was trained for stage {saved_stage}%, "
            f"but stage {stage}% was requested. "
            f"Please use the matching model or stage."
        )

    prefix = make_prefix(text, stage)

    if not prefix or not prefix.strip():
        raise ValueError("The generated conversation prefix is empty.")

    # Convert conversation text into TF-IDF features.
    features = vectorizer.transform([prefix])

    print("[INFO] TF-IDF feature shape:", features.shape)
    print("[INFO] Model stage:", stage)

    # Predict label.
    prediction = int(classifier.predict(features)[0])

    # SGDClassifier(loss="log_loss") supports predict_proba().
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(features)[0]
        classes = list(classifier.classes_)

        if 1 not in classes:
            raise ValueError(
                f"Positive class 1 not found. Classes: {classes}"
            )

        positive_index = classes.index(1)
        score = float(probabilities[positive_index])

    else:
        raise AttributeError(
            "The classifier does not support predict_proba()."
        )

    return prediction, score, stage


def predict_roberta(model_path, text, stage, max_length):
    """Predict using a fine-tuned RoBERTa model."""

    prefix = make_prefix(text, stage)

    if not prefix or not prefix.strip():
        raise ValueError("The generated conversation prefix is empty.")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.eval()

    inputs = tokenizer(
        prefix,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    if probabilities.numel() != 2:
        raise ValueError(
            f"Expected 2 output classes, got {probabilities.numel()}."
        )

    score = float(probabilities[1].item())
    prediction = int(score >= 0.5)

    return prediction, score, stage


def main():

    parser = argparse.ArgumentParser(
        description="Predict personal attacks in an online conversation."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to .joblib file or RoBERTa model directory",
    )

    parser.add_argument(
        "--text-file",
        required=True,
        help="Path to UTF-8 conversation text file",
    )

    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        choices=[20, 40, 60, 80, 100],
        help="Conversation prefix percentage. "
             "Traditional models use the saved stage by default.",
    )

    parser.add_argument(
        "--model-type",
        choices=["traditional", "roberta"],
        default="traditional",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    # Check input file.
    text_path = Path(args.text_file)

    if not text_path.is_file():
        raise FileNotFoundError(
            f"Conversation file not found: {text_path}"
        )

    # Read UTF-8 text. utf-8-sig also handles UTF-8 BOM files.
    text = text_path.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not text:
        raise ValueError("Conversation text file is empty.")

    print("[INFO] Conversation file:", text_path)
    print("[INFO] Character count:", len(text))

    # Run inference.
    if args.model_type == "traditional":

        prediction, score, stage = predict_traditional(
            args.model,
            text,
            args.stage,
        )

    else:

        stage = args.stage if args.stage is not None else 100

        prediction, score, stage = predict_roberta(
            args.model,
            text,
            stage,
            args.max_length,
        )

    # Display results.
    print("\n========== PREDICTION RESULT ==========")

    print({
        "stage_percent": stage,
        "predicted_label": prediction,
        "predicted_label_meaning": (
            "personal_attack"
            if prediction == 1
            else "no_personal_attack"
        ),
        "personal_attack_probability": round(score, 6),
    })

    print("=======================================")


if __name__ == "__main__":
    main()
