"""Run a saved traditional or RoBERTa model on one conversation text file."""
import argparse
from pathlib import Path
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils.common import make_prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Traditional .joblib or RoBERTa model directory")
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--stage", type=int, default=100, choices=[20, 40, 60, 80, 100])
    parser.add_argument("--model-type", choices=["traditional", "roberta"], default="traditional")
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    text = Path(args.text_file).read_text(encoding="utf-8")
    prefix = make_prefix(text, args.stage)

    if args.model_type == "traditional":
        bundle = joblib.load(args.model)
        x = bundle["vectorizer"].transform([prefix])
        pred = int(bundle["classifier"].predict(x)[0])
        score = float(bundle["classifier"].predict_proba(x)[0, 1])
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(args.model)
        model.eval()
        inputs = tokenizer(prefix, return_tensors="pt", truncation=True,
                           padding=True, max_length=args.max_length)
        with torch.no_grad():
            logits = model(**inputs).logits
            score = float(torch.softmax(logits, dim=1)[0, 1].item())
        pred = int(score >= 0.5)

    print({
        "stage_percent": args.stage,
        "predicted_label": pred,
        "predicted_label_meaning": "personal_attack" if pred == 1 else "no_personal_attack",
        "personal_attack_probability": round(score, 6),
    })


if __name__ == "__main__":
    main()
