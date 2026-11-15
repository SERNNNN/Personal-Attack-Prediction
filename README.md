# Early Prediction of Personal Attacks in Online Discussions

A reproducible research codebase for evaluating whether personal attacks can be
predicted **before an online discussion has ended**.

## Research question

> How effectively can traditional machine learning models and RoBERTa predict
> personal attacks in ongoing online discussions, and how does prediction
> performance vary with the proportion of conversation text available?

## Task

This repository implements binary conversation-level classification:

- `0`: no personal attack
- `1`: personal attack

For early prediction, each conversation is truncated to its first 20%, 40%,
60%, 80%, or 100% of utterances. Each prefix stage is trained and evaluated as
a separate experiment. The `80%` stage means **80% of each conversation's
utterances are visible**, not that 80% of the dataset is assigned to testing.

## Repository layout

```text
Personal-Attack-Prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── CITATION.cff
├── configs/
│   ├── traditional_config.json
│   └── roberta_config.json
├── data/
│   ├── README.md
│   └── conversations_clean.csv
├── src/
│   ├── data_preparation/
│   │   └── validate_dataset.py
│   ├── training/
│   │   ├── traditional_early_prediction.py
│   │   └── roberta_early_prediction.py
│   ├── evaluation/
│   │   └── summarize_results.py
│   ├── inference/
│   │   └── predict.py
│   └── utils/
│       └── common.py
├── tests/
│   └── test_common.py
├── results/
│   ├── traditional/
│   └── roberta/
└── checkpoints/
    └── README.md
```

## Dataset

The scripts expect a **conversation-level CSV** with the following columns:

| Column            | Description                                                     |
| ----------------- | --------------------------------------------------------------- |
| `conversation_id` | Unique conversation identifier                                  |
| `text`            | Full conversation, utterances joined by `▪◦▪`                   |
| `label`           | Binary label: `0` or `1`                                        |
| `split`           | Existing partition: `train`, `val`, or `test`                   |
| `pair_id`         | Optional pair identifier; paired rows should stay in one split  |
| `n_utterances`    | Optional utterance count; checked against the text when present |

Put the file at `data/conversations_clean.csv`, or pass another path with
`--data`. The scripts use the provided split assignments and do not silently
create new splits.

**Data rights:** do not commit the original dataset or any restricted data to a
public repository unless its license explicitly permits redistribution. Follow
the dataset provider's access and citation requirements. See `data/README.md`.



## Environment

Recommended: Python 3.10 or 3.11.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with the machine's CUDA
driver/runtime. The course environment is expected to provide an NVIDIA RTX
A5000, but the author is using a version compatible with an RTX 2060 laptop.

## Validate the dataset

```bash
python src/data_preparation/validate_dataset.py --data data/conversations_clean.csv
```

This checks required columns, labels, split values, duplicate conversation
IDs, utterance counts, and pair/split integrity where `pair_id` is present.

## Train the traditional model

```bash
python src/training/traditional_early_prediction.py \
  --config configs/traditional_config.json
```

Windows PowerShell may use the same command on one line:

```powershell
python src/training/traditional_early_prediction.py --config configs/traditional_config.json
```

This runs TF-IDF + `SGDClassifier(loss="log_loss")` separately for each
conversation-prefix stage. `max_iter=20` and `tol=null` request 20 passes over
the training data (epochs) for the SGD classifier.

This is **not** the same algorithm as the earlier LogisticRegression baseline.
If reproducing that baseline, preserve its exact estimator and configuration
and name it separately in results.

## Fine-tune RoBERTa

```bash
python src/training/roberta_early_prediction.py --config configs/roberta_config.json
```

The default model is `FacebookAI/roberta-base`. The model weights are downloaded
from Hugging Face on first use unless already cached. GPU memory requirements
depend on sequence length and batch size. Lower `batch_size` or
`max_sequence_length` if memory is insufficient; record any changes.

## Summarize results

```bash
python src/evaluation/summarize_results.py --results-dir results
```

The script combines stage-level files into one summary table.

## Predict with a saved model

Traditional model:

```bash
python src/inference/predict.py \
  --model results/traditional/stage_80/model.joblib \
  --text-file path/to/conversation.txt
```

RoBERTa model:

```bash
python src/inference/predict.py \
  --model results/roberta/stage_80/model \
  --text-file path/to/conversation.txt
```

The input text file should contain utterances separated by `▪◦▪`. Prediction
uses the same prefix convention as training.

## Outputs

Each stage writes a folder under the configured output directory, for example:

```text
results/traditional/
├── stage_20/
│   ├── metrics.json
│   ├── predictions.csv
│   └── model.joblib
├── stage_40/
├── stage_60/
├── stage_80/
├── stage_100/
└── metrics.csv
```

RoBERTa additionally saves training logs and a Hugging Face model directory for
each stage. Avoid committing large checkpoints unless required and permitted;
provide reproducible download/training instructions instead.

## Reproducibility protocol

For every reported result, preserve:

1. Dataset name, version, provenance, checksum and class counts.
2. Exact train/validation/test split definitions.
3. Prefix construction rule and utterance separator.
4. Model name, tokenizer/checkpoint, hyperparameters and random seed.
5. Python, package, PyTorch, CUDA and GPU versions.
6. Per-stage metrics, confusion matrix and predictions.
7. The exact command and configuration used.
8. Any failed runs, threshold issues, and deviations from the plan.

Metrics to report: Accuracy, Precision, Recall, F1, ROC-AUC and PR-AUC. Include
test-set size and class distribution. ROC-AUC can be meaningful even when
default-threshold predictions collapse to one class; always inspect the
confusion matrix and F1 as well.

## Limitations

This repository is a reproducible implementation scaffold. Its default
hyperparameters are based on settings discussed during project development;
they are not proof that a previous run used those exact settings. Before
submitting, reconcile each config with the actual experiment logs and scripts.
Do not claim results until the scripts have been run on the stated dataset.
