# Dataset instructions

The training scripts expect one CSV row per conversation.

Required columns:
- `conversation_id`
- `text`
- `label`
- `split`

Optional columns:
- `pair_id`
- `n_utterances`

Utterances inside `text` must be separated by the exact delimiter `▪◦▪`.
Labels must be numeric `0` and `1`. Splits must be `train`, `val`, and `test`.

Save the prepared CSV as `data/conversations_clean.csv`, or supply a different
path with `--data`.

## Source data and licensing

Obtain the real dataset through its official source and follow its license,
terms, and citation instructions. This repository intentionally does not
include source data. Do not upload private, restricted, or non-redistributable
data to GitHub.

## Synthetic dataset

If you use the project-generated `synthetic_personal_attack_6000.csv`, keep it
separate from real-data experiments and explicitly mark all outputs as
synthetic. The dataset was programmatically/template generated and has not
been established as human-validated ground truth. It may contain artificial
surface patterns that inflate scores.
