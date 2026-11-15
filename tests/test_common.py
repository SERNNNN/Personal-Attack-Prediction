import pandas as pd
from src.utils.common import make_prefix, split_utterances, validate_dataframe


def test_prefix_keeps_ceil_fraction():
    text = "one▪◦▪two▪◦▪three▪◦▪four▪◦▪five"
    assert make_prefix(text, 20) == "one"


def test_prefix_80_percent():
    text = "one▪◦▪two▪◦▪three▪◦▪four▪◦▪five"
    assert len(split_utterances(make_prefix(text, 80))) == 4


def test_validation_passes_for_minimal_frame():
    df = pd.DataFrame({
        "conversation_id": ["a", "b"],
        "text": ["hello▪◦▪world", "foo▪◦▪bar"],
        "label": [0, 1],
        "split": ["train", "test"],
    })
    assert validate_dataframe(df) == []
