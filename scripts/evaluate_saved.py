"""Evaluate the committed LSTM on FD001 and shifted FD002 without retraining."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from failsight import data as D
from scripts.train_eval import sequences, score


def main():
    meta = json.loads((ROOT / "models/failsight_lstm_meta.json").read_text())
    model = tf.keras.models.load_model(ROOT / "models/failsight_lstm.keras")
    results = {}
    for dataset in ("FD001", "FD002"):
        _, test, labels = D.load(dataset)
        X, y = sequences(D.label_test(test, labels), meta["sensors"],
                         meta["mu"], meta["sd"], last_only=True)
        assert np.array_equal(y, np.minimum(labels, D.RUL_CAP))
        predictions = model(X, training=False).numpy().ravel()
        results[dataset] = score(y, predictions)
    results["scope"] = "Committed LSTM, no retraining; official last-cycle engine predictions"
    (ROOT / "results/saved_lstm_evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
