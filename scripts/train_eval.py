"""Train and evaluate Remaining-Useful-Life models on NASA C-MAPSS FD001; check telemetry drift on FD002.

Protocol: model selection on 20% of training engines held out by engine (no engine in both splits); the chosen
settings are refit on all 100 training engines and scored once on the official test set (100 engines, one
prediction at each engine's last observed cycle, against NASA's true RUL file). Writes results/metrics.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from failsight import data as D  # noqa: E402
from failsight.drift import DriftMonitor, early_life  # noqa: E402
from failsight.metrics import nasa_score, rmse, warning_metrics  # noqa: E402

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[1]
OUT, MODELS = ROOT / "results", ROOT / "models"
SEED, SEQ = 0, 30


def fit_hgb(X, y):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=SEED).fit(X, y)


def fit_ridge(X, y):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)


def sequences(df, sensors, mu, sd, last_only=False):
    """Sliding windows of the last SEQ cycles of standardized sensors (front-padded with the first row)."""
    Xs, ys = [], []
    for _, g in df.sort_values(["engine_id", "cycle"]).groupby("engine_id"):
        v = ((g[sensors].to_numpy() - mu) / sd).astype("float32")
        if len(v) < SEQ:
            v = np.vstack([np.repeat(v[:1], SEQ - len(v), 0), v])
        r = g["rul"].to_numpy()[-len(v):] if len(g) >= SEQ else np.r_[np.repeat(g["rul"].iloc[0], SEQ - len(g)), g["rul"]]
        ends = [len(v)] if last_only else range(SEQ, len(v) + 1)
        for e in ends:
            Xs.append(v[e - SEQ:e])
            ys.append(r[e - 1])
    return np.stack(Xs), np.asarray(ys, "float32")


def fit_lstm(Xs, ys, epochs=30):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    m = tf.keras.Sequential([tf.keras.Input(Xs.shape[1:]), tf.keras.layers.LSTM(64, return_sequences=True),
                             tf.keras.layers.LSTM(32), tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.Dense(16, activation="relu"), tf.keras.layers.Dense(1)])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    m.fit(Xs, ys, epochs=epochs, batch_size=256, verbose=0, validation_split=0.1,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    return m


def score(y, p):
    p = np.clip(p, 0, D.RUL_CAP)
    return {"rmse": round(rmse(y, p), 2), "nasa_score": round(nasa_score(y, p), 1),
            "early_warning_30_cycles": warning_metrics(y, p, D.WARN_CYCLES)}


def main():
    t0 = time.time()
    tr_raw, te_raw, rul = D.load("FD001")
    sensors = D.informative_sensors(tr_raw)
    cols = D.feature_columns(sensors)
    train = D.add_features(D.label_train(tr_raw), sensors)
    test = D.add_features(D.label_test(te_raw, rul), sensors)
    test_last = D.last_cycle(test)
    y_test = test_last["rul"].to_numpy()
    assert np.array_equal(np.minimum(rul, D.RUL_CAP), y_test)  # labels line up with NASA's RUL file

    # ---- model selection on held-out training engines ----
    fit_part, val = D.engine_split(train)
    val_last = D.last_cycle(val)
    mu, sd = fit_part[sensors].mean().to_numpy(), fit_part[sensors].std().to_numpy()
    val_scores = {
        "mean_baseline": rmse(val["rul"], np.full(len(val), fit_part["rul"].mean())),
        "ridge_raw_sensors": rmse(val["rul"], fit_ridge(fit_part[["cycle"] + sensors], fit_part["rul"])
                                  .predict(val[["cycle"] + sensors])),
        "gradient_boosting_features": rmse(val["rul"], fit_hgb(fit_part[cols], fit_part["rul"]).predict(val[cols])),
    }
    Xs, ys = sequences(fit_part, sensors, mu, sd)
    Xv, yv = sequences(val, sensors, mu, sd)
    val_scores["lstm_tensorflow"] = rmse(yv, fit_lstm(Xs, ys).predict(Xv, verbose=0).ravel())
    val_scores = {k: round(v, 2) for k, v in val_scores.items()}

    # ---- refit on all training engines, score once on the official test set ----
    mu, sd = train[sensors].mean().to_numpy(), train[sensors].std().to_numpy()
    hgb = fit_hgb(train[cols], train["rul"])
    ridge = fit_ridge(train[["cycle"] + sensors], train["rul"])
    Xs, ys = sequences(train, sensors, mu, sd)
    lstm = fit_lstm(Xs, ys)
    Xt, yt = sequences(test, sensors, mu, sd, last_only=True)
    assert np.array_equal(yt, y_test)
    p = {"mean_baseline": np.full(len(y_test), train["rul"].mean()),
         "ridge_raw_sensors": ridge.predict(test_last[["cycle"] + sensors]),
         "gradient_boosting_features": hgb.predict(test_last[cols]),
         "lstm_tensorflow": lstm.predict(Xt, verbose=0).ravel()}
    p["ensemble_gb_lstm"] = (p["gradient_boosting_features"] + p["lstm_tensorflow"]) / 2
    test_scores = {k: score(y_test, v) for k, v in p.items()}

    # ---- drift monitoring: FD001 training telemetry as the reference ----
    naive = DriftMonitor(tr_raw, sensors).check(te_raw)          # whole histories: confounded by wear
    mon = DriftMonitor(early_life(tr_raw), sensors)
    d_same = mon.check(early_life(te_raw))
    tr2, te2, rul2 = D.load("FD002")
    d_shift = mon.check(early_life(te2))
    test2_last = D.last_cycle(D.add_features(D.label_test(te2, rul2), sensors))
    shifted = score(test2_last["rul"].to_numpy(), hgb.predict(test2_last[cols]))

    res = {"dataset": "NASA C-MAPSS FD001 (train 100 engines / 20,631 cycles; test 100 engines / 13,096 cycles)",
           "rul_cap": D.RUL_CAP, "informative_sensors": sensors, "n_features": len(cols),
           "validation_rmse_by_engine_holdout": val_scores, "test": test_scores,
           "drift": {"alert_psi": mon.alert, "baseline": "first 30 cycles of each engine",
                     "fd001_test_vs_train_whole_history_naive": {k: naive[k] for k in ("alert", "max_psi", "drifted_sensors")},
                     "fd001_test_vs_train": {k: d_same[k] for k in ("alert", "max_psi", "drifted_sensors")},
                     "fd002_test_vs_fd001_train": {k: d_shift[k] for k in ("alert", "max_psi", "drifted_sensors")},
                     "gradient_boosting_on_fd002_without_retraining": shifted},
           "runtime_s": round(time.time() - t0, 1)}
    OUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2))
    best = min(val_scores, key=val_scores.get)
    res["selected_on_validation"] = best
    lstm.save(MODELS / "failsight_lstm.keras")
    joblib.dump({"mu": mu, "sd": sd, "sensors": sensors, "seq": SEQ, "rul_cap": D.RUL_CAP,
                 "reference": early_life(tr_raw)[sensors]}, MODELS / "failsight_lstm_meta.joblib")
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2))
    joblib.dump({"model": hgb, "sensors": sensors, "columns": cols, "reference": tr_raw[sensors]},
                MODELS / "failsight_gb.joblib")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
