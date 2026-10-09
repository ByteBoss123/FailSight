"""NASA C-MAPSS turbofan telemetry: loading, Remaining-Useful-Life labels, causal features.

Each engine streams 21 sensors and 3 operating settings once per cycle until it fails (train) or until
an unknown point before failure (test, with the true RUL given separately). FD001: one operating condition,
one fault mode. FD002: six operating conditions, same fault mode (used here as a distribution shift).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
COLUMNS = ["engine_id", "cycle"] + [f"setting_{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
RUL_CAP = 125          # standard piecewise-linear target: early-life cycles are all "healthy"
WARN_CYCLES = 30       # early-warning horizon: failure within 30 cycles
WINDOWS = (5, 15)


def load(subset="FD001"):
    def read(name):
        return pd.read_csv(RAW / f"{name}_{subset}.txt", sep=r"\s+", header=None)

    train, test = read("train"), read("test")
    train.columns = test.columns = COLUMNS
    rul = read("RUL")[0].to_numpy()
    return train, test, rul


def informative_sensors(train, min_std=1e-4):
    """Sensors that actually vary in training data (7 of 21 are flat in FD001)."""
    return [c for c in COLUMNS[5:] if train[c].std() > min_std]


def label_train(train):
    df = train.copy()
    last = df.groupby("engine_id")["cycle"].transform("max")
    df["rul"] = (last - df["cycle"]).clip(upper=RUL_CAP)
    return df


def label_test(test, rul):
    df = test.copy()
    last = df.groupby("engine_id")["cycle"].transform("max")
    true_end = df["engine_id"].map(dict(zip(range(1, len(rul) + 1), rul)))
    df["rul"] = (true_end + last - df["cycle"]).clip(upper=RUL_CAP)
    return df


def add_features(df, sensors):
    """Per-engine rolling mean/std and slope over past cycles only (no look-ahead), plus cycle count."""
    out = df.sort_values(["engine_id", "cycle"]).copy()
    g = out.groupby("engine_id")
    feats = {}
    for s in sensors:
        for w in WINDOWS:
            r = g[s].rolling(w, min_periods=1)
            feats[f"{s}_mean{w}"] = r.mean().reset_index(level=0, drop=True)
            feats[f"{s}_std{w}"] = r.std().reset_index(level=0, drop=True).fillna(0.0)
        feats[f"{s}_slope15"] = (out[s] - g[s].shift(14)).fillna(0.0) / 14
    out = pd.concat([out, pd.DataFrame(feats, index=out.index)], axis=1)
    return out


def feature_columns(sensors):
    cols = ["cycle"] + list(sensors)
    for s in sensors:
        cols += [f"{s}_mean{w}" for w in WINDOWS] + [f"{s}_std{w}" for w in WINDOWS] + [f"{s}_slope15"]
    return cols


def last_cycle(df):
    """The official test protocol scores one prediction per engine, at its last observed cycle."""
    return df.loc[df.groupby("engine_id")["cycle"].idxmax()].sort_values("engine_id")


def engine_split(df, frac=0.2, seed=0):
    ids = np.array(sorted(df["engine_id"].unique()))
    rng = np.random.default_rng(seed)
    val = set(rng.choice(ids, size=int(len(ids) * frac), replace=False).tolist())
    return df[~df["engine_id"].isin(val)], df[df["engine_id"].isin(val)]
