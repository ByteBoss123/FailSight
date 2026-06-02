"""
Data Pipeline — FailSight
Loads NASA CMAPSS turbofan degradation dataset
Labels Remaining Useful Life (RUL) per engine
Engineers degradation features for failure prediction

Dataset: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
Files needed: train_FD001.txt, test_FD001.txt, RUL_FD001.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/")
PROCESSED_PATH = Path("data/processed/")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# NASA CMAPSS column names
COLUMNS = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors that actually carry degradation signal (from research literature)
# Others are near-constant and add noise
USEFUL_SENSORS = [
    "sensor_2",   # Total temperature at LPC outlet
    "sensor_3",   # Total temperature at HPC outlet
    "sensor_4",   # Total temperature at LPT outlet
    "sensor_7",   # Total pressure at HPC outlet
    "sensor_8",   # Physical fan speed
    "sensor_9",   # Physical core speed
    "sensor_11",  # Static pressure at HPC outlet
    "sensor_12",  # Ratio of fuel flow to Ps30
    "sensor_13",  # Corrected fan speed
    "sensor_14",  # Corrected core speed
    "sensor_15",  # Bypass ratio
    "sensor_17",  # Bleed enthalpy
    "sensor_20",  # High-pressure turbine coolant bleed
    "sensor_21",  # Low-pressure turbine coolant bleed
]

FEATURE_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"] + USEFUL_SENSORS


def load_cmapss(subset: str = "FD001") -> tuple:
    """
    Load NASA CMAPSS dataset.
    FD001: Single fault mode, sea level conditions — cleanest subset to start.
    """
    train_file = RAW_PATH / f"train_{subset}.txt"
    test_file = RAW_PATH / f"test_{subset}.txt"
    rul_file = RAW_PATH / f"RUL_{subset}.txt"

    if not train_file.exists():
        logger.warning(f"{train_file} not found — generating synthetic data")
        return _generate_synthetic_cmapss()

    train = pd.read_csv(train_file, sep=" ", header=None, names=COLUMNS)
    test = pd.read_csv(test_file, sep=" ", header=None, names=COLUMNS)
    rul = pd.read_csv(rul_file, sep=" ", header=None, names=["RUL"])

    # Drop empty columns (CMAPSS has trailing spaces)
    train = train.dropna(axis=1, how="all")
    test = test.dropna(axis=1, how="all")

    logger.info(f"Train: {train.shape} | Test: {test.shape}")
    return train, test, rul


def label_rul(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    """
    Label each row with Remaining Useful Life (RUL).

    RUL = cycles remaining until engine failure.
    Cap at 125 — engines in healthy state treated as equally healthy.
    This is standard practice in predictive maintenance literature.

    Example:
        Engine 1 fails at cycle 192
        At cycle 1:   RUL = 191
        At cycle 100: RUL = 92
        At cycle 191: RUL = 1  ← imminent failure
    """
    df = df.copy()
    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    df = df.merge(max_cycles, on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=cap)
    df = df.drop(columns=["max_cycle"])

    # Failure label: RUL <= 30 cycles = imminent failure zone
    df["failure_imminent"] = (df["RUL"] <= 30).astype(int)

    # Risk tier for dashboard
    df["risk_tier"] = pd.cut(
        df["RUL"],
        bins=[-1, 15, 30, 60, 125],
        labels=["critical", "high", "medium", "healthy"],
    )

    logger.info(f"RUL labeled. Failure imminent: {df['failure_imminent'].sum()} rows")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer degradation features from raw sensor readings.

    Key insight: raw sensor values matter less than their TREND over time.
    A temperature reading of 650°C is normal. A temperature RISING 5° per cycle
    for 20 cycles straight is a failure signal.
    """
    df = df.copy()

    for sensor in USEFUL_SENSORS:
        if sensor not in df.columns:
            continue

        # Rolling mean — smooths noise, reveals trend
        df[f"{sensor}_roll7_mean"] = (
            df.groupby("engine_id")[sensor]
            .transform(lambda x: x.rolling(7, min_periods=1).mean())
        )

        # Rolling std — rising variance = degradation signal
        df[f"{sensor}_roll7_std"] = (
            df.groupby("engine_id")[sensor]
            .transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0))
        )

        # Rate of change — acceleration of degradation
        df[f"{sensor}_diff"] = (
            df.groupby("engine_id")[sensor].transform(lambda x: x.diff().fillna(0))
        )

    # Cycle normalized per engine (how far into life is this engine?)
    max_cycles = df.groupby("engine_id")["cycle"].transform("max")
    df["cycle_normalized"] = df["cycle"] / max_cycles

    # Drop rows with NaN from rolling features
    df = df.dropna()
    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


def normalize_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    Normalize sensor features using MinMaxScaler fit on training data only.
    Critical: never fit scaler on test data — that's data leakage.
    """
    feature_cols = [c for c in train.columns if c.startswith("sensor_") or c.startswith("op_")]
    feature_cols += ["cycle_normalized"]
    feature_cols = [c for c in feature_cols if c in train.columns]

    scaler = MinMaxScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])

    # Save scaler for API inference
    joblib.dump(scaler, PROCESSED_PATH / "scaler.pkl")
    joblib.dump(feature_cols, PROCESSED_PATH / "feature_cols.pkl")

    logger.info(f"Normalized {len(feature_cols)} features")
    return train, test, scaler, feature_cols


def _generate_synthetic_cmapss(n_engines: int = 100) -> tuple:
    """
    Generate realistic synthetic degradation data when NASA dataset unavailable.
    Simulates gradual sensor drift + failure pattern.
    """
    logger.info("Generating synthetic CMAPSS-like data...")
    np.random.seed(42)
    records = []

    for engine_id in range(1, n_engines + 1):
        max_cycles = np.random.randint(150, 350)

        for cycle in range(1, max_cycles + 1):
            degradation = cycle / max_cycles  # 0 to 1 as engine degrades

            row = {
                "engine_id": engine_id,
                "cycle": cycle,
                "op_setting_1": np.random.normal(0, 0.002),
                "op_setting_2": np.random.normal(0, 0.0003),
                "op_setting_3": np.random.choice([100.0]),
            }

            # Sensors drift upward as engine degrades (realistic pattern)
            sensor_bases = {
                "sensor_2": 642, "sensor_3": 1590, "sensor_4": 1400,
                "sensor_7": 554, "sensor_8": 2388, "sensor_9": 9046,
                "sensor_11": 47, "sensor_12": 522, "sensor_13": 2388,
                "sensor_14": 8138, "sensor_15": 8.4, "sensor_17": 391,
                "sensor_20": 38, "sensor_21": 23,
            }

            for sensor, base in sensor_bases.items():
                drift = base * 0.05 * degradation  # 5% drift at end of life
                noise = np.random.normal(0, base * 0.005)
                row[sensor] = base + drift + noise

            records.append(row)

    train_df = pd.DataFrame(records)

    # Test set: engines stopped before failure (no ground truth RUL)
    test_records = []
    rul_values = []
    for engine_id in range(n_engines + 1, n_engines + 21):
        max_cycles = np.random.randint(150, 350)
        stop_at = np.random.randint(100, max_cycles - 20)
        true_rul = max_cycles - stop_at

        for cycle in range(1, stop_at + 1):
            degradation = cycle / max_cycles
            row = {"engine_id": engine_id, "cycle": cycle,
                   "op_setting_1": np.random.normal(0, 0.002),
                   "op_setting_2": np.random.normal(0, 0.0003),
                   "op_setting_3": 100.0}
            for sensor, base in {"sensor_2": 642, "sensor_3": 1590, "sensor_4": 1400,
                                  "sensor_7": 554, "sensor_8": 2388, "sensor_9": 9046,
                                  "sensor_11": 47, "sensor_12": 522, "sensor_13": 2388,
                                  "sensor_14": 8138, "sensor_15": 8.4, "sensor_17": 391,
                                  "sensor_20": 38, "sensor_21": 23}.items():
                drift = base * 0.05 * degradation
                noise = np.random.normal(0, base * 0.005)
                row[sensor] = base + drift + noise
            test_records.append(row)

        rul_values.append(true_rul)

    test_df = pd.DataFrame(test_records)
    rul_df = pd.DataFrame({"RUL": rul_values})

    logger.info(f"Synthetic data: {len(train_df)} train rows, {len(test_df)} test rows")
    return train_df, test_df, rul_df


def run_pipeline():
    """Execute full data pipeline."""
    train_raw, test_raw, rul = load_cmapss()
    train_labeled = label_rul(train_raw)
    train_features = engineer_features(train_labeled)
    test_features = engineer_features(label_rul(test_raw))
    train_norm, test_norm, scaler, feat_cols = normalize_features(train_features, test_features)

    train_norm.to_csv(PROCESSED_PATH / "train.csv", index=False)
    test_norm.to_csv(PROCESSED_PATH / "test.csv", index=False)
    rul.to_csv(PROCESSED_PATH / "rul.csv", index=False)

    logger.info("Pipeline complete. Files in data/processed/")
    return train_norm, test_norm, rul


if __name__ == "__main__":
    run_pipeline()
