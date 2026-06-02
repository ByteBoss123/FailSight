"""
Drift Monitoring — FailSight
Evidently AI for sensor drift + model performance monitoring

Why this matters for semiconductor equipment:
Equipment ages. Sensors degrade. A model trained on new equipment
will drift as the fleet ages. This monitors when retraining is needed.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_PATH = Path("data/processed/")
REPORTS_PATH = Path("monitoring/reports/")
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

SENSOR_COLS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "sensor_8", "sensor_9", "sensor_11", "sensor_12",
]


def load_reference():
    """Load training data as reference distribution."""
    path = PROCESSED_PATH / "train.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df[SENSOR_COLS].dropna().iloc[:500]
    return _synthetic_reference()


def simulate_production_data(reference: pd.DataFrame, scenario: str = "normal") -> pd.DataFrame:
    """
    Simulate incoming production sensor data.

    Scenarios:
    - normal: equipment operating as expected
    - aging: sensors slowly drifting (normal equipment wear)
    - degrading: sharp sensor excursions (imminent failure)
    """
    np.random.seed(42)
    new_data = reference.copy().iloc[:100]

    if scenario == "aging":
        # Gradual drift — equipment getting older
        for col in SENSOR_COLS:
            new_data[col] = new_data[col] * np.random.uniform(1.02, 1.08, len(new_data))

    elif scenario == "degrading":
        # Sharp excursions — failure approaching
        for col in SENSOR_COLS[:4]:
            new_data[col] = new_data[col] * np.random.uniform(1.2, 1.6, len(new_data))

    return new_data


def _synthetic_reference() -> pd.DataFrame:
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "sensor_2": np.random.normal(642, 3, n),
        "sensor_3": np.random.normal(1590, 5, n),
        "sensor_4": np.random.normal(1400, 8, n),
        "sensor_7": np.random.normal(554, 2, n),
        "sensor_8": np.random.normal(2388, 10, n),
        "sensor_9": np.random.normal(9046, 20, n),
        "sensor_11": np.random.normal(47, 0.5, n),
        "sensor_12": np.random.normal(522, 3, n),
    })


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame, name: str = None) -> dict:
    """Generate Evidently drift report and return summary."""
    if name is None:
        name = f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    cols = [c for c in SENSOR_COLS if c in reference.columns and c in current.columns]
    ref = reference[cols]
    cur = current[cols]

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=ref, current_data=cur)
    report.save_html(str(REPORTS_PATH / f"{name}.html"))

    # Extract summary
    report_dict = report.as_dict()
    summary = {
        "generated_at": datetime.now().isoformat(),
        "drift_detected": False,
        "drift_share": 0.0,
        "drifted_sensors": [],
        "recommendation": "Normal",
    }

    for metric in report_dict.get("metrics", []):
        if metric.get("metric") == "DatasetDriftMetric":
            result = metric.get("result", {})
            summary["drift_detected"] = result.get("dataset_drift", False)
            summary["drift_share"] = result.get("share_of_drifted_columns", 0.0)

    if summary["drift_detected"]:
        if summary["drift_share"] > 0.5:
            summary["recommendation"] = "CRITICAL: Retrain model immediately. Major sensor distribution shift detected."
        else:
            summary["recommendation"] = "WARNING: Schedule model retraining. Sensor drift detected."
    else:
        summary["recommendation"] = "Stable: No retraining needed."

    with open(REPORTS_PATH / f"{name}.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Report saved: {name} | Drift: {summary['drift_detected']}")
    return summary


def monitor(scenario: str = "normal"):
    """Run monitoring for a given scenario."""
    reference = load_reference()
    current = simulate_production_data(reference, scenario)
    summary = run_drift_report(reference, current, f"drift_{scenario}")

    print(f"Scenario: {scenario.upper()}")
    print(f"Drift Detected: {summary['drift_detected']}")
    print(f"Drift Share: {summary['drift_share']:.1%}")
    print(f"Recommendation: {summary['recommendation']}")
    return summary


if __name__ == "__main__":
    for scenario in ["normal", "aging", "degrading"]:
        print(f"\n{'='*50}")
        monitor(scenario)
    print(f"\nReports saved to: monitoring/reports/")
