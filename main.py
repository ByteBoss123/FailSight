"""
FastAPI — FailSight Risk Score API
Serves real-time equipment failure risk scores
Integrates with fab MES systems via REST

Deploy: Render.com (free tier)
Run locally: uvicorn api.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import joblib
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FailSight — Equipment Failure Prediction API",
    description="""
    Real-time semiconductor equipment failure prediction.
    Returns Remaining Useful Life (RUL) estimate and risk tier
    from multivariate sensor telemetry.

    Designed for integration with fab MES / SCADA systems.
    """,
    version="1.0.0",
)

PROCESSED_PATH = Path("data/processed/")
MODEL_PATH = Path("models/survival/")


# ── Schemas ───────────────────────────────────────────────────────────────────

class SensorReading(BaseModel):
    engine_id: str
    cycle: int
    op_setting_1: float = 0.0
    op_setting_2: float = 0.0
    op_setting_3: float = 100.0
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_17: float
    sensor_20: float
    sensor_21: float

    class Config:
        json_schema_extra = {
            "example": {
                "engine_id": "CHAMBER_03",
                "cycle": 187,
                "op_setting_1": -0.0007,
                "op_setting_2": -0.0004,
                "op_setting_3": 100.0,
                "sensor_2": 641.82,
                "sensor_3": 1589.7,
                "sensor_4": 1400.6,
                "sensor_7": 553.9,
                "sensor_8": 2388.1,
                "sensor_9": 9045.2,
                "sensor_11": 47.1,
                "sensor_12": 521.7,
                "sensor_13": 2388.1,
                "sensor_14": 8137.6,
                "sensor_15": 8.4,
                "sensor_17": 391.0,
                "sensor_20": 38.8,
                "sensor_21": 23.4,
            }
        }


class RiskScoreResponse(BaseModel):
    engine_id: str
    cycle: int
    predicted_rul: float
    risk_tier: str
    risk_score: float          # 0-100, higher = more urgent
    confidence: str
    recommendation: str
    generated_at: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    api_version: str


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_artifacts():
    artifacts = {}
    try:
        scaler_path = PROCESSED_PATH / "scaler.pkl"
        feat_path = PROCESSED_PATH / "feature_cols.pkl"
        gb_path = MODEL_PATH / "gb_rul.pkl"

        if scaler_path.exists():
            artifacts["scaler"] = joblib.load(scaler_path)
        if feat_path.exists():
            artifacts["feature_cols"] = joblib.load(feat_path)
        if gb_path.exists():
            artifacts["gb_model"] = joblib.load(gb_path)
            logger.info("GradientBoosting model loaded")
    except Exception as e:
        logger.error(f"Artifact loading error: {e}")
    return artifacts


ARTIFACTS = load_artifacts()


# ── Risk Logic ────────────────────────────────────────────────────────────────

def rul_to_risk_tier(rul: float) -> tuple:
    """Convert RUL to risk tier + recommendation."""
    if rul <= 15:
        return (
            "critical",
            min(100, 100 - rul * 2),
            "IMMEDIATE ACTION: Schedule maintenance within 24 hours. Risk of unplanned failure is high.",
        )
    elif rul <= 30:
        return (
            "high",
            85 - (rul - 15) * 1.5,
            "URGENT: Plan maintenance within 1 week. Monitor sensor trends closely.",
        )
    elif rul <= 60:
        return (
            "medium",
            50 - (rul - 30) * 0.8,
            "MONITOR: Schedule preventive maintenance within 30 days.",
        )
    else:
        return (
            "healthy",
            max(5, 30 - (rul - 60) * 0.2),
            "NORMAL: Equipment operating within expected parameters.",
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "system": "FailSight",
        "description": "Semiconductor Equipment Failure Prediction",
        "version": "1.0.0",
        "endpoints": ["/predict", "/predict/batch", "/health", "/docs"],
    }


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = list(ARTIFACTS.keys())
    return HealthResponse(
        status="healthy" if "gb_model" in ARTIFACTS else "degraded",
        models_loaded=loaded,
        api_version="1.0.0",
    )


@app.post("/predict", response_model=RiskScoreResponse)
def predict_failure_risk(reading: SensorReading):
    """
    Predict equipment failure risk from current sensor reading.

    Returns:
    - predicted_rul: estimated cycles remaining before failure
    - risk_tier: critical / high / medium / healthy
    - risk_score: 0-100 urgency score
    - recommendation: actionable engineer guidance
    """
    try:
        # Build feature vector
        sensor_data = {
            "op_setting_1": reading.op_setting_1,
            "op_setting_2": reading.op_setting_2,
            "op_setting_3": reading.op_setting_3,
            "sensor_2": reading.sensor_2,
            "sensor_3": reading.sensor_3,
            "sensor_4": reading.sensor_4,
            "sensor_7": reading.sensor_7,
            "sensor_8": reading.sensor_8,
            "sensor_9": reading.sensor_9,
            "sensor_11": reading.sensor_11,
            "sensor_12": reading.sensor_12,
            "sensor_13": reading.sensor_13,
            "sensor_14": reading.sensor_14,
            "sensor_15": reading.sensor_15,
            "sensor_17": reading.sensor_17,
            "sensor_20": reading.sensor_20,
            "sensor_21": reading.sensor_21,
            "cycle_normalized": min(reading.cycle / 300.0, 1.0),
        }

        df_input = pd.DataFrame([sensor_data])

        # Scale if scaler available
        if "scaler" in ARTIFACTS and "feature_cols" in ARTIFACTS:
            feat_cols = [c for c in ARTIFACTS["feature_cols"] if c in df_input.columns]
            if feat_cols:
                df_input[feat_cols] = ARTIFACTS["scaler"].transform(df_input[feat_cols])

        # Predict RUL
        if "gb_model" in ARTIFACTS:
            feature_cols = ARTIFACTS.get("feature_cols", list(df_input.columns))
            available_cols = [c for c in feature_cols if c in df_input.columns]
            rul = float(ARTIFACTS["gb_model"].predict(df_input[available_cols])[0])
            rul = np.clip(rul, 0, 125)
            confidence = "model-based"
        else:
            # Heuristic fallback if model not loaded
            degradation_proxy = (reading.sensor_3 - 1580) / 30
            rul = float(np.clip(125 * (1 - degradation_proxy), 0, 125))
            confidence = "heuristic-fallback"

        risk_tier, risk_score, recommendation = rul_to_risk_tier(rul)

        return RiskScoreResponse(
            engine_id=reading.engine_id,
            cycle=reading.cycle,
            predicted_rul=round(rul, 1),
            risk_tier=risk_tier,
            risk_score=round(risk_score, 1),
            confidence=confidence,
            recommendation=recommendation,
            generated_at=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[RiskScoreResponse])
def predict_batch(readings: List[SensorReading]):
    """Batch prediction for multiple equipment units simultaneously."""
    if len(readings) > 100:
        raise HTTPException(status_code=400, detail="Max 100 readings per batch")
    return [predict_failure_risk(r) for r in readings]
