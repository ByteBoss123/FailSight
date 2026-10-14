"""REST API: POST an engine's recent sensor readings, get remaining useful life and an early-warning flag.

    uvicorn failsight.api:app
    POST /predict {"readings": [{"s2": 642.1, "s3": 1589.7, ...}, ...]}   # oldest first, >= 1 cycle
    POST /drift   {"readings": [...]}   # a fleet's early-life readings: PSI per sensor vs training telemetry
"""
import os
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from failsight.drift import DriftMonitor

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
MODELS = Path(__file__).resolve().parents[1] / "models"
WARN_CYCLES = 30
app = FastAPI(title="FailSight")


class Readings(BaseModel):
    readings: list[dict[str, float]] = Field(min_length=1)


@lru_cache
def _model():
    import tensorflow as tf
    meta = json.loads((MODELS / "failsight_lstm_meta.json").read_text())
    meta["reference"] = pd.DataFrame(meta["reference"])
    return tf.keras.models.load_model(MODELS / "failsight_lstm.keras"), meta


def _frame(readings, sensors):
    df = pd.DataFrame(readings)
    missing = [s for s in sensors if s not in df]
    if missing:
        raise HTTPException(422, f"missing sensors: {missing}")
    if not np.isfinite(df[sensors].to_numpy()).all():
        raise HTTPException(422, "every reading must contain finite values for all required sensors")
    return df[sensors]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(body: Readings):
    model, meta = _model()
    v = ((_frame(body.readings, meta["sensors"]).to_numpy() - meta["mu"]) / meta["sd"]).astype("float32")
    seq = meta["seq"]
    v = v[-seq:] if len(v) >= seq else np.vstack([np.repeat(v[:1], seq - len(v), 0), v])
    rul = float(np.clip(model(v[None], training=False).numpy().ravel()[0], 0, meta["rul_cap"]))
    tier = "critical" if rul <= 15 else "high" if rul <= WARN_CYCLES else "medium" if rul <= 60 else "healthy"
    return {"remaining_useful_life_cycles": round(rul, 1), "failure_within_30_cycles": rul <= WARN_CYCLES,
            "risk_tier": tier, "cycles_used": min(len(body.readings), seq)}


@app.post("/drift")
def drift(body: Readings):
    _, meta = _model()
    mon = DriftMonitor(meta["reference"], meta["sensors"])
    out = mon.check(_frame(body.readings, meta["sensors"]))
    return {k: out[k] for k in ("alert", "max_psi", "drifted_sensors")}
