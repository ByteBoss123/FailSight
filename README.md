# FailSight: Equipment Failure Prediction from Sensor Telemetry

Predicts how many operating cycles a machine has left (remaining useful life, RUL) from its sensor telemetry,
raises an early warning when failure is within 30 cycles, monitors incoming telemetry for drift, and serves
predictions through a REST API. Trained and scored on NASA's C-MAPSS turbofan degradation dataset with the
official test protocol.

## Results at a glance

| Area | Result | Evidence |
|---|---|---|
| RUL prediction, official test set (100 engines) | TensorFlow LSTM RMSE **12.8** cycles vs 41.9 for a mean baseline; chosen on held-out training engines, not on the test set | `results/metrics.json` |
| Early warning (fails within 30 cycles) | Caught 21 of 25 engines, 1 false alarm (precision 0.955, recall 0.84) | `results/metrics.json` |
| Drift monitoring | No alert on same-condition telemetry (max PSI 0.04); alert on a 6-condition fleet (FD002, max PSI 3.6), where the saved LSTM's RMSE rises to 33.88 and it catches 0 of 61 imminent failures | `results/metrics.json` |
| Serving | FastAPI `/predict` and `/drift`; p50 133 ms per prediction on CPU (in-process test client) | `failsight/api.py` |
| Tests | 9 pytest tests: data checksums, label alignment with NASA's RUL file, no look-ahead in features, engine-level split, metrics, drift, API | `tests/` |

## Data

NASA C-MAPSS (Saxena et al., PHM 2008), committed in `data/raw/` with SHA-256 checksums (two independent public
mirrors were byte-identical). FD001: 100 training engines run to failure (20,631 cycles), 100 test engines cut
off before failure (13,096 cycles) with NASA's true RUL. FD002 (six operating conditions) is used only as an
unseen, shifted fleet for drift monitoring.

## Method

- **Labels.** RUL capped at 125 cycles (standard piecewise-linear target; early life counts as healthy).
- **Features** (`failsight/data.py`): the 15 sensors that vary (6 are flat in FD001), plus per-engine rolling
  mean and standard deviation over 5 and 15 cycles and a 15-cycle slope, computed from past cycles only.
- **Models** (`scripts/train_eval.py`): mean baseline, ridge regression, gradient boosting on 91 engineered
  features, and a two-layer TensorFlow LSTM on 30-cycle sensor windows.
- **Protocol.** 20% of training engines held out *by engine* for model selection; the selected setup is refit on
  all training engines and scored once on the official test set (one prediction per engine, at its last cycle).

| Model | Validation RMSE | Test RMSE | NASA score (lower is better) | Warnings caught / false alarms |
|---|---|---|---|---|
| Mean baseline | 41.5 | 41.9 | 33,355 | 0 / 0 |
| Ridge, raw sensors | 22.6 | 21.3 | 1,188 | 10 / 0 |
| Gradient boosting, features | 17.6 | 15.0 | 373 | 18 / 1 |
| **TensorFlow LSTM (selected)** | **14.4** | **12.8** | **285** | **21 / 1** |
| GB + LSTM average | – | 12.8 | 275 | 18 / 1 |

The average of the two models scores slightly better on RMSE but catches fewer imminent failures, so the LSTM alone
is served.

## Drift monitoring

`failsight/drift.py` computes the Population Stability Index (PSI) per sensor against training telemetry and alerts
above 0.25.

The first version compared whole engine histories and **raised a false alarm on FD001's own test set** (max PSI 0.35):
test engines are cut off earlier in life, so they look less worn than training engines that ran to failure. Wear is
what the model predicts, not drift. The monitor now compares each engine's first 30 cycles (its healthy baseline):

| Fleet | Max PSI | Alert | Saved LSTM on that fleet |
|---|---|---|---|
| FD001 test (same conditions) | 0.04 | no | RMSE 12.8 |
| FD002 test (6 operating conditions) | 3.62 | yes, 15 sensors | RMSE 33.88, 0 of 61 imminent failures caught |

On these two benchmark fleets, the alert accompanies a substantial degradation in prediction quality. This does not prove that the monitor detects every future model failure.

The original `metrics.json` shifted-fleet score (RMSE 35.51) is for gradient boosting. The saved LSTM is now evaluated separately in [saved_lstm_evaluation.json](results/saved_lstm_evaluation.json).

## Run

```bash
pip install -r requirements.txt
python scripts/train_eval.py        # ~2 min on CPU; writes results/metrics.json and models/
pytest -q tests
uvicorn failsight.api:app
```

## Limitations

- Turbofan engines, not semiconductor tools: the method (telemetry windows, RUL, early warning, drift) transfers,
  the sensors do not.
- One fault mode (FD001). Multi-condition fleets (FD002/FD004) need per-condition normalization and retraining;
  the drift monitor is what tells you so.
- Single training run per model (seed 0); seed-to-seed variance was not measured.

This repo previously held an unfinished version (results marked "TBD", synthetic-data fallback). It was rebuilt
on the real dataset; the unfinished RAG and DistilBERT components were removed rather than shipped untested.

## Decision supported and reproducibility review

**Decision:** which equipment should a maintenance planner review first, and when should a model be held back pending validation on changed operating conditions?
This is a personal project on NASA's simulated turbofan degradation benchmark, not factory deployment or measured downtime savings.
The FD002 result is a reason to require validation/retraining before using this model on a different fleet, not a universal maintenance claim.

October 14 review: the committed LSTM reproduces FD001 RMSE 12.8, 21/25 imminent failures caught, and one false alarm.
Nine tests pass, including API execution, partial-sensor rejection, labels, timing checks, and drift.
Model metadata now loads from portable JSON instead of a pandas-version-dependent pickle; the legacy joblib file remains for provenance.

```bash
python scripts/evaluate_saved.py  # original saved LSTM, no retraining; writes both fleet scores
python -m pytest -q
```

This review did not retrain the LSTM or measure seed-to-seed variability. Sensor incompleteness/nonfinite values are rejected with HTTP 422.
