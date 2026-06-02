# FailSight — Semiconductor Equipment Failure Prediction System

> *"Unplanned downtime on a chip fab line costs $1M+ per hour.  
> FailSight predicts equipment failures before they happen."*

---

## The Problem

Semiconductor fabrication equipment — etch systems, CVD chambers, ion implanters —
generates thousands of sensor readings per second. When a machine fails unexpectedly,
the fab line stops. Engineers scramble. Chips are scrapped. Customers lose millions.

Most fabs react to failures. **FailSight predicts them.**

---

## What It Does

1. **Ingests real-time sensor telemetry** from fab equipment (temperature, pressure, voltage, vibration, RF power)
2. **Predicts time-to-failure** using survival analysis + ML on historical sensor patterns
3. **Classifies failure type** using a fine-tuned DistilBERT on maintenance logs + error codes
4. **Answers engineer questions** via a RAG agent: *"What failed last time Chamber 3 showed this pressure spike?"*
5. **Monitors model drift** so predictions stay accurate as equipment ages
6. **Serves risk scores** via REST API for integration with fab MES systems

---

## Architecture

```
Sensor Telemetry (NASA CMAPSS Dataset — turbofan degradation)
         ↓
Data Pipeline (Python, Pandas, NumPy)
         ↓
Feature Engineering (rolling stats, degradation signals, RUL labels)
         ↓
┌─────────────────────────────────────────────────────┐
│  Failure Prediction Models                          │
│  ├── Random Survival Forest (time-to-failure)       │
│  ├── LSTM (sequence-based RUL prediction)           │
│  └── Gradient Boosting (binary failure classifier)  │
└─────────────────────────────────────────────────────┘
         ↓
MLflow (experiment tracking + model versioning)
         ↓
FastAPI (real-time risk score endpoint)
         ↓
┌─────────────────────────────────────────────────────┐
│  FailSight Agent                                    │
│  ├── FAISS vector store (maintenance logs)          │
│  ├── LangChain RAG chain                            │
│  └── Fine-tuned DistilBERT (failure classifier)     │
└─────────────────────────────────────────────────────┘
         ↓
Evidently AI (sensor drift + model performance monitoring)
         ↓
Streamlit Dashboard (unified engineer interface)
```

---

## Dataset

**NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)**
- Real degradation dataset used in predictive maintenance research
- 100+ engines with multivariate sensor readings over time
- Ground truth: Remaining Useful Life (RUL) per engine
- Download: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

*Why this dataset?* It's the industry standard benchmark for predictive maintenance ML.
Semiconductor fab equipment degrades the same way — gradual sensor drift until failure.
Using this dataset shows domain understanding, not just tutorial-following.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Survival Analysis | scikit-survival, lifelines |
| ML Models | scikit-learn, XGBoost, PyTorch (LSTM) |
| MLOps | MLflow, FastAPI, Uvicorn |
| GenAI / RAG | LangChain, FAISS, sentence-transformers |
| Fine-tuning | HuggingFace Transformers, DistilBERT, PyTorch |
| Monitoring | Evidently AI |
| Dashboard | Streamlit, Plotly |

---

## Project Structure

```
failsight/
├── data/
│   ├── raw/                    # NASA CMAPSS dataset
│   ├── processed/              # Engineered features + RUL labels
│   └── embeddings/             # FAISS maintenance log index
├── models/
│   ├── survival/               # RSF + GBM + LSTM artifacts
│   └── classifier/             # Fine-tuned DistilBERT
├── api/
│   └── main.py                 # FastAPI risk score endpoint
├── agent/
│   ├── rag_pipeline.py         # LangChain RAG over maintenance logs
│   └── failsight_agent.py      # Agentic diagnostic workflow
├── monitoring/
│   └── drift_monitor.py        # Evidently AI sensor + model drift
├── dashboard/
│   └── app.py                  # Streamlit engineer interface
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_rul_prediction.ipynb
│   └── 03_finetuning.ipynb
├── requirements.txt
├── HOW_TO_RUN.txt
└── README.md
```

---

## Key Results (update after training)

| Model | RMSE (RUL) | MAE (RUL) | Notes |
|---|---|---|---|
| Gradient Boosting | TBD | TBD | Fast, interpretable |
| Random Survival Forest | TBD | TBD | Handles censored data |
| LSTM | TBD | TBD | Best on sequence patterns |
| Ensemble | TBD | TBD | Production model |

---

## Live Demo
- API: [Render URL]
- Dashboard: [Streamlit Cloud URL]
- MLflow Screenshots: /assets/

---

## Interview Story (2 minutes)

*"Semiconductor fab equipment like etch chambers and CVD systems
degrade over time through thousands of hours of operation.
When they fail unexpectedly, fab lines stop — at $1M+ per hour.

I built FailSight, a predictive maintenance system that ingests
sensor telemetry, predicts remaining useful life using survival
analysis and gradient boosting tracked with MLflow, and serves
real-time risk scores via a FastAPI endpoint.

On top of that, I built a RAG agent that lets field engineers
ask natural language questions over historical maintenance logs —
things like 'what failed last time Chamber 3 showed this
pressure pattern?' — using LangChain and FAISS embeddings.

The failure type classifier uses a fine-tuned DistilBERT on
maintenance text, and Evidently AI monitors for sensor drift
so the model stays accurate as equipment ages.

The whole system is deployed and publicly accessible."*

---

**Author:** Amarnath | MS Data Analytics Engineering, George Mason University
