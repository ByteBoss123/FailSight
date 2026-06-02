"""
Failure Prediction Models — FailSight
Gradient Boosting + Random Survival Forest + LSTM
Full MLflow tracking for all experiments

Predicts Remaining Useful Life (RUL) of semiconductor fab equipment
from multivariate sensor telemetry.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import warnings
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

PROCESSED_PATH = Path("data/processed/")
MODEL_PATH = Path("models/survival/")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "failsight_rul_prediction"
SEQUENCE_LENGTH = 30  # Look back 30 cycles to predict RUL


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Score function from PHM challenge — penalizes late predictions more
    diff = y_pred - y_true
    score = np.sum(
        np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)
    )
    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "PHM_Score": round(float(score), 2),
    }


# ── Gradient Boosting ─────────────────────────────────────────────────────────

def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """
    XGBoost-style Gradient Boosting for RUL regression.
    Fast, interpretable, strong baseline.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "min_samples_leaf": 10,
        "loss": "huber",  # Robust to outliers — important for sensor data
    }

    with mlflow.start_run(run_name="GradientBoosting_RUL"):
        mlflow.log_params(params)
        mlflow.log_param("model_type", "GradientBoostingRegressor")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("sequence_features", "rolling_mean_std_diff")

        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val).clip(0, 125)
        metrics = compute_metrics(y_val.values, preds)
        mlflow.log_metrics(metrics)

        # Feature importance — useful for explainability
        importance = pd.Series(
            model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)
        top_features = importance.head(10).to_dict()
        mlflow.log_dict(top_features, "top_10_features.json")

        # Save model
        model_path = MODEL_PATH / "gb_rul.pkl"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "gradient_boosting_rul")

        print(f"GradientBoosting → MAE: {metrics['MAE']} | RMSE: {metrics['RMSE']} | PHM: {metrics['PHM_Score']}")

    return {"model": model, "metrics": metrics, "name": "GradientBoosting", "preds": preds}


# ── LSTM ──────────────────────────────────────────────────────────────────────

class SensorSequenceDataset(Dataset):
    """
    PyTorch Dataset for LSTM sequence modeling.
    Each sample = last N cycles of sensor readings for one engine at one timestep.
    """
    def __init__(self, df: pd.DataFrame, feature_cols: list, seq_len: int = SEQUENCE_LENGTH):
        self.sequences = []
        self.targets = []

        for engine_id, group in df.groupby("engine_id"):
            group = group.sort_values("cycle")
            features = group[feature_cols].values
            targets = group["RUL"].values

            for i in range(seq_len, len(features)):
                self.sequences.append(features[i - seq_len : i])
                self.targets.append(targets[i])

        self.sequences = torch.FloatTensor(np.array(self.sequences))
        self.targets = torch.FloatTensor(np.array(self.targets))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMFailurePredictor(nn.Module):
    """
    LSTM network for remaining useful life prediction.
    Captures temporal degradation patterns that tabular models miss.
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Take last timestep
        return out.squeeze()


def train_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
    epochs: int = 30,
) -> dict:
    """
    Train LSTM on sensor sequences with MLflow tracking.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training LSTM on: {device}")

    train_ds = SensorSequenceDataset(train_df, feature_cols)
    val_ds = SensorSequenceDataset(val_df, feature_cols)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = LSTMFailurePredictor(input_size=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.HuberLoss()

    with mlflow.start_run(run_name="LSTM_RUL"):
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_param("hidden_size", 64)
        mlflow.log_param("num_layers", 2)
        mlflow.log_param("sequence_length", SEQUENCE_LENGTH)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("loss", "HuberLoss")

        best_val_loss = float("inf")

        for epoch in range(epochs):
            # Train
            model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # Validate
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    preds = model(X_batch).cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(y_batch.numpy())

            train_loss = np.mean(train_losses)
            val_preds_clipped = np.clip(val_preds, 0, 125)
            metrics = compute_metrics(np.array(val_targets), val_preds_clipped)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_MAE", metrics["MAE"], step=epoch)
            mlflow.log_metric("val_RMSE", metrics["RMSE"], step=epoch)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | MAE: {metrics['MAE']}")

            # Save best model
            val_loss = np.mean((np.array(val_targets) - val_preds_clipped) ** 2)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_PATH / "lstm_best.pt")

            scheduler.step()

        # Final metrics
        mlflow.log_metrics({f"final_{k}": v for k, v in metrics.items()})
        print(f"LSTM → MAE: {metrics['MAE']} | RMSE: {metrics['RMSE']} | PHM: {metrics['PHM_Score']}")

    return {"model": model, "metrics": metrics, "name": "LSTM", "preds": val_preds_clipped}


# ── Ensemble ──────────────────────────────────────────────────────────────────

def ensemble_predictions(gb_preds, lstm_preds, y_true, weights=(0.5, 0.5)) -> dict:
    """Weighted ensemble of GB + LSTM."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    min_len = min(len(gb_preds), len(lstm_preds), len(y_true))

    with mlflow.start_run(run_name="Ensemble_GB_LSTM"):
        mlflow.log_param("model_type", "Ensemble")
        mlflow.log_param("weight_gb", weights[0])
        mlflow.log_param("weight_lstm", weights[1])

        ensemble = weights[0] * gb_preds[:min_len] + weights[1] * lstm_preds[:min_len]
        ensemble = np.clip(ensemble, 0, 125)
        metrics = compute_metrics(np.array(y_true[:min_len]), ensemble)
        mlflow.log_metrics(metrics)

        print(f"Ensemble   → MAE: {metrics['MAE']} | RMSE: {metrics['RMSE']} | PHM: {metrics['PHM_Score']}")

    return {"preds": ensemble, "metrics": metrics, "name": "Ensemble"}


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_training_pipeline():
    """Run full FailSight training pipeline."""
    print("=" * 60)
    print("FAILSIGHT — EQUIPMENT FAILURE PREDICTION PIPELINE")
    print("=" * 60)

    train_path = PROCESSED_PATH / "train.csv"
    if not train_path.exists():
        print("Run data/pipeline.py first.")
        return

    df = pd.read_csv(train_path)
    feature_cols = joblib.load(PROCESSED_PATH / "feature_cols.pkl")
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Last observation per engine for tabular models
    latest = df.sort_values("cycle").groupby("engine_id").last().reset_index()
    X = latest[feature_cols]
    y = latest["RUL"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_df, val_df = (
        df[df["engine_id"].isin(X_train.index)],
        df[df["engine_id"].isin(X_val.index)],
    )

    print(f"\nEngines — Train: {len(X_train)} | Val: {len(X_val)}")
    print()

    # Train all models
    gb_result = train_gradient_boosting(X_train, y_train, X_val, y_val)
    lstm_result = train_lstm(df, df, feature_cols, epochs=30)

    # Align predictions for ensemble
    min_len = min(len(gb_result["preds"]), len(lstm_result["preds"]))
    ensemble_result = ensemble_predictions(
        gb_result["preds"][:min_len],
        lstm_result["preds"][:min_len],
        y_val.values[:min_len],
    )

    print()
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    for result in [gb_result, lstm_result, ensemble_result]:
        m = result["metrics"]
        print(f"{result['name']:20} | MAE: {m['MAE']:7.4f} | RMSE: {m['RMSE']:7.4f}")

    print(f"\nMLflow: run 'mlflow ui' → http://localhost:5000")


if __name__ == "__main__":
    run_training_pipeline()
