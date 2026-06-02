"""
DistilBERT Fine-tuning — FailSight Failure Type Classifier
Classifies maintenance log text into failure categories

Gap 3 fix: fine-tuning a transformer on domain-specific text
PyTorch backend — checks the DL framework requirement

Failure classes:
  0 - pressure_spike
  1 - temperature_drift
  2 - rf_power_instability
  3 - particle_event
  4 - mechanical_wear
  5 - general_maintenance
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple
import logging
import mlflow

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/classifier/")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

LABELS = [
    "pressure_spike",
    "temperature_drift",
    "rf_power_instability",
    "particle_event",
    "mechanical_wear",
    "general_maintenance",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
BASE_MODEL = "distilbert-base-uncased"


def create_training_data() -> List[Dict]:
    """
    Training examples built from realistic semiconductor
    fab maintenance log language. Each maps a text description
    to a failure category that a field engineer would assign.
    """
    examples = [
        # pressure_spike
        {"text": "Pressure excursion above setpoint during process. O-ring failure on gas inlet.", "label": "pressure_spike"},
        {"text": "Chamber pressure spiked to 3x setpoint. Throttle valve actuator stuck open.", "label": "pressure_spike"},
        {"text": "Sudden pressure rise in load lock. Gate valve seat worn, replaced assembly.", "label": "pressure_spike"},
        {"text": "Process pressure unstable, fluctuating ±15%. Pressure transducer calibration drift.", "label": "pressure_spike"},
        {"text": "Vacuum excursion in end station. Leak detected at conflat flange.", "label": "pressure_spike"},
        {"text": "Foreline pressure high. Roughing pump oil degraded. Oil change and pump service.", "label": "pressure_spike"},
        {"text": "Pressure interlock triggered during etch. MFC flow controller stuck at full open.", "label": "pressure_spike"},
        {"text": "Chamber base pressure could not reach spec. Quartz window cracked, replaced.", "label": "pressure_spike"},

        # temperature_drift
        {"text": "Heater zone 3 drifting 15 degrees above setpoint. Thermocouple replaced.", "label": "temperature_drift"},
        {"text": "Chuck temperature unstable. Coolant flow sensor degraded, calibration lost.", "label": "temperature_drift"},
        {"text": "Process temperature uniformity out of spec. Zone 2 heater element open circuit.", "label": "temperature_drift"},
        {"text": "Wafer temperature monitoring alarm. Pyrometer window coated, cleaning required.", "label": "temperature_drift"},
        {"text": "Thermal furnace setpoint overshoot. PID controller parameters drifted.", "label": "temperature_drift"},
        {"text": "Slurry temperature above target. Heat exchanger fouled, throughput reduced.", "label": "temperature_drift"},
        {"text": "Temperature controller oscillating. SSR relay degraded, replaced solid state relay.", "label": "temperature_drift"},

        # rf_power_instability
        {"text": "RF power fluctuating during deposition. Matching network capacitor failed.", "label": "rf_power_instability"},
        {"text": "Plasma instability during etch. ESC voltage drifting, capacitor bank replaced.", "label": "rf_power_instability"},
        {"text": "RF generator arc detected. Power supply output filter degraded.", "label": "rf_power_instability"},
        {"text": "Reflected power high on RF match. Coax cable connector corroded.", "label": "rf_power_instability"},
        {"text": "Plasma ignition failure. RF generator output below threshold, module replaced.", "label": "rf_power_instability"},
        {"text": "Inconsistent plasma density. RF match network auto-tuning range exceeded.", "label": "rf_power_instability"},

        # particle_event
        {"text": "High particle count on monitor wafer. Chamber wall coating delamination found.", "label": "particle_event"},
        {"text": "Scratch defects on product wafers. CMP pad conditioner worn, replaced.", "label": "particle_event"},
        {"text": "Adder defects above spec. Robot end effector contaminated, cleaned.", "label": "particle_event"},
        {"text": "Particle shower event. Shower head orifice plugged with process byproduct.", "label": "particle_event"},
        {"text": "Metal contamination detected. Sputtered material from worn shield set.", "label": "particle_event"},
        {"text": "Wafer backside particles. Lift pin tip broken, depositing particles.", "label": "particle_event"},

        # mechanical_wear
        {"text": "Wafer carrier membrane ruptured. Replaced carrier head assembly.", "label": "mechanical_wear"},
        {"text": "Beam current drop 40 percent. Ion source filament at end of life, replaced.", "label": "mechanical_wear"},
        {"text": "Scan motor encoder drift causing dose non-uniformity. Encoder replaced.", "label": "mechanical_wear"},
        {"text": "Wafer handler drop event. Robot blade warped from repeated thermal cycling.", "label": "mechanical_wear"},
        {"text": "Pump bearing noise detected. Preventive replacement before failure.", "label": "mechanical_wear"},
        {"text": "Throttle valve actuator not reaching position. Actuator motor brushes worn.", "label": "mechanical_wear"},
        {"text": "Load lock door seal leaking. Door O-ring compressed beyond spec, replaced.", "label": "mechanical_wear"},

        # general_maintenance
        {"text": "Scheduled 500-cycle PM completed. All parameters within spec.", "label": "general_maintenance"},
        {"text": "Quarterly preventive maintenance performed. No anomalies found.", "label": "general_maintenance"},
        {"text": "Annual calibration of all sensors completed. Certificates updated.", "label": "general_maintenance"},
        {"text": "PM performed per schedule. Chamber cleaned, seals inspected, process qualified.", "label": "general_maintenance"},
        {"text": "Routine maintenance: filters replaced, coolant flushed, alignment verified.", "label": "general_maintenance"},
        {"text": "Tool qualification after PM. Process results within control limits.", "label": "general_maintenance"},
    ]

    return examples


def prepare_datasets(examples: List[Dict]) -> Tuple:
    tokenizer = DistilBertTokenizer.from_pretrained(BASE_MODEL)

    texts = [ex["text"] for ex in examples]
    labels = [LABEL2ID[ex["label"]] for ex in examples]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(tokenize, batched=True)
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(tokenize, batched=True)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return train_ds, val_ds, tokenizer


def compute_metrics(pred: EvalPrediction) -> Dict:
    preds = np.argmax(pred.predictions, axis=-1)
    acc = (preds == pred.label_ids).mean()
    return {"accuracy": float(acc)}


def train_failure_classifier():
    """Fine-tune DistilBERT on fab maintenance log text."""
    logger.info("Fine-tuning DistilBERT for failure type classification...")

    examples = create_training_data()
    train_ds, val_ds, tokenizer = prepare_datasets(examples)

    model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=str(MODEL_PATH / "checkpoints"),
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=10,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        logging_steps=5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    mlflow.set_experiment("failsight_rul_prediction")
    with mlflow.start_run(run_name="DistilBERT_FailureClassifier"):
        mlflow.log_param("model_type", "DistilBERT-fine-tuned")
        mlflow.log_param("base_model", BASE_MODEL)
        mlflow.log_param("num_classes", len(LABELS))
        mlflow.log_param("classes", str(LABELS))
        mlflow.log_param("epochs", 8)
        mlflow.log_param("domain", "semiconductor_fab_maintenance")

        trainer.train()
        eval_result = trainer.evaluate()

        mlflow.log_metric("eval_accuracy", eval_result["eval_accuracy"])
        mlflow.log_metric("eval_loss", eval_result["eval_loss"])

        print(f"\nClassifier accuracy: {eval_result['eval_accuracy']:.1%}")

    model.save_pretrained(str(MODEL_PATH / "final"))
    tokenizer.save_pretrained(str(MODEL_PATH / "final"))
    with open(MODEL_PATH / "label_config.json", "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f)

    logger.info(f"Classifier saved to {MODEL_PATH / 'final'}")
    return model, tokenizer


def predict_failure_type(text: str, model=None, tokenizer=None) -> Dict:
    """Classify a maintenance log entry into failure type."""
    if model is None:
        model_dir = MODEL_PATH / "final"
        if not model_dir.exists():
            return {"label": "unknown", "confidence": 0.0}
        tokenizer = DistilBertTokenizer.from_pretrained(str(model_dir))
        model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs).item()

    return {
        "failure_type": ID2LABEL[pred_id],
        "confidence": round(probs[0][pred_id].item(), 4),
        "all_scores": {ID2LABEL[i]: round(probs[0][i].item(), 4) for i in range(len(LABELS))},
    }


if __name__ == "__main__":
    model, tokenizer = train_failure_classifier()

    print("\n--- Failure Type Classifier Demo ---")
    test_cases = [
        "Pressure excursion, O-ring failed on gas inlet manifold",
        "Heater zone temperature drifting above setpoint",
        "RF power fluctuating during plasma deposition",
        "High particle count found on monitor wafer",
        "Scheduled 500-cycle PM completed, all parameters within spec",
    ]

    for text in test_cases:
        result = predict_failure_type(text, model, tokenizer)
        print(f"\n{text[:55]}...")
        print(f"→ {result['failure_type'].upper()} ({result['confidence']:.1%} confidence)")
