"""Telemetry drift monitoring: Population Stability Index per sensor against a training reference.

Compared on healthy early-life cycles (see early_life). PSI < 0.1 stable, 0.1-0.25 moderate, > 0.25 major shift (common industry thresholds). An alert fires when
any monitored sensor exceeds 0.25, i.e. the model is being asked about telemetry unlike what it learned from.
"""
import numpy as np

ALERT_PSI = 0.25


def psi(reference, current, bins=10, eps=1e-4):
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant reference: compare the share that falls outside it
        ref_share = 1.0
        cur_share = float(np.mean(np.isclose(current, reference[0], atol=1e-6)))
        return float((ref_share - cur_share) * np.log((ref_share + eps) / (cur_share + eps)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference) + eps
    c = np.histogram(current, edges)[0] / len(current) + eps
    return float(np.sum((c - r) * np.log(c / r)))


HEALTHY_CYCLES = 30


def early_life(df, cycles=HEALTHY_CYCLES):
    """First `cycles` cycles of each engine: its healthy baseline. Degradation later in life is the signal the
    model predicts, so comparing whole histories would flag normal wear as drift."""
    return df[df.groupby("engine_id")["cycle"].rank(method="first") <= cycles]


class DriftMonitor:
    def __init__(self, reference_df, columns, alert=ALERT_PSI):
        self.columns, self.alert = list(columns), alert
        self.reference = {c: reference_df[c].to_numpy() for c in self.columns}

    def check(self, current_df):
        scores = {c: round(psi(self.reference[c], current_df[c].to_numpy()), 4) for c in self.columns}
        drifted = sorted((c for c, v in scores.items() if v > self.alert), key=lambda c: -scores[c])
        return {"alert": bool(drifted), "max_psi": max(scores.values()), "drifted_sensors": drifted, "psi": scores}
