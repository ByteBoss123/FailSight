import numpy as np


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def nasa_score(y, p):
    """PHM08 asymmetric score: late predictions (overestimating remaining life) cost more than early ones."""
    d = np.asarray(p) - np.asarray(y)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def warning_metrics(y, p, horizon):
    """Early warning: does the model say "fails within `horizon` cycles" for the engines that do?"""
    y, p = np.asarray(y), np.asarray(p)
    t, f = y <= horizon, p <= horizon
    tp, fp, fn = int((t & f).sum()), int((~t & f).sum()), int((t & ~f).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"engines_failing_within_horizon": int(t.sum()), "caught": tp, "missed": fn, "false_alarms": fp,
            "precision": round(prec, 3), "recall": round(rec, 3)}
