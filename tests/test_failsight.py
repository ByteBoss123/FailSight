import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest

from failsight import data as D
from failsight.drift import DriftMonitor, early_life, psi
from failsight.metrics import nasa_score, rmse, warning_metrics

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "results" / "metrics.json").read_text())


def test_raw_data_matches_checksums():
    for line in (ROOT / "data" / "raw" / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((ROOT / "data" / "raw" / name).read_bytes()).hexdigest() == digest


def test_official_sizes_and_labels():
    tr, te, rul = D.load("FD001")
    assert len(tr) == 20631 and len(te) == 13096 and len(rul) == 100
    lab = D.label_train(tr)
    assert lab.groupby("engine_id")["rul"].min().eq(0).all()  # every training engine runs to failure
    last = D.last_cycle(D.label_test(te, rul))
    assert np.array_equal(last["rul"].to_numpy(), np.minimum(rul, D.RUL_CAP))


def test_features_do_not_look_ahead():
    tr, _, _ = D.load("FD001")
    s = D.informative_sensors(tr)
    full = D.add_features(tr, s).set_index(["engine_id", "cycle"])
    cut = tr[tr["cycle"] <= 50]
    part = D.add_features(cut, s).set_index(["engine_id", "cycle"])
    cols = [c for c in D.feature_columns(s) if c != "cycle"]
    assert np.allclose(full.loc[part.index, cols].to_numpy(), part[cols].to_numpy())


def test_engine_split_has_no_overlap():
    tr, _, _ = D.load("FD001")
    a, b = D.engine_split(tr)
    assert not set(a.engine_id) & set(b.engine_id) and len(set(b.engine_id)) == 20


def test_metrics():
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    assert nasa_score([10], [20]) > nasa_score([20], [10])  # late is penalized more than early
    w = warning_metrics([5, 50, 10], [8, 20, 40], 30)
    assert (w["caught"], w["false_alarms"], w["missed"]) == (1, 1, 1)


def test_psi_detects_shift_but_not_resample():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.1
    assert psi(ref, rng.normal(1.5, 1, 5000)) > 0.25


def test_published_results():
    t = RES["test"]
    assert RES["selected_on_validation"] == "lstm_tensorflow"
    assert t["lstm_tensorflow"]["rmse"] < t["gradient_boosting_features"]["rmse"] < t["ridge_raw_sensors"]["rmse"]
    assert t["lstm_tensorflow"]["rmse"] < t["mean_baseline"]["rmse"] / 3
    d = RES["drift"]
    assert d["fd001_test_vs_train_whole_history_naive"]["alert"]   # the false alarm the healthy baseline fixes
    assert not d["fd001_test_vs_train"]["alert"] and d["fd002_test_vs_fd001_train"]["alert"]


def test_early_life_monitor_reproduces():
    tr, te, _ = D.load("FD001")
    s = D.informative_sensors(tr)
    mon = DriftMonitor(early_life(tr), s)
    assert not mon.check(early_life(te))["alert"]
    _, te2, _ = D.load("FD002")
    assert mon.check(early_life(te2))["alert"]


@pytest.fixture(scope="module")
def client():
    if not (ROOT / "models" / "failsight_lstm.keras").exists():
        pytest.skip("run scripts/train_eval.py first")
    from fastapi.testclient import TestClient
    from failsight.api import app
    return TestClient(app)


def test_api_predict_and_drift(client):
    _, te, rul = D.load("FD001")
    s = D.informative_sensors(D.load("FD001")[0])
    eng = te[te.engine_id == 1]
    r = client.post("/predict", json={"readings": eng[s].to_dict("records")})
    assert r.status_code == 200 and 0 <= r.json()["remaining_useful_life_cycles"] <= D.RUL_CAP
    t = time.perf_counter()
    for _ in range(20):
        client.post("/predict", json={"readings": eng[s].tail(30).to_dict("records")})
    assert (time.perf_counter() - t) / 20 < 1.0
    assert client.post("/predict", json={"readings": [{"s2": 1.0}]}).status_code == 422
    _, te2, _ = D.load("FD002")
    assert client.post("/drift", json={"readings": early_life(te2)[s].to_dict("records")}).json()["alert"]
