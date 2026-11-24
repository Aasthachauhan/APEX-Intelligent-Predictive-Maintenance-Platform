import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from ml.anomaly_detection.detector import fit, score
from ml.data.ai4i import TARGET, load_real, simulate
from ml.features.engineering import AI4IFeatures, build_preprocessor
from ml.training.train import metrics, run


def test_simulator_deterministic_and_imbalanced():
    a, b = simulate(3000, 7), simulate(3000, 7)
    pd.testing.assert_frame_equal(a, b)
    assert 0.01 < a[TARGET].mean() < 0.15


def test_metrics_hand_computed():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1, 0.2, 0.3])
    m = metrics(y, p, 0.5)  # pred=[1,0,1,0,0,0] -> tp1 fn1 fp1 tn3
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 3)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["specificity"] == 0.75


def test_feature_engineering_values():
    x = pd.DataFrame([dict(air_temp=300, process_temp=310, rpm=60, torque=2, tool_wear=10, type="L")])
    f = AI4IFeatures().transform(x).iloc[0]
    assert f.temp_diff == 10 and f.strain == 20 and f.power_w == pytest.approx(2 * 2 * np.pi)


def test_preprocessor_fit_on_train_only():
    df = simulate(2000, 3)
    tr, te = train_test_split(df, test_size=0.3, random_state=0)
    p = build_preprocessor().fit(tr.drop(columns=[TARGET]))
    sc = p.named_steps["ct"].named_transformers_["num"]
    assert sc.n_samples_seen_ == len(tr)  # scaler never saw test rows


def test_unseen_category_does_not_crash():
    df = simulate(2000, 3)
    p = build_preprocessor().fit(df)
    p.transform(df.head(2).assign(type="X"))


def test_training_rejects_insufficient_data(tmp_path):
    with pytest.raises(ValueError):
        run(simulate(100, 1), tmp_path, "t")
    with pytest.raises(ValueError):
        run(simulate(2000, 1).assign(machine_failure=0), tmp_path, "t")


def test_training_reproducible_and_beats_chance(trained, tmp_path):
    again = run(simulate(6000, 1), tmp_path, "SIMULATED-test")
    assert again["selected_test_metrics"]["pr_auc"] == pytest.approx(trained["selected_test_metrics"]["pr_auc"])
    assert trained["selected_test_metrics"]["pr_auc"] > 3 * trained["positive_rate"]
    assert len(trained["results"]) == 5


def test_real_loader_rejects_wrong_schema(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a,b\n1,2")
    with pytest.raises(ValueError, match="missing columns"):
        load_real(str(f))


def test_anomaly_detector_separates_extremes():
    df = simulate(4000, 5)
    normal = df[df.machine_failure == 0]
    m = fit(normal, "isolation_forest")
    weird = df.head(20).assign(torque=200.0, rpm=300, air_temp=340.0)
    assert score(m, weird).is_anomaly.mean() > 0.9
    assert score(m, normal.head(500)).is_anomaly.mean() < 0.1
