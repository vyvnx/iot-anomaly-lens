"""Testes unitários com dados SINTÉTICOS (correção do software, não desempenho no CICIoT2023)."""

import numpy as np
import pandas as pd
import pytest

from tc2_iot.calibration import calibrate, predict
from tc2_iot.config import CATEGORIES, ConfigError
from tc2_iot.data.acquire import looks_like_html, safe_extract_zip, sanitize_url
from tc2_iot.data.inspect import check_headers, mapping_problems
from tc2_iot.data.preprocess import Preprocessor
from tc2_iot.data.split import assign_partitions, resolve_effective_conflicts
from tc2_iot.evaluation import binary_metrics, category_metrics
from tc2_iot.models.autoencoder import Autoencoder
from tc2_iot.models.isolation_forest import IsolationForestModel
from tc2_iot.models.sgd_ocsvm import SGDOneClassSVMModel

from conftest import make_cfg

RNG = np.random.default_rng(0)
NORMAL = RNG.standard_normal((600, 10))
OUTLIERS = RNG.standard_normal((20, 10)) + 8


# ---------------------------------------------------------------- labels, schema, config

def test_unknown_label_and_absent_category_detected():
    mapping = {"status": "confirmed", "labels": {"Benign": {"benign": True, "category": None},
                                                 "DDoS-X": {"benign": False, "category": "DDoS"}}}
    p = mapping_problems(mapping, ["Benign", "DDoS-X", "Novo-Rotulo"])
    assert p["unknown_labels"] == ["Novo-Rotulo"]
    assert set(p["absent_categories"]) == set(CATEGORIES) - {"DDoS"}


def test_duplicate_column_names_rejected_even_after_normalization():
    with pytest.raises(ValueError, match="normalização"):
        check_headers({"a.csv": ["Tot sum", "tot_sum", "label"]})


def test_synthetic_config_must_stay_in_tests_folder(tmp_path):
    with pytest.raises(ConfigError):
        make_cfg(tmp_path, output_dir=str(tmp_path / "runs"))


def test_confirmatory_cannot_be_enabled_by_config(tmp_path):
    with pytest.raises(ConfigError):
        make_cfg(tmp_path, experiment={"confirmatory_evaluation": True})


# ---------------------------------------------------------------- acquisition helpers

def test_html_saved_as_csv_detected(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("<!DOCTYPE html><html><form>login</form></html>")
    assert looks_like_html(f)
    g = tmp_path / "y.csv"
    g.write_text("a,b,label\n1,2,Benign\n")
    assert not looks_like_html(g)


def test_url_credentials_removed():
    assert sanitize_url("https://user:pw@host.org/a/b.csv?token=SECRET#x") == "https://host.org/a/b.csv"


def test_zip_slip_blocked(tmp_path):
    import zipfile
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../fora.csv", "a")
    (tmp_path / "dest").mkdir()
    with pytest.raises(ValueError, match="sai da pasta"):
        safe_extract_zip(z, tmp_path / "dest")


# ---------------------------------------------------------------- splits

def _sample(n_benign=200, n_attack=40):
    rows = [{"vector_id": f"b{i:04d}", "record_id": f"f#{i}", "label": "B", "benign": True, "category": None,
             "stratum": "Benigno"} for i in range(n_benign)]
    rows += [{"vector_id": f"a{i:04d}", "record_id": f"g#{i}", "label": f"A{i % 3}", "benign": False,
              "category": "DDoS", "stratum": "DDoS"} for i in range(n_attack)]
    return pd.DataFrame(rows)


def test_systematic_split_balances_stratum_and_labels(tmp_path):
    p = assign_partitions(_sample(n_attack=41), make_cfg(tmp_path))
    att = p[~p["benign"]]
    ti, tc = (att["partition"] == "test_initial").sum(), (att["partition"] == "test_confirmatory").sum()
    assert ti + tc == 41 and abs(ti - tc) <= 1
    per_label = att.groupby("label")["partition"].apply(lambda x: abs((x == "test_initial").sum() - (x == "test_confirmatory").sum()))
    assert (per_label <= 1).all()
    assert p.loc[p["benign"], "partition"].value_counts()["train"] == 120


def test_partitions_independent_of_row_order(tmp_path):
    cfg = make_cfg(tmp_path)
    s = _sample()
    a = assign_partitions(s, cfg).set_index("vector_id")["partition"]
    b = assign_partitions(s.sample(frac=1, random_state=3), cfg).set_index("vector_id")["partition"]
    assert a.sort_index().equals(b.sort_index())


def test_fit_and_calibration_partitions_are_benign_only(tmp_path):
    p = assign_partitions(_sample(), make_cfg(tmp_path))
    assert p.loc[p["partition"].isin(["train", "validation_fit", "validation_calibration"]), "benign"].all()
    assert set(p.loc[~p["benign"], "partition"]) <= {"test_initial", "test_confirmatory"}


def test_effective_space_rule_same_class_keeps_best_partition():
    df = pd.DataFrame({"vector_id": ["v1", "v2", "v3"], "record_id": ["r1", "r2", "r3"], "label": ["B", "B", "A"],
                       "stratum": ["Benigno", "Benigno", "DDoS"], "benign": [True, True, False],
                       "partition": ["test_initial", "train", "test_confirmatory"]})
    Z = np.array([[1.0, 2.0], [1.0, 2.0], [5.0, 5.0]], dtype=np.float32)  # v1 == v2 after transformation
    kept, removed = resolve_effective_conflicts(df, Z)
    assert list(kept["vector_id"]) == ["v2", "v3"]
    assert list(removed["vector_id"]) == ["v1"] and removed["motivo"].iloc[0].startswith("colisao_numerica")


def test_effective_space_rule_binary_ambiguity_removes_all_members():
    df = pd.DataFrame({"vector_id": ["v1", "v2", "v3"], "record_id": ["r1", "r2", "r3"], "label": ["B", "R", "B"],
                       "stratum": ["Benigno", "Recon", "Benigno"], "benign": [True, False, True],
                       "partition": ["train", "test_confirmatory", "train"]})
    Z = np.array([[1.0, 2.0], [1.0, 2.0], [5.0, 5.0]], dtype=np.float32)  # benign (train) == attack (test)
    kept, removed = resolve_effective_conflicts(df, Z)
    assert list(kept["vector_id"]) == ["v3"]
    assert set(removed["vector_id"]) == {"v1", "v2"} and (removed["motivo"] == "ambiguidade_binaria_apos_transformacao").all()


def test_train_change_refits_preprocessing(tmp_path):
    """A train benign identical to a test attack after dropping a train-constant column: both leave, refit."""
    from tc2_iot.protocol import split_and_preprocess
    cfg = make_cfg(tmp_path)
    rng = np.random.default_rng(1)
    b = _sample(n_benign=200, n_attack=0)
    for f in ("x1", "x2"):
        b[f] = rng.standard_normal(len(b))
    b["c"] = 0.0  # constant in train
    donor = assign_partitions(b, cfg).query("partition == 'train'").iloc[0]
    atk = pd.DataFrame([{"vector_id": "zz", "record_id": "z#0", "label": "R", "benign": False, "category": "Recon",
                         "stratum": "Recon", "x1": donor["x1"], "x2": donor["x2"], "c": 5.0}])
    parts, pre, Z, removed, fits = split_and_preprocess(pd.concat([b, atk], ignore_index=True), cfg, ["x1", "x2", "c"])
    assert set(removed["vector_id"]) == {donor["vector_id"], "zz"} and fits == 2
    assert pre.scaler.n_samples_seen_ == 119 and "c" in pre.dropped


# ---------------------------------------------------------------- preprocessing

def test_scaler_ignores_test_only_values():
    X = RNG.standard_normal((100, 4))
    X[:, 2] = 3.0  # constant in train
    train = np.arange(100) < 60
    a = Preprocessor(list("abcd")).fit(X[train])
    X2 = X.copy()
    X2[~train] = 1e6  # change only non-train rows
    b = Preprocessor(list("abcd")).fit(X2[train])
    assert a.dropped == b.dropped == ["c"]
    assert np.array_equal(a.scaler.mean_, b.scaler.mean_) and np.array_equal(a.scaler.scale_, b.scaler.scale_)


def test_float32_overflow_detected():
    p = Preprocessor(["a"]).fit(np.array([[0.0], [1.0]]))
    with pytest.raises(OverflowError):
        p.transform(np.array([[1e300]]))


# ---------------------------------------------------------------- models: score orientation and persistence

MODEL_PARAMS = {
    "if": lambda: IsolationForestModel({"n_estimators": 50, "max_samples": 256, "max_features": 1.0,
                                        "bootstrap": False, "contamination": "auto"}, 1, 1, len(NORMAL)),
    "sgd": lambda: SGDOneClassSVMModel({"variant": "nystroem_rbf", "components": 64, "gamma": "inverse_feature_count",
                                        "nu": 0.01, "max_iter": 200, "tol": 1e-4, "average": True, "shuffle": True},
                                       1, len(NORMAL), 10),
    "ae": lambda: Autoencoder({"hidden_dimensions": [32, 8, 32], "batch_size": 64, "max_epochs": 30, "learning_rate": 1e-2,
                               "weight_decay": 0.0, "patience": 5, "min_delta": 1e-5}, 1, 10),
}


@pytest.mark.parametrize("name", list(MODEL_PARAMS))
def test_higher_score_means_more_anomalous_and_save_load(name, tmp_path):
    m = MODEL_PARAMS[name]().fit(NORMAL.astype(np.float32), NORMAL[:100].astype(np.float32))
    s_norm = m.anomaly_score(NORMAL[:100].astype(np.float32))
    s_out = m.anomaly_score(OUTLIERS.astype(np.float32))
    assert np.median(s_out) > np.median(s_norm)
    m.save(tmp_path)
    again = type(m).load(tmp_path).anomaly_score(OUTLIERS.astype(np.float32))
    np.testing.assert_allclose(again, s_out, rtol=1e-6, atol=1e-8)


def test_autoencoder_refuses_no_bottleneck():
    with pytest.raises(ValueError, match="gargalo"):
        Autoencoder({"hidden_dimensions": [32, 8, 32], "batch_size": 8, "max_epochs": 1, "learning_rate": 1e-3,
                     "weight_decay": 0.0, "patience": 1, "min_delta": 0}, 1, 8)


def test_autoencoder_restores_best_checkpoint():
    m = MODEL_PARAMS["ae"]().fit(NORMAL.astype(np.float32), NORMAL[:100].astype(np.float32))
    best = min(h["validation_fit_mse"] for h in m.history)
    now = float(m.anomaly_score(NORMAL[:100].astype(np.float32)).mean())
    assert now == pytest.approx(best, rel=1e-5)


# ---------------------------------------------------------------- calibration

def test_calibration_ties_not_broken():
    s = np.array([0.0] * 95 + [1.0] * 5)  # quantile 0.99 falls on a tie block
    c = calibrate(s, 0.01, min_warning=1000)
    assert c["threshold"] == 1.0 and c["ties_at_threshold"] == 5 and c["calibration_fpr"] == 0.0
    assert c["warnings"]  # fewer than 1000 records


def test_calibration_exact_quantile():
    s = np.arange(1000, dtype=float)
    c = calibrate(s, 0.01, min_warning=1000)
    assert c["threshold"] == 990.0 and c["calibration_fp"] == 9 and not c["warnings"]
    assert list(predict(np.array([990.0, 990.5]), c["threshold"])) == [0, 1]


# ---------------------------------------------------------------- metrics

def test_metrics_manual_case():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
    score = np.array([.1, .2, .3, .9, .8, .7, .95, .6, .15, .25])
    m = binary_metrics(y, score, pred)
    assert (m["TN"], m["FP"], m["FN"], m["TP"]) == (3, 1, 2, 4)
    assert m["recall"] == pytest.approx(4 / 6) and m["fpr"] == pytest.approx(1 / 4)
    assert m["precision"] == pytest.approx(4 / 5) and m["f1"] == pytest.approx(8 / 11)
    assert m["balanced_accuracy"] == pytest.approx((4 / 6 + 3 / 4) / 2)
    assert m["auroc"] == pytest.approx(16 / 24)  # attack>benign pairs counted by hand


def test_metrics_null_denominators():
    m = binary_metrics(np.array([1, 1]), np.array([.1, .2]), np.array([0, 0]))
    assert m["fpr"] is None and m["precision"] is None and m["auroc"] is None
    assert "sem benignos" in m["undefined_reasons"]


def test_category_metrics_absent_category_is_undefined():
    df = pd.DataFrame({"benign": [False, False, True, False], "category": ["DDoS", "DDoS", None, None],
                       "stratum": ["DDoS", "DDoS", "Benigno", "attack_category_ambiguous"],
                       "label": ["d1", "d1", "b", "d1+x1"], "pred": [1, 0, 0, 1]})
    rows = {r["grupo"]: r for r in category_metrics(df)}
    assert rows["DDoS"]["recall"] == 0.5 and rows["Mirai"]["recall"] is None
    assert rows["attack_category_ambiguous"]["recall"] == 1.0
    macro = [r for r in category_metrics(df) if r["nivel"] == "macro_categorias"][0]
    assert macro["recall"] is None and macro["N"] == 2  # ambiguous vectors are outside the 7-category macro
