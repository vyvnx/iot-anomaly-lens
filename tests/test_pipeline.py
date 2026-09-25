"""Fluxo completo com fixtures SINTÉTICAS (data_kind=synthetic)."""

import io
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tc2_iot.config import MODELS
from tc2_iot.data.clean import DataBlocked
from tc2_iot.protocol import (IntegrityError, calibrate, evaluate, recompute_tables, train, verify_data, verify_run)
from tc2_iot.provenance import exposures
from tc2_iot.reporting import report

from conftest import build_data


def test_conflict_policy_v2(pipeline):
    """A excluded with identifiers kept; B kept as ambiguous attacks; C kept in the category stratum."""
    dq = json.loads((pipeline["pdir"] / "data_quality.json").read_text())
    sit = dq["duplicates"]["by_situation"]
    assert sit["A"] == {"groups": 1, "rows": 2} and sit["B"] == {"groups": 6, "rows": 12} and sit["C"] == {"groups": 6, "rows": 12}
    assert dq["policy_version"] == 2 and dq["duplicates"]["hash_collisions"] == 0
    assert dq["exclusions"]["by_reason_overlapping"] == {"label_missing": 0, "missing": 1, "non_numeric": 1, "non_finite": 1}
    excl = pd.read_parquet(pipeline["pdir"] / "excluded_binary_ambiguous_rows.parquet")
    assert sorted(excl["label"]) == ["SYN_BenignTraffic", "SYN_DoS-A"] and excl["group_id"].nunique() == 1
    parts = pd.read_parquet(pipeline["pdir"] / "partitions.parquet")
    b = parts[parts["situation"] == "B"]
    assert len(b) == 6 and (b["stratum"] == "attack_category_ambiguous").all() and b["category"].isna().all()
    assert set(b["label"]) == {"SYN_DDoS-A+SYN_DoS-A"} and not b["benign"].any()
    c = parts[parts["situation"] == "C"]
    assert (c["stratum"] == "DDoS").all() and (c["category"] == "DDoS").all()
    assert set(b["partition"]) <= {"test_initial", "test_confirmatory"}
    assert abs((b["partition"] == "test_initial").sum() - (b["partition"] == "test_confirmatory").sum()) <= 1


def test_ambiguous_stratum_outside_category_macro(pipeline):
    cm = pd.read_csv(pipeline["pdir"] / "category_metrics.csv")
    amb = cm[cm["nivel"] == "categoria_ambigua"]
    assert len(amb) == 3 and (amb["N"] > 0).all()  # one per model
    macro = cm[cm["nivel"] == "macro_categorias"].iloc[0]
    assert macro["N"] == cm[(cm["nivel"] == "categoria") & (cm["model"] == macro["model"])]["N"].sum()


def test_sample_independent_of_file_layout(tmp_path):
    """Same rows split into 2 or 3 differently named files -> same sampled vectors and partitions."""
    a = build_data(tmp_path / "a", n_files=3)
    b = build_data(tmp_path / "b", n_files=2)
    sa = pd.read_parquet(Path(a["data"]["processed_dir"]) / "sample.parquet")
    sb = pd.read_parquet(Path(b["data"]["processed_dir"]) / "sample.parquet")
    assert set(sa["vector_id"]) == set(sb["vector_id"])


def test_full_flow_artifacts_and_verification(pipeline):
    pdir = pipeline["pdir"]
    for f in ["protocol.json", "partitions.parquet", "preprocessor.joblib", "thresholds.json", "models_manifest.json",
              "metrics.csv", "category_metrics.csv", "timings.csv", "ae_training_history.csv", "sample_counts.csv"]:
        assert (pdir / f).exists(), f
    res = verify_run(pipeline["runs"], pipeline["pid"])
    assert res["ok"], res["failed"]
    m = pd.read_csv(pdir / "metrics.csv")
    assert set(m["model"]) == set(MODELS) and set(m["split"]) == {"test_initial"}
    assert json.loads((pdir / "protocol.json").read_text())["data_kind"] == "synthetic"


def test_all_models_share_records_and_order(pipeline):
    preds = [pd.read_parquet(f) for f in sorted((pipeline["pdir"] / "predictions" / "test_initial").glob("*.parquet"))]
    assert len(preds) == 3
    assert all(p["record_id"].tolist() == preds[0]["record_id"].tolist() for p in preds)


def test_metrics_recomposed_from_predictions(pipeline):
    before = pd.read_csv(pipeline["pdir"] / "metrics.csv")
    after = recompute_tables(pipeline["pdir"])["metrics"]
    pd.testing.assert_frame_equal(before, pd.read_csv(io.StringIO(after.to_csv(index=False))))


def test_threshold_independent_of_test_data(clone):
    thr = json.loads((clone["pdir"] / "thresholds.json").read_text())
    f = next((clone["pdir"] / "predictions" / "test_initial").glob("isolation_forest*.parquet"))
    df = pd.read_parquet(f)
    df["y_true"] = 1 - df["y_true"]
    df["score"] = df["score"] * 10
    df.to_parquet(f)
    recompute_tables(clone["pdir"])
    assert json.loads((clone["pdir"] / "thresholds.json").read_text()) == thr


def test_altered_frozen_artifact_invalidates(clone):
    p = clone["pdir"] / "partitions.parquet"
    df = pd.read_parquet(p)
    df.loc[0, "partition"] = "train" if df.loc[0, "partition"] != "train" else "test_initial"
    df.to_parquet(p, index=False)
    with pytest.raises(IntegrityError):
        evaluate(clone["runs"], clone["pid"], "test_initial")


def test_altered_raw_manifest_invalidates(clone):
    p = clone["pdir"] / "raw_manifest.json"
    m = json.loads(p.read_text())
    m["files"][0]["sha256"] = "0" * 64
    p.write_text(json.dumps(m))
    with pytest.raises(IntegrityError):
        train(clone["runs"], clone["pid"], MODELS, [42])


def test_resume_does_not_reuse_stale_model(clone):
    d = clone["pdir"] / "models" / "autoencoder" / "seed_42"
    (d / "state_dict.pt").write_bytes(b"corrompido")
    with pytest.raises(IntegrityError):  # calibrated entries are immutable: tampering is an error, not a silent retrain
        train(clone["runs"], clone["pid"], ["autoencoder"], [42])
    (clone["pdir"] / "models_manifest.json").unlink()
    (clone["pdir"] / "thresholds.json").unlink()
    train(clone["runs"], clone["pid"], ["autoencoder"], [42], log=lambda *a: None)  # stale artifact retrained
    fit = json.loads((d / "fit.json").read_text())
    assert all(clone["pdir"].joinpath(f).read_bytes() != b"corrompido" for f in fit["files_sha256"])
    calibrate(clone["runs"], clone["pid"], log=lambda *a: None)


def test_confirmatory_requires_release_and_is_logged(clone):
    with pytest.raises(PermissionError):
        evaluate(clone["runs"], clone["pid"], "test_confirmatory")
    assert not list((clone["pdir"] / "predictions").glob("test_confirmatory/*"))
    evaluate(clone["runs"], clone["pid"], "test_confirmatory", release_confirmatory=True, log=lambda *a: None)
    ex = exposures(clone["runs"])
    assert any(e["split"] == "test_confirmatory" for e in ex)
    assert verify_run(clone["runs"], clone["pid"])["confirmatory_not_evaluated_without_release"]


def test_run_initial_never_touches_confirmatory(tmp_path):
    from tc2_iot.cli import main
    cfg = build_data(tmp_path)
    main(["run-initial", "--config", cfg["_config_path"]])
    runs = Path(cfg["output_dir"])
    pdir = next(runs.glob("synthetic-*"))
    assert not list(pdir.glob("predictions/test_confirmatory/*"))
    assert {e["split"] for e in exposures(runs)} == {"test_initial"}
    md = (pdir / "results_initial.md").read_text()
    assert md.startswith("# ⚠ DADOS SINTÉTICOS")
    assert "tests" in pdir.parts  # synthetic outputs stay separate


def test_report_regenerated(clone):
    path = report(clone["runs"], clone["pid"])
    assert path.exists() and (clone["pdir"] / "figures" / "roc_pr_test_initial.pdf").exists()


def test_forbidden_attribute_and_unconfirmed_mapping_block(tmp_path):
    cfg = build_data(tmp_path)
    bad = tmp_path / "allow.yaml"
    a = yaml.safe_load(Path(cfg["data"]["feature_allowlist"]).read_text())
    a["features"].append("Classe Sintetica")
    bad.write_text(yaml.safe_dump(a, allow_unicode=True))
    from tc2_iot.data.sample import prepare
    with pytest.raises(DataBlocked, match="rótulo"):
        prepare({**cfg, "data": {**cfg["data"], "feature_allowlist": str(bad)}}, 1, log=lambda *x: None)
    m = yaml.safe_load(Path(cfg["data"]["label_mapping"]).read_text())
    del m["labels"]["SYN_Mirai-A"]
    mp = tmp_path / "map.yaml"
    mp.write_text(yaml.safe_dump(m, allow_unicode=True))
    with pytest.raises(DataBlocked, match="SYN_Mirai-A"):
        prepare({**cfg, "data": {**cfg["data"], "label_mapping": str(mp)}}, 1, log=lambda *x: None)


def test_category_below_minimum_blocks(tmp_path):
    cfg = build_data(tmp_path)
    with pytest.raises(DataBlocked, match="abaixo do mínimo"):
        verify_data({**cfg, "sample": {**cfg["sample"], "min_attack_per_category_per_test": 10_000}})


def test_pending_split_mode_blocks_freeze(tmp_path):
    from tc2_iot.protocol import freeze
    cfg = build_data(tmp_path)
    cfg = {**cfg, "split": {**cfg["split"], "mode": "pending_provenance_inspection"}}
    with pytest.raises(DataBlocked, match="pendente"):
        freeze(cfg, Path(cfg["output_dir"]), log=lambda *a: None)
