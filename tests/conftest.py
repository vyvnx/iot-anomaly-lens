"""Fixtures SINTÉTICAS: verificam a correção do software, não a detecção no CICIoT2023."""

import copy
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

from make_synthetic import write_fixture  # noqa: E402

from tc2_iot import config as config_mod  # noqa: E402


def make_cfg(base: Path, files="all", **overrides) -> dict:
    """Small synthetic config rooted under base/tests/ (synthetic outputs must live under a tests folder)."""
    raw = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text())
    root = base / "tests" / "output"
    raw["output_dir"] = str(root / "runs")
    raw["data"].update(raw_dir=str(root / "raw"), processed_dir=str(root / "processed"),
                       source_manifest=str(root / "raw_manifest.json"),
                       label_mapping=str(ROOT / "tests/fixtures/label_mapping_synthetic.yaml"),
                       feature_allowlist=str(ROOT / "tests/fixtures/feature_allowlist_synthetic.yaml"), files=files)
    raw["sample"].update(benign_cap=3000, ambiguous_cap=6,
                         category_caps={"DDoS": 200, "DoS": 100, "Recon": 100, "Web-based": 100, "Brute Force": 100,
                                        "Spoofing": 100, "Mirai": 100})
    raw["experiment"]["model_seeds"] = [42]
    raw["models"]["autoencoder"]["max_epochs"] = 3
    raw["models"]["isolation_forest"]["n_estimators"] = 20
    for k, v in overrides.items():
        raw[k] = {**raw[k], **v} if isinstance(v, dict) else v
    p = base / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True))
    return config_mod.load(p)


def build_data(base: Path, n_files=3, **cfg_overrides) -> dict:
    from tc2_iot.data.acquire import import_data
    from tc2_iot.data.inspect import inspect
    from tc2_iot.data.sample import prepare

    write_fixture(base / "source", n_benign=3000, n_attack=120, n_files=n_files)
    cfg = make_cfg(base, **cfg_overrides)
    import_data(base / "source", Path(cfg["data"]["raw_dir"]), Path(cfg["data"]["source_manifest"]))
    inspect(cfg, log=lambda *a: None)
    prepare(cfg, 2, log=lambda *a: None)
    return cfg


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    """Full small flow with the three models, run once per session."""
    from tc2_iot.config import MODELS
    from tc2_iot.protocol import calibrate, evaluate, freeze, train

    base = tmp_path_factory.mktemp("flow")
    cfg = build_data(base)
    runs = Path(cfg["output_dir"])
    pdir = freeze(cfg, runs, log=lambda *a: None)
    pid = pdir.name
    train(runs, pid, MODELS, [42], log=lambda *a: None)
    calibrate(runs, pid, log=lambda *a: None)
    evaluate(runs, pid, "test_initial", log=lambda *a: None)
    return {"cfg": cfg, "runs": runs, "pid": pid, "pdir": pdir, "base": base}


@pytest.fixture
def clone(pipeline, tmp_path):
    """Copy of the protocol folder for tests that tamper with artifacts."""
    import shutil
    runs = tmp_path / "tests" / "runs"
    shutil.copytree(pipeline["pdir"], runs / pipeline["pid"])
    return {**pipeline, "runs": runs, "pdir": runs / pipeline["pid"], "cfg": copy.deepcopy(pipeline["cfg"])}
