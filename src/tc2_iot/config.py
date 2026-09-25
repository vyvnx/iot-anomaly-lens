"""Leitura, validação e resolução da configuração experimental."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

PARTITIONS = ["train", "validation_fit", "validation_calibration", "test_initial", "test_confirmatory"]
BENIGN_ONLY = PARTITIONS[:3]
TEST_SPLITS = PARTITIONS[3:]
CATEGORIES = ["DDoS", "DoS", "Recon", "Web-based", "Brute Force", "Spoofing", "Mirai"]
BENIGN_STRATUM = "Benigno"
# attack vectors whose identical copies carry labels of two or more categories: not an 8th category
AMBIGUOUS = "attack_category_ambiguous"
STRATA = [BENIGN_STRATUM, *CATEGORIES, AMBIGUOUS]
# v1 (drop every conflicting group, cap per original label) was never used by a frozen protocol
POLICY_VERSION = 2
MODELS = ["isolation_forest", "sgd_one_class_svm", "autoencoder"]
SPLIT_MODES = ["pending_provenance_inspection", "random_unique_vectors", "provenance_group"]

# every key the config may contain, with its expected type; unknown keys are rejected
SCHEMA = {
    "project": str,
    "experiment_kind": str,
    "data_kind": str,
    "output_dir": str,
    "data": {
        "raw_dir": str, "processed_dir": str, "source_manifest": str, "label_mapping": str,
        "feature_allowlist": str, "label_column": str, "files": (str, list),
        "invalid_policy": str, "duplicate_policy": str, "conflicting_labels": str,
    },
    "sample": {
        "benign_cap": int, "category_caps": dict, "ambiguous_cap": int, "seed": int,
        "min_attack_per_category_per_test": int,
    },
    "split": {
        "mode": str, "seed": int, "justification": str,
        "benign_fractions": dict, "attack_fractions": dict,
    },
    "preprocessing": {"drop_train_constants": bool, "scaler": str},
    "experiment": {
        "model_seeds": list, "target_fpr": float, "quantile_method": str,
        "decision_operator": str, "device": str, "threads": (str, int),
        "confirmatory_evaluation": bool, "calibration_min_warning": int,
    },
    "models": {
        "isolation_forest": {"n_estimators": int, "max_samples": int, "contamination": str,
                             "max_features": float, "bootstrap": bool},
        "sgd_one_class_svm": {"variant": str, "components": int, "gamma": (str, float), "nu": float,
                              "max_iter": int, "tol": float, "average": bool, "shuffle": bool},
        "autoencoder": {"hidden_dimensions": list, "batch_size": int, "max_epochs": int,
                        "learning_rate": float, "weight_decay": float, "patience": int,
                        "min_delta": float},
    },
    "benchmark": {"warmup_passes": int, "repeats": int, "inference_batch_size": int},
}

# values the spec fixes as textual rules but omits from the yaml example; exposed and recorded
DEFAULTS = {
    "output_dir": "runs",
    "data": {"label_column": "auto", "files": "all", "feature_allowlist": "configs/feature_allowlist.yaml"},
    "sample": {"min_attack_per_category_per_test": 100},
    "split": {"justification": ""},
    "experiment": {"calibration_min_warning": 1000},
    "models": {
        "isolation_forest": {"max_features": 1.0, "bootstrap": False},
        "sgd_one_class_svm": {"shuffle": True},
        "autoencoder": {"weight_decay": 0.0},
    },
}


class ConfigError(ValueError):
    pass


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _check(cfg: dict, schema: dict, where: str) -> None:
    for k, v in cfg.items():
        if k not in schema:
            raise ConfigError(f"chave desconhecida na configuração: {where}{k}")
        expected = schema[k]
        if isinstance(expected, dict):
            if not isinstance(v, dict):
                raise ConfigError(f"{where}{k} deve ser um mapeamento")
            _check(v, expected, f"{where}{k}.")
        else:
            types = expected if isinstance(expected, tuple) else (expected,)
            # yaml gives int for "1" where a float is expected; accept it
            if float in types and isinstance(v, int) and not isinstance(v, bool):
                continue
            if not isinstance(v, types) or (isinstance(v, bool) and bool not in types):
                raise ConfigError(f"{where}{k} tem tipo inválido: {type(v).__name__}")
    for k, expected in schema.items():
        if k not in cfg:
            raise ConfigError(f"chave obrigatória ausente: {where}{k}")


def validate(cfg: dict) -> None:
    _check(cfg, SCHEMA, "")
    if cfg["data_kind"] not in ("real", "synthetic"):
        raise ConfigError("data_kind deve ser 'real' ou 'synthetic'")
    if cfg["data_kind"] == "synthetic" and "tests" not in Path(cfg["output_dir"]).parts:
        raise ConfigError("dados sintéticos devem gravar saídas sob uma pasta 'tests/' para não se misturarem aos reais")
    if cfg["data_kind"] == "real" and "tests" in Path(cfg["output_dir"]).parts:
        raise ConfigError("dados reais não podem gravar saídas na pasta de testes")
    d = cfg["data"]
    if d["invalid_policy"] != "drop":
        raise ConfigError("somente invalid_policy=drop está implementado; imputação exige nova configuração")
    if d["duplicate_policy"] != "unique_feature_vectors" or d["conflicting_labels"] != "drop_binary_ambiguous":
        raise ConfigError("somente duplicate_policy=unique_feature_vectors e conflicting_labels=drop_binary_ambiguous "
                          f"(política v{POLICY_VERSION}) estão implementados")
    caps = cfg["sample"]["category_caps"]
    if set(caps) != set(CATEGORIES) or not all(isinstance(v, int) and v >= 0 for v in caps.values()):
        raise ConfigError(f"sample.category_caps deve ter um inteiro >= 0 para cada categoria: {CATEGORIES}")
    cfg["sample"]["category_caps"] = {c: caps[c] for c in CATEGORIES}
    s = cfg["split"]
    if s["mode"] not in SPLIT_MODES:
        raise ConfigError(f"split.mode deve ser um de {SPLIT_MODES}")
    for name, keys in (("benign_fractions", PARTITIONS), ("attack_fractions", TEST_SPLITS)):
        if set(s[name]) != set(keys):
            raise ConfigError(f"split.{name} deve conter exatamente {keys}")
        # canonical order: rounding ties and assignment follow it
        s[name] = {k: s[name][k] for k in keys}
    for name in ("benign_fractions", "attack_fractions"):
        fr = s[name]
        if any(not 0 <= float(v) <= 1 for v in fr.values()) or abs(sum(map(float, fr.values())) - 1) > 1e-9:
            raise ConfigError(f"split.{name} deve somar 1")
    e = cfg["experiment"]
    if e["quantile_method"] != "higher" or e["decision_operator"] != "greater_than":
        raise ConfigError("somente quantile_method=higher e decision_operator=greater_than estão implementados")
    if not 0 < e["target_fpr"] < 1:
        raise ConfigError("experiment.target_fpr deve estar em (0, 1)")
    if e["device"] not in ("cpu", "cuda"):
        raise ConfigError("experiment.device deve ser cpu ou cuda")
    if e["confirmatory_evaluation"]:
        raise ConfigError("confirmatory_evaluation deve ser false; a avaliação confirmatória é liberada só por comando explícito")
    if not e["model_seeds"] or not all(isinstance(x, int) for x in e["model_seeds"]):
        raise ConfigError("experiment.model_seeds deve ser uma lista de inteiros")
    m = cfg["models"]
    if m["isolation_forest"]["contamination"] != "auto":
        raise ConfigError("isolation_forest.contamination deve ser 'auto'; a decisão usa a calibração comum")
    if m["sgd_one_class_svm"]["variant"] not in ("nystroem_rbf", "linear"):
        raise ConfigError("sgd_one_class_svm.variant deve ser nystroem_rbf ou linear")
    g = m["sgd_one_class_svm"]["gamma"]
    if isinstance(g, str) and g != "inverse_feature_count":
        raise ConfigError("sgd_one_class_svm.gamma deve ser inverse_feature_count ou número")
    if cfg["preprocessing"]["scaler"] != "standard":
        raise ConfigError("somente scaler=standard está implementado")


def load(path: str | Path) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, raw)
    validate(cfg)
    cfg["_config_path"] = str(path)
    return cfg


def canonical_hash(obj) -> str:
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def public(cfg: dict) -> dict:
    """Configuration without runtime-only keys (underscore-prefixed)."""
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()
