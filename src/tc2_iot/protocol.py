"""Ciclo de vida de um protocolo: verify-data, freeze, train, calibrate, evaluate, verify-run.

Cada protocolo vive em <runs>/<protocol_id>/. Artefatos são gravados uma única vez; a
retomada só reaproveita um artefato cujo hash confere com o registro.
"""

from __future__ import annotations

import copy
import hashlib
import io
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import benchmark, calibration, evaluation
from .config import (AMBIGUOUS, BENIGN_STRATUM, CATEGORIES, MODELS, PARTITIONS, POLICY_VERSION, STRATA,
                     TEST_SPLITS, canonical_hash, public, sha256_file)
from .data.clean import DataBlocked
from .data.preprocess import Preprocessor
from .data.sample import prepare_is_current
from .data.split import assign_partitions, check_partitions, resolve_effective_conflicts
from .environment import apply_thread_limits, collect, resolve_threads
from .models import base as model_base
from .provenance import code_version, exposures, now, read_json, record_exposure, write_json

SYNTHETIC_BANNER = "DADOS SINTÉTICOS — verificação de software; NÃO são resultados do CICIoT2023"
FROZEN_FILES = ["partitions.parquet", "features.parquet", "preprocessor.joblib", "label_mapping.yaml",
                "raw_manifest.json", "data_quality.json", "feature_catalog.csv", "effective_space_exclusions.csv",
                "excluded_binary_ambiguous_rows.parquet"]


class IntegrityError(RuntimeError):
    pass


# ---------------------------------------------------------------- verify-data

def split_and_preprocess(sample: pd.DataFrame, cfg: dict, feats: list[str]):
    """Partitions, preprocessing fitted on train only and the P4 rule in the transformed space.

    If the rule removes training rows, the preprocessing is refitted on the new train and the check is
    repeated until train no longer changes. Removing rows never creates new collisions.
    """
    parts = assign_partitions(sample, cfg)
    removed_all = []
    for it in range(1, 11):
        X = parts[feats].to_numpy(np.float64)
        pre = Preprocessor(feats, cfg["preprocessing"]["drop_train_constants"]).fit(X[(parts["partition"] == "train").to_numpy()])
        parts, removed = resolve_effective_conflicts(parts, pre.transform(X))
        removed_all.append(removed.assign(iteracao=it))
        if not (removed["partition"] == "train").any():
            break
    else:
        raise IntegrityError("o treino não estabilizou após 10 reajustes do pré-processamento")
    Z = pre.transform(parts[feats].to_numpy(np.float64))
    return parts, pre, Z, pd.concat(removed_all, ignore_index=True), it


def partition_table(parts: pd.DataFrame) -> pd.DataFrame:
    t = pd.crosstab(parts["partition"], parts["stratum"]).reindex(index=PARTITIONS, columns=STRATA, fill_value=0)
    t["ataques"] = t[[*CATEGORIES, AMBIGUOUS]].sum(axis=1)
    t["total"] = t[BENIGN_STRATUM] + t["ataques"]
    t.loc["total"] = t.sum()
    return t


def verify_data(cfg: dict) -> dict:
    proc = Path(cfg["data"]["processed_dir"])
    checks, errors, warnings = {}, [], []
    checks["prepare_current"] = prepare_is_current(cfg)
    if not checks["prepare_current"]:
        raise DataBlocked("amostra ausente ou desatualizada em relação a arquivos/mapeamento/allowlist/config; execute prepare")
    s = pd.read_parquet(proc / "sample.parquet")
    from .data.inspect import normalize
    feats = [normalize(f) for f in read_json(proc / "prepare_state.json")["features"]]
    X = s[feats].to_numpy(np.float64)
    checks["vector_id_unique"] = bool(s["vector_id"].is_unique)
    checks["record_id_unique"] = bool(s["record_id"].is_unique)
    checks["features_finite"] = bool(np.isfinite(X).all())
    checks["hash_recheck_ok"] = bool(s["hash_ok"].all())
    checks["has_benign"] = bool(s["benign"].any())
    checks["no_binary_ambiguity_in_sample"] = bool(~s["situation"].eq("A").any())
    checks["strata_valid"] = bool(s["stratum"].isin(STRATA).all())
    for k, v in checks.items():
        if not v:
            errors.append(f"falhou: {k}")
    table, removed, iterations = None, None, None
    if cfg["split"]["mode"] == "pending_provenance_inspection":
        warnings.append("split.mode pendente: partições e mínimos por teste não verificados")
    else:
        parts, pre, Z, removed, iterations = split_and_preprocess(s, cfg, feats)
        table = partition_table(parts)
        table.to_csv(proc / "partition_preview.csv")
        removed.to_csv(proc / "effective_space_exclusions_preview.csv", index=False)
        minimum = cfg["sample"]["min_attack_per_category_per_test"]
        short = {t: {c: int(table.at[t, c]) for c in CATEGORIES if table.at[t, c] < minimum} for t in TEST_SPLITS}
        checks["categories_min_per_test"] = not any(short.values())
        if any(short.values()):
            errors.append(f"categorias abaixo do mínimo de {minimum} por teste (sem duplicar vetores; revisar a amostragem): {short}")
    result = {"data_kind": cfg["data_kind"], "checked_at_utc": now(), "checks": checks, "errors": errors,
              "partition_table": None if table is None else table.astype(int).to_dict(orient="index"),
              "effective_space_exclusions": None if removed is None else {
                  "total": len(removed), "preprocessing_fits": iterations,
                  "by_reason_partition": removed.groupby(["motivo", "partition"]).size().rename("n").reset_index().to_dict("records")},
              "sample_total": len(s),
              "warnings": read_json(proc / "data_quality.json")["warnings"] + warnings}
    write_json(proc / "verify_data.json", result)
    if errors:
        raise DataBlocked("verify-data falhou: " + " | ".join(errors))
    return result


# ---------------------------------------------------------------- freeze

def freeze_fingerprint(cfg: dict) -> str:
    proc = Path(cfg["data"]["processed_dir"])
    st = read_json(proc / "prepare_state.json")
    return canonical_hash({"config": public(cfg), "sample": st["sample_sha256"], "inputs": st["input_fingerprint"],
                           "code": code_version()["source_tree_sha256"]})


def find_protocol(runs: Path, fingerprint: str) -> Path | None:
    for p in sorted(runs.glob("*/protocol.json")):
        if read_json(p).get("freeze_fingerprint") == fingerprint:
            return p.parent
    return None


def resolve_config(cfg: dict, n_train: int, d: int) -> dict:
    r = copy.deepcopy(public(cfg))
    r["experiment"]["threads"] = resolve_threads(cfg["experiment"]["threads"])
    r["models"]["isolation_forest"]["max_samples"] = min(cfg["models"]["isolation_forest"]["max_samples"], n_train)
    sgd = r["models"]["sgd_one_class_svm"]
    sgd["components"] = min(sgd["components"], n_train)
    sgd["gamma_resolved"] = 1.0 / d if sgd["gamma"] == "inverse_feature_count" else float(sgd["gamma"])
    r["resolved_dimensions"] = {"d_after_preprocessing": d, "n_train": n_train}
    return r


def split_content_hash(parts: pd.DataFrame, split: str) -> str:
    ids = sorted(parts.loc[parts["partition"] == split, "record_id"])
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def freeze(cfg: dict, runs: Path, log=print) -> Path:
    vd = verify_data(cfg)
    proc = Path(cfg["data"]["processed_dir"])
    state = read_json(proc / "prepare_state.json")
    from .data.inspect import normalize
    feats = [normalize(f) for f in state["features"]]
    sample = pd.read_parquet(proc / "sample.parquet")
    t0 = time.perf_counter()
    parts, pre, Z, removed, iterations = split_and_preprocess(sample, cfg, feats)
    prep_seconds = time.perf_counter() - t0
    checks = check_partitions(parts, Z)
    if not all(checks.values()):
        raise IntegrityError(f"verificações de partição falharam: {checks}")
    table = partition_table(parts)
    minimum = cfg["sample"]["min_attack_per_category_per_test"]
    for t in TEST_SPLITS:
        for c in CATEGORIES:
            if table.at[t, c] < minimum:
                raise DataBlocked(f"após conflitos, {t}/{c} tem {table.at[t, c]} < {minimum}; não redistribuo para melhorar resultados")
    counts = parts.groupby(["partition", "stratum", "label"]).size().rename("registros").reset_index()

    fp = freeze_fingerprint(cfg)
    pid = f"{cfg['data_kind']}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{fp[:8]}"
    pdir = runs / pid
    pdir.mkdir(parents=True, exist_ok=False)
    parts.drop(columns=feats + ["hash_ok"]).to_parquet(pdir / "partitions.parquet", index=False)
    parts[["vector_id", *feats]].to_parquet(pdir / "features.parquet", index=False)
    pre.save(pdir / "preprocessor.joblib")
    write_json(pdir / "preprocessing.json", {**pre.summary(), "fit_on": "train", "fit_and_transform_seconds": prep_seconds})
    counts.rename(columns={"partition": "particao", "stratum": "estrato", "label": "rotulos_originais"}).to_csv(
        pdir / "sample_counts.csv", index=False)
    table.to_csv(pdir / "partition_table.csv")
    removed.to_csv(pdir / "effective_space_exclusions.csv", index=False)
    shutil.copy(proc / "strata_counts.csv", pdir / "strata_counts.csv")
    shutil.copy(proc / "excluded_binary_ambiguous_rows.parquet", pdir / "excluded_binary_ambiguous_rows.parquet")
    shutil.copy(proc / "sample_counts.csv", pdir / "sample_counts_before_split.csv")
    shutil.copy(proc / "data_quality.json", pdir / "data_quality.json")
    shutil.copy(cfg["data"]["label_mapping"], pdir / "label_mapping.yaml")
    shutil.copy(cfg["data"]["source_manifest"], pdir / "raw_manifest.json")
    cat = pd.read_csv(proc / "inspection" / "feature_catalog.csv")
    cat["decisao_final"] = cat["original_name"].map(
        lambda c: "incluído" if normalize(c) in pre.kept else
        ("removido: constante no treino" if normalize(c) in pre.dropped else "excluído pela allowlist"))
    cat.to_csv(pdir / "feature_catalog.csv", index=False)
    env = collect(cfg["data"]["processed_dir"])
    write_json(pdir / "environment.json", env)

    n_train = int((parts["partition"] == "train").sum())
    resolved = resolve_config(cfg, n_train, pre.n_features_out)
    effective = parts.groupby(["benign", "partition"]).size()
    protocol = {
        "protocol_id": pid,
        "created_at_utc": now(),
        "data_kind": cfg["data_kind"],
        "experiment_kind": cfg["experiment_kind"],
        "config_source": cfg["_config_path"],
        "config_resolved": resolved,
        "config_hash": canonical_hash(public(cfg)),
        "freeze_fingerprint": fp,
        "code_version": code_version(),
        "features_input": feats, "features_used": pre.kept, "features_dropped_train_constant": pre.dropped,
        "features_hash": canonical_hash(pre.kept),
        "label_mapping_hash": sha256_file(pdir / "label_mapping.yaml"),
        "raw_input_hash": canonical_hash(read_json(pdir / "raw_manifest.json")["files"]),
        "files_sha256": {f: sha256_file(pdir / f) for f in FROZEN_FILES},
        "split": {"mode": cfg["split"]["mode"], "seed": cfg["split"]["seed"], "justification": cfg["split"]["justification"],
                  "effective_counts": {f"{'benigno' if b else 'ataque'}/{p}": int(n) for (b, p), n in effective.items()},
                  "allocation": "sistemática por estrato: ordenação por rótulo e prioridade SHA-256(semente:vector_id); "
                                "cada vetor vai para a partição com maior déficit em relação à fração",
                  "effective_space_rule": "vetores idênticos após o pré-processamento: benigno+ataque -> todos excluídos; "
                                          "mesma classe binária -> mantidos na partição de maior prioridade "
                                          "(" + " > ".join(PARTITIONS) + "); treino alterado -> pré-processamento reajustado",
                  "effective_space_exclusions": {"total": len(removed), "preprocessing_fits": iterations,
                                                 "file": "effective_space_exclusions.csv"},
                  "partition_table": table.astype(int).to_dict(orient="index"),
                  "checks": checks,
                  "content_sha256": {t: split_content_hash(parts, t) for t in PARTITIONS}},
        "verify_data": vd["checks"],
        "policy_version": POLICY_VERSION,
        "limitations": [
            "unidade de análise: vetores de atributos distintos, não a distribuição natural do tráfego",
            "partição aleatória de vetores distintos não comprova independência temporal, por dispositivo ou de observações relacionadas"
            if cfg["split"]["mode"] == "random_unique_vectors" else "partição por grupos de proveniência",
            "vetores com benigno e ataque (ambiguidade binária na representação adotada) foram excluídos; podem ser justamente "
            "as observações mais difíceis de distinguir, o que pode enviesar as métricas",
            "vetores de ataque com categorias conflitantes formam o estrato attack_category_ambiguous (não é uma 8ª categoria)",
            "tetos por estrato são desenho experimental; a composição por rótulo original não reproduz a base",
            *vd["warnings"],
        ],
        "confirmatory_released": False,
    }
    write_json(pdir / "protocol.json", protocol)
    log(f"protocolo congelado: {pid}")
    return pdir


# ---------------------------------------------------------------- loading and integrity

def load_protocol(runs: Path, pid: str) -> tuple[Path, dict]:
    pdir = runs / pid
    if not (pdir / "protocol.json").exists():
        raise FileNotFoundError(f"protocolo {pid} não encontrado em {runs}")
    return pdir, read_json(pdir / "protocol.json")


def check_integrity(pdir: Path, protocol: dict, strict_code: bool) -> list[str]:
    problems = [f"{f} alterado desde o congelamento" for f, h in protocol["files_sha256"].items()
                if not (pdir / f).exists() or sha256_file(pdir / f) != h]
    if strict_code and code_version()["source_tree_sha256"] != protocol["code_version"]["source_tree_sha256"]:
        problems.append("código-fonte difere do registrado no congelamento; crie um novo protocolo")
    manifest = pdir / "models_manifest.json"
    if manifest.exists():
        for key, e in read_json(manifest)["entries"].items():
            for f, h in e["files_sha256"].items():
                if not (pdir / f).exists() or sha256_file(pdir / f) != h:
                    problems.append(f"artefato de modelo alterado: {f}")
    return problems


def require_integrity(pdir: Path, protocol: dict, strict_code: bool = True) -> None:
    problems = check_integrity(pdir, protocol, strict_code)
    if problems:
        raise IntegrityError("; ".join(problems))


def load_data(pdir: Path, protocol: dict):
    parts = pd.read_parquet(pdir / "partitions.parquet").sort_values("vector_id").reset_index(drop=True)
    feats = pd.read_parquet(pdir / "features.parquet").set_index("vector_id").loc[parts["vector_id"]]
    X = feats[protocol["features_input"]].to_numpy(np.float64)
    return parts, X, Preprocessor.load(pdir / "preprocessor.joblib")


def apply_threads(protocol: dict) -> dict:
    return apply_thread_limits(protocol["config_resolved"]["experiment"]["threads"])


def model_dir(model: str, seed: int) -> str:
    return f"models/{model}/seed_{seed}"


# ---------------------------------------------------------------- train

def _fit_is_valid(pdir: Path, rel: str) -> bool:
    fj = pdir / rel / "fit.json"
    if not fj.exists():
        return False
    info = read_json(fj)
    return all((pdir / f).exists() and sha256_file(pdir / f) == h for f, h in info["files_sha256"].items())


def train(runs: Path, pid: str, models: list[str], seeds: list[int], log=print) -> None:
    pdir, protocol = load_protocol(runs, pid)
    require_integrity(pdir, protocol)
    cfg = protocol["config_resolved"]
    bad_seeds = [s for s in seeds if s not in cfg["experiment"]["model_seeds"]]
    if bad_seeds:
        raise ValueError(f"sementes {bad_seeds} não estão no protocolo ({cfg['experiment']['model_seeds']})")
    parts, X, pre = load_data(pdir, protocol)
    Ztr = pre.transform(X[(parts["partition"] == "train").to_numpy()])
    Zvf = pre.transform(X[(parts["partition"] == "validation_fit").to_numpy()])
    manifest = read_json(pdir / "models_manifest.json")["entries"] if (pdir / "models_manifest.json").exists() else {}
    threads = apply_threads(protocol)
    for m in models:
        warmed = False
        for seed in seeds:
            rel = model_dir(m, seed)
            if _fit_is_valid(pdir, rel):
                log(f"{m} semente {seed}: já treinado e verificado")
                continue
            if f"{m}/{seed}" in manifest:
                raise IntegrityError(f"{m}/{seed} está no manifesto, mas seus artefatos não conferem")
            if (pdir / rel).exists():
                shutil.rmtree(pdir / rel)  # stale/incomplete, never reused
            (pdir / rel).mkdir(parents=True)
            log(f"treinando {m} semente {seed} ({len(Ztr)} registros de treino)")
            if not warmed:
                # untimed, discarded fit on a small slice: library/runtime initialization is not charged to the first seed
                w = min(len(Ztr), 256)
                model_base.build(m, cfg, seed, cfg["experiment"]["threads"], w, Ztr.shape[1]).fit(
                    Ztr[:w], Zvf[:w] if m == "autoencoder" else None)
                warmed = True
            model = model_base.build(m, cfg, seed, cfg["experiment"]["threads"], len(Ztr), Ztr.shape[1])
            t0 = time.perf_counter()
            model.fit(Ztr, Zvf if m == "autoencoder" else None)
            fit_s = time.perf_counter() - t0
            files = model.save(pdir / rel)
            write_json(pdir / rel / "fit.json", {
                "model": m, "seed": seed, "n_train": len(Ztr),
                "n_validation_fit": len(Zvf) if m == "autoencoder" else 0,
                "fit_seconds": fit_s,
                "fit_seconds_includes": {"isolation_forest": "construção das árvores",
                                         "sgd_one_class_svm": "ajuste Nyström (landmarks) + transformação + SGD",
                                         "autoencoder": "treino e validação por época"}[m]
                + "; exclui aquecimento de bibliotecas (ajuste descartado em até 256 registros)",
                "params": model.resolved_params(), "fit_info": model.fit_info(),
                "threads": threads, "trained_at_utc": now(),
                "files_sha256": {f.relative_to(pdir).as_posix(): sha256_file(f) for f in files},
            })


# ---------------------------------------------------------------- calibrate

def calibrate(runs: Path, pid: str, log=print) -> dict:
    pdir, protocol = load_protocol(runs, pid)
    require_integrity(pdir, protocol)
    cfg = protocol["config_resolved"]
    parts, X, pre = load_data(pdir, protocol)
    vc = (parts["partition"] == "validation_calibration").to_numpy()
    assert parts.loc[vc, "benign"].all()
    Zvc = pre.transform(X[vc])
    mpath = pdir / "models_manifest.json"
    manifest = read_json(mpath) if mpath.exists() else {"protocol_id": pid, "entries": {}}
    tpath = pdir / "thresholds.json"
    thresholds = read_json(tpath) if tpath.exists() else {}
    _ = apply_threads(protocol)
    for fj in sorted(pdir.glob("models/*/seed_*/fit.json")):
        info = read_json(fj)
        key = f"{info['model']}/{info['seed']}"
        if key in manifest["entries"]:
            continue  # frozen entries are never modified
        if not _fit_is_valid(pdir, fj.parent.relative_to(pdir).as_posix()):
            raise IntegrityError(f"{key}: artefatos não conferem com fit.json")
        model = model_base.load(info["model"], fj.parent)
        t0 = time.perf_counter()
        scores = model.anomaly_score(Zvc)
        score_s = time.perf_counter() - t0
        c = calibration.calibrate(scores, cfg["experiment"]["target_fpr"], cfg["experiment"]["calibration_min_warning"])
        c["scoring_seconds"] = score_s
        thresholds[key] = c
        manifest["entries"][key] = {
            "model": info["model"], "seed": info["seed"], "threshold": c["threshold"],
            "files_sha256": {**info["files_sha256"], fj.relative_to(pdir).as_posix(): sha256_file(fj)},
            "calibrated_at_utc": now(),
        }
        log(f"{key}: limiar {c['threshold']:.6g}, FPR calibração {c['calibration_fpr']:.4%} ({c['calibration_fp']}/{c['n_calibration']}), empates {c['ties_at_threshold']}")
    write_json(tpath, thresholds)
    manifest["thresholds_sha256"] = canonical_hash({k: v["threshold"] for k, v in thresholds.items()})
    write_json(mpath, manifest)
    return thresholds


# ---------------------------------------------------------------- evaluate

def evaluate(runs: Path, pid: str, split: str, release_confirmatory: bool = False, log=print) -> None:
    if split not in TEST_SPLITS:
        raise ValueError(f"split deve ser um de {TEST_SPLITS}")
    pdir, protocol = load_protocol(runs, pid)
    if split == "test_confirmatory" and not release_confirmatory:
        raise PermissionError("o teste confirmatório exige liberação explícita do autor (--release-confirmatory)")
    require_integrity(pdir, protocol, strict_code=True)
    if not (pdir / "models_manifest.json").exists():
        raise IntegrityError("execute calibrate antes de evaluate")
    cfg = protocol["config_resolved"]
    manifest = read_json(pdir / "models_manifest.json")
    parts, X, pre = load_data(pdir, protocol)
    mask = (parts["partition"] == split).to_numpy()
    ev = parts[mask].reset_index(drop=True)
    Xs = X[mask]
    record_exposure(runs, pid, split, protocol["split"]["content_sha256"][split])
    if split == "test_confirmatory":
        protocol["confirmatory_released"] = True
        # protocol.json itself is not hashed; the release flag is also in the append-only exposure log
        write_json(pdir / "protocol.json", protocol)
    out = pdir / "predictions" / split
    out.mkdir(parents=True, exist_ok=True)
    b = cfg["benchmark"]
    Zs = pre.transform(Xs)
    threads = apply_threads(protocol)
    for key, e in sorted(manifest["entries"].items()):
        fname = out / f"{e['model']}__seed{e['seed']}.parquet"
        if fname.exists():
            log(f"{key}: já avaliado em {split}")
            continue
        model = model_base.load(e["model"], pdir / model_dir(e["model"], e["seed"]))
        scores = model.anomaly_score(Zs)
        if not np.isfinite(scores).all():
            raise ValueError(f"{key}: escores não finitos")
        thr = e["threshold"]
        pred = calibration.predict(scores, thr)
        df = pd.DataFrame({"record_id": ev["record_id"], "vector_id": ev["vector_id"], "label": ev["label"],
                           "category": ev["category"], "stratum": ev["stratum"], "benign": ev["benign"], "y_true": (~ev["benign"]).astype(np.int8),
                           "model": e["model"], "seed": e["seed"], "score": scores, "threshold": thr, "pred": pred})
        bs = b["inference_batch_size"]
        t_scores = benchmark.repeat_timed(lambda: benchmark.batched(model.anomaly_score, Zs, bs), b["warmup_passes"], b["repeats"])
        full = lambda: benchmark.batched(lambda x: calibration.predict(model.anomaly_score(pre.transform(x)), thr), Xs, bs)
        t_full = benchmark.repeat_timed(full, b["warmup_passes"], b["repeats"])
        timing = {"split": split, "model": e["model"], "seed": e["seed"], "batch_size": bs, "threads": threads,
                  "device": cfg["experiment"]["device"],
                  "scores_only": benchmark.summarize(t_scores, len(Zs)),
                  "preprocess_score_threshold": benchmark.summarize(t_full, len(Xs)),
                  "excludes": "leitura de disco; escrita de resultados", "measured_at_utc": now()}
        write_json(out / f"{e['model']}__seed{e['seed']}.timing.json", timing)
        df.to_parquet(fname, index=False)
        log(f"{key}: {split} avaliado ({len(df)} registros)")
    recompute_tables(pdir)


# ---------------------------------------------------------------- tables from saved artifacts

def recompute_tables(pdir: Path) -> dict[str, pd.DataFrame]:
    """metrics/category/timings/history tables are always rebuilt from predictions and records."""
    metrics, cats, timings, hist = [], [], [], []
    for f in sorted(pdir.glob("predictions/*/*.parquet")):
        df = pd.read_parquet(f)
        split, model, seed = f.parent.name, df["model"].iloc[0], int(df["seed"].iloc[0])
        m = evaluation.binary_metrics(df["y_true"].to_numpy(), df["score"].to_numpy(), df["pred"].to_numpy())
        metrics.append({"split": split, "model": model, "seed": seed, "threshold": float(df["threshold"].iloc[0]), **m})
        for r in evaluation.category_metrics(df):
            cats.append({"split": split, "model": model, "seed": seed, **r})
    for fj in sorted(pdir.glob("models/*/seed_*/fit.json")):
        info = read_json(fj)
        timings.append({"fase": "ajuste", "split": "train", "model": info["model"], "seed": info["seed"],
                        "records": info["n_train"], "seconds_median": info["fit_seconds"], "repeats": 1,
                        "threads": info["threads"]["threads_budget"], "inclui": info["fit_seconds_includes"]})
        for h in (read_json(fj.parent / "autoencoder.json")["history"] if info["model"] == "autoencoder" else []):
            hist.append({"model": info["model"], "seed": info["seed"], **h, "best_epoch": info["fit_info"]["best_epoch"]})
    if (pdir / "thresholds.json").exists():
        for key, c in read_json(pdir / "thresholds.json").items():
            model, seed = key.split("/")
            timings.append({"fase": "calibracao_escores", "split": "validation_calibration", "model": model, "seed": int(seed),
                            "records": c["n_calibration"], "seconds_median": c["scoring_seconds"], "repeats": 1,
                            "inclui": "escores sobre dados já transformados"})
    for tj in sorted(pdir.glob("predictions/*/*.timing.json")):
        t = read_json(tj)
        for phase, label in [("scores_only", "inferencia_escores"), ("preprocess_score_threshold", "inferencia_completa")]:
            timings.append({"fase": label, "split": t["split"], "model": t["model"], "seed": t["seed"],
                            **t[phase], "threads": t["threads"]["threads_budget"], "batch_size": t["batch_size"],
                            "inclui": "escores com dados transformados" if phase == "scores_only"
                            else "pré-processamento + escores + limiar, sem leitura de disco"})
    if (pdir / "preprocessing.json").exists():
        p = read_json(pdir / "preprocessing.json")
        timings.append({"fase": "preparacao_comum", "split": "train", "model": "todos", "seed": None,
                        "records": p["fit_rows"], "seconds_median": p["fit_and_transform_seconds"], "repeats": 1,
                        "inclui": "ajuste do scaler em train + transformação da amostra"})
    timings = pd.DataFrame(timings)
    if len(timings):
        timings["seed"] = timings["seed"].astype("Int64")
    tables = {"metrics": pd.DataFrame(metrics), "category_metrics": pd.DataFrame(cats),
              "timings": timings, "ae_training_history": pd.DataFrame(hist)}
    for name, df in tables.items():
        if len(df):
            df.to_csv(pdir / f"{name}.csv", index=False)
    return tables


def combined_predictions(pdir: Path, split: str) -> pd.DataFrame | None:
    files = sorted((pdir / "predictions" / split).glob("*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.to_parquet(pdir / f"predictions_{split}.parquet", index=False)
    return df


# ---------------------------------------------------------------- verify-run

def verify_run(runs: Path, pid: str) -> dict:
    pdir, protocol = load_protocol(runs, pid)
    res: dict[str, object] = {}
    problems = check_integrity(pdir, protocol, strict_code=False)
    res["frozen_artifacts_intact"] = not problems
    res["integrity_problems"] = problems
    res["code_matches_freeze"] = code_version()["source_tree_sha256"] == protocol["code_version"]["source_tree_sha256"]
    parts, X, pre = load_data(pdir, protocol)
    Z = pre.transform(X)
    res.update(check_partitions(parts, Z))
    tr = (parts["partition"] == "train").to_numpy()
    refit = Preprocessor(protocol["features_input"], pre.drop_train_constants).fit(X[tr])
    res["preprocessor_fit_only_on_train"] = bool(
        refit.kept == pre.kept and np.allclose(refit.scaler.mean_, pre.scaler.mean_)
        and np.allclose(refit.scaler.scale_, pre.scaler.scale_))
    res["split_hashes_match"] = all(split_content_hash(parts, t) == h for t, h in protocol["split"]["content_sha256"].items())

    thresholds = read_json(pdir / "thresholds.json") if (pdir / "thresholds.json").exists() else {}
    manifest = read_json(pdir / "models_manifest.json")["entries"] if (pdir / "models_manifest.json").exists() else {}
    vc = (parts["partition"] == "validation_calibration").to_numpy()
    cfg = protocol["config_resolved"]
    thr_ok, reload_ok, pred_ok, order_ok = True, True, True, True
    _ = apply_threads(protocol)
    for key, e in manifest.items():
        model = model_base.load(e["model"], pdir / model_dir(e["model"], e["seed"]))
        c = calibration.calibrate(model.anomaly_score(Z[vc]), cfg["experiment"]["target_fpr"])
        thr_ok &= np.isclose(c["threshold"], e["threshold"], rtol=1e-9, atol=1e-12) and thresholds[key]["threshold"] == e["threshold"]
    # reload tolerance: scores recomputed from saved artifacts, over the whole split exactly as evaluate does,
    # must match stored predictions; batch-size sensitivity (float32/BLAS blocking) is reported separately
    batch_sens = []
    for split in TEST_SPLITS:
        files = sorted((pdir / "predictions" / split).glob("*.parquet"))
        orders = []
        for f in files:
            df = pd.read_parquet(f)
            e = manifest[f"{df['model'].iloc[0]}/{int(df['seed'].iloc[0])}"]
            pred_ok &= bool((df["threshold"] == e["threshold"]).all() and ((df["score"] > df["threshold"]).astype(int) == df["pred"]).all())
            orders.append(tuple(df["record_id"]))
            model = model_base.load(e["model"], pdir / model_dir(e["model"], e["seed"]))
            Zs = Z[(parts["partition"] == split).to_numpy()]
            s = model.anomaly_score(Zs)
            reload_ok &= bool(np.allclose(s, df["score"].to_numpy(), rtol=1e-9, atol=1e-12))
            sb = benchmark.batched(model.anomaly_score, Zs, cfg["benchmark"]["inference_batch_size"])
            batch_sens.append({"split": split, "model": e["model"], "seed": e["seed"],
                               "max_abs_score_diff": float(np.abs(sb - s).max()),
                               "decisions_changed": int(((sb > e["threshold"]) != (s > e["threshold"])).sum()),
                               "records": len(s)})
        order_ok &= len(set(orders)) <= 1
    res["thresholds_reproducible_from_calibration_only"] = bool(thr_ok)
    res["predictions_use_frozen_thresholds"] = bool(pred_ok)
    res["same_records_and_order_for_all_models"] = bool(order_ok)
    res["save_load_scores_within_tolerance"] = bool(reload_ok)
    res["save_load_tolerance"] = "rtol=1e-9, atol=1e-12, teste inteiro em uma chamada (como em evaluate)"
    res["batch_size_sensitivity_info"] = {
        "note": "informativo: diferença entre pontuar o teste inteiro e em lotes de inference_batch_size; "
                "predições e calibração oficiais usam o conjunto inteiro",
        "by_model": batch_sens}
    before = {k: pd.read_csv(pdir / f"{k}.csv") for k in ("metrics", "category_metrics") if (pdir / f"{k}.csv").exists()}
    after = recompute_tables(pdir)
    res["metrics_recomputable_from_predictions"] = bool(before) and all(
        pd.read_csv(io.StringIO(after[k].to_csv(index=False))).equals(before[k]) for k in before)
    conf_files = list((pdir / "predictions" / "test_confirmatory").glob("*.parquet"))
    released = any(x["protocol_id"] == pid and x["split"] == "test_confirmatory" for x in exposures(runs))
    res["confirmatory_not_evaluated_without_release"] = (not conf_files) or released
    res["confirmatory_evaluated"] = bool(conf_files)
    res["test_exposures"] = [x for x in exposures(runs) if x["split_content_sha256"] in protocol["split"]["content_sha256"].values()]
    res["models_calibrated"] = sorted(manifest)
    res["models_expected"] = [f"{m}/{s}" for m in MODELS for s in cfg["experiment"]["model_seeds"]]
    res["data_kind"] = protocol["data_kind"]
    bools = {k: v for k, v in res.items() if isinstance(v, bool) and k not in ("confirmatory_evaluated", "code_matches_freeze")}
    res["failed"] = [k for k, v in bools.items() if not v]
    res["ok"] = not res["failed"]
    res["checked_at_utc"] = now()
    write_json(pdir / "verification.json", res)
    return res
