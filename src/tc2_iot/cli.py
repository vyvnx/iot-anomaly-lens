"""Interface de linha de comando tc2-iot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config as config_mod
from .config import MODELS


def _cfg(a):
    return config_mod.load(a.config)


def cmd_doctor(a):
    from .environment import doctor, human_summary
    env = doctor(a.out, a.data_dir)
    print(human_summary(env))
    print(f"\nGravado: {Path(a.out) / 'environment.json'}")


def cmd_import(a):
    from .data.acquire import import_data
    cfg = _cfg(a)
    m = import_data(Path(a.source), Path(cfg["data"]["raw_dir"]), Path(cfg["data"]["source_manifest"]),
                    a.mode, a.obtained_at)
    total = sum(f["size_bytes"] for f in m["files"])
    print(f"{len(m['files'])} arquivos registrados em {cfg['data']['source_manifest']} ({total / 2**30:.2f} GiB)")


def cmd_download(a):
    from .data.acquire import download
    cfg = _cfg(a)
    urls = [u.strip() for u in Path(a.urls_file).read_text(encoding="utf-8").splitlines() if u.strip() and not u.startswith("#")]
    res = download(urls, Path(cfg["data"]["raw_dir"]), Path(cfg["data"]["source_manifest"]),
                   int(a.max_gib * 2**30), dry_run=a.dry_run)
    print(json.dumps(res if a.dry_run else {"files": len(res["files"])}, indent=2, ensure_ascii=False))


def cmd_inspect(a):
    from .data.inspect import inspect
    cfg = _cfg(a)
    res = inspect(cfg)
    m = res["mapping"]
    print(f"{res['total_rows']} registros; coluna de rótulo: {res['label_column']}; {len(res['observed_labels'])} rótulos observados")
    print(f"Saídas: {res['out']}")
    print(f"Mapeamento ({cfg['data']['label_mapping']}): status={m['status']}; sem mapeamento: {m['unknown_labels']}; "
          f"categorias ausentes: {m['absent_categories']}")


def cmd_prepare(a):
    from .data.sample import prepare
    from .environment import resolve_threads
    cfg = _cfg(a)
    res = prepare(cfg, resolve_threads(cfg["experiment"]["threads"]))
    print(json.dumps(res, indent=2, ensure_ascii=False))


def cmd_verify_data(a):
    from .protocol import verify_data
    res = verify_data(_cfg(a))
    print(json.dumps({k: res[k] for k in ("checks", "partition_table", "effective_space_exclusions", "warnings")},
                     indent=2, ensure_ascii=False))


def cmd_freeze(a):
    from .protocol import freeze
    cfg = _cfg(a)
    pdir = freeze(cfg, Path(cfg["output_dir"]))
    print(pdir.name)


def cmd_train(a):
    from .protocol import train
    train(Path(a.runs_dir), a.protocol, MODELS if a.models == ["all"] else a.models, a.seeds or _seeds(a))


def _seeds(a):
    from .protocol import load_protocol
    return load_protocol(Path(a.runs_dir), a.protocol)[1]["config_resolved"]["experiment"]["model_seeds"]


def cmd_calibrate(a):
    from .protocol import calibrate
    calibrate(Path(a.runs_dir), a.protocol)


def cmd_evaluate(a):
    from .protocol import evaluate
    if a.split == "test_confirmatory" and a.release_confirmatory:
        print("ATENÇÃO: liberando o teste confirmatório. O registro de exposição é permanente.", file=sys.stderr)
    evaluate(Path(a.runs_dir), a.protocol, a.split, a.release_confirmatory)


def cmd_report(a):
    from .reporting import report
    print(report(Path(a.runs_dir), a.protocol, a.split))


def cmd_verify_run(a):
    from .protocol import verify_run
    res = verify_run(Path(a.runs_dir), a.protocol)
    print(json.dumps({k: v for k, v in res.items() if k != "test_exposures"}, indent=2, ensure_ascii=False))
    if not res["ok"]:
        sys.exit(1)


def cmd_run_initial(a):
    from .data.inspect import selected_files
    from .data.sample import prepare, prepare_is_current
    from .environment import resolve_threads
    from .protocol import (calibrate, evaluate, find_protocol, freeze, freeze_fingerprint, train, verify_data,
                           verify_run)
    from .provenance import read_json
    from .reporting import report

    cfg = _cfg(a)
    runs = Path(cfg["output_dir"])
    proc = Path(cfg["data"]["processed_dir"])
    schema = proc / "inspection" / "schema.json"
    names = [f["relative_name"] for f in selected_files(cfg)]
    need_inspect = not schema.exists() or read_json(schema)["files"] != names
    need_prepare = need_inspect or not prepare_is_current(cfg)
    seeds = cfg["experiment"]["model_seeds"]
    if a.dry_run:
        from .protocol import SYNTHETIC_BANNER
        print(f"data_kind={cfg['data_kind']}" + (f"  ({SYNTHETIC_BANNER})" if cfg["data_kind"] == "synthetic" else ""))
        print(f"arquivos selecionados: {len(names)}; threads: {resolve_threads(cfg['experiment']['threads'])}")
        steps = [("inspect", need_inspect, f"{proc}/inspection/"), ("prepare", need_prepare, f"{proc}/sample.parquet"),
                 ("verify-data", True, f"{proc}/verify_data.json"), ("freeze", True, f"{runs}/<protocol_id>/"),
                 ("train", True, f"{len(MODELS)} modelos x sementes {seeds}"), ("calibrate", True, "thresholds.json"),
                 ("evaluate test_initial", True, "predictions/test_initial/"), ("report", True, "results_initial.md, figures/"),
                 ("verify-run", True, "verification.json")]
        for name, todo, out in steps:
            print(f"  {'executar' if todo else 'pronto  '}  {name:22s} -> {out}")
        print("  NUNCA executado aqui: evaluate test_confirmatory")
        if not need_prepare:
            dq = read_json(proc / "data_quality.json")["sample"]
            print(f"amostra atual: {dq['selected_total']} vetores; por estrato: {dq['selected_by_stratum']}")
            pdir = find_protocol(runs, freeze_fingerprint(cfg))
            print(f"protocolo existente para retomada: {pdir.name if pdir else 'nenhum (freeze criará um novo)'}")
        return
    if need_inspect:
        from .data.inspect import inspect
        inspect(cfg)
    if need_prepare:
        prepare(cfg, resolve_threads(cfg["experiment"]["threads"]))
    verify_data(cfg)
    pdir = find_protocol(runs, freeze_fingerprint(cfg))
    if pdir:
        print(f"retomando protocolo {pdir.name}")
    else:
        pdir = freeze(cfg, runs)
    pid = pdir.name
    train(runs, pid, MODELS, seeds)
    calibrate(runs, pid)
    evaluate(runs, pid, "test_initial")
    print(report(runs, pid, "test_initial"))
    res = verify_run(runs, pid)
    print(f"protocolo {pid}: verificação {'OK' if res['ok'] else 'FALHOU: ' + str(res['failed'])}")
    if not res["ok"]:
        sys.exit(1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="tc2-iot", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_, config=False, protocol=False):
        s = sub.add_parser(name, help=help_)
        if config:
            s.add_argument("--config", required=True, help="arquivo YAML (ex.: configs/initial.yaml)")
        if protocol:
            s.add_argument("--protocol", required=True, help="ID do protocolo congelado")
            s.add_argument("--runs-dir", default="runs", help="pasta dos protocolos (padrão: runs)")
        s.set_defaults(fn=fn)
        return s

    s = add("doctor", cmd_doctor, "inspeção do ambiente (somente leitura) -> environment.json")
    s.add_argument("--out", default="runs")
    s.add_argument("--data-dir", default="data")
    s = add("import-data", cmd_import, "inventariar e importar CSVs baixados manualmente", config=True)
    s.add_argument("--source", required=True)
    s.add_argument("--mode", choices=["copy", "link"], default="copy")
    s.add_argument("--obtained-at", default=None, help="data do download manual (ISO 8601), se conhecida")
    s = add("download", cmd_download, "baixar URLs autorizadas listadas em arquivo", config=True)
    s.add_argument("--urls-file", required=True)
    s.add_argument("--max-gib", type=float, required=True, help="limite total de download em GiB")
    s.add_argument("--dry-run", action="store_true")
    add("inspect", cmd_inspect, "esquema, rótulos, contagens e qualidade", config=True)
    add("prepare", cmd_prepare, "limpeza, deduplicação global e amostragem", config=True)
    add("verify-data", cmd_verify_data, "verificações da amostra preparada", config=True)
    add("freeze", cmd_freeze, "partições, pré-processamento e congelamento do protocolo", config=True)
    s = add("train", cmd_train, "ajustar modelos (somente benignos)", protocol=True)
    s.add_argument("--models", nargs="+", default=["all"], choices=["all", *MODELS])
    s.add_argument("--seeds", nargs="+", type=int)
    add("calibrate", cmd_calibrate, "limiares em validation_calibration", protocol=True)
    s = add("evaluate", cmd_evaluate, "escores, decisões, métricas e tempos em um teste", protocol=True)
    s.add_argument("--split", required=True, choices=["test_initial", "test_confirmatory"])
    s.add_argument("--release-confirmatory", action="store_true", help="liberação explícita do teste confirmatório")
    s = add("report", cmd_report, "tabelas, figuras e results_*.md", protocol=True)
    s.add_argument("--split", default="test_initial", choices=["test_initial", "test_confirmatory"])
    add("verify-run", cmd_verify_run, "verificações automáticas do protocolo", protocol=True)
    s = add("run-initial", cmd_run_initial, "orquestra até o teste inicial (nunca o confirmatório)", config=True)
    s.add_argument("--dry-run", action="store_true")

    a = p.parse_args(argv)
    from .data.clean import DataBlocked
    try:
        a.fn(a)
    except (DataBlocked, config_mod.ConfigError, FileNotFoundError, PermissionError) as e:
        print(f"BLOQUEADO: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
