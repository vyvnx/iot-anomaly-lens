"""Inspeção dos CSVs: esquema, rótulos, contagens e qualidade, sem carregar tudo em RAM."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import yaml

from ..config import CATEGORIES, sha256_file

# proposal only: the author must confirm against official material before status becomes confirmed
CATEGORY_RULES = [
    ("prefix", "DDoS", "DDoS"), ("prefix", "DoS", "DoS"), ("prefix", "Recon", "Recon"),
    ("prefix", "Mirai", "Mirai"), ("contains", "VulnerabilityScan", "Recon"),
    ("contains", "BruteForce", "Brute Force"), ("contains", "Spoofing", "Spoofing"),
    ("contains", "SqlInjection", "Web-based"), ("contains", "XSS", "Web-based"),
    ("contains", "CommandInjection", "Web-based"), ("contains", "Backdoor", "Web-based"),
    ("contains", "Uploading", "Web-based"), ("contains", "BrowserHijacking", "Web-based"),
]

# interpretation by the agent from the CICIoT2023 naming; to be checked against the official paper/readme
KNOWN_DESCRIPTIONS = {
    "flow_duration": "duração do fluxo", "header_length": "comprimento de cabeçalho",
    "protocol_type": "tipo de protocolo (codificado numericamente)", "duration": "tempo de vida (TTL)", "time_to_live": "tempo de vida (TTL)",
    "rate": "taxa de transmissão de pacotes", "srate": "taxa de pacotes de saída",
    "drate": "taxa de pacotes de entrada", "ack_count": "contagem de pacotes com ACK",
    "syn_count": "contagem de pacotes com SYN", "fin_count": "contagem de pacotes com FIN",
    "urg_count": "contagem de pacotes com URG", "rst_count": "contagem de pacotes com RST",
    "tot_sum": "soma dos comprimentos de pacote", "min": "comprimento mínimo de pacote",
    "max": "comprimento máximo de pacote", "avg": "comprimento médio de pacote",
    "std": "desvio-padrão do comprimento de pacote", "tot_size": "tamanho de pacote",
    "iat": "tempo entre chegadas de pacotes", "number": "número de pacotes no fluxo",
    "magnitue": "magnitude (médias de entrada/saída)", "radius": "raio (variâncias de entrada/saída)",
    "covariance": "covariância entrada/saída", "variance": "razão de variâncias entrada/saída",
    "weight": "peso (produto de contagens entrada/saída)",
}
FLAG_SUFFIX = "_flag_number"
PROTOCOL_INDICATORS = {"http", "https", "dns", "telnet", "smtp", "ssh", "irc", "tcp", "udp",
                       "dhcp", "arp", "icmp", "ipv", "llc", "igmp"}
IDENTIFIER_PATTERN = re.compile(r"(^|_)(id|index|idx|unnamed|file|filename|timestamp|ts|src_ip|dst_ip|ip|mac|port)($|_)")
# columns flagged for explicit author decision: possible proxies of the capture process
CAPTURE_PROXY_SUSPECTS = {"iat": "valores podem refletir o instante/processo de captura; decisão do autor necessária"}


def normalize(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", name.strip().lower()).strip("_")


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def selected_files(cfg: dict) -> list[dict]:
    manifest_path = Path(cfg["data"]["source_manifest"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} não existe; execute import-data ou download primeiro")
    files = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    sel = cfg["data"]["files"]
    if sel != "all":
        by_name = {f["relative_name"]: f for f in files}
        missing = [s for s in sel if s not in by_name]
        if missing:
            raise FileNotFoundError(f"arquivos selecionados ausentes do manifesto: {missing}")
        files = [by_name[s] for s in sel]
    if not files:
        raise FileNotFoundError("nenhum arquivo selecionado")
    return sorted(files, key=lambda f: f["relative_name"])


def verify_raw(cfg: dict, files: list[dict]) -> None:
    raw = Path(cfg["data"]["raw_dir"])
    for f in files:
        p = raw / f["relative_name"]
        if not p.exists():
            raise FileNotFoundError(f"arquivo bruto ausente: {p}")
        if sha256_file(p) != f["sha256"]:
            raise ValueError(f"SHA-256 de {p} difere do raw_manifest.json; arquivo alterado")


def read_header(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return next(csv.reader(f))


def check_headers(headers: dict[str, list[str]]) -> list[str]:
    first_name, first = next(iter(headers.items()))
    for name, h in headers.items():
        if h != first:
            raise ValueError(f"cabeçalho de {name} difere de {first_name}: "
                             f"somente em um: {sorted(set(h) ^ set(first))}; ordem igual: {sorted(h) == sorted(first)}")
    dup = {c for c in first if first.count(c) > 1}
    if dup:
        raise ValueError(f"nomes de coluna duplicados: {sorted(dup)}")
    norm = [normalize(c) for c in first]
    dup_n = {n for n in norm if norm.count(n) > 1 or not n}
    if dup_n:
        raise ValueError(f"nomes de coluna duplicados/vazios após normalização: {sorted(dup_n)}")
    return first


def stage_file(src: Path, dst: Path, relative_name: str, columns: list[str]) -> dict:
    """Stream one CSV into parquet (all strings) adding source file and 0-based data-row index."""
    bad_rows = []

    def on_invalid(row):
        bad_rows.append({"line": row.number, "expected_columns": row.expected_columns, "actual_columns": row.actual_columns})
        return "skip"

    reader = pacsv.open_csv(
        src,
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        parse_options=pacsv.ParseOptions(invalid_row_handler=on_invalid),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in columns},
                                             strings_can_be_null=True, quoted_strings_can_be_null=False),
    )
    tmp = dst.with_suffix(".parquet.part")
    writer, offset = None, 0
    for batch in reader:
        n = batch.num_rows
        tbl = pa.Table.from_batches([batch]).select(columns)
        tbl = tbl.append_column("__source_file", pa.array([relative_name] * n, pa.string()))
        tbl = tbl.append_column("__row_index", pa.array(range(offset, offset + n), pa.int64()))
        offset += n
        if writer is None:
            writer = pq.ParquetWriter(tmp, tbl.schema, compression="zstd")
        writer.write_table(tbl)
    if writer is None:
        raise ValueError(f"{relative_name} não contém registros")
    writer.close()
    tmp.rename(dst)
    return {"rows_read": offset, "parse_errors": len(bad_rows), "parse_error_examples": bad_rows[:20]}


def stage_all(cfg: dict, files: list[dict], columns: list[str], log=print) -> dict:
    """Staging is keyed by file SHA-256, so unchanged files are not re-read."""
    staging = Path(cfg["data"]["processed_dir"]) / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    info = {}
    for i, f in enumerate(files, 1):
        dst = staging / f"{f['sha256'][:32]}.parquet"
        meta = dst.with_suffix(".json")
        if dst.exists() and meta.exists():
            info[f["relative_name"]] = json.loads(meta.read_text())
            continue
        log(f"[{i}/{len(files)}] lendo {f['relative_name']}")
        res = stage_file(Path(cfg["data"]["raw_dir"]) / f["relative_name"], dst, f["relative_name"], columns)
        meta.write_text(json.dumps(res, indent=2))
        info[f["relative_name"]] = res
    return info


def staged_glob(cfg: dict, files: list[dict]) -> list[str]:
    staging = Path(cfg["data"]["processed_dir"]) / "staging"
    return [str(staging / f"{f['sha256'][:32]}.parquet") for f in files]


def detect_label_column(con, paths: list[str], columns: list[str], configured: str) -> str:
    if configured != "auto":
        if configured not in columns:
            raise ValueError(f"data.label_column={configured!r} não existe; colunas: {columns}")
        return configured
    candidates = []
    for c in columns:
        n, bad = con.execute(
            f"SELECT count({q(c)}), count({q(c)}) - count(TRY_CAST({q(c)} AS DOUBLE)) "
            f"FROM (SELECT {q(c)} FROM read_parquet(?) LIMIT 100000)", [paths]).fetchone()
        if n and bad / n > 0.5:
            candidates.append(c)
    if len(candidates) != 1:
        raise ValueError(f"não foi possível identificar a coluna de rótulo sem ambiguidade (candidatas: {candidates}); "
                         "defina data.label_column na configuração")
    return candidates[0]


def load_label_mapping(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    m = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    for label, e in (m.get("labels") or {}).items():
        if not isinstance(e, dict) or not isinstance(e.get("benign"), bool):
            raise ValueError(f"mapeamento inválido para {label!r}: requer benign: true/false")
        if e["benign"] and e.get("category") is not None:
            raise ValueError(f"rótulo benigno {label!r} não pode ter categoria de ataque")
        if not e["benign"] and e.get("category") not in CATEGORIES:
            raise ValueError(f"rótulo {label!r}: categoria {e.get('category')!r} fora de {CATEGORIES}")
    return m


def propose_category(label: str) -> tuple[bool | None, str | None]:
    if "benign" in label.lower():
        return True, None
    for kind, key, cat in CATEGORY_RULES:
        if (kind == "prefix" and label.lower().startswith(key.lower())) or (kind == "contains" and key.lower() in label.lower()):
            return False, cat
    return None, None


def mapping_problems(mapping: dict | None, observed: list[str]) -> dict:
    """Unknown labels, absent mapped labels and absent categories."""
    if mapping is None:
        return {"status": "ausente", "unknown_labels": observed, "absent_labels": [], "absent_categories": CATEGORIES,
                "has_benign": False}
    labels = mapping.get("labels") or {}
    unknown = sorted(set(observed) - set(labels))
    absent = sorted(set(labels) - set(observed))
    present_cats = {labels[l]["category"] for l in observed if l in labels and not labels[l]["benign"]}
    return {
        "status": mapping.get("status"),
        "unknown_labels": unknown,
        "absent_labels": absent,
        "absent_categories": [c for c in CATEGORIES if c not in present_cats],
        "has_benign": any(labels[l]["benign"] for l in observed if l in labels),
    }


def feature_catalog(columns: list[str], label_col: str, numeric_frac: dict[str, float]) -> list[dict]:
    rows = []
    for c in columns:
        n = normalize(c)
        desc, origin = KNOWN_DESCRIPTIONS.get(n), "interpretação do agente pela nomenclatura; conferir no material oficial"
        if n.endswith(FLAG_SUFFIX):
            desc = f"indicador/proporção da flag TCP {n[:-len(FLAG_SUFFIX)].upper()}"
        elif n in PROTOCOL_INDICATORS:
            desc = f"indicador de protocolo {n.upper()}"
        if desc is None:
            origin = "sem descrição conhecida"
        if c == label_col:
            include, reason = False, "coluna de rótulo: nunca entra em X"
        elif IDENTIFIER_PATTERN.search(n):
            include, reason = False, "possível identificador de linha/arquivo/captura: excluído (revisar)"
        elif numeric_frac.get(c, 0) < 0.99:
            include, reason = False, f"não numérica em {100 * (1 - numeric_frac.get(c, 0)):.2f}% dos valores"
        elif n in CAPTURE_PROXY_SUSPECTS:
            include, reason = True, "PENDENTE DE DECISÃO: " + CAPTURE_PROXY_SUSPECTS[n]
        else:
            include, reason = True, "característica de tráfego extraída"
        rows.append({"original_name": c, "normalized_name": n, "description": desc or "",
                     "description_origin": origin, "observed_type": "numeric" if numeric_frac.get(c, 0) >= 0.99 else "text",
                     "numeric_fraction": round(numeric_frac.get(c, 0), 6), "proposed_include": include, "reason": reason})
    return rows


def inspect(cfg: dict, log=print) -> dict:
    files = selected_files(cfg)
    verify_raw(cfg, files)
    raw = Path(cfg["data"]["raw_dir"])
    headers = {f["relative_name"]: read_header(raw / f["relative_name"]) for f in files}
    columns = check_headers(headers)
    stage_info = stage_all(cfg, files, columns, log)
    paths = staged_glob(cfg, files)
    out = Path(cfg["data"]["processed_dir"]) / "inspection"
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    label_col = detect_label_column(con, paths, columns, cfg["data"]["label_column"])
    src = f"read_parquet({paths!r})"

    # counts per file and original label
    counts = con.execute(f"SELECT __source_file, {q(label_col)}, count(*) FROM {src} GROUP BY ALL ORDER BY ALL").fetchall()
    observed = sorted({r[1] for r in counts if r[1] is not None})
    mapping = load_label_mapping(cfg["data"]["label_mapping"])
    labels_map = (mapping or {}).get("labels") or {}
    with open(out / "raw_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "rotulo_original", "benigno", "categoria", "registros"])
        for file, label, n in counts:
            e = labels_map.get(label)
            w.writerow([file, label, "" if e is None else e["benign"],
                        "NÃO MAPEADO" if e is None else (e["category"] or "Benigno"), n])

    # per-column quality over all rows of all selected files
    feats = [c for c in columns if c != label_col]
    quality_cols, numeric_frac = {}, {}
    exprs = []
    for c in feats:
        x, d = q(c), f"TRY_CAST({q(c)} AS DOUBLE)"
        exprs += [f"count(*) FILTER (WHERE {x} IS NULL OR trim({x}) = '')",
                  f"count(*) FILTER (WHERE {x} IS NOT NULL AND trim({x}) <> '' AND {d} IS NULL)",
                  f"count(*) FILTER (WHERE isnan({d}))", f"count(*) FILTER (WHERE isinf({d}))"]
    total, label_null, *vals = con.execute(
        f"SELECT count(*), count(*) FILTER (WHERE {q(label_col)} IS NULL OR trim({q(label_col)}) = ''), {', '.join(exprs)} FROM {src}"
    ).fetchone()
    for i, c in enumerate(feats):
        missing, nonnum, nan, inf = vals[4 * i: 4 * i + 4]
        quality_cols[c] = {"missing": missing, "non_numeric": nonnum, "nan": nan, "infinite": inf}
        numeric_frac[c] = 1 - nonnum / max(1, total - missing)

    catalog = feature_catalog(columns, label_col, numeric_frac)
    with open(out / "feature_catalog.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(catalog[0]))
        w.writeheader()
        w.writerows(catalog)

    schema = {
        "files": [f["relative_name"] for f in files],
        "label_column": label_col,
        "label_column_detection": "configurada" if cfg["data"]["label_column"] != "auto" else "automática (única coluna não numérica)",
        "columns": [{"original_name": r["original_name"], "normalized_name": r["normalized_name"],
                     "observed_type": r["observed_type"], "description": r["description"],
                     "description_origin": r["description_origin"]} for r in catalog],
        "provenance_columns_found": [r["original_name"] for r in catalog if IDENTIFIER_PATTERN.search(r["normalized_name"])],
    }
    (out / "schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")

    problems = mapping_problems(mapping, observed)
    quality = {
        "total_rows": total, "files": len(files), "label_missing": label_null,
        "parse_errors_by_file": {k: v["parse_errors"] for k, v in stage_info.items()},
        "parse_error_examples": {k: v["parse_error_examples"] for k, v in stage_info.items() if v["parse_errors"]},
        "columns": quality_cols,
        "observed_labels": observed,
        "label_mapping_check": problems,
        "note": "duplicações e conflitos globais são calculados em prepare (data_quality.json)",
    }
    (out / "quality_report.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")

    # proposals for the author to review; never used until copied and confirmed
    proposal = {"status": "proposed",
                "source": "proposta automática por regras de nome (CATEGORY_RULES); conferir com a documentação oficial do CICIoT2023",
                "labels": {}}
    for lab in observed:
        benign, cat = propose_category(lab)
        proposal["labels"][lab] = {"benign": benign, "category": cat} if benign is not None else {"benign": None, "category": None, "REVISAR": True}
    (out / "label_mapping.proposed.yaml").write_text(yaml.safe_dump(proposal, allow_unicode=True, sort_keys=False), encoding="utf-8")
    allow = {"status": "proposed", "label_column": label_col,
             "features": [r["original_name"] for r in catalog if r["proposed_include"]],
             "excluded": {r["original_name"]: r["reason"] for r in catalog if not r["proposed_include"]},
             "pending_decisions": {r["original_name"]: r["reason"] for r in catalog if r["reason"].startswith("PENDENTE")}}
    (out / "feature_allowlist.proposed.yaml").write_text(yaml.safe_dump(allow, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {"out": str(out), "total_rows": total, "label_column": label_col, "observed_labels": observed,
            "mapping": problems}
