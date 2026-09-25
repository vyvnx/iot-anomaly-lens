"""Limpeza e deduplicação global em disco (DuckDB).

Identidade do vetor: SHA-256 da forma canônica dos atributos aprovados, em ordem fixa,
com -0.0 normalizado para 0.0. A forma canônica usa a conversão double->texto do DuckDB,
que é de ida e volta exata (verificado em testes); colisões são verificadas comparando
o texto canônico mínimo e máximo de cada grupo (flag hash_collision). O rótulo nunca entra no hash.
"""

from __future__ import annotations

from pathlib import Path

import psutil
import yaml

from ..config import AMBIGUOUS, CATEGORIES, POLICY_VERSION, canonical_hash
from .inspect import IDENTIFIER_PATTERN, load_label_mapping, mapping_problems, normalize, q, read_header, selected_files, staged_glob


class DataBlocked(RuntimeError):
    """Raised when the experiment must stop until the author resolves an input."""


def load_allowlist(cfg: dict, columns: list[str], label_col: str) -> list[str]:
    p = Path(cfg["data"]["feature_allowlist"])
    if not p.exists():
        raise DataBlocked(f"{p} não existe; revise a proposta gerada por inspect e grave-a com status: confirmed")
    a = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if a.get("status") != "confirmed":
        raise DataBlocked(f"{p} não está com status: confirmed")
    feats = a.get("features") or []
    if a.get("pending_decisions"):
        raise DataBlocked(f"{p} ainda tem pending_decisions: {list(a['pending_decisions'])}")
    missing = [f for f in feats if f not in columns]
    if missing:
        raise DataBlocked(f"atributos da allowlist ausentes nos CSVs: {missing}")
    if label_col in feats:
        raise DataBlocked("a coluna de rótulo não pode estar na allowlist")
    forbidden = [f for f in feats if IDENTIFIER_PATTERN.search(normalize(f))]
    if forbidden:
        raise DataBlocked(f"atributos proibidos (identificadores de linha/arquivo/captura) na allowlist: {forbidden}")
    if len(set(feats)) != len(feats) or len({normalize(f) for f in feats}) != len(feats):
        raise DataBlocked("allowlist com atributos duplicados")
    if not feats:
        raise DataBlocked("allowlist vazia")
    return feats


def load_confirmed_mapping(cfg: dict, observed: list[str]) -> dict:
    m = load_label_mapping(cfg["data"]["label_mapping"])
    if m is None or m.get("status") != "confirmed":
        raise DataBlocked(f"{cfg['data']['label_mapping']} ausente ou sem status: confirmed")
    probs = mapping_problems(m, observed)
    if probs["unknown_labels"]:
        raise DataBlocked(f"rótulos sem mapeamento (o experimento fica bloqueado): {probs['unknown_labels']}")
    if not probs["has_benign"]:
        raise DataBlocked("nenhum rótulo benigno observado")
    return m


def connect(cfg: dict, threads: int):
    """Persistent work database on disk: intermediate tables never have to fit in RAM."""
    import duckdb

    work = Path(cfg["data"]["processed_dir"])
    tmp = work / "duckdb_tmp"
    db = work / "work.duckdb"
    for p in (db, db.with_suffix(".duckdb.wal")):
        p.unlink(missing_ok=True)  # always rebuilt from staging; never reused
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db))
    vm = psutil.virtual_memory()
    mem = int(min(vm.available * 0.6, vm.total * 0.5))
    con.execute(f"SET memory_limit='{mem // (1 << 20)}MB'")
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    con.execute(f"SET threads={threads}")
    con.execute("SET preserve_insertion_order=false")
    return con


def build_views(con, cfg: dict, feats: list[str], label_col: str, mapping: dict) -> None:
    files = selected_files(cfg)
    paths = staged_glob(cfg, files)
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise DataBlocked("staging ausente; execute inspect antes de prepare")
    con.execute("CREATE OR REPLACE TABLE label_map(label_no INTEGER, label VARCHAR, benign BOOLEAN, category VARCHAR, cat_no INTEGER)")
    con.executemany("INSERT INTO label_map VALUES (?, ?, ?, ?, ?)",
                    [(i, k, v["benign"], v["category"], None if v["benign"] else CATEGORIES.index(v["category"]))
                     for i, (k, v) in enumerate(sorted(mapping["labels"].items()))])
    con.execute("CREATE OR REPLACE TABLE files(file_no INTEGER, source_file VARCHAR)")
    con.executemany("INSERT INTO files VALUES (?, ?)", [(i, f["relative_name"]) for i, f in enumerate(files)])
    raw = [q(f) for f in feats]
    norm = [q(normalize(f)) for f in feats]
    # cast once in an inner select; flags and values reuse the casts
    casts = ", ".join(f"TRY_CAST({r} AS DOUBLE) AS {n}, {r} AS {q('__s_' + normalize(f))}"
                      for r, n, f in zip(raw, norm, feats))
    s = [q("__s_" + normalize(f)) for f in feats]
    missing_any = " OR ".join(f"{x} IS NULL OR trim({x}) = ''" for x in s)
    nonnum_any = " OR ".join(f"({x} IS NOT NULL AND trim({x}) <> '' AND {n} IS NULL)" for x, n in zip(s, norm))
    nonfinite_any = " OR ".join(f"NOT isfinite({n})" for n in norm)
    values = ", ".join(f"CASE WHEN {n} = 0 THEN 0.0::DOUBLE ELSE {n} END AS {n}" for n in norm)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW records AS
        SELECT f.file_no, c.__source_file AS source_file, c.__row_index AS row_index, c.label, m.label_no,
               coalesce(c.label IS NULL OR trim(c.label) = '', false) AS r_label_missing,
               coalesce({missing_any}, false) AS r_missing,
               coalesce({nonnum_any}, false) AS r_non_numeric,
               coalesce({nonfinite_any}, false) AS r_non_finite,
               {values}
        FROM (SELECT __source_file, __row_index, {q(label_col)} AS label, {casts} FROM read_parquet({paths!r})) c
        JOIN files f ON f.source_file = c.__source_file
        LEFT JOIN label_map m ON m.label = c.label""")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW valid AS
        SELECT *, {canon_expr(feats)} AS canon FROM records
        WHERE NOT (r_label_missing OR r_missing OR r_non_numeric OR r_non_finite)""")


def canon_expr(feats: list[str]) -> str:
    return "concat_ws('|', " + ", ".join(f"{q(normalize(f))}::VARCHAR" for f in feats) + ")"


def materialize_rows(con, feats: list[str]) -> None:
    """Single pass over the raw records into a narrow, fixed-width table on disk.

    key = md5 (128 bits) of the canonical text; chk = xxhash (64 bits) of the same text,
    used only to detect key collisions between different vectors.
    """
    con.execute(f"""
        CREATE OR REPLACE TABLE rows AS
        SELECT file_no, row_index, label_no, r_label_missing, r_missing, r_non_numeric, r_non_finite,
               CASE WHEN NOT (r_label_missing OR r_missing OR r_non_numeric OR r_non_finite) THEN md5_number(canon) END AS key,
               CASE WHEN NOT (r_label_missing OR r_missing OR r_non_numeric OR r_non_finite) THEN hash(canon) END AS chk
        FROM (SELECT *, {canon_expr(feats)} AS canon FROM records)""")
    if con.execute("SELECT count(*) FROM rows WHERE label_no IS NULL AND NOT r_label_missing").fetchone()[0]:
        raise DataBlocked("rótulos sem mapeamento encontrados na leitura completa")


def exclusion_stats(con) -> dict:
    rows = con.execute("""
        SELECT m.label, m.category, m.benign, r_label_missing, r_missing, r_non_numeric, r_non_finite, count(*)
        FROM rows r LEFT JOIN label_map m USING (label_no) GROUP BY ALL ORDER BY ALL""").fetchall()
    by_reason = {"label_missing": 0, "missing": 0, "non_numeric": 0, "non_finite": 0}
    combos, per_label, total, removed = {}, {}, 0, 0
    for label, cat, benign, *flags, n in rows:
        total += n
        names = [k for k, f in zip(by_reason, flags) if f]
        per_label.setdefault(label, {"category": cat if not benign else "Benigno", "rows": 0, "removed_invalid": 0})
        per_label[label]["rows"] += n
        if names:
            removed += n
            per_label[label]["removed_invalid"] += n
            combos["+".join(names)] = combos.get("+".join(names), 0) + n
            for k in names:
                by_reason[k] += n
    return {"total_rows": total, "removed_unique": removed, "by_reason_overlapping": by_reason,
            "by_reason_combination": combos, "per_label": per_label}


SITUATION_SQL = """CASE WHEN g.n_labels = 1 THEN 'consistent'
                         WHEN g.has_benign AND NOT g.all_benign THEN 'A'
                         WHEN g.all_benign THEN 'consistent'
                         WHEN g.cat_min = g.cat_max THEN 'C' ELSE 'B' END"""


def group_vectors(con) -> None:
    """One row per distinct feature vector, over all selected files jointly (fixed-size states only).

    situation: consistent (one label), A (benign and attack: binary ambiguity, excluded),
    B (attacks of different categories: kept, category ambiguous), C (attacks of one category: kept).
    """
    # representative = smallest (file, row); file_no follows the sorted file names, not the read order
    con.execute("""
        CREATE OR REPLACE TABLE pairs AS
        SELECT key, label_no, count(*) AS n, min(file_no::BIGINT * 4294967296 + row_index) AS rep,
               min(chk) AS chk_min, max(chk) AS chk_max
        FROM rows WHERE key IS NOT NULL GROUP BY key, label_no""")
    con.execute(f"""
        CREATE OR REPLACE TABLE groups AS
        SELECT g.*, {SITUATION_SQL} AS situation FROM (
          SELECT p.key, count(*) AS n_labels, min(p.label_no) AS label_no, sum(p.n) AS multiplicity, min(p.rep) AS rep,
                 min(p.chk_min) <> max(p.chk_max) AS hash_collision,
                 bool_or(m.benign) AS has_benign, bool_and(m.benign) AS all_benign,
                 min(m.cat_no) AS cat_min, max(m.cat_no) AS cat_max
          FROM pairs p JOIN label_map m USING (label_no) GROUP BY p.key) g""")


def duplicate_stats(con, out_dir: Path) -> dict:
    collisions = con.execute("SELECT count(*) FROM groups WHERE hash_collision").fetchone()[0]
    if collisions:
        raise RuntimeError(f"{collisions} colisões de hash com vetores diferentes; interrompido")
    distinct, valid_rows = con.execute("SELECT count(*), sum(multiplicity) FROM groups").fetchone()
    by_sit = {s: {"groups": g, "rows": int(r)} for s, g, r in con.execute(
        "SELECT situation, count(*), sum(multiplicity) FROM groups GROUP BY 1 ORDER BY 1").fetchall()}
    # per-label impact; multi-label groups are few, so the join has a small build side
    per_label = con.execute("""
        WITH multi AS (SELECT key, situation FROM groups WHERE n_labels > 1),
             lab AS (SELECT label_no, sum(n) AS rows_valid, count(*) AS pairs FROM pairs GROUP BY label_no),
             lab_m AS (SELECT p.label_no,
                              sum(p.n) FILTER (WHERE situation = 'A') AS rows_a, count(*) FILTER (WHERE situation = 'A') AS vec_a,
                              sum(p.n) FILTER (WHERE situation = 'B') AS rows_b, count(*) FILTER (WHERE situation = 'B') AS vec_b,
                              sum(p.n) FILTER (WHERE situation = 'C') AS rows_c, count(*) FILTER (WHERE situation = 'C') AS vec_c
                       FROM pairs p JOIN multi USING (key) GROUP BY p.label_no)
        SELECT m.label, lab.rows_valid, lab.pairs, coalesce(rows_a, 0), coalesce(vec_a, 0), coalesce(rows_b, 0),
               coalesce(vec_b, 0), coalesce(rows_c, 0), coalesce(vec_c, 0)
        FROM lab JOIN label_map m USING (label_no) LEFT JOIN lab_m USING (label_no) ORDER BY m.label""").fetchall()
    # every excluded row keeps its identifiers and labels in the exclusion report
    excluded = out_dir / "excluded_binary_ambiguous_rows.parquet"
    con.execute(f"""
        COPY (SELECT f.source_file, r.row_index, m.label, coalesce(m.category, 'Benigno') AS category,
                     sha256(r.key::VARCHAR) AS group_id
              FROM rows r JOIN (SELECT key FROM groups WHERE situation = 'A') a USING (key)
              JOIN files f USING (file_no) JOIN label_map m USING (label_no)
              ORDER BY group_id, f.source_file, r.row_index)
        TO '{excluded.as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
    return {
        "identity": "chave md5 (128 bits) do texto canônico dos atributos (ordem fixa, -0.0 normalizado), sem rótulo "
                    "nem arquivo; colisões verificadas por xxhash (64 bits) do mesmo texto; vector_id publicado = "
                    "sha256 do texto canônico",
        "hash_collisions": collisions,
        "valid_rows": int(valid_rows or 0),
        "distinct_vectors": distinct,
        "duplicate_rows_collapsed": int((valid_rows or 0) - distinct),
        "by_situation": by_sit,
        "policy": {
            "version": POLICY_VERSION,
            "A": "benigno e ataque no mesmo vetor: grupo inteiro excluído por ambiguidade binária na representação "
                 "adotada (não se afirma que os rótulos originais estejam errados)",
            "B": f"ataques de categorias diferentes: mantidos como ataque, estrato {AMBIGUOUS} (não é uma 8ª categoria)",
            "C": "ataques de uma mesma categoria: mantidos no estrato da categoria",
        },
        "excluded_rows_file": excluded.name,
        "per_label": {r[0]: {"rows_valid": int(r[1]), "vectors_with_label": int(r[2]),
                             "rows_in_A_excluded": int(r[3]), "vectors_in_A": int(r[4]),
                             "rows_in_B_kept": int(r[5]), "vectors_in_B": int(r[6]),
                             "rows_in_C_kept": int(r[7]), "vectors_in_C": int(r[8])}
                      for r in per_label},
    }


def input_fingerprint(cfg: dict, feats: list[str], label_col: str, mapping: dict) -> str:
    files = selected_files(cfg)
    return canonical_hash({
        "files": [(f["relative_name"], f["sha256"]) for f in files], "features": feats, "label_column": label_col,
        "mapping": mapping, "data": {k: v for k, v in cfg["data"].items()},
        # the per-category minimum is checked by verify-data and does not change the sample
        "sample": {k: v for k, v in cfg["sample"].items() if k != "min_attack_per_category_per_test"},
    })


def resolve_inputs(cfg: dict) -> tuple[list[str], str, dict]:
    import json

    schema_p = Path(cfg["data"]["processed_dir"]) / "inspection" / "schema.json"
    quality_p = schema_p.with_name("quality_report.json")
    if not schema_p.exists():
        raise DataBlocked("execute inspect antes de prepare")
    schema = json.loads(schema_p.read_text(encoding="utf-8"))
    if schema["files"] != [f["relative_name"] for f in selected_files(cfg)]:
        raise DataBlocked("a inspeção foi feita sobre outra seleção de arquivos; execute inspect novamente")
    label_col = schema["label_column"]
    columns = read_header(Path(cfg["data"]["raw_dir"]) / schema["files"][0])
    feats = load_allowlist(cfg, columns, label_col)
    observed = json.loads(quality_p.read_text(encoding="utf-8"))["observed_labels"]
    mapping = load_confirmed_mapping(cfg, observed)
    return feats, label_col, mapping
