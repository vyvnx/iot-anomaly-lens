"""Amostragem estratificada por rótulo original, com prioridade estável por hash criptográfico.

A prioridade de cada vetor é MD5(semente || ':' || chave do vetor), um hash criptográfico estável.
Em cada estrato são selecionados os menores valores: amostragem uniforme sem reposição,
independente da ordem de leitura dos arquivos (a chave não depende dela).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

from ..config import AMBIGUOUS, BENIGN_STRATUM, CATEGORIES, POLICY_VERSION, canonical_hash, sha256_file
from ..provenance import code_version
from .clean import (DataBlocked, build_views, connect, duplicate_stats, exclusion_stats, group_vectors,
                    input_fingerprint, materialize_rows, resolve_inputs)
from .inspect import normalize, selected_files


STRATUM_SQL = f"""CASE WHEN g.all_benign THEN '{BENIGN_STRATUM}'
                         WHEN g.situation = 'B' THEN '{AMBIGUOUS}'
                         ELSE c.category END"""


def strata_caps(cfg: dict) -> dict[str, int]:
    s = cfg["sample"]
    return {BENIGN_STRATUM: s["benign_cap"], **s["category_caps"], AMBIGUOUS: s["ambiguous_cap"]}


def select_sample(con, cfg: dict) -> None:
    """Per stratum keep the `cap` smallest priorities with arg_min(..., n): no global sort of all vectors."""
    seed = cfg["sample"]["seed"]
    # category of single-category groups (consistent or C); benign and B groups get theirs from STRATUM_SQL
    con.execute(f"""
        CREATE OR REPLACE TABLE eligible AS
        SELECT g.key, g.rep, g.multiplicity, g.situation, {STRATUM_SQL} AS stratum
        FROM groups g LEFT JOIN (SELECT cat_no, any_value(category) AS category FROM label_map
                                 WHERE cat_no IS NOT NULL GROUP BY cat_no) c ON c.cat_no = g.cat_min
        WHERE g.situation <> 'A'""")
    con.execute("CREATE OR REPLACE TABLE sampled(key UHUGEINT, rep BIGINT, multiplicity BIGINT, situation VARCHAR, stratum VARCHAR)")
    for stratum, cap in strata_caps(cfg).items():
        if cap == 0:
            continue
        con.execute(f"""
            INSERT INTO sampled
            SELECT x.key, x.rep, x.multiplicity, x.situation, ? FROM (
              SELECT unnest(arg_min(struct_pack(key := key, rep := rep, multiplicity := multiplicity, situation := situation),
                                    struct_pack(p := md5_number('{seed}:' || key::VARCHAR), k := key), {int(cap)})) AS x
              FROM eligible WHERE stratum = ?)""", [stratum, stratum])
    # complete label/category sets of each sampled vector, kept as metadata
    con.execute("""
        CREATE OR REPLACE TABLE sampled_meta AS
        SELECT s.*, string_agg(m.label, '+' ORDER BY m.label) AS labels,
               string_agg(DISTINCT coalesce(m.category, 'Benigno'), '+' ORDER BY coalesce(m.category, 'Benigno')) AS categories,
               bool_and(m.benign) AS benign
        FROM sampled s JOIN pairs p USING (key) JOIN label_map m USING (label_no)
        GROUP BY ALL""")


def export_sample(con, feats: list[str], out: Path) -> int:
    """Join representatives back to the raw records to fetch exact doubles, re-checking the key."""
    norm = [normalize(f) for f in feats]
    con.execute(f"""
        COPY (
          SELECT sha256(v.canon) AS vector_id, v.source_file || '#' || v.row_index AS record_id, v.source_file,
                 v.row_index, s.labels AS label, s.benign,
                 CASE WHEN s.benign OR s.stratum = '{AMBIGUOUS}' THEN NULL ELSE s.stratum END AS category,
                 s.stratum, s.situation, s.categories AS category_set, s.multiplicity,
                 (md5_number(v.canon) = s.key) AS hash_ok,
                 {', '.join('v."' + n + '"' for n in norm)}
          FROM sampled_meta s
          JOIN valid v ON v.file_no = (s.rep // 4294967296)::INTEGER AND v.row_index = (s.rep % 4294967296)::BIGINT
          ORDER BY vector_id
        ) TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
    n_bad, n = con.execute(f"SELECT count(*) FILTER (WHERE NOT hash_ok), count(*) FROM read_parquet('{out.as_posix()}')").fetchone()
    expected = con.execute("SELECT count(*) FROM sampled").fetchone()[0]
    if n_bad or n != expected:
        raise RuntimeError(f"verificação da amostra falhou: {n_bad} chaves divergentes, {n} de {expected} registros")
    return n


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def prepare(cfg: dict, threads: int, log=print) -> dict:
    feats, label_col, mapping = resolve_inputs(cfg)
    proc = Path(cfg["data"]["processed_dir"])
    con = connect(cfg, threads)
    build_views(con, cfg, feats, label_col, mapping)
    log("leitura única: conversão, exclusões e chaves dos vetores")
    materialize_rows(con, feats)
    excl = exclusion_stats(con)
    log("agrupando vetores idênticos (todos os arquivos selecionados)")
    group_vectors(con)
    dup = duplicate_stats(con, proc)
    log("selecionando amostra")
    select_sample(con, cfg)
    sample_p = proc / "sample.parquet"
    n = export_sample(con, feats, sample_p)

    # strata: available distinct vectors, cap, selected and shortfall
    avail = dict(con.execute("SELECT stratum, count(*) FROM eligible GROUP BY 1").fetchall())
    picked = dict(con.execute("SELECT stratum, count(*) FROM sampled GROUP BY 1").fetchall())
    strata = [{"estrato": st, "vetores_distintos_disponiveis": avail.get(st, 0), "teto": cap,
               "amostrados": picked.get(st, 0), "falta_em_relacao_ao_teto": cap - picked.get(st, 0)}
              for st, cap in strata_caps(cfg).items()]
    _write_csv(proc / "strata_counts.csv", strata)
    # composition by original label: single-label vectors and multi-label vectors that contain the label
    comp = con.execute("""
        SELECT m.label, count(*) FILTER (WHERE s.situation = 'consistent'),
               count(*) FILTER (WHERE s.situation = 'C'), count(*) FILTER (WHERE s.situation = 'B')
        FROM sampled s JOIN pairs p USING (key) JOIN label_map m USING (label_no) GROUP BY 1""").fetchall()
    comp = {r[0]: r[1:] for r in comp}
    rows = []
    for label, e in sorted(mapping["labels"].items()):
        el = excl["per_label"].get(label, {"rows": 0, "removed_invalid": 0})
        dl = dup["per_label"].get(label, {})
        c1, c_c, c_b = comp.get(label, (0, 0, 0))
        rows.append({"rotulo_original": label, "benigno": e["benign"], "categoria": e["category"] or "Benigno",
                     "registros_brutos": el["rows"], "removidos_invalidos": el["removed_invalid"],
                     "registros_excluidos_ambiguidade_binaria_A": dl.get("rows_in_A_excluded", 0),
                     "registros_em_grupos_B_mantidos": dl.get("rows_in_B_kept", 0),
                     "registros_em_grupos_C_mantidos": dl.get("rows_in_C_kept", 0),
                     "amostrados_rotulo_unico": c1, "amostrados_multirrotulo_mesma_categoria_C": c_c,
                     "amostrados_multirrotulo_categoria_ambigua_B": c_b})
    _write_csv(proc / "sample_counts.csv", rows)

    cat_counts = {st: picked.get(st, 0) for st in strata_caps(cfg)}
    absent_labels = [r["rotulo_original"] for r in rows
                     if not (r["amostrados_rotulo_unico"] or r["amostrados_multirrotulo_mesma_categoria_C"]
                             or r["amostrados_multirrotulo_categoria_ambigua_B"])]
    warnings = []
    if cfg["data"]["files"] != "all":
        warnings.append("somente parte dos arquivos foi selecionada; a amostra não é probabilística em relação ao CICIoT2023 completo")
    if absent_labels:
        warnings.append(f"rótulos originais sem nenhum vetor na amostra: {absent_labels}")
    short = {r["estrato"]: r["falta_em_relacao_ao_teto"] for r in strata if r["falta_em_relacao_ao_teto"] > 0}
    if short:
        warnings.append(f"estratos abaixo do teto (usada a quantidade disponível): {short}")
    if any(cat_counts[c] == 0 for c in CATEGORIES):
        warnings.append(f"categorias ausentes: {[c for c in CATEGORIES if cat_counts[c] == 0]}; o escopo completo não é coberto")
    quality = {
        "data_kind": cfg["data_kind"],
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "policy_version": POLICY_VERSION,
        "features": feats, "label_column": label_col,
        "invalid_policy": "drop: registros com ausentes, não numéricos ou não finitos nos atributos aprovados são removidos",
        "exclusions": excl, "duplicates": dup,
        "sample": {"caps": strata_caps(cfg), "seed": cfg["sample"]["seed"],
                   "method": "prioridade MD5(semente:chave do vetor), menores valores por estrato (top-k), sem reposição",
                   "strata": "Benigno, as 7 categorias (inclui vetores multirrótulo C) e attack_category_ambiguous (B)",
                   "note": "tetos são desenho experimental, não distribuição natural da base",
                   "selected_total": n, "selected_by_stratum": cat_counts},
        "analysis_unit": "vetores de atributos distintos (não é a distribuição natural do tráfego)",
        "warnings": warnings,
    }
    (proc / "data_quality.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")
    state = {
        "input_fingerprint": input_fingerprint(cfg, feats, label_col, mapping),
        "sample_sha256": sha256_file(sample_p),
        "code_version": code_version(),
        "policy_version": POLICY_VERSION,
        "files": [f["relative_name"] for f in selected_files(cfg)],
        "features": feats, "label_column": label_col, "mapping_hash": canonical_hash(mapping),
    }
    (proc / "prepare_state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    con.close()
    # the work database is rebuilt from staging on every run; its statistics are in data_quality.json
    (proc / "work.duckdb").unlink(missing_ok=True)
    return {"sample": str(sample_p), "n": n, "by_stratum": cat_counts, "warnings": warnings}


def prepare_is_current(cfg: dict) -> bool:
    proc = Path(cfg["data"]["processed_dir"])
    st, sp = proc / "prepare_state.json", proc / "sample.parquet"
    if not (st.exists() and sp.exists()):
        return False
    state = json.loads(st.read_text(encoding="utf-8"))
    try:
        feats, label_col, mapping = resolve_inputs(cfg)
    except DataBlocked:
        return False
    return (state["input_fingerprint"] == input_fingerprint(cfg, feats, label_col, mapping)
            and state["sample_sha256"] == sha256_file(sp))
