"""Auditoria dos grupos com rótulos conflitantes (com e sem IAT).

A prévia das partições e das colisões pós-transformação agora é feita por `tc2-iot verify-data`.

Não altera a amostra nem congela protocolo; não usa modelos nem métricas.
Uso: python scripts/audit_conflicts.py --config configs/initial.yaml
Saídas: <processed_dir>/audit/*.json|csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from tc2_iot import config as config_mod
from tc2_iot.data.clean import build_views, canon_expr, connect, resolve_inputs
from tc2_iot.data.inspect import q, read_header, selected_files, staged_glob
from tc2_iot.environment import resolve_threads
from tc2_iot.provenance import code_version, now

EXTRA = "IAT"  # attribute excluded by decision D3; audited only to measure the decision's consequence


def situation_sql() -> str:
    # A: benign together with attack; B: attacks from different categories; C: attacks from the same category
    return """CASE WHEN bool_or(m.benign) THEN 'A_benigno_e_ataque'
                   WHEN count(DISTINCT m.category) > 1 THEN 'B_ataques_categorias_diferentes'
                   ELSE 'C_ataques_mesma_categoria' END"""


def full_audit(cfg: dict, out: Path, log=print) -> dict:
    feats, label_col, mapping = resolve_inputs(cfg)
    header = read_header(Path(cfg["data"]["raw_dir"]) / json.loads(
        (Path(cfg["data"]["processed_dir"]) / "inspection" / "schema.json").read_text())["files"][0])
    feats39 = [c for c in header if c in feats or c == EXTRA]  # header order, same as the allowlist order
    assert [f for f in feats39 if f != EXTRA] == feats, "ordem dos atributos difere da allowlist"
    con = connect(cfg, resolve_threads(cfg["experiment"]["threads"]))
    build_views(con, cfg, feats39, label_col, mapping)  # validity flags over 39 columns
    log("leitura única: chaves com e sem IAT")
    con.execute(f"""
        CREATE OR REPLACE TABLE rows AS
        SELECT label_no, r_label_missing OR r_missing OR r_non_numeric OR r_non_finite AS invalid,
               md5_number({canon_expr(feats)}) AS k38, hash({canon_expr(feats)}) AS c38,
               md5_number({canon_expr(feats39)}) AS k39, hash({canon_expr(feats39)}) AS c39
        FROM records""")
    # validity with IAT must match validity without it, otherwise the two universes differ
    inv39 = con.execute("SELECT count(*) FILTER (WHERE invalid) FROM rows").fetchone()[0]
    inv_iat_only = None
    if EXTRA in header:
        iat = q(EXTRA)
        inv_iat_only = con.execute(f"""
            SELECT count(*) FROM (SELECT {iat} AS s, TRY_CAST({iat} AS DOUBLE) AS d
                                  FROM read_parquet({staged_glob(cfg, selected_files(cfg))!r}))
            WHERE s IS NULL OR trim(s) = '' OR d IS NULL OR NOT isfinite(d)""").fetchone()[0]
    con.execute("DELETE FROM rows WHERE invalid")
    res = {"valid_rows": con.execute("SELECT count(*) FROM rows").fetchone()[0],
           "invalid_rows_39_columns": inv39, "iat_invalid_values_total": inv_iat_only}
    for k, c in (("k38", "c38"), ("k39", "c39")):
        log(f"agrupando por {k}")
        con.execute(f"""CREATE OR REPLACE TABLE p_{k} AS
            SELECT {k} AS key, label_no, count(*) AS n, min({c}) AS cmin, max({c}) AS cmax FROM rows GROUP BY ALL""")
        con.execute(f"""CREATE OR REPLACE TABLE g_{k} AS
            SELECT key, count(*) AS n_labels, sum(n) AS rows, min(cmin) <> max(cmax) AS collision FROM p_{k} GROUP BY key""")
        tot, conf, conf_rows, coll = con.execute(f"""SELECT count(*), count(*) FILTER (WHERE n_labels > 1),
            coalesce(sum(rows) FILTER (WHERE n_labels > 1), 0), count(*) FILTER (WHERE collision) FROM g_{k}""").fetchone()
        res[k] = {"distinct_vectors": tot, "conflicting_groups": conf, "conflicting_rows": int(conf_rows), "hash_collisions": coll}
        # conflicting groups are few: detailed analysis on a small table
        con.execute(f"""CREATE OR REPLACE TABLE c_{k} AS
            SELECT p.key, p.label_no, p.n, m.label, m.benign, coalesce(m.category, 'Benigno') AS category
            FROM p_{k} p JOIN (SELECT key FROM g_{k} WHERE n_labels > 1) USING (key) JOIN label_map m USING (label_no)""")
        con.execute(f"""CREATE OR REPLACE TABLE s_{k} AS
            SELECT key, {situation_sql()} AS situation, sum(n) AS rows,
                   string_agg(label, ' + ' ORDER BY label) AS labels,
                   string_agg(DISTINCT category, ' + ' ORDER BY category) AS categories
            FROM c_{k} m GROUP BY key""")
        res[k]["by_situation"] = {s: {"groups": g, "rows": int(r)} for s, g, r in con.execute(
            f"SELECT situation, count(*), sum(rows) FROM s_{k} GROUP BY 1 ORDER BY 1").fetchall()}
        con.execute(f"""COPY (SELECT s.situation, c.category, c.label, count(DISTINCT c.key) AS grupos_com_o_rotulo, sum(c.n) AS registros
            FROM c_{k} c JOIN s_{k} s USING (key) GROUP BY ALL ORDER BY ALL)
            TO '{(out / f'conflitos_{k}_por_rotulo.csv').as_posix()}' (HEADER)""")
        # distinct groups per category (a group with two labels of one category counts once)
        con.execute(f"""COPY (SELECT s.situation, c.category, count(DISTINCT c.key) AS grupos_distintos, sum(c.n) AS registros
            FROM c_{k} c JOIN s_{k} s USING (key) GROUP BY ALL ORDER BY ALL)
            TO '{(out / f'conflitos_{k}_por_categoria.csv').as_posix()}' (HEADER)""")
        con.execute(f"""COPY (SELECT situation, labels, categories, count(*) AS grupos, sum(rows) AS registros
            FROM s_{k} GROUP BY ALL ORDER BY registros DESC)
            TO '{(out / f'conflitos_{k}_por_combinacao.csv').as_posix()}' (HEADER)""")
    # consequence of excluding IAT at row level: identical with IAT implies identical without it
    log("comparando com e sem IAT")
    con.execute("""CREATE OR REPLACE TABLE rowflags AS
        SELECT r.label_no, (g38.n_labels > 1) AS conf38, (g39.n_labels > 1) AS conf39
        FROM rows r JOIN g_k38 g38 ON g38.key = r.k38 JOIN g_k39 g39 ON g39.key = r.k39""")
    res["iat_effect_rows"] = {f"{a}/{b}": n for a, b, n in con.execute("""
        SELECT CASE WHEN conf39 THEN 'conflito_com_IAT' ELSE 'sem_conflito_com_IAT' END,
               CASE WHEN conf38 THEN 'conflito_sem_IAT' ELSE 'sem_conflito_sem_IAT' END, count(*)
        FROM rowflags GROUP BY ALL ORDER BY ALL""").fetchall()}
    con.execute(f"""COPY (SELECT coalesce(m.category, 'Benigno') AS categoria, m.label AS rotulo,
            count(*) FILTER (WHERE conf39) AS registros_conflito_com_iat,
            count(*) FILTER (WHERE conf38) AS registros_conflito_sem_iat,
            count(*) FILTER (WHERE conf38 AND NOT conf39) AS registros_conflito_criado_pela_exclusao
        FROM rowflags JOIN label_map m USING (label_no) GROUP BY ALL ORDER BY ALL)
        TO '{(out / 'efeito_iat_por_rotulo.csv').as_posix()}' (HEADER)""")
    con.close()
    (Path(cfg["data"]["processed_dir"]) / "work.duckdb").unlink(missing_ok=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = config_mod.load(a.config)
    out = Path(cfg["data"]["processed_dir"]) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    res = {"generated_at_utc": now(), "code_version": code_version(), "config": a.config,
           "situations": {"A_benigno_e_ataque": "grupo contém BENIGN e ao menos um ataque",
                          "B_ataques_categorias_diferentes": "só ataques, de duas ou mais categorias",
                          "C_ataques_mesma_categoria": "só ataques, todos da mesma categoria"}}
    res["full"] = full_audit(cfg, out)
    (out / "auditoria_conflitos.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
