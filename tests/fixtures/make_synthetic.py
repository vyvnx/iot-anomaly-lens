"""Gera CSVs SINTÉTICOS para testar o software. Não representam o CICIoT2023.

Inclui de propósito: duplicatas entre arquivos, grupos multirrótulo das situações A (benigno e
ataque), B (ataques de categorias diferentes) e C (ataques da mesma categoria), valores
ausentes, não numéricos e infinitos, uma coluna constante e uma linha malformada.
Uso: python tests/fixtures/make_synthetic.py <pasta_destino>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

FEATURES = [f"feat {i:02d}" for i in range(14)] + ["const_col"]
LABEL_COLUMN = "Classe Sintetica"
ATTACKS = {"SYN_DDoS-A": 3.0, "SYN_DDoS-B": 2.5, "SYN_DoS-A": 2.0, "SYN_Recon-A": 1.5, "SYN_Web-A": 1.2,
           "SYN_BruteForce-A": 1.8, "SYN_Spoofing-A": 1.4, "SYN_Mirai-A": 4.0}
BENIGN = "SYN_BenignTraffic"


def rows_for(label: str, n: int, rng: np.random.Generator) -> list[list]:
    shift = ATTACKS.get(label, 0.0)
    X = rng.standard_normal((n, len(FEATURES) - 1)) + shift * (np.arange(len(FEATURES) - 1) % 3 == 0)
    X = np.round(X, 6)
    return [[*map(float, x), 0.0, label] for x in X]


def write_fixture(dest: Path, seed: int = 7, n_benign: int = 6000, n_attack: int = 300, n_files: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    dest.mkdir(parents=True, exist_ok=True)
    rows = rows_for(BENIGN, n_benign, rng)
    for lab in ATTACKS:
        rows += rows_for(lab, n_attack, rng)
    rng.shuffle(rows)
    dup_benign = list(rows[0]) if rows[0][-1] == BENIGN else [*rows_for(BENIGN, 1, rng)[0]]
    conflict = [*rows_for(BENIGN, 1, rng)[0]]
    conflict_attack = conflict[:-1] + ["SYN_DoS-A"]
    # situation B: attack vectors labelled with two categories; situation C: two labels of one category
    b_vecs = [r[:-1] for r in rows_for("SYN_DDoS-A", 6, rng)]
    c_vecs = [r[:-1] for r in rows_for("SYN_DDoS-A", 6, rng)]
    special = [
        dup_benign, dup_benign,                       # exact duplicates, will land in different files
        conflict, conflict_attack,                    # situation A: same vector, benign and attack
        *[v + ["SYN_DDoS-A"] for v in b_vecs], *[v + ["SYN_DoS-A"] for v in b_vecs],
        *[v + ["SYN_DDoS-A"] for v in c_vecs], *[v + ["SYN_DDoS-B"] for v in c_vecs],
        [*rows[1][:3], "", *rows[1][4:]],             # missing value
        [*rows[2][:2], "abc", *rows[2][3:]],          # non-numeric
        [*rows[3][:1], "inf", *rows[3][2:]],          # infinite
    ]
    rows += special
    chunks = [rows[i::n_files] for i in range(n_files)]
    # guarantee the duplicate pair crosses files
    chunks[0].append(dup_benign)
    chunks[-1].append(dup_benign)
    for i, chunk in enumerate(chunks):
        with open(dest / f"part-{i:05d}-synthetic.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([*FEATURES, LABEL_COLUMN])
            w.writerows(chunk)
            if i == 0:
                f.write("1.0,2.0\n")  # malformed row (wrong column count)
    return {"files": n_files, "labels": [BENIGN, *ATTACKS], "label_column": LABEL_COLUMN}


if __name__ == "__main__":
    print(write_fixture(Path(sys.argv[1])))
