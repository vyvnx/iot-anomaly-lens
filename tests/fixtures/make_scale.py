"""Gera um conjunto SINTÉTICO grande (padrão: 2 milhões de linhas) para medir tempo/memória de inspect/prepare.

Não representa o CICIoT2023. Uso: python tests/fixtures/make_scale.py <pasta> [linhas_por_arquivo] [arquivos]
Grava <pasta>/src/*.csv, <pasta>/scale.yaml, <pasta>/map.yaml e <pasta>/allow.yaml.
"""

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pc
import yaml

ROOT = Path(__file__).resolve().parents[2]
CATS = {"DDoS": "DDoS", "DoS": "DoS", "Recon": "Recon", "Web": "Web-based", "BruteForce": "Brute Force",
        "Spoofing": "Spoofing", "Mirai": "Mirai"}


def main(dest: Path, rows: int = 400_000, n_files: int = 5) -> None:
    rng = np.random.default_rng(1)
    cols = [f"f{i}" for i in range(46)]
    labels = ["SYN_BenignTraffic"] + [f"SYN_{c}-{j}" for c in CATS for j in range(4)]
    (dest / "src").mkdir(parents=True, exist_ok=True)
    for k in range(n_files):
        X = np.round(rng.gamma(2, 50, (rows, 46)), 3)
        X[:, ::5] = rng.integers(0, 2, (rows, 10))  # binary flags -> some duplicates
        lab = np.array(labels)[rng.integers(0, len(labels), rows)]
        pc.write_csv(pa.table({**{c: X[:, i] for i, c in enumerate(cols)}, "label": lab}), dest / "src" / f"part-{k:05d}.csv")
    cfg = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text())
    real = yaml.safe_load((ROOT / "configs" / "initial.yaml").read_text())
    out = "tests/output/scale"
    cfg["output_dir"] = out + "/runs"
    cfg["data"].update(raw_dir=out + "/raw", processed_dir=out + "/processed", source_manifest=out + "/raw_manifest.json",
                       label_mapping=str(dest / "map.yaml"), feature_allowlist=str(dest / "allow.yaml"))
    cfg["sample"].update(benign_cap=100000, category_caps=real["sample"]["category_caps"], ambiguous_cap=2000)
    cfg["models"], cfg["benchmark"] = real["models"], real["benchmark"]
    cfg["experiment"]["model_seeds"] = [42]
    (dest / "scale.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
    mapping = {l: ({"benign": True, "category": None} if "Benign" in l else
                   {"benign": False, "category": CATS[l[4:].split("-")[0]]}) for l in labels}
    (dest / "map.yaml").write_text(yaml.safe_dump({"status": "confirmed", "labels": mapping}))
    (dest / "allow.yaml").write_text(yaml.safe_dump({"status": "confirmed", "features": cols}))


if __name__ == "__main__":
    main(Path(sys.argv[1]), *map(int, sys.argv[2:]))
