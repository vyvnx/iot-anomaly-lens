"""Versão do código, registro de exposição dos testes e utilidades de rastreabilidade."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

SRC = Path(__file__).resolve().parent


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def code_version() -> dict:
    h = hashlib.sha256()
    for p in sorted(SRC.rglob("*.py")):
        h.update(p.relative_to(SRC).as_posix().encode())
        h.update(p.read_bytes())
    git = None
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=SRC, capture_output=True, text=True,
                             timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return {"source_tree_sha256": h.hexdigest(), "git_commit": git}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exposure_log(output_dir: Path) -> Path:
    return Path(output_dir) / "test_exposure_log.jsonl"


def record_exposure(output_dir: Path, protocol_id: str, split: str, content_hash: str) -> None:
    """Append-only: a new protocol id never erases the fact that a test subset was consulted."""
    p = exposure_log(output_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"at_utc": now(), "protocol_id": protocol_id, "split": split,
                            "split_content_sha256": content_hash}) + "\n")


def exposures(output_dir: Path) -> list[dict]:
    p = exposure_log(output_dir)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
