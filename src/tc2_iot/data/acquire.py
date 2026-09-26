"""Obtenção dos CSVs: importação local (import-data) e download de URLs autorizadas."""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from ..config import sha256_file

HTML_MARKERS = (b"<!doctype html", b"<html", b"<head", b"<form", b"<?xml")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def looks_like_html(path: Path) -> bool:
    with open(path, "rb") as f:
        head = f.read(2048).lstrip().lower()
    return any(head.startswith(m) or m in head[:512] for m in HTML_MARKERS)


def sanitize_url(url: str) -> str:
    """Drop userinfo, query and fragment: they may carry tokens or temporary credentials."""
    p = urllib.parse.urlsplit(url)
    host = p.hostname or ""
    if p.port:
        host += f":{p.port}"
    return urllib.parse.urlunsplit((p.scheme, host, p.path, "", ""))


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"files": []}


def save_manifest(path: Path, manifest: dict) -> None:
    manifest["files"].sort(key=lambda f: f["relative_name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _upsert(manifest: dict, entry: dict) -> None:
    manifest["files"] = [f for f in manifest["files"] if f["relative_name"] != entry["relative_name"]]
    manifest["files"].append(entry)


def import_data(source: Path, raw_dir: Path, manifest_path: Path, mode: str = "copy",
                obtained_at: str | None = None, origin: str = "download manual pelo autor") -> dict:
    """Inventory CSVs under `source` and copy/link them into raw_dir without modifying originals."""
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"pasta de origem não encontrada: {source}")
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() == ".csv")
    if not files:
        raise FileNotFoundError(f"nenhum arquivo .csv encontrado em {source}")
    html = [str(p.relative_to(source)) for p in files if looks_like_html(p)]
    if html:
        raise ValueError(f"arquivos .csv que parecem páginas HTML/formulário (download incompleto?): {html}")
    need = sum(p.stat().st_size for p in files) if mode == "copy" else 0
    raw_dir.mkdir(parents=True, exist_ok=True)
    if need and shutil.disk_usage(raw_dir).free < need * 1.05:
        raise OSError(f"espaço insuficiente para copiar {need} bytes em {raw_dir}")
    manifest = load_manifest(manifest_path)
    for p in files:
        rel = p.relative_to(source).as_posix()
        dst = raw_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            if sha256_file(dst) != sha256_file(p):
                raise FileExistsError(f"{dst} já existe com conteúdo diferente; não será substituído")
        elif mode == "copy":
            shutil.copy2(p, dst)
        elif mode == "link":
            os.symlink(p, dst)
        else:
            raise ValueError("mode deve ser copy ou link")
        _upsert(manifest, {
            "relative_name": rel,
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "origin": origin,
            "import_mode": mode,
            "obtained_at": obtained_at,  # null when the author does not know the download date
            "registered_at": _now(),
        })
    save_manifest(manifest_path, manifest)
    return manifest


def _head(url: str, timeout: int) -> tuple[int | None, bool]:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            size = r.headers.get("Content-Length")
            return (int(size) if size else None), r.headers.get("Accept-Ranges", "").lower() == "bytes"
    except Exception:
        return None, False


def safe_extract_zip(archive: Path, dest: Path) -> list[Path]:
    dest = dest.resolve()
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        total = sum(m.file_size for m in members)
        if shutil.disk_usage(dest).free < total * 1.05:
            raise OSError(f"espaço insuficiente para extrair {total} bytes de {archive.name}")
        for m in members:
            target = (dest / m.filename).resolve()
            if not target.is_relative_to(dest):
                raise ValueError(f"entrada do arquivo compactado sai da pasta de destino: {m.filename}")
        z.extractall(dest)
        return [dest / m.filename for m in members if not m.is_dir()]


def download(urls: list[str], raw_dir: Path, manifest_path: Path, max_bytes: int,
             timeout: int = 60, dry_run: bool = False, log=print) -> dict:
    """Stream authorized URLs into raw_dir/<name>.part and promote after validation."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    plan = []
    for url in urls:
        size, ranges = _head(url, timeout)
        name = Path(urllib.parse.urlsplit(url).path).name
        if not name:
            raise ValueError(f"URL sem nome de arquivo: {sanitize_url(url)}")
        plan.append((url, name, size, ranges))
        log(f"{name}: tamanho {'desconhecido' if size is None else f'{size} bytes'}; retomada {'suportada' if ranges else 'não suportada'}")
    unknown = [n for _, n, s, _ in plan if s is None]
    known = sum(s for *_, s, _ in plan if s)
    if unknown:
        raise RuntimeError(
            f"tamanho desconhecido para {unknown}; não inicio download ilimitado. "
            "Baixe manualmente e use import-data, ou forneça URLs com Content-Length."
        )
    if known > max_bytes:
        raise RuntimeError(f"total {known} bytes excede o limite configurado de {max_bytes} bytes")
    if shutil.disk_usage(raw_dir).free < known * 1.05:
        raise OSError("espaço em disco insuficiente para o download")
    if dry_run:
        return {"planned_bytes": known, "files": [n for _, n, _, _ in plan]}
    manifest = load_manifest(manifest_path)
    for url, name, size, ranges in plan:
        final, part = raw_dir / name, raw_dir / (name + ".part")
        if final.exists():
            log(f"{name} já existe; mantido")
            continue
        start = part.stat().st_size if part.exists() and ranges else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ctype = r.headers.get("Content-Type", "").lower()
            if "text/html" in ctype:
                raise RuntimeError(f"{name}: servidor devolveu HTML (provável formulário/cadastro), não o arquivo")
            if start and not (r.status == 206 and r.headers.get("Content-Range", "").startswith(f"bytes {start}-")):
                start = 0  # server ignored the range; restart from scratch
            with open(part, "ab" if start else "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
        if part.stat().st_size != size:
            raise RuntimeError(f"{name}: recebido {part.stat().st_size} bytes, esperado {size}; .part mantido para retomada")
        if looks_like_html(part):
            raise RuntimeError(f"{name}: conteúdo parece HTML/formulário")
        part.rename(final)
        extracted = safe_extract_zip(final, raw_dir) if zipfile.is_zipfile(final) else [final]
        for p in extracted:
            if p.suffix.lower() != ".csv":
                continue
            if looks_like_html(p):
                raise RuntimeError(f"{p.name}: conteúdo parece HTML/formulário")
            _upsert(manifest, {
                "relative_name": p.relative_to(raw_dir.resolve() if p.is_absolute() else raw_dir).as_posix(),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "origin": sanitize_url(url),
                "import_mode": "download",
                "obtained_at": _now(),
                "registered_at": _now(),
            })
        save_manifest(manifest_path, manifest)
    return manifest
