"""Inspeção local do ambiente (somente leitura) e limites de threads."""

from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

LIBS = ["numpy", "scikit-learn", "scipy", "torch", "pyarrow", "duckdb", "pandas",
        "matplotlib", "psutil", "pyyaml", "joblib", "threadpoolctl", "pytest"]


def resolve_threads(value) -> int:
    if value == "auto":
        return psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
    return int(value)


def apply_thread_limits(n: int) -> dict:
    """Limit BLAS/OpenMP and torch to n threads for the rest of the process; returns a record."""
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits

    threadpool_limits(limits=n)
    torch.set_num_threads(n)
    record = {
        "threads_budget": n,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "sklearn_n_jobs": n,
        "threadpools": [{k: p.get(k) for k in ("user_api", "internal_api", "num_threads")} for p in threadpool_info()],
    }
    return record


def _gpu() -> list[dict] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        name, mem, drv = [x.strip() for x in line.split(",")]
        gpus.append({"name": name, "memory_total_mib": int(float(mem)), "driver_version": drv})
    return gpus


def _cpu_model() -> str | None:
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            return None
    return platform.processor() or None


def _is_wsl() -> bool:
    return "microsoft" in platform.release().lower()


def collect(data_dir: str | Path = "data") -> dict:
    import torch

    data_dir = Path(data_dir)
    probe = data_dir if data_dir.exists() else Path.cwd()
    vm = psutil.virtual_memory()
    versions = {}
    for lib in LIBS:
        try:
            versions[lib] = md.version(lib)
        except md.PackageNotFoundError:
            versions[lib] = None
    now = dt.datetime.now(dt.timezone.utc)
    cuda = torch.cuda.is_available()
    # no hostname, user name or absolute paths: this file is meant to be shared
    return {
        "collected_at_utc": now.isoformat(timespec="seconds"),
        "local_timezone": time.strftime("%Z"),
        "local_utc_offset": dt.datetime.now().astimezone().strftime("%z"),
        "system": platform.system(),
        "system_release": platform.release(),
        "wsl": _is_wsl(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "cpu_model": _cpu_model(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "memory_total_bytes": vm.total,
        "memory_available_bytes": vm.available,
        "disk_free_bytes_data_dir": shutil.disk_usage(probe).free,
        "disk_probe": "data/" if data_dir.exists() else "diretório do projeto (data/ ainda não existe)",
        "gpu_nvidia_smi": _gpu(),
        "torch_cuda_available": cuda,
        "torch_cuda_version": torch.version.cuda,
        "library_versions": versions,
        "execution_device_main_comparison": "cpu",
        "threads_auto_resolves_to": resolve_threads("auto"),
    }


def human_summary(env: dict) -> str:
    gib = 1024 ** 3
    gpus = env["gpu_nvidia_smi"]
    gpu_txt = "não detectada" if not gpus else "; ".join(f"{g['name']} ({g['memory_total_mib']} MiB)" for g in gpus)
    lines = [
        f"Coletado em (UTC): {env['collected_at_utc']}  fuso local: {env['local_timezone']} ({env['local_utc_offset']})",
        f"Sistema: {env['system']} {env['system_release']}{' (WSL)' if env['wsl'] else ''} {env['machine']}",
        f"Python: {env['python']}",
        f"CPU: {env['cpu_model']} — {env['cpu_physical_cores']} núcleos físicos / {env['cpu_logical_cores']} lógicos",
        f"Memória: total {env['memory_total_bytes'] / gib:.1f} GiB, disponível {env['memory_available_bytes'] / gib:.1f} GiB",
        f"Disco livre ({env['disk_probe']}): {env['disk_free_bytes_data_dir'] / gib:.1f} GiB",
        f"GPU (nvidia-smi): {gpu_txt}; CUDA no PyTorch instalado: {env['torch_cuda_available']}",
        f"Threads (auto): {env['threads_auto_resolves_to']}",
        "Bibliotecas: " + ", ".join(f"{k}={v}" for k, v in env["library_versions"].items()),
    ]
    if gpus and not env["torch_cuda_available"]:
        lines.append("Obs.: há GPU, mas o PyTorch instalado é CPU; a comparação principal é em CPU (CUDA é opcional).")
    return "\n".join(lines)


def doctor(out_dir: str | Path = ".", data_dir: str | Path = "data") -> dict:
    env = collect(data_dir)
    out = Path(out_dir) / "environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    return env
