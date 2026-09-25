"""Medição de tempos em lote (perf_counter): aquecimento não cronometrado + repetições cronometradas."""

from __future__ import annotations

import time

import numpy as np


def batched(fn, X: np.ndarray, batch: int) -> np.ndarray:
    return np.concatenate([fn(X[i:i + batch]) for i in range(0, len(X), batch)]) if len(X) else np.empty(0)


def repeat_timed(fn, warmup: int, repeats: int, sync=None) -> list[float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append(time.perf_counter() - t0)
    return times


def summarize(times: list[float], n_records: int) -> dict:
    t = np.asarray(times)
    med = float(np.median(t))
    return {"repeats": len(t), "seconds_median": med, "seconds_min": float(t.min()), "seconds_max": float(t.max()),
            "seconds_iqr": float(np.percentile(t, 75) - np.percentile(t, 25)),
            "records": n_records,
            "records_per_second": n_records / med if med > 0 else None,
            "microseconds_per_record": 1e6 * med / n_records if n_records else None}
