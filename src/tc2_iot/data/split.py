"""Partições determinísticas dos vetores distintos e política de conflitos no espaço transformado."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..config import BENIGN_ONLY, BENIGN_STRATUM, PARTITIONS, TEST_SPLITS
from .clean import DataBlocked


def priority(seed: int, vector_ids: pd.Series) -> pd.Series:
    return vector_ids.map(lambda v: hashlib.sha256(f"{seed}:{v}".encode()).hexdigest())


def assign_partitions(sample: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s = cfg["split"]
    if s["mode"] == "pending_provenance_inspection":
        raise DataBlocked("split.mode ainda está pendente; defina random_unique_vectors ou provenance_group com justificativa")
    if not s["justification"].strip():
        raise DataBlocked("split.justification é obrigatória para congelar o protocolo")
    if s["mode"] == "provenance_group":
        # ponytail: not implemented because no documented capture/device/period column is known in the csv
        # distribution; implement group-wise allocation here if inspection finds one
        raise DataBlocked("provenance_group não implementado: nenhum identificador confiável de captura foi encontrado")
    df = sample.copy()
    df["_prio"] = priority(s["seed"], df["vector_id"])
    df["partition"] = None
    for stratum, grp in df.groupby("stratum", sort=True):
        fr = s["benign_fractions"] if stratum == BENIGN_STRATUM else s["attack_fractions"]
        order = list(fr)
        counts = dict.fromkeys(fr, 0)
        # systematic allocation: sorted by label then random priority, each row goes to the partition with the
        # largest deficit (ties: configured order); the stratum and each label within it follow the fractions
        for i, idx in enumerate(grp.sort_values(["label", "_prio", "vector_id"]).index, 1):
            part = max(order, key=lambda k: (i * float(fr[k]) - counts[k], -order.index(k)))
            counts[part] += 1
            df.at[idx, "partition"] = part
    return df.drop(columns="_prio").sort_values("vector_id").reset_index(drop=True)


def effective_keys(Z: np.ndarray) -> pd.Series:
    return pd.Series([hashlib.sha256(r.tobytes()).hexdigest() for r in np.ascontiguousarray(Z)])


def resolve_effective_conflicts(df: pd.DataFrame, Z: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows identical in the transformed space (what the models receive).

    - benign and attack in one group: every member is excluded (binary ambiguity, as in P1);
    - same binary class across partitions: keep the members in the highest-priority partition
      (train > validation_fit > validation_calibration > test_initial > test_confirmatory), drop the others.
    Returns kept rows and one row per removed record with its reason.
    """
    df = df.reset_index(drop=True)
    keys = effective_keys(Z)
    rank = df["partition"].map({p: i for i, p in enumerate(PARTITIONS)})
    size = keys.map(keys.value_counts())
    mixed = df["benign"].groupby(keys).transform("nunique") > 1
    cross = rank > rank.groupby(keys).transform("min")
    reason = pd.Series(None, index=df.index, dtype=object)
    reason[(size > 1) & mixed] = "ambiguidade_binaria_apos_transformacao"
    reason[(size > 1) & ~mixed & cross] = "colisao_numerica_mesma_classe_particao_de_menor_prioridade"
    removed = df.loc[reason.notna(), ["record_id", "vector_id", "label", "stratum", "partition"]].assign(
        motivo=reason[reason.notna()], grupo_efetivo=keys[reason.notna()].str[:16])
    return df[reason.isna()].reset_index(drop=True), removed.reset_index(drop=True)


def check_partitions(df: pd.DataFrame, Z: np.ndarray | None = None) -> dict:
    """Non-intersection of ids and vectors across partitions; benign-only fit/calibration subsets."""
    checks = {}
    for col in ("record_id", "vector_id"):
        checks[f"{col}_unique"] = bool(df[col].is_unique)
    checks["benign_only_in_" + "_".join(BENIGN_ONLY)] = bool(df.loc[df["partition"].isin(BENIGN_ONLY), "benign"].all())
    checks["attacks_only_in_tests"] = bool(df.loc[~df["benign"], "partition"].isin(TEST_SPLITS).all())
    checks["all_rows_assigned"] = bool(df["partition"].isin(PARTITIONS).all())
    if Z is not None:
        keys = effective_keys(Z)
        checks["no_effective_vector_crosses_partitions"] = bool((pd.Series(df["partition"].values).groupby(keys).nunique() <= 1).all())
        checks["no_effective_vector_with_both_binary_classes"] = bool((pd.Series(df["benign"].values).groupby(keys).nunique() <= 1).all())
    return checks
