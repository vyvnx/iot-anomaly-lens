"""Tabelas, gráficos (PNG + PDF) e results_initial.md, recriáveis a partir das predições salvas."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from .config import AMBIGUOUS, CATEGORIES, MODELS, PARTITIONS, STRATA  # noqa: E402
from .protocol import SYNTHETIC_BANNER, combined_predictions, load_protocol, recompute_tables  # noqa: E402
from .provenance import now, read_json  # noqa: E402

NAMES = {"isolation_forest": "Isolation Forest",
         "sgd_one_class_svm": "SGD One-Class SVM (Nyström RBF)",
         "autoencoder": "Autoencoder"}
# reference palette, first three categorical slots (validated all-pairs, light surface)
COLORS = {"isolation_forest": "#2a78d6", "sgd_one_class_svm": "#eb6834", "autoencoder": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STRATUM_PT = {AMBIGUOUS: "Ataque, categoria ambígua"}
PART_PT = {"train": "treino", "validation_fit": "validação (ajuste)", "validation_calibration": "validação (calibração)",
           "test_initial": "teste inicial", "test_confirmatory": "teste confirmatório"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 200})


def _save(fig, out: Path, name: str, synthetic: bool) -> str:
    if synthetic:
        fig.text(0.5, 0.005, SYNTHETIC_BANNER, ha="center", va="bottom", fontsize=8, color="#b00020")
    fig.tight_layout(rect=(0, 0.04, 1, 1) if synthetic else None)
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    return f"figures/{name}.png"


def fig_composition(parts: pd.DataFrame, out: Path, syn: bool) -> str:
    tab = pd.crosstab(parts["stratum"], parts["partition"]).reindex(index=STRATA, columns=PARTITIONS, fill_value=0)
    rows = [STRATUM_PT.get(r, r) for r in STRATA]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.grid(False)
    ax.imshow(np.log10(tab.to_numpy() + 1), cmap="Blues", aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.iat[i, j]
            ax.text(j, i, f"{v:,}".replace(",", "."), ha="center", va="center", fontsize=8,
                    color="white" if v > tab.to_numpy().max() ** 0.8 else INK)
    ax.set_xticks(range(len(PARTITIONS)), [PART_PT[p] for p in PARTITIONS], rotation=15)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title("Registros (vetores distintos) por estrato e partição", loc="left", color=INK)
    return _save(fig, out, "composicao_particoes", syn)


def fig_confusion(pred: pd.DataFrame, seed: int, split: str, out: Path, syn: bool) -> str:
    models = [m for m in MODELS if m in set(pred["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.1), squeeze=False)
    for ax, m in zip(axes[0], models):
        d = pred[(pred["model"] == m) & (pred["seed"] == seed)]
        cm = pd.crosstab(d["y_true"], d["pred"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy()
        ax.grid(False)
        ax.imshow(cm / cm.sum(axis=1, keepdims=True).clip(1), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / max(1, cm[i].sum())
                ax.text(j, i, f"{cm[i, j]:,}\n({pct:.1%})".replace(",", "."), ha="center", va="center",
                        color="white" if pct > 0.6 else INK, fontsize=8)
        ax.set_xticks([0, 1], ["benigno", "anômalo"])
        ax.set_yticks([0, 1], ["benigno", "ataque"])
        ax.set_xlabel("Decisão do modelo")
        ax.set_ylabel("Rótulo real")
        ax.set_title(NAMES[m], fontsize=9, color=INK)
    fig.suptitle(f"Matrizes de confusão — {PART_PT[split]}, semente {seed} (% por linha)", x=0.01, ha="left", fontsize=9)
    return _save(fig, out, f"matriz_confusao_{split}", syn)


def fig_category_recall(cats: pd.DataFrame, split: str, out: Path, syn: bool) -> str:
    d = cats[(cats["split"] == split) & (cats["nivel"] == "categoria")]
    models = [m for m in MODELS if m in set(d["model"])]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        dm = d[d["model"] == m]
        mean = dm.groupby("grupo")["recall"].mean().reindex(CATEGORIES)
        x = np.arange(len(CATEGORIES)) + (k - (len(models) - 1) / 2) * w
        ax.bar(x, mean.to_numpy(), width=w * 0.9, color=COLORS[m], label=NAMES[m])
        for _, r in dm.iterrows():  # every seed shown as a point
            ax.plot(x[CATEGORIES.index(r["grupo"])], r["recall"], "o", ms=3, color=INK, alpha=0.6)
    ax.set_xticks(range(len(CATEGORIES)), CATEGORIES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall (TP / N da categoria)")
    ax.set_title(f"Recall por categoria de ataque — {PART_PT[split]} (barra = média; pontos = sementes)", loc="left", color=INK)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return _save(fig, out, f"recall_categoria_{split}", syn)


def fig_fpr(metrics: pd.DataFrame, split: str, target: float, out: Path, syn: bool) -> str:
    d = metrics[metrics["split"] == split]
    models = [m for m in MODELS if m in set(d["model"])]
    fig, ax = plt.subplots(figsize=(7, 3))
    for k, m in enumerate(models):
        v = d[d["model"] == m]["fpr"] * 100
        ax.bar(k, v.mean(), color=COLORS[m], width=0.55)
        ax.plot([k] * len(v), v, "o", ms=4, color=INK, alpha=0.7)
        ax.text(k, v.mean(), f"{v.mean():.2f}%", ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(target * 100, ls="--", color=MUTED, lw=1)
    ax.text(len(models) - 0.45, target * 100, f"referência {target:.0%}", va="bottom", ha="left", fontsize=8, color=MUTED)
    ax.set_xlim(-0.5, len(models) + 0.3)
    ax.set_xticks(range(len(models)), [NAMES[m].replace(" (", "\n(") for m in models], fontsize=8)
    ax.set_ylabel("FPR observada (%)")
    ax.set_title(f"FPR nos benignos do {PART_PT[split]} com limiar calibrado", loc="left", color=INK)
    return _save(fig, out, f"fpr_{split}", syn)


def fig_curves(pred: pd.DataFrame, seed: int, split: str, out: Path, syn: bool) -> str:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.4))
    for m in [m for m in MODELS if m in set(pred["model"])]:
        d = pred[(pred["model"] == m) & (pred["seed"] == seed)]
        if d["y_true"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(d["y_true"], d["score"])
        p, r, _ = precision_recall_curve(d["y_true"], d["score"])
        a1.plot(fpr, tpr, lw=2, color=COLORS[m], label=NAMES[m])
        a2.plot(r, p, lw=2, color=COLORS[m], label=NAMES[m], drawstyle="steps-post")
    a1.plot([0, 1], [0, 1], ls=":", color=MUTED, lw=1)
    a1.set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos (recall)", title="Curva ROC")
    a2.set(xlabel="Recall", ylabel="Precisão", title="Curva precisão–recall", ylim=(0, 1.02))
    a1.legend(loc="lower right", fontsize=7)
    fig.suptitle(f"{PART_PT[split]}, semente {seed}; precisão depende da proporção amostrada", x=0.01, ha="left", fontsize=8, color=MUTED)
    return _save(fig, out, f"roc_pr_{split}", syn)


def fig_ae_loss(hist: pd.DataFrame, seed: int, out: Path, syn: bool) -> str | None:
    d = hist[hist["seed"] == seed]
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(d["epoch"], d["train_mse"], lw=2, color=COLORS["isolation_forest"], label="treino")
    ax.plot(d["epoch"], d["validation_fit_mse"], lw=2, color=COLORS["sgd_one_class_svm"], label="validação (ajuste)")
    best = int(d["best_epoch"].iloc[0])
    ax.axvline(best, ls="--", color=MUTED, lw=1)
    ax.text(best, ax.get_ylim()[1], f" checkpoint: época {best}", va="top", fontsize=8, color=MUTED)
    ax.set(xlabel="Época", ylabel="MSE (atributos padronizados)")
    ax.set_title(f"Perda do Autoencoder por época — semente {seed}", loc="left", color=INK)
    ax.legend()
    return _save(fig, out, "ae_perda", syn)


def fig_timings(tim: pd.DataFrame, split: str, out: Path, syn: bool, repeats: int) -> str:
    """Two panels, never a dual axis: fit seconds and inference throughput."""
    fit = tim[tim["fase"] == "ajuste"]
    inf = tim[(tim["fase"] == "inferencia_completa") & (tim["split"] == split)]
    models = [m for m in MODELS if m in set(fit["model"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.5, 3))
    for k, m in enumerate(models):
        v = fit[fit["model"] == m]["seconds_median"]
        a1.bar(k, v.mean(), color=COLORS[m], width=0.55)
        a1.plot([k] * len(v), v, "o", ms=4, color=INK, alpha=0.7)
        if not inf.empty:
            w = inf[inf["model"] == m]["records_per_second"]
            a2.bar(k, w.mean(), color=COLORS[m], width=0.55)
            a2.plot([k] * len(w), w, "o", ms=4, color=INK, alpha=0.7)
    for a in (a1, a2):
        a.set_xticks(range(len(models)), [NAMES[m].split(" (")[0] for m in models], fontsize=8)
    a1.set(ylabel="segundos (CPU)", title="Tempo de ajuste")
    a2.set(ylabel=f"registros/s (mediana de {repeats})", title=f"Vazão em lote — {PART_PT[split]}")
    fig.suptitle("Medidas offline em lote; não equivalem a operação em tempo real", x=0.01, ha="left", fontsize=8, color=MUTED)
    return _save(fig, out, f"tempos_{split}", syn)


def _fmt(v, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "não definido"
    return f"{v:.2%}" if pct else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def report(runs: Path, pid: str, split: str = "test_initial") -> Path:
    pdir, protocol = load_protocol(runs, pid)
    syn = protocol["data_kind"] == "synthetic"
    cfg = protocol["config_resolved"]
    tables = recompute_tables(pdir)
    pred = combined_predictions(pdir, split)
    if pred is None:
        raise FileNotFoundError(f"sem predições para {split}; execute evaluate")
    out = pdir / "figures"
    out.mkdir(exist_ok=True)
    seed = cfg["experiment"]["model_seeds"][0]
    parts = pd.read_parquet(pdir / "partitions.parquet")
    m, cats, tim, hist = (tables[k] for k in ("metrics", "category_metrics", "timings", "ae_training_history"))
    figs = [fig_composition(parts, out, syn), fig_confusion(pred, seed, split, out, syn),
            fig_category_recall(cats, split, out, syn), fig_fpr(m, split, cfg["experiment"]["target_fpr"], out, syn),
            fig_curves(pred, seed, split, out, syn), fig_timings(tim, split, out, syn, cfg["benchmark"]["repeats"])]
    if not hist.empty:
        figs.append(fig_ae_loss(hist, seed, out, syn))

    ms = m[m["split"] == split].copy()
    run_counts = ms.groupby("model").size()
    per_run = pd.DataFrame({
        "modelo": ms["model"].map(NAMES), "semente": ms["seed"],
        "TN": ms["TN"], "FP": ms["FP"], "FN": ms["FN"], "TP": ms["TP"],
        "recall": [_fmt(v, True) for v in ms["recall"]], "FPR": [_fmt(v, True) for v in ms["fpr"]],
        "precisão": [_fmt(v, True) for v in ms["precision"]], "F1": [_fmt(v) for v in ms["f1"]],
        "acurácia balanceada": [_fmt(v) for v in ms["balanced_accuracy"]],
        "AUROC": [_fmt(v) for v in ms["auroc"]], "AP": [_fmt(v) for v in ms["average_precision"]]})
    agg_lines = []
    for model, g in ms.groupby("model"):
        if len(g) >= 2:
            agg_lines.append(f"| {NAMES[model]} | {len(g)} | " + " | ".join(
                f"{g[c].mean():.4f} ± {g[c].std(ddof=1):.4f}" for c in ("recall", "fpr", "auroc", "average_precision")) + " |")
    fprs = ms.groupby("model")["fpr"].mean()
    fpr_note = ""
    if len(fprs) > 1 and fprs.min() > 0 and fprs.max() / fprs.min() > 1.5:
        fpr_note = ("\n> A FPR observada no teste difere bastante entre os modelos "
                    f"({', '.join(f'{NAMES[k]}: {v:.2%}' for k, v in fprs.items())}). "
                    "Os modelos **não** foram comparados à mesma FPR de teste; compare recall junto com a FPR observada.\n")
    ct = cats[(cats["split"] == split) & (cats["nivel"].isin(["categoria", "macro_categorias", "categoria_ambigua"]))]
    cat_tab = ct.pivot_table(index="grupo", columns=["model", "seed"], values="recall", aggfunc="first", dropna=False)
    cat_tab = cat_tab.map(lambda v: _fmt(v, True))
    ncat = ct.drop_duplicates("grupo").set_index("grupo")["N"]
    cat_tab.insert(0, "N", ncat.reindex(cat_tab.index).astype(int))
    cat_tab.columns = ["N"] + [f"{NAMES[a].split(' (')[0]} s{b}" for a, b in cat_tab.columns[1:]]
    thr = read_json(pdir / "thresholds.json")
    thr_tab = pd.DataFrame([{"modelo/semente": k, "n calibração": v["n_calibration"], "limiar": f"{v['threshold']:.6g}",
                             "empates": v["ties_at_threshold"], "FPR calibração": f"{v['calibration_fpr']:.2%} ({v['calibration_fp']})",
                             "avisos": "; ".join(v["warnings"]) or "—"} for k, v in thr.items()])
    t_inf = tim[(tim["split"] == split) & tim["fase"].str.startswith("inferencia")]
    t_tab = t_inf[["model", "seed", "fase", "records", "seconds_median", "seconds_iqr", "records_per_second", "microseconds_per_record"]].round(4) if not t_inf.empty else pd.DataFrame()
    fit = tim[tim["fase"] == "ajuste"][["model", "seed", "records", "seconds_median", "inclui"]].round(3)
    counts = pd.read_csv(pdir / "sample_counts.csv")
    comp = counts.pivot_table(index="estrato", columns="particao", values="registros", aggfunc="sum", fill_value=0)
    fit_infos = {f"{read_json(f)['model']}/{read_json(f)['seed']}": read_json(f)["fit_info"] for f in sorted(pdir.glob("models/*/seed_*/fit.json"))}
    warnings = [f"{k}: {'; '.join(v.get('convergence_warnings', []))}" for k, v in fit_infos.items() if v.get("convergence_warnings")]
    banner = f"# ⚠ {SYNTHETIC_BANNER}\n\n" if syn else ""
    md = f"""{banner}# Resultados iniciais — protocolo `{pid}`

Gerado em {now()} (UTC) a partir de `predictions_{split}.parquet`, `metrics.csv`, `category_metrics.csv`,
`timings.csv` e `thresholds.json` desta pasta. Tipo de dado: **{protocol['data_kind']}**. Subconjunto: **{split}**.
Execuções concluídas por modelo: {', '.join(f'{NAMES[k]}: {v}' for k, v in run_counts.items())}.

## Composição dos dados (vetores distintos)

{_md_table(comp.reset_index())}

Modo de partição: `{protocol['split']['mode']}` (semente {protocol['split']['seed']}). Justificativa registrada:
{protocol['split']['justification']}

## Calibração (somente benignos de validation_calibration)

Regra: `{next(iter(thr.values()))['rule']}`. Alvo de FPR: {cfg['experiment']['target_fpr']:.0%}. Empates não são quebrados.

{_md_table(thr_tab)}

## Métricas por execução ({split}; ataque = classe positiva)

{_md_table(per_run)}
{fpr_note}
AP = average precision (soma em degraus do scikit-learn), não a área trapezoidal sob a curva PR.
Precisão e F1 dependem da composição amostrada e não se extrapolam para prevalências reais.

{"## Média ± desvio-padrão amostral entre sementes" + chr(10) + chr(10) + "| modelo | n | recall | FPR | AUROC | AP |" + chr(10) + "|---|---|---|---|---|---|" + chr(10) + chr(10).join(agg_lines) if agg_lines else "Apenas uma execução por modelo até aqui (n_runs=1); nenhuma dispersão é apresentada."}

## Recall por categoria (sem FPR: conjuntos só de ataques)

Macro = média simples dos recalls das 7 categorias; não é o recall global. A linha `attack_category_ambiguous`
reúne ataques cujos vetores idênticos têm rótulos de categorias diferentes: entra nas métricas binárias, fica fora
do macro e não é uma 8ª categoria nem uma saída do modelo.

{_md_table(cat_tab.reset_index())}

## Tempos (CPU, {cfg['experiment']['threads']} threads; medidas offline em lote)

Ajuste:

{_md_table(fit)}

Inferência (1 aquecimento + {cfg['benchmark']['repeats']} repetições; lote {cfg['benchmark']['inference_batch_size']}; sem leitura de disco):

{_md_table(t_tab) if not t_tab.empty else '—'}

{"Avisos de convergência do SGD: " + " | ".join(warnings) if warnings else "Nenhum aviso de convergência registrado."}

## Figuras

{chr(10).join(f'- [{Path(f).stem}]({f})' for f in figs if f)}

Curvas e matrizes usam a primeira semente predefinida ({seed}); as tabelas trazem todas as sementes.

## Limitações registradas no protocolo

{chr(10).join('- ' + x for x in protocol['limitations'])}
- Uma configuração fixa de cada família não comprova superioridade universal do algoritmo.
- Não há teste de significância nem intervalos de confiança: a unidade de independência ainda não foi justificada.
- Este relatório não aponta um "melhor modelo" por uma única métrica.
"""
    path = pdir / f"results_{'initial' if split == 'test_initial' else 'confirmatory'}.md"
    path.write_text(md, encoding="utf-8")
    return path
