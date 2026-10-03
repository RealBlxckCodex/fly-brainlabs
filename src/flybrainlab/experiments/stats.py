"""Statistics for comparing two runs, metric by metric (computed, never estimated by the LLM)."""
from __future__ import annotations

import numpy as np
from scipy import stats

from .store import RunStore

BINARY_HINT = {"takeoff", "survived", "reached", "landed", "chose", "chose_bright", "chose_left", "tumbled", "crashed"}


def _is_binary(x: np.ndarray, name: str) -> bool:
    return name in BINARY_HINT or (len(x) > 0 and set(np.unique(x)).issubset({0.0, 1.0}))


def compare_arrays(a: np.ndarray, b: np.ndarray, name: str = "", n_boot: int = 5000, seed: int = 0) -> dict:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    res = {"metric": name, "n_a": int(len(a)), "n_b": int(len(b))}
    if len(a) < 2 or len(b) < 2:
        res["note"] = "fewer than 2 valid trials in a run; no test"
        res.update(mean_a=float(a.mean()) if len(a) else None, mean_b=float(b.mean()) if len(b) else None)
        return res
    res.update(mean_a=float(a.mean()), mean_b=float(b.mean()), std_a=float(a.std(ddof=1)), std_b=float(b.std(ddof=1)),
               diff=float(b.mean() - a.mean()))
    rng = np.random.default_rng(seed)
    boots = rng.choice(b, (n_boot, len(b))).mean(1) - rng.choice(a, (n_boot, len(a))).mean(1)
    res["diff_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    if _is_binary(np.r_[a, b], name):
        table = [[int(a.sum()), int(len(a) - a.sum())], [int(b.sum()), int(len(b) - b.sum())]]
        res["test"] = "fisher_exact"
        res["p_value"] = float(stats.fisher_exact(table)[1])
        res["proportion_a"], res["proportion_b"] = float(a.mean()), float(b.mean())
    else:
        if np.ptp(np.r_[a, b]) == 0:
            res.update(test="none (identical constant values)", p_value=1.0)
        else:
            res["test"] = "welch_t + mann_whitney"
            res["p_value"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
            res["p_mannwhitney"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        res["cohens_d"] = float((b.mean() - a.mean()) / sp) if sp > 0 else None
    return res


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m, running = len(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def compare_runs(run_a: str, run_b: str, metrics: list[str] | None = None, store: RunStore | None = None) -> dict:
    store = store or RunStore()
    ta, tb = store.trials(run_a), store.trials(run_b)
    skip = {"trial", "seed", "wall_s"}
    cols = metrics or [c for c in ta.columns if c in tb.columns and c not in skip and np.issubdtype(ta[c].dtype, np.number)]
    rows = [compare_arrays(ta[c].to_numpy(float), tb[c].to_numpy(float), c) for c in cols if c in ta and c in tb]
    tested = [r for r in rows if "p_value" in r]
    for r, padj in zip(tested, holm([r["p_value"] for r in tested])):
        r["p_holm"] = padj
    ca, cb = store.config(run_a), store.config(run_b)
    return {"run_a": run_a, "run_b": run_b, "label_a": ca.get("label"), "label_b": cb.get("label"),
            "scenario": [ca.get("scenario"), cb.get("scenario")], "metrics": rows,
            "note": "diff = mean_b - mean_a; p_holm corrected across the metrics listed"}
