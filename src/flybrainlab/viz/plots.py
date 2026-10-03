"""Plots for runs: trajectories, DN activity, metric comparison."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..experiments.store import RunStore


def plot_run(run_id: str, store: RunStore | None = None, out: Path | None = None) -> Path:
    store = store or RunStore()
    p = store.path(run_id)
    cfg = store.config(run_id)
    trajs = sorted(p.glob("traj_*.npz"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    for f in trajs:
        d = np.load(f, allow_pickle=True)
        pos = d["pos"]
        ax.plot(pos[:, 0], pos[:, 1], lw=1.2, label=f.stem)
        ax.plot(*pos[0, :2], "go", ms=4)
        if "spider" in d.files:
            sp = d["spider"]
            ax.plot(sp[:, 0], sp[:, 1], "k--", lw=0.8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"top view ({cfg['scenario']})")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.legend(fontsize=7)
    ax = axes[1]
    if trajs:
        d = np.load(trajs[0], allow_pickle=True)
        t = d["t"]
        rates = d["dn_rates"]
        groups = [str(g) for g in d["dn_groups"]]
        for k, g in enumerate(groups):
            if rates[:, k].max() > 0:
                ax.plot(t[: len(rates)], rates[:, k], lw=1, label=g)
        ax.set_title("readout DN group rates (trial 0)")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("Hz")
        ax.legend(fontsize=7, ncol=2)
        ax2 = axes[2]
        ax2.plot(t, d["pos"][:, 2], label="height z [cm]")
        if "up" in d.files:
            tilt = np.degrees(np.arccos(np.clip(d["up"][:, 2], -1, 1)))
            ax2.plot(t, tilt / 90.0, label="tilt / 90 deg")
        ax2.set_title("trial 0: height and tilt")
        ax2.set_xlabel("t [s]")
        ax2.legend(fontsize=7)
    fig.suptitle(f"{run_id}  |  {cfg.get('label')}", fontsize=9)
    fig.tight_layout()
    out = out or p / "overview.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_comparison(run_ids: list[str], metrics: list[str], store: RunStore | None = None,
                    out: Path | None = None, labels: list[str] | None = None) -> Path:
    store = store or RunStore()
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4), squeeze=False)
    labels = labels or [store.config(r).get("label") or r for r in run_ids]
    rng = np.random.default_rng(0)
    for ax, m in zip(axes[0], metrics):
        for k, r in enumerate(run_ids):
            x = store.trials(r)[m].to_numpy(float)
            x = x[np.isfinite(x)]
            ax.bar(k, x.mean() if len(x) else 0, color="#c9d3e6", edgecolor="#4a5d85")
            if len(x) > 1:
                ax.errorbar(k, x.mean(), yerr=x.std(ddof=1), color="#4a5d85", capsize=4)
            ax.scatter(k + rng.uniform(-0.15, 0.15, len(x)), x, s=12, color="#1f2d4d", zorder=3)
        ax.set_xticks(range(len(run_ids)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(m)
    fig.tight_layout()
    out = out or store.root / f"compare_{'_'.join(r[-12:] for r in run_ids)}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
