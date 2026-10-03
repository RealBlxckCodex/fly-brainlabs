"""M1 validation: the brain runs and reproduces a known result.

1. Reference check (FlyWire v783, Shiu et al. 2024 example): activate 20 sugar GRNs at
   150 Hz for 1 s. Expected: MN9 (proboscis extension motor neuron) is driven.
   Run the same protocol in the Brian2 reference implementation and in our step-wise
   engine; compare MN9 rate and the rates of all responsive neurons.
2. MaleCNS v1.0: sugar GRNs -> MN9, and LPLC2 -> Giant Fiber (DNp01) -> TTMn (jump
   motor neuron in the VNC), plus shuffled-connectome controls.

Usage: python scripts/validate_m1.py [--brian-trials 6] [--trials 30]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flybrainlab.brain import LIFNetwork, LIFParams
from flybrainlab.config import default_config, deep_merge
from flybrainlab.connectome import GroupResolver, load_connectome
from flybrainlab.connectome.interventions import shuffle
from flybrainlab.paths import ROOT

OUT = ROOT / "docs" / "validation"
_G: dict = {}


def params_for(ds: str) -> LIFParams:
    cfg = default_config()
    cfg = deep_merge(cfg, cfg["dataset_overrides"][ds])
    return LIFParams.from_dict(cfg["brain"])


def ours(seed: int):
    W, exc, rate, dur, p = _G["W"], _G["exc"], _G["rate"], _G["dur"], _G["p"]
    net = LIFNetwork(W, p, seed=seed)
    net.set_poisson_rates(exc, rate)
    net.run(dur)
    return net.rates_hz(dur)


def brian(seed: int):
    from flybrainlab.brain.brian2_ref import run_brian2_trial

    return run_brian2_trial(_G["W"], _G["exc"], _G["rate"], _G["dur"], _G["p"], seed=seed)


def run_many(fn, W, exc, rate, dur, p, n, workers):
    _G.update(W=W, exc=exc, rate=rate, dur=dur, p=p)
    with mp.get_context("fork").Pool(workers) as pool:
        return np.array(pool.map(fn, range(n)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--brian-trials", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--brian-workers", type=int, default=2)
    a = ap.parse_args()
    res = {"protocol": "Poisson activation 150 Hz (sugar) / 100 Hz (LPLC2), 1000 ms, w/o lesion"}

    # ---------------- 1. FlyWire reference
    c = load_connectome("flywire_v783")
    g = GroupResolver(c)
    p = params_for("flywire_v783")
    sugar, mn9 = g.resolve("sugar_grn"), g.resolve("mn9")
    t0 = time.time()
    R_ours = run_many(ours, c.W, sugar, 150.0, 1000.0, p, a.trials, a.workers)
    t_ours = time.time() - t0
    t0 = time.time()
    R_br = run_many(brian, c.W, sugar, 150.0, 1000.0, p, a.brian_trials, a.brian_workers)
    t_br = time.time() - t0
    mo, mb = R_ours.mean(0), R_br.mean(0)
    act = (mo > 0) | (mb > 0)
    resp = act.copy()
    resp[sugar] = False
    corr = float(np.corrcoef(mo[resp], mb[resp])[0, 1])
    sh = shuffle(c, seed=1)
    R_sh = run_many(ours, sh.W, sugar, 150.0, 1000.0, p, 8, a.workers)
    res["flywire_v783"] = {
        "n_sugar_grn": int(len(sugar)),
        "mn9_rate_ours_hz": [float(R_ours[:, mn9].mean()), float(R_ours[:, mn9].std())],
        "mn9_rate_brian2_hz": [float(R_br[:, mn9].mean()), float(R_br[:, mn9].std())],
        "mn9_rate_shuffled_hz": [float(R_sh[:, mn9].mean()), float(R_sh[:, mn9].std())],
        "n_trials_ours": a.trials, "n_trials_brian2": a.brian_trials,
        "n_active_ours": int((mo > 0).sum()), "n_active_brian2": int((mb > 0).sum()),
        "rate_correlation_responsive_neurons": corr, "n_responsive_neurons": int(resp.sum()),
        "wall_s_per_trial_ours": t_ours / a.trials * a.workers, "wall_s_per_trial_brian2": t_br / a.brian_trials * a.brian_workers,
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.scatter(mb[resp], mo[resp], s=8, alpha=0.6)
    lim = max(mo[resp].max(), mb[resp].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.scatter(mb[mn9], mo[mn9], s=40, c="r", label="MN9")
    ax.set_xlabel(f"Brian2 reference rate [Hz] (n={a.brian_trials})")
    ax.set_ylabel(f"flybrainlab engine rate [Hz] (n={a.trials})")
    ax.set_title(f"FlyWire v783, sugar GRNs 150 Hz: r = {corr:.3f}")
    ax.legend()

    # ---------------- 2. MaleCNS
    c = load_connectome("malecns_v1.0")
    g = GroupResolver(c)
    p = params_for("malecns_v1.0")
    out = {}
    for name, src, rate, targets in (("sugar", "sugar_grn", 150.0, ["mn9"]),
                                     ("looming", "lplc2", 100.0, ["giant_fiber", "ttmn"])):
        exc = g.resolve(src)
        R = run_many(ours, c.W, exc, rate, 1000.0, p, 8, a.workers)
        sh = shuffle(c, seed=1)
        Rs = run_many(ours, sh.W, exc, rate, 1000.0, p, 8, a.workers)
        out[name] = {"source": src, "n_source": int(len(exc)), "rate_hz": rate, "n_active": int((R.mean(0) > 0).sum())}
        for t in targets:
            idx = g.resolve(t)
            out[name][t] = {"intact_hz": [float(R[:, idx].mean()), float(R[:, idx].mean(1).std())],
                            "shuffled_hz": [float(Rs[:, idx].mean()), float(Rs[:, idx].mean(1).std())]}
    res["malecns_v1.0"] = out
    ax = axes[1]
    labels, vi, vs = [], [], []
    for name, d in out.items():
        for t, v in d.items():
            if isinstance(v, dict):
                labels.append(f"{d['source']}->{t}")
                vi.append(v["intact_hz"][0])
                vs.append(v["shuffled_hz"][0])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, vi, 0.4, label="intact")
    ax.bar(x + 0.2, vs, 0.4, label="shuffled")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("mean rate [Hz]")
    ax.set_title(f"MaleCNS v1.0 (w_syn {p.w_syn} mV)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "m1_validation.png", dpi=120)
    (OUT / "m1_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
