"""Fit (and honestly evaluate) a trained escape-direction readout.

Protocol (open loop, fly fixed at the origin facing +x):
  for each looming azimuth a (n_az evenly spaced, random jitter) and seed:
    run brain + looming encoder until the Giant Fiber trace crosses the takeoff threshold
    (same rule as the closed loop), record the filtered rate of every descending neuron
  target = direction away from the stimulus, relative to heading: (cos(a+pi), sin(a+pi))
Fit ridge regression (standardised features), evaluate with leave-azimuth-out CV:
  mean absolute angular error vs. chance (90 deg), intact vs. shuffled connectome.
The intact fit is saved to configs/readouts/escape_<dataset>.json; enable with
  --set readout.jump_yaw_fitted.path=configs/readouts/escape_<dataset>.json

Usage: python scripts/fit_escape_readout.py [--dataset malecns_v1.0] [--n-az 24] [--seeds 2]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import numpy as np

from flybrainlab.brain import LIFNetwork, LIFParams
from flybrainlab.config import deep_merge, default_config
from flybrainlab.connectome import GroupResolver, load_connectome
from flybrainlab.connectome.interventions import shuffle
from flybrainlab.motor.readout import Readout
from flybrainlab.paths import ROOT
from flybrainlab.sensory.encoders import LoomingObject, Percept, SensoryEncoder

_G: dict = {}


def episode(args):
    az, seed = args
    c, g, p, dn = _G["c"], _G["g"], _G["p"], _G["dn"]
    enc = SensoryEncoder(g, {"tonic": {}})
    ro = Readout(g, {}, dt_ms=p.dt)
    net = LIFNetwork(c.W, p, seed=seed)
    eye = {"L": np.array([0.085, 0.03, 0.095]), "R": np.array([0.085, -0.03, 0.095])}
    d = np.array([np.cos(az) * np.cos(0.35), np.sin(az) * np.cos(0.35), np.sin(0.35)])
    pos_dn = np.full(c.n_neurons, -1)
    pos_dn[dn] = np.arange(len(dn))
    trace = np.zeros(len(dn))
    decay = np.exp(-p.dt / 20.0)
    dist, speed, r = 10.0, 25.0, 0.5
    for step in range(int(0.5 / (p.dt * 1e-3))):
        t = step * p.dt * 1e-3
        if step % 10 == 0:
            dd = max(dist - speed * t, r)
            P = Percept(t=t, R=np.eye(3), eye_pos=eye, antenna_pos=eye, velocity=np.zeros(3), angvel=np.zeros(3),
                        gravity=np.array([0, 0, -981.0]), objects=[LoomingObject("o", np.array([0, 0, 0.095]) + d * dd, r)])
            idx, rates = enc.encode(P, dt=1e-3)
            net.set_poisson_rates(idx, rates)
            if ro.command().takeoff:
                return trace / 0.02, True
        spk = net.step()
        ro.observe(spk)
        trace *= decay
        k = pos_dn[spk]
        k = k[k >= 0]
        if k.size:
            np.add.at(trace, k, 1.0)
    return trace / 0.02, False


def collect(c, g, p, dn, azs, seeds, workers):
    _G.update(c=c, g=g, p=p, dn=dn)
    jobs = [(a, s) for a in azs for s in seeds]
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(episode, jobs)
    X = np.array([r[0] for r in res])
    took = np.array([r[1] for r in res])
    a = np.array([j[0] for j in jobs])
    return X, took, a


def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    W = np.linalg.solve(A, Z.T @ (Y - Y.mean(0)))
    return W.T, Y.mean(0), mu, sd


def ang_err(pred, target):
    return np.degrees(np.abs(np.angle(np.exp(1j * (pred - target)))))


def cv_error(X, a, groups, lam):
    target = a + np.pi
    Y = np.c_[np.cos(target), np.sin(target)]
    errs = np.zeros(len(a))
    for gi in np.unique(groups):
        te = groups == gi
        coef, b, mu, sd = ridge_fit(X[~te], Y[~te], lam)
        y = ((X[te] - mu) / sd) @ coef.T + b
        errs[te] = ang_err(np.arctan2(y[:, 1], y[:, 0]), target[te])
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="malecns_v1.0")
    ap.add_argument("--n-az", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--lam", type=float, default=50.0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cfg = default_config()
    cfg = deep_merge(cfg, cfg["dataset_overrides"][a.dataset])
    p = LIFParams.from_dict(cfg["brain"])
    c = load_connectome(a.dataset)
    g = GroupResolver(c)
    dn = g.resolve("descending")
    rng = np.random.default_rng(0)
    azs = np.linspace(-np.pi, np.pi, a.n_az, endpoint=False) + rng.uniform(-0.05, 0.05, a.n_az)
    seeds = list(range(a.seeds))
    out = {"dataset": a.dataset, "n_descending": int(len(dn)), "n_azimuths": a.n_az, "seeds": a.seeds, "lambda": a.lam}
    for label, conn in (("intact", c), ("shuffled", shuffle(c, seed=123))):
        X, took, az = collect(conn, g, p, dn, azs, seeds, a.workers)
        grp = np.repeat(np.arange(a.n_az), a.seeds)
        errs = cv_error(X, az, grp, a.lam)
        out[label] = {"takeoff_fraction": float(took.mean()), "cv_mean_abs_error_deg": float(errs.mean()),
                      "cv_median_abs_error_deg": float(np.median(errs)), "n_active_dn": int((X.sum(0) > 0).sum())}
        print(label, out[label], flush=True)
        if label == "intact":
            target = az + np.pi
            coef, b, mu, sd = ridge_fit(X, np.c_[np.cos(target), np.sin(target)], a.lam)
            keep = X.sum(0) > 0  # only DNs that ever fired carry weight
            w = {"dataset": a.dataset, "neuron_ids": c.neurons["id"].to_numpy()[dn[keep]].tolist(),
                 "coef": coef[:, keep].tolist(), "intercept": b.tolist(), "feature_mean": mu[keep].tolist(),
                 "feature_scale": sd[keep].tolist(), "lambda": a.lam,
                 "note": "trained readout: DN rates at GF trigger -> escape direction (fit_escape_readout.py)"}
            d = ROOT / "configs" / "readouts"
            d.mkdir(exist_ok=True)
            (d / f"escape_{a.dataset}.json").write_text(json.dumps(w))
    out["chance_mean_abs_error_deg"] = 90.0
    (ROOT / "docs" / "validation" / f"escape_readout_{a.dataset}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
