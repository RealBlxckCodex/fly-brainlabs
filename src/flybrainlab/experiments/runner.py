"""Experiment runner: config resolution, interventions, parallel trials, logging.

``run_experiment`` is the single entry point used by the CLI, notebooks and the LLM
tool layer. Every run is reproducible from its ``config.json`` (config + seeds + git
commit + package versions).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from .. import provenance
from ..config import deep_merge, default_config, scenario_config
from ..connectome import GroupResolver, load_connectome
from ..connectome.interventions import apply_interventions
from ..loop import run_trial
from ..paths import RUNS
from .store import dump_json, summarize

_CACHE: dict = {}  # dataset -> (connectome, groups); shared with forked workers


def get_connectome(dataset: str):
    if dataset not in _CACHE:
        c = load_connectome(dataset)
        _CACHE[dataset] = (c, GroupResolver(c))
    return _CACHE[dataset]


def resolve_config(scenario: str, dataset: str | None = None, gravity: float | None = None, lesion=None,
                   rewiring=None, shuffle=False, seed: int = 0, n_trials: int | None = None,
                   ablate_senses=None, overrides: dict | None = None, label: str | None = None) -> dict:
    cfg = default_config()
    ds = dataset or cfg["dataset"]
    cfg = deep_merge(cfg, cfg.get("dataset_overrides", {}).get(ds, {}))
    cfg = deep_merge(cfg, scenario_config(scenario))
    cfg = deep_merge(cfg, overrides or {})
    cfg.update(dataset=ds, scenario=scenario, seed=int(seed), label=label)
    if gravity is not None:
        cfg["world"]["gravity"] = float(gravity)
    if n_trials is not None:
        cfg["experiment"]["n_trials"] = int(n_trials)
    if ablate_senses is not None:
        cfg["ablate_senses"] = list(ablate_senses)
    cfg["intervention"] = {"lesion": lesion, "rewiring": rewiring, "shuffle": shuffle}
    cfg.pop("dataset_overrides", None)
    return cfg


def _trial_seed(base: int, k: int) -> int:
    return int(base) * 1000 + int(k)


def _prepare_connectome(cfg: dict, trial_seed: int):
    c, g = get_connectome(cfg["dataset"])
    iv = cfg["intervention"]
    shuffle = iv.get("shuffle")
    if shuffle:
        shuffle = cfg["experiment"].get("shuffle_mode", "targets") if shuffle is True else shuffle
    c2, silenced = apply_interventions(c, g, lesion_spec=iv.get("lesion"), rewiring=iv.get("rewiring"),
                                       shuffle_spec=shuffle, seed=trial_seed if shuffle else cfg["seed"],
                                       model_mods=cfg.get("model_mods"))
    return c2, g, silenced


_SHARED: dict = {}


def _worker(args):
    k, cfg = args
    seed = _trial_seed(cfg["seed"], k)
    try:
        if cfg["intervention"].get("shuffle"):
            c, g, silenced = _prepare_connectome(cfg, seed)  # independent shuffle per trial
        else:
            c, g, silenced = _SHARED["connectome"]
        keep = k < cfg["experiment"].get("keep_trajectories", 3)
        res = run_trial(c, g, cfg, seed=seed, silenced=silenced, keep_log=keep)
        return k, seed, res, None
    except Exception:  # pragma: no cover - reported in the run log
        return k, seed, None, traceback.format_exc()


def _new_run_id(cfg: dict, suffix: str = "") -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}_{cfg['scenario']}_{provenance.config_hash(cfg)[:6]}{suffix}"


def _execute(cfg: dict, run_id: str, workers: int, verbose: bool) -> dict:
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    n = int(cfg["experiment"]["n_trials"])
    t0 = time.time()
    c, g = get_connectome(cfg["dataset"])
    if not cfg["intervention"].get("shuffle"):
        _SHARED["connectome"] = _prepare_connectome(cfg, cfg["seed"])
        csum = _SHARED["connectome"][0].summary()
        n_sil = len(_SHARED["connectome"][2])
    else:
        csum = {**c.summary(), "modifications": ["shuffle per trial (seed = trial seed + 10000)"]}
        n_sil = None
    cfg_logged = {**cfg, "run_id": run_id, "created": provenance.utc_now(), "git": provenance.git_commit(),
                  "environment": provenance.environment(), "connectome": csum, "n_silenced_neurons": n_sil,
                  "trial_seeds": [_trial_seed(cfg["seed"], k) for k in range(n)]}
    dump_json(cfg_logged, out / "config.json")
    jobs = [(k, cfg) for k in range(n)]
    results = []
    workers = max(1, min(workers, n))
    if workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            for r in pool.imap_unordered(_worker, jobs):
                results.append(r)
                if verbose:
                    print(f"  [{run_id}] trial {r[0]} done ({len(results)}/{n})", flush=True)
    else:
        for j in jobs:
            results.append(_worker(j))
            if verbose:
                print(f"  [{run_id}] trial {j[0]} done", flush=True)
    results.sort(key=lambda r: r[0])
    rows, events, errors = [], {}, []
    for k, seed, res, err in results:
        if err:
            errors.append({"trial": k, "seed": seed, "error": err})
            continue
        rows.append({"trial": k, "seed": seed, **res.metrics, "wall_s": round(res.wall_s, 2)})
        events[k] = res.events
        if res.log:
            np.savez_compressed(out / f"traj_{k}.npz", **{key: np.asarray(v) for key, v in res.log.items()
                                                         if key != "mode"},
                                mode=np.asarray(res.log.get("mode", []), dtype=str),
                                dn_rates=res.neural["dn_rates"], dn_groups=np.asarray(res.neural["readout_groups"]),
                                raster_t=res.neural["raster"][0], raster_i=res.neural["raster"][1],
                                events=np.asarray([(t, e) for t, e in res.events], dtype=object))
    df = pd.DataFrame(rows)
    df.to_csv(out / "trials.csv", index=False)
    summary = {"run_id": run_id, "label": cfg.get("label"), "scenario": cfg["scenario"], "dataset": cfg["dataset"],
               "n_trials_ok": len(rows), "n_errors": len(errors), "wall_s": round(time.time() - t0, 1),
               "metrics": summarize(df)}
    dump_json(summary, out / "summary.json")
    dump_json({"events": events, "errors": errors}, out / "events.json")
    if errors and verbose:
        print(errors[0]["error"])
    return summary


def run_experiment(scenario: str, dataset: str | None = None, gravity: float | None = None, lesion=None,
                   rewiring=None, shuffle=False, seed: int = 0, n_trials: int | None = None, ablate_senses=None,
                   overrides: dict | None = None, label: str | None = None, shuffle_control: bool | None = None,
                   workers: int | None = None, verbose: bool = True) -> dict:
    """Run an experiment (n_trials seeds) and, by default, its shuffled-connectome control.

    Returns ``{"run_id", "summary", "control_run_id", "control_summary"}``.
    """
    cfg = resolve_config(scenario, dataset, gravity, lesion, rewiring, shuffle, seed, n_trials, ablate_senses,
                         overrides, label)
    workers = workers or int(os.environ.get("FLYLAB_WORKERS", cfg["experiment"].get("workers", 4)))
    run_id = _new_run_id(cfg)
    if verbose:
        print(f"[run] {run_id}: {scenario} on {cfg['dataset']} x{cfg['experiment']['n_trials']} trials", flush=True)
    summary = _execute(cfg, run_id, workers, verbose)
    result = {"run_id": run_id, "summary": summary, "control_run_id": None, "control_summary": None}
    do_ctl = cfg["experiment"].get("shuffle_control", True) if shuffle_control is None else shuffle_control
    if do_ctl and not shuffle:
        ccfg = {**cfg, "intervention": {**cfg["intervention"], "shuffle": True}, "control_of": run_id,
                "label": f"{label or scenario} [shuffle control]"}
        cid = run_id + "_shuffle"
        if verbose:
            print(f"[run] {cid}: shuffled-connectome control", flush=True)
        result["control_run_id"] = cid
        result["control_summary"] = _execute(ccfg, cid, workers, verbose)
    return result
