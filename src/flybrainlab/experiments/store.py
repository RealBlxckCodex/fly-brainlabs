"""Run storage. Every run lives in runs/<run_id>/ with:

config.json     fully resolved config + provenance (git commit, seeds, versions)
trials.csv      one row of computed metrics per trial (seed, wall time ...)
summary.json    aggregated statistics per metric (mean, std, sem, median, 95% CI, n)
events.json     per-trial event lists (takeoff, landing ...)
traj_<k>.npz    full trajectory + DN rates for the first trials
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import RUNS


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and np.isnan(o):
        return None
    return str(o)


def dump_json(obj, path: Path) -> None:
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, float) and np.isnan(x):
            return None
        if isinstance(x, (np.floating, np.integer, np.ndarray)):
            return _jsonable(x)
        return x

    path.write_text(json.dumps(clean(obj), indent=2, default=_jsonable))


def summarize(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    skip = {"trial", "seed", "wall_s"}
    for col in df.columns:
        if col in skip or not np.issubdtype(df[col].dtype, np.number):
            continue
        x = df[col].to_numpy(float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            out[col] = {"n": 0, "n_missing": int(len(df))}
            continue
        boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1) if len(x) > 1 else np.array([x[0]])
        out[col] = {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                    "sem": float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
                    "median": float(np.median(x)), "min": float(x.min()), "max": float(x.max()),
                    "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                    "n": int(len(x)), "n_missing": int(len(df) - len(x))}
    return out


class RunStore:
    def __init__(self, root: Path = RUNS):
        self.root = Path(root)

    def path(self, run_id: str) -> Path:
        p = self.root / run_id
        if not p.exists():
            raise KeyError(f"unknown run_id {run_id!r}")
        return p

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*/config.json")):
            cfg = json.loads(p.read_text())
            out.append({"run_id": p.parent.name, "scenario": cfg.get("scenario"), "dataset": cfg.get("dataset"),
                        "label": cfg.get("label"), "control_of": cfg.get("control_of"),
                        "n_trials": cfg.get("experiment", {}).get("n_trials")})
        return out

    def config(self, run_id: str) -> dict:
        return json.loads((self.path(run_id) / "config.json").read_text())

    def trials(self, run_id: str) -> pd.DataFrame:
        return pd.read_csv(self.path(run_id) / "trials.csv")

    def summary(self, run_id: str) -> dict:
        return json.loads((self.path(run_id) / "summary.json").read_text())

    def trajectory(self, run_id: str, k: int = 0) -> dict:
        d = np.load(self.path(run_id) / f"traj_{k}.npz", allow_pickle=True)
        return {key: d[key] for key in d.files}
