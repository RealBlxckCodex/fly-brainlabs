"""Reference experiment series (M3-M6) on the default dataset. Results feed docs/results.md.

Every condition runs n seeds plus its own shuffled-connectome control. Same seed across
conditions of one study, so scenario randomisation (stimulus direction, spider start,
bright side) is matched trial by trial.

Usage: python scripts/reference_experiments.py [--dataset malecns_v1.0] [--only spider zerog]
"""
from __future__ import annotations

import argparse
import json

from flybrainlab.experiments import compare_runs, run_experiment
from flybrainlab.paths import ROOT
from flybrainlab.viz.plots import plot_comparison, plot_run

STUDIES = {
    "looming": dict(n=12, metrics=["takeoff", "time_before_collision_ms", "theta_at_takeoff_deg", "escape_dir_error_deg", "gf_spikes"],
                    conditions=[("intact", {}), ("giant fiber lesion", {"lesion": ["giant_fiber"]}),
                                ("LPLC2+LC4 lesion", {"lesion": ["looming_vpn"]})]),
    "spider": dict(n=12, metrics=["survived", "survival_time_s", "takeoff", "escape_latency_ms", "escape_dir_error_deg"],
                   conditions=[("intact", {}), ("LPLC2+LC4 lesion", {"lesion": ["looming_vpn"]}),
                               ("giant fiber lesion", {"lesion": ["giant_fiber"]})]),
    "zerog": dict(n=10, metrics=["mean_angspeed_rad_s", "mean_tilt_deg", "corrective_roll_corr", "control_effort",
                                 "corrective_yaw_corr", "drift_cm", "upright_fraction"],
                  conditions=[("1 g intact", {"gravity": 981.0}), ("0 g intact", {"gravity": 0.0}),
                              ("0 g HS/VS lesion", {"gravity": 0.0, "lesion": ["hs_L", "hs_R", "vs_L", "vs_R"]}),
                              ("1 g JO ablated", {"gravity": 981.0, "ablate_senses": ["jo"]})]),
    "lightchoice": dict(n=12, metrics=["chose", "chose_bright", "chose_left", "progress_x_cm"],
                        conditions=[("intact", {}), ("vision ablated", {"ablate_senses": ["photoreceptors"]})]),
    "foraging": dict(n=8, metrics=["reached", "final_distance_cm", "approach_cm", "path_efficiency"],
                     conditions=[("intact", {}), ("olfaction ablated", {"ablate_senses": ["olfaction"]})]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out_path = ROOT / "docs" / "validation" / "reference_runs.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name, st in STUDIES.items():
        if a.only and name not in a.only:
            continue
        runs = []
        for label, kw in st["conditions"]:
            r = run_experiment(name, dataset=a.dataset, seed=a.seed, n_trials=st["n"], label=f"{name}: {label}",
                               workers=a.workers, **kw)
            runs.append({"label": label, "run_id": r["run_id"], "control_run_id": r["control_run_id"]})
            plot_run(r["run_id"])
        base = runs[0]["run_id"]
        comps = {"vs_shuffle": compare_runs(base, runs[0]["control_run_id"], st["metrics"])}
        for rr in runs[1:]:
            comps[f"vs_{rr['label']}"] = compare_runs(base, rr["run_id"], st["metrics"])
        ids = [x["run_id"] for x in runs] + [runs[0]["control_run_id"]]
        labels = [x["label"] for x in runs] + ["intact, shuffled"]
        fig = plot_comparison(ids, st["metrics"][:4], labels=labels,
                              out=ROOT / "docs" / "validation" / f"ref_{name}.png")
        results[name] = {"runs": runs, "comparisons": comps, "figure": str(fig.relative_to(ROOT))}
        out_path.write_text(json.dumps(results, indent=1, default=str))
        print(f"== {name} done", flush=True)


if __name__ == "__main__":
    main()
