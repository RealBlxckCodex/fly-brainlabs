"""Command line interface: ``flylab <command>`` (or ``python -m flybrainlab.cli``)."""
from __future__ import annotations

import argparse
import json
import sys

import yaml


def _parse_set(items: list[str] | None) -> dict:
    """--set brain.w_syn=0.2 --set scenario_cfg.speed=40 -> nested dict (YAML-typed values)."""
    out: dict = {}
    for it in items or []:
        key, val = it.split("=", 1)
        d = out
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(val)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="flylab", description="Fly Brain Lab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download raw data")
    d.add_argument("--dataset", choices=["malecns_v1.0", "flywire_v783", "all"], default="all")
    d.add_argument("--flybody", action="store_true", help="also clone the flybody MuJoCo model")

    s = sub.add_parser("sanity", help="build/load connectome and check counts against the release")
    s.add_argument("--dataset", default=None)
    s.add_argument("--rebuild", action="store_true")

    g = sub.add_parser("groups", help="list or search neuron groups")
    g.add_argument("query", nargs="?", default="")
    g.add_argument("--dataset", default=None)

    r = sub.add_parser("run", help="run an experiment (+ shuffle control)")
    r.add_argument("scenario")
    r.add_argument("--dataset", default=None)
    r.add_argument("--trials", type=int, default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--gravity", type=float, default=None)
    r.add_argument("--lesion", nargs="*", default=None, help="group names")
    r.add_argument("--rewiring", default=None, help="YAML/JSON list of rewiring ops")
    r.add_argument("--ablate", nargs="*", default=None, help="senses to ablate")
    r.add_argument("--shuffle", action="store_true", help="run on a shuffled connectome")
    r.add_argument("--no-control", action="store_true", help="skip the automatic shuffle control")
    r.add_argument("--label", default=None)
    r.add_argument("--workers", type=int, default=None)
    r.add_argument("--set", action="append", help="config override key.path=value")
    r.add_argument("--plot", action="store_true")

    c = sub.add_parser("compare", help="compare two runs")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--metrics", nargs="*")

    sub.add_parser("runs", help="list runs")
    p = sub.add_parser("plot", help="plot a run")
    p.add_argument("run_id")

    a = sub.add_parser("agent", help="LLM experimenter (needs Anthropic credentials)")
    a.add_argument("question")
    a.add_argument("--dataset", default=None)
    a.add_argument("--model", default=None)
    a.add_argument("--budget", type=int, default=120, help="max simulated trials incl. controls")
    a.add_argument("--effort", default="high")
    a.add_argument("--workers", type=int, default=None)

    st = sub.add_parser("study", help="scripted reference study (offline experimenter)")
    st.add_argument("name", choices=["giant-fiber"])
    st.add_argument("--dataset", default=None)
    st.add_argument("--scenario", default="looming", choices=["looming", "spider"])
    st.add_argument("--trials", type=int, default=8)
    st.add_argument("--seed", type=int, default=1)
    st.add_argument("--workers", type=int, default=None)

    args = ap.parse_args(argv)

    if args.cmd == "download":
        from . import download

        if args.dataset in ("malecns_v1.0", "all"):
            download.download_malecns()
        if args.dataset in ("flywire_v783", "all"):
            download.download_flywire()
        if args.flybody:
            download.download_flybody()
    elif args.cmd == "sanity":
        from .config import default_config
        from .connectome import load_connectome
        from .connectome.loaders import sanity_check

        c = load_connectome(args.dataset or default_config()["dataset"], rebuild=args.rebuild)
        res = sanity_check(c)
        print(json.dumps(res, indent=2, default=str))
        return 0 if res["ok"] else 1
    elif args.cmd == "groups":
        from .agent.tools import ExperimentTools

        print(json.dumps(ExperimentTools(dataset=args.dataset, verbose=False).call("list_neuron_groups",
                                                                                   {"query": args.query}), indent=1))
    elif args.cmd == "run":
        from .experiments import run_experiment

        rew = yaml.safe_load(args.rewiring) if args.rewiring else None
        res = run_experiment(args.scenario, dataset=args.dataset, gravity=args.gravity, lesion=args.lesion,
                             rewiring=rew, shuffle=args.shuffle, seed=args.seed, n_trials=args.trials,
                             ablate_senses=args.ablate, overrides=_parse_set(args.set), label=args.label,
                             shuffle_control=False if args.no_control else None, workers=args.workers)
        for key in ("summary", "control_summary"):
            sm = res[key]
            if not sm:
                continue
            print(f"\n== {sm['run_id']} ({sm['label']}) ok={sm['n_trials_ok']} errors={sm['n_errors']} "
                  f"wall={sm['wall_s']}s")
            for m, v in sm["metrics"].items():
                if v.get("n"):
                    print(f"  {m:28s} mean={v['mean']:.4g}  std={v['std']:.3g}  n={v['n']}")
        if args.plot:
            from .viz.plots import plot_run

            print(plot_run(res["run_id"]))
    elif args.cmd == "compare":
        from .experiments import compare_runs

        print(json.dumps(compare_runs(args.run_a, args.run_b, args.metrics), indent=1))
    elif args.cmd == "runs":
        from .experiments import RunStore

        for r in RunStore().list():
            print(f"{r['run_id']:55s} {r['scenario']:12s} {r['dataset']:14s} {r['label']}")
    elif args.cmd == "plot":
        from .viz.plots import plot_run

        print(plot_run(args.run_id))
    elif args.cmd == "agent":
        from .agent.llm import DEFAULT_MODEL, run_agent

        log = run_agent(args.question, dataset=args.dataset, model=args.model or DEFAULT_MODEL,
                        trial_budget=args.budget, effort=args.effort, workers=args.workers)
        print("\nreports:", log["reports"], "\nsession log:", log["session_log"])
    elif args.cmd == "study":
        from .agent.scripted import giant_fiber_study

        print(json.dumps(giant_fiber_study(args.dataset, n_trials=args.trials, seed=args.seed,
                                           workers=args.workers, scenario=args.scenario), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
