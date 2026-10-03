"""Tool layer for the LLM experimenter.

The LLM never simulates or estimates anything itself. It can only:
  list_neuron_groups  find named groups / cell types
  describe_scenarios  read the scenario catalogue and metric definitions
  run_experiment      launch a simulation (always paired with a shuffle control)
  get_metrics         read *computed* metrics of a run
  compare             statistics between two runs (computed here)
  submit_report       file the final report (validated: numbers must come from tool output)
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from ..config import default_config
from ..env.scenarios import SCENARIOS
from ..experiments import RunStore, compare_runs, run_experiment
from ..experiments.runner import get_connectome
from .report import ReportError, validate_and_write_report

_STR_LIST = {"type": "array", "items": {"type": "string"}}

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "list_neuron_groups",
        "description": (
            "Search named neuron groups and connectome cell types (case-insensitive substring). "
            "Returns group names with neuron counts and matching cell types with a selector that can be used "
            "as a lesion target. Use an empty query to list all named groups."),
        "input_schema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "substring, e.g. 'DNa', 'lplc2', 'JO'"}},
            "required": ["query"], "additionalProperties": False},
    },
    {
        "name": "describe_scenarios",
        "description": "List the available scenarios, their default parameters and the metrics each one computes.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "run_experiment",
        "description": (
            "Run a closed-loop simulation experiment with n_trials independent seeds. A shuffled-connectome "
            "control (same settings, synaptic targets randomised per neuron) is run automatically and its "
            "run_id is returned as control_run_id. Returns compact computed summaries. Expensive: each trial "
            "takes seconds to a minute of compute; total trials are budgeted."),
        "input_schema": {"type": "object", "properties": {
            "scenario": {"type": "string", "enum": sorted(SCENARIOS)},
            "label": {"type": "string", "description": "short human-readable label for this condition"},
            "n_trials": {"type": "integer", "minimum": 2, "maximum": 30},
            "seed": {"type": "integer", "description": "base seed; trial k uses seed*1000+k"},
            "gravity": {"type": "number", "description": "cm/s^2; 981 Earth, 0 weightless (air unchanged)"},
            "lesion": {**_STR_LIST, "description": "neuron group names to silence (see list_neuron_groups)"},
            "rewiring": {"type": "array", "description": (
                "list of ops: {op: scale|cut|add|shift|flip_sign, pre: group, post: group, factor?: number, "
                "synapses?: number, to?: group}"), "items": {"type": "object"}},
            "ablate_senses": {**_STR_LIST, "description": "vision, vision_L, vision_R, looming, photoreceptors, olfaction, jo"},
            "scenario_overrides": {"type": "object", "description": "override scenario parameters, e.g. {speed: 40}"},
        }, "required": ["scenario", "label", "n_trials", "seed"], "additionalProperties": False},
    },
    {
        "name": "get_metrics",
        "description": "Computed metrics of a run: per-metric mean/std/sem/median/95% CI/n plus per-trial values.",
        "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}},
                         "required": ["run_id"], "additionalProperties": False},
    },
    {
        "name": "compare",
        "description": (
            "Statistical comparison of two runs per metric: means, std, difference (b - a) with bootstrap 95% CI, "
            "Welch t-test + Mann-Whitney (continuous) or Fisher exact (binary), Cohen's d, Holm-corrected p."),
        "input_schema": {"type": "object", "properties": {
            "run_a": {"type": "string"}, "run_b": {"type": "string"},
            "metrics": {**_STR_LIST, "description": "optional subset of metric names"}},
            "required": ["run_a", "run_b"], "additionalProperties": False},
    },
    {
        "name": "submit_report",
        "description": (
            "Submit the final report. It is validated: every number in any text field must appear in an earlier "
            "tool result (rounded as written), and every finding must cite run_id + metric + statistic whose value "
            "is checked against the stored results. Rejected reports return the offending items; fix and resubmit."),
        "input_schema": {"type": "object", "properties": {
            "title": {"type": "string"},
            "hypothesis": {"type": "string"},
            "intervention": {"type": "string", "description": "what was changed (lesion/rewiring/scenario)"},
            "runs": {**_STR_LIST, "description": "run_ids of the experimental conditions"},
            "controls": {**_STR_LIST, "description": "run_ids of controls (shuffle, intact baseline)"},
            "findings": {"type": "array", "items": {"type": "object", "properties": {
                "statement": {"type": "string"}, "run_id": {"type": "string"}, "metric": {"type": "string"},
                "statistic": {"type": "string", "enum": ["mean", "std", "sem", "median", "n", "ci95_low", "ci95_high"]},
                "value": {"type": "number"}},
                "required": ["statement", "run_id", "metric", "statistic", "value"], "additionalProperties": False}},
            "comparisons": {"type": "array", "items": {"type": "object", "properties": {
                "run_a": {"type": "string"}, "run_b": {"type": "string"}, "metric": {"type": "string"},
                "p_value": {"type": "number"}, "interpretation": {"type": "string"}},
                "required": ["run_a", "run_b", "metric", "p_value", "interpretation"], "additionalProperties": False}},
            "spread": {"type": "string", "description": "variability across seeds (std / CI) in words"},
            "control_assessment": {"type": "string", "description": "what the shuffle/other controls show"},
            "limitations": {"type": "string"},
            "conclusion": {"type": "string", "description": "phrased as a hypothesis, not a proof"},
        }, "required": ["title", "hypothesis", "intervention", "runs", "controls", "findings", "spread",
                        "control_assessment", "limitations", "conclusion"], "additionalProperties": False},
    },
]


def _round_floats(x, nd=4):
    if isinstance(x, float):
        return None if math.isnan(x) else round(x, nd)
    if isinstance(x, dict):
        return {k: _round_floats(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [_round_floats(v, nd) for v in x]
    return x


def _compact_summary(summary: dict | None) -> dict | None:
    if not summary:
        return None
    m = {k: {s: v.get(s) for s in ("mean", "std", "n") if s in v} for k, v in summary["metrics"].items()}
    return {"run_id": summary["run_id"], "label": summary.get("label"), "n_trials_ok": summary["n_trials_ok"],
            "n_errors": summary["n_errors"], "metrics": m}


class ExperimentTools:
    def __init__(self, dataset: str | None = None, trial_budget: int = 200, workers: int | None = None,
                 store: RunStore | None = None, verbose: bool = True):
        self.dataset = dataset or default_config()["dataset"]
        self.trial_budget = trial_budget
        self.trials_used = 0
        self.workers = workers
        self.store = store or RunStore()
        self.verbose = verbose
        self.transcript: list[dict] = []  # every tool call and result (used for number provenance)
        self.reports: list[str] = []

    # ------------------------------------------------------------------
    def call(self, name: str, args: dict) -> Any:
        fn = getattr(self, f"tool_{name}", None)
        if fn is None:
            raise ValueError(f"unknown tool {name}")
        result = _round_floats(fn(**args))
        self.transcript.append({"tool": name, "input": args, "result": result})
        return result

    def call_json(self, name: str, args: dict) -> tuple[str, bool]:
        try:
            return json.dumps(self.call(name, args), default=str), False
        except ReportError as e:
            self.transcript.append({"tool": name, "input": args, "error": str(e)})
            return json.dumps({"error": "report rejected", "problems": e.problems}), True
        except Exception as e:  # tool errors go back to the model
            self.transcript.append({"tool": name, "input": args, "error": repr(e)})
            return json.dumps({"error": repr(e)}), True

    # ------------------------------------------------------------------ tools
    def tool_list_neuron_groups(self, query: str = "") -> dict:
        _, g = get_connectome(self.dataset)
        if not query:
            return {"dataset": self.dataset, "groups": [g.describe(n) for n in g.names()]}
        return {"dataset": self.dataset, "matches": g.search(query)}

    def tool_describe_scenarios(self) -> dict:
        return {name: {"doc": (cls.__doc__ or "").strip(), "defaults": cls.defaults, "metrics": cls.metric_info()}
                for name, cls in SCENARIOS.items()}

    def tool_run_experiment(self, scenario: str, label: str, n_trials: int, seed: int, gravity: float | None = None,
                            lesion: list | None = None, rewiring: list | None = None,
                            ablate_senses: list | None = None, scenario_overrides: dict | None = None) -> dict:
        cost = 2 * n_trials  # + shuffle control
        if self.trials_used + cost > self.trial_budget:
            raise RuntimeError(f"trial budget exceeded: used {self.trials_used}, request {cost}, "
                               f"budget {self.trial_budget}")
        self.trials_used += cost
        overrides = {"scenario_cfg": scenario_overrides} if scenario_overrides else None
        r = run_experiment(scenario, dataset=self.dataset, gravity=gravity, lesion=lesion or None,
                           rewiring=rewiring or None, seed=seed, n_trials=n_trials, ablate_senses=ablate_senses,
                           overrides=overrides, label=label, shuffle_control=True, workers=self.workers,
                           verbose=self.verbose)
        return {"run_id": r["run_id"], "control_run_id": r["control_run_id"],
                "summary": _compact_summary(r["summary"]), "control_summary": _compact_summary(r["control_summary"]),
                "trials_used": self.trials_used, "trial_budget": self.trial_budget}

    def tool_get_metrics(self, run_id: str) -> dict:
        cfg = self.store.config(run_id)
        trials = self.store.trials(run_id).replace({np.nan: None})
        return {"run_id": run_id, "label": cfg.get("label"), "scenario": cfg["scenario"],
                "intervention": cfg.get("intervention"), "gravity": cfg["world"]["gravity"],
                "ablate_senses": cfg.get("ablate_senses"), "control_of": cfg.get("control_of"),
                "summary": self.store.summary(run_id)["metrics"], "trials": trials.to_dict(orient="records")}

    def tool_compare(self, run_a: str, run_b: str, metrics: list | None = None) -> dict:
        return compare_runs(run_a, run_b, metrics, store=self.store)

    def tool_submit_report(self, **report) -> dict:
        path = validate_and_write_report(report, self.transcript, self.store)
        self.reports.append(str(path))
        return {"accepted": True, "path": str(path)}
