"""Offline experimenter with a fixed plan (no API key needed).

Exercises exactly the same tool layer and report validation as the LLM. Useful for CI,
for reproducing the reference study, and as a template for what a good report contains.
"""
from __future__ import annotations

from .tools import ExperimentTools


def _f(x, nd=3):
    return "nan" if x is None else f"{x:.{nd}f}"


def giant_fiber_study(dataset: str | None = None, n_trials: int = 8, seed: int = 1, workers: int | None = None,
                      scenario: str = "looming") -> dict:
    """Hypothesis: looming-evoked takeoff depends on the Giant Fiber (DNp01) and on real wiring."""
    T = ExperimentTools(dataset=dataset, trial_budget=4 * n_trials + 4, workers=workers)
    T.call("list_neuron_groups", {"query": "giant_fiber"})
    T.call("describe_scenarios", {})
    base = T.call("run_experiment", {"scenario": scenario, "label": "intact", "n_trials": n_trials, "seed": seed})
    les = T.call("run_experiment", {"scenario": scenario, "label": "giant fiber lesion (DNp01)",
                                     "n_trials": n_trials, "seed": seed, "lesion": ["giant_fiber"]})
    a, s, l_ = base["run_id"], base["control_run_id"], les["run_id"]
    ma, ms, ml = (T.call("get_metrics", {"run_id": r}) for r in (a, s, l_))
    metric = "takeoff" if scenario == "looming" else "survived"
    c_les = T.call("compare", {"run_a": a, "run_b": l_, "metrics": [metric, "gf_spikes"]})
    c_shu = T.call("compare", {"run_a": a, "run_b": s, "metrics": [metric, "gf_spikes"]})

    def mean(m, k):
        return m["summary"][k]["mean"]

    def std(m, k):
        return m["summary"][k]["std"]

    p_les = next(r for r in c_les["metrics"] if r["metric"] == metric)
    p_shu = next(r for r in c_shu["metrics"] if r["metric"] == metric)
    findings = [
        {"statement": f"Intact connectome: mean {metric}", "run_id": a, "metric": metric, "statistic": "mean",
         "value": mean(ma, metric)},
        {"statement": f"Giant fiber lesion: mean {metric}", "run_id": l_, "metric": metric, "statistic": "mean",
         "value": mean(ml, metric)},
        {"statement": f"Shuffled connectome: mean {metric}", "run_id": s, "metric": metric, "statistic": "mean",
         "value": mean(ms, metric)},
        {"statement": "Intact: Giant Fiber spikes per trial", "run_id": a, "metric": "gf_spikes", "statistic": "mean",
         "value": mean(ma, "gf_spikes")},
        {"statement": "Shuffled: Giant Fiber spikes per trial", "run_id": s, "metric": "gf_spikes",
         "statistic": "mean", "value": mean(ms, "gf_spikes")},
    ]
    comparisons = []
    for c, lab in ((p_les, "intact vs GF lesion"), (p_shu, "intact vs shuffle")):
        if "p_value" in c:
            comparisons.append({"run_a": c_les["run_a"] if c is p_les else c_shu["run_a"],
                                "run_b": c_les["run_b"] if c is p_les else c_shu["run_b"], "metric": metric,
                                "p_value": c["p_value"], "interpretation": f"{lab}: Fisher exact test on {metric}"})
    report = {
        "title": f"Giant Fiber dependence of looming escape ({T.dataset}, {scenario})",
        "hypothesis": "Looming-evoked takeoff in the model is mediated by the Giant Fiber (DNp01) and requires the "
                      "real synaptic wiring from LPLC2/LC4 onto it.",
        "intervention": "Silence both DNp01 neurons (all their synapses removed, input clamped). Control: synaptic "
                        "targets shuffled per presynaptic neuron (out-degree, weights and signs preserved).",
        "runs": [a, l_], "controls": [s, les["control_run_id"]],
        "findings": findings, "comparisons": comparisons,
        "spread": f"Std across seeds of {metric}: intact {_f(std(ma, metric))}, lesion {_f(std(ml, metric))}, "
                  f"shuffle {_f(std(ms, metric))}. Std of gf_spikes intact {_f(std(ma, 'gf_spikes'))}.",
        "control_assessment": f"Shuffling synaptic targets changed mean {metric} from {_f(mean(ma, metric))} to "
                              f"{_f(mean(ms, metric))} (Fisher p = {_f(p_shu.get('p_value'), 4)}), and Giant Fiber "
                              f"spikes from {_f(mean(ma, 'gf_spikes'))} to {_f(mean(ms, 'gf_spikes'))}.",
        "limitations": "Takeoff is read out from DNp01 firing by a hand-set threshold, so the lesion result partly "
                       "follows from the readout design; the informative part is that LPLC2/LC4 drive reaches DNp01 "
                       "through the real wiring but not through the shuffled one. The jump direction readout is "
                       "hand-set and not validated.",
        "conclusion": "In this model, looming-evoked takeoff depends on the identified Giant Fiber pathway and on "
                      "the specific connectome wiring (lost after target shuffling). This is a statement about the "
                      "wiring diagram plus model assumptions, not about real flies.",
    }
    res = T.call("submit_report", report)
    return {"report": res["path"], "runs": [a, l_, s], "trials_used": T.trials_used}
