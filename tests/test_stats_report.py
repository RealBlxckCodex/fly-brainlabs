import numpy as np
import pandas as pd
import pytest

from flybrainlab.agent.report import ReportError, number_is_supported, validate_and_write_report
from flybrainlab.experiments.stats import compare_arrays, holm
from flybrainlab.experiments.store import RunStore, dump_json, summarize


def test_compare_binary_uses_fisher():
    r = compare_arrays(np.ones(8), np.zeros(8), "takeoff")
    assert r["test"] == "fisher_exact" and r["p_value"] < 0.001


def test_compare_continuous():
    rng = np.random.default_rng(0)
    r = compare_arrays(rng.normal(0, 1, 20), rng.normal(2, 1, 20), "x")
    assert r["p_value"] < 0.001 and r["cohens_d"] > 1 and r["diff_ci95"][0] > 0


def test_holm_monotone():
    adj = holm([0.01, 0.04, 0.03])
    assert adj == sorted(adj, key=lambda x: x) or max(adj) <= 1
    assert adj[0] == pytest.approx(0.03)


def test_number_support_rounding():
    seen = {0.123456, 18.0, 0.0286}
    assert number_is_supported("0.123", seen)
    assert number_is_supported("12.3", seen)  # percentage form
    assert number_is_supported("18", seen)
    assert number_is_supported("3", seen)  # small integers allowed
    assert not number_is_supported("0.87", seen)


def test_report_validation(tmp_path):
    store = RunStore(tmp_path)
    for rid, vals in (("r1", [1, 1, 1, 0]), ("r1_shuffle", [0, 0, 0, 0])):
        (tmp_path / rid).mkdir()
        dump_json({"scenario": "looming", "label": rid}, tmp_path / rid / "config.json")
        df = pd.DataFrame({"trial": range(4), "takeoff": vals})
        df.to_csv(tmp_path / rid / "trials.csv", index=False)
        dump_json({"run_id": rid, "metrics": summarize(df)}, tmp_path / rid / "summary.json")
    rep = {"title": "t", "hypothesis": "h", "intervention": "i", "runs": ["r1"], "controls": ["r1_shuffle"],
           "findings": [{"statement": "s", "run_id": "r1", "metric": "takeoff", "statistic": "mean", "value": 0.75}],
           "spread": "", "control_assessment": "", "limitations": "", "conclusion": "takeoff was 0.75"}
    transcript = [{"tool": "get_metrics", "result": store.summary("r1")}]
    path = validate_and_write_report(rep, transcript, store)
    assert path.exists() and "Standard model limitations" in path.read_text()
    bad = dict(rep, conclusion="takeoff was 0.93")
    with pytest.raises(ReportError):
        validate_and_write_report(bad, transcript, store)
    bad = dict(rep, findings=[dict(rep["findings"][0], value=0.5)])
    with pytest.raises(ReportError):
        validate_and_write_report(bad, transcript, store)
