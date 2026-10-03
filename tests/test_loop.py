"""Whole closed loop on the toy connectome: looming -> LPLC2 -> GF -> takeoff, shuffle abolishes it."""
import numpy as np

from flybrainlab.config import deep_merge, default_config, scenario_config
from flybrainlab.connectome.interventions import apply_interventions
from flybrainlab.loop import run_trial


def _cfg(scenario):
    cfg = default_config()
    cfg = deep_merge(cfg, scenario_config(scenario))
    cfg["scenario"] = scenario
    return cfg


def test_looming_escape_and_shuffle_control(toy):
    c, g = toy
    cfg = _cfg("looming")
    cfg["scenario_cfg"]["azimuth_deg"] = 60.0
    r = run_trial(c, g, cfg, seed=0)
    assert r.metrics["takeoff"] == 1.0 and r.metrics["gf_spikes"] > 0
    cs, _ = apply_interventions(c, g, shuffle_spec="targets", seed=5)
    rs = run_trial(cs, g, cfg, seed=0)
    assert rs.metrics["gf_spikes"] < r.metrics["gf_spikes"]


def test_lesion_blocks_escape(toy):
    c, g = toy
    cfg = _cfg("looming")
    cl, sil = apply_interventions(c, g, lesion_spec=["giant_fiber"])
    r = run_trial(cl, g, cfg, seed=1, silenced=sil)
    assert r.metrics["takeoff"] == 0.0 and r.metrics["gf_spikes"] == 0


def test_all_scenarios_run(toy):
    c, g = toy
    for sc in ("spider", "foraging", "lightchoice", "zerog"):
        cfg = _cfg(sc)
        cfg["scenario_cfg"]["duration"] = 0.1
        r = run_trial(c, g, cfg, seed=2)
        assert np.isfinite(r.metrics["sim_time_s"]) and r.metrics["sim_time_s"] > 0.05, sc
