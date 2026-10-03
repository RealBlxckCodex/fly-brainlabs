"""Closed loop: world -> senses -> connectome LIF -> DN readout -> motor primitives -> body.

Timing: brain and physics both advance with dt = 0.1 ms. Sensory input rates and the
motor command are refreshed every ``control_period_ms`` (default 1 ms).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np

from .body.fly import BodyParams, MotorCommand
from .brain.lif import LIFNetwork, LIFParams
from .connectome.groups import GroupResolver
from .connectome.model import Connectome
from .env.scenarios import make_scenario
from .env.world import build_world
from .motor.readout import Readout
from .sensory.encoders import Percept, SensoryEncoder


@dataclass
class TrialResult:
    metrics: dict
    events: list
    log: dict
    neural: dict
    wall_s: float


def run_trial(connectome: Connectome, groups: GroupResolver, cfg: dict, seed: int,
              silenced: np.ndarray | None = None, keep_log: bool = True) -> TrialResult:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    lif = LIFParams.from_dict(cfg["brain"])
    dt_s = lif.dt * 1e-3
    scen = make_scenario(cfg["scenario"], cfg.get("scenario_cfg"), rng)
    world = {**scen.world(), **cfg.get("world", {})}
    if "floor" in scen.world():
        world["floor"] = scen.world()["floor"]
    bp = BodyParams.from_dict(cfg.get("body"))
    pos, heading, mode = scen.init_pose()
    model, data, body = build_world(world, bp, pos, heading, mode, timestep=dt_s)
    scen.reset(model, data, body)

    encoder = SensoryEncoder(groups, cfg.get("sensory"), ablate=cfg.get("ablate_senses"))
    readout = Readout(groups, cfg.get("readout"), dt_ms=lif.dt)
    record = np.unique(np.concatenate([readout.all_neurons(), groups.resolve("giant_fiber")]))
    net = LIFNetwork(connectome.W, lif, seed=seed, silenced=silenced, record=record)

    period = max(1, int(round(cfg.get("control_period_ms", 1.0) / lif.dt)))
    log_every = max(1, int(round(cfg.get("log_period_ms", 5.0) / lif.dt)))
    log = {k: [] for k in ("t", "pos", "heading", "up", "angvel", "mode", "effort", "cmd")}
    extra_keys: set = set()
    dn_rates = []
    cmd = MotorCommand()
    gravity = model.opt.gravity.copy()
    step = 0
    effort_acc = []
    while True:
        t = data.time
        if step % period == 0:
            scen.update(t)
            if scen.done(t):
                break
            P = Percept(t=t, R=body.R, eye_pos={"L": body.site_pos("eye_left"), "R": body.site_pos("eye_right")},
                        antenna_pos={"L": body.site_pos("antenna_left"), "R": body.site_pos("antenna_right")},
                        velocity=body.vel, angvel=body.angvel_body, gravity=gravity, wind=scen.wind(), objects=scen.objects(t),
                        lights=scen.lights(t), ambient=scen.cfg.get("ambient", 0.0), odor=scen.odor())
            idx, rates = encoder.encode(P, dt=period * dt_s)
            net.set_poisson_rates(idx, rates)
            cmd = readout.command()
            effort_acc.append(cmd.effort())
        spk = net.step()
        readout.observe(spk)
        body.apply(cmd)
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            body.events.append((data.time, "physics_diverged"))
            break
        if step % log_every == 0:
            log["t"].append(t)
            log["pos"].append(body.pos)
            log["heading"].append(body.heading)
            log["up"].append(body.R[:, 2])
            log["angvel"].append(body.angvel_body)
            log["mode"].append(body.mode)
            log["effort"].append(cmd.effort())
            log["cmd"].append([cmd.forward, cmd.turn, cmd.power, cmd.roll, cmd.pitch, cmd.yaw, cmd.jump_yaw])
            for k, v in scen.log_extra(t).items():
                log.setdefault(k, []).append(v)
                extra_keys.add(k)
            dn_rates.append(list(readout.rates().values()))
        step += 1
    if not log["t"]:  # ensure at least one log entry
        log["t"].append(data.time)
        log["pos"].append(body.pos)
        log["up"].append(body.R[:, 2])
        log["angvel"].append(body.angvel_body)
        log["effort"].append(0.0)
        for k, v in scen.log_extra(data.time).items():
            log.setdefault(k, []).append(v)
    metrics = scen.metrics(log, body.events)
    dur_ms = net.t_ms
    gf = groups.resolve("giant_fiber")
    rates = net.rates_hz(dur_ms)
    metrics.update({
        "gf_spikes": int(net.spike_counts[gf].sum()),
        "n_active_neurons": int((net.spike_counts > 0).sum()),
        "mean_rate_active_hz": float(rates[net.spike_counts > 0].mean()) if (net.spike_counts > 0).any() else 0.0,
        "sim_time_s": float(data.time),
    })
    neural = {"readout_groups": readout.names, "dn_rates": np.asarray(dn_rates),
              "group_mean_rate_hz": {g: float(rates[readout.idx[g]].mean()) for g in readout.names}}
    if keep_log:
        neural["raster"] = net.recorded_spikes()
        neural["spike_counts"] = net.spike_counts.copy()
    if not keep_log:
        log = {}
    return TrialResult(metrics=metrics, events=list(body.events), log=log, neural=neural, wall_s=time.time() - t0)
