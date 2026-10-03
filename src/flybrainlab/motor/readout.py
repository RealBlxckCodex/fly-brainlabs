"""Motor readout: descending-neuron (DN) activity -> MotorCommand.

This layer is NOT connectome. It is a fixed, hand-specified linear/tanh mapping from
the filtered firing rates of identified DN groups onto motor primitives (configured in
``configs/default.yaml -> readout``). Optionally a *fitted* linear readout (weights
learned by ``scripts/fit_readout.py``) replaces individual channels; reports must then
say "connectome + trained readout".

Optional fitted channel ``jump_yaw_fitted: {path: <json>}`` (scripts/fit_escape_readout.py):
a ridge regression from the filtered rates of individual descending neurons to the
(cos, sin) of the escape direction. Reports must then say "connectome + trained readout".

Default mapping (literature basis in docs/assumptions.md):
  takeoff   Giant Fiber DNp01 rate > threshold      (von Reyn et al. 2014)
  jump_yaw  (L - R) of DNp02/DNp11 ("escape steering", hand-set sign)
  forward   DNp09 (P9-like) - MDN                    (Bidaye et al. 2014/2020)
  turn      (L - R) of DNa02 + DNa01                 (Rayshubskiy et al. 2020)
  power     DNg02 above baseline                     (Namiki et al. 2022)
  yaw       (L - R) of DNa02 in flight (HS -> DNa02 is ipsilateral in the connectome)
  roll      -(L - R) of DNp20 + DNb06 (VS targets; sign chosen corrective)
"""
from __future__ import annotations

import numpy as np

from ..body.fly import MotorCommand
from ..connectome.groups import GroupResolver

DEFAULT_READOUT = {
    "tau_ms": 20.0,
    "takeoff": {"groups": ["giant_fiber"], "threshold_hz": 20.0},  # one GF spike (2 cells, tau 20 ms) = 25 Hz
    "jump_yaw": {"left": ["dnp02_L", "dnp11_L"], "right": ["dnp02_R", "dnp11_R"], "scale_hz": 50.0,
                 "max_rad": 1.6, "sign": -1.0},
    "forward": {"pos": ["dnp09"], "neg": ["mdn"], "scale_hz": 30.0},
    "turn": {"left": ["dna02_L", "dna01_L"], "right": ["dna02_R", "dna01_R"], "scale_hz": 30.0},
    "power": {"pos": ["dng02"], "baseline_hz": 0.0, "scale_hz": 50.0},
    # VS -> DNp20/DNb06 are the strongest VS->DN routes in MaleCNS. The sign of this channel
    # is a design choice (set corrective: left-VS activity -> roll back), see A-14.
    "roll": {"left": ["dnp20_L", "dnb06_L"], "right": ["dnp20_R", "dnb06_R"], "scale_hz": 30.0, "sign": -1.0},
    "pitch": None,
    "yaw": {"left": ["dna02_L"], "right": ["dna02_R"], "scale_hz": 30.0},
}


class Readout:
    def __init__(self, groups: GroupResolver, cfg: dict | None = None, dt_ms: float = 0.1):
        from ..config import deep_merge

        self.cfg = deep_merge(DEFAULT_READOUT, cfg or {})
        self.dt_ms = dt_ms
        self.decay = float(np.exp(-dt_ms / self.cfg["tau_ms"]))
        names = set()
        for ch in ("takeoff", "jump_yaw", "forward", "turn", "power", "roll", "pitch", "yaw"):
            spec = self.cfg.get(ch)
            if spec:
                for k in ("groups", "left", "right", "pos", "neg"):
                    names.update(spec.get(k, []))
        self.names = sorted(names)
        self.idx = {n: groups.resolve(n) for n in self.names}
        n = groups.c.n_neurons
        self._is_member = np.zeros(n, bool)
        self._slots: dict[int, list[int]] = {}  # neuron -> group slots (may be several)
        for k, g in enumerate(self.names):
            self._is_member[self.idx[g]] = True
            for i in self.idx[g]:
                self._slots.setdefault(int(i), []).append(k)
        self.sizes = np.array([max(len(self.idx[g]), 1) for g in self.names], float)
        self.trace = np.zeros(len(self.names))  # exponentially filtered spike count per group
        self.weights = self.cfg.get("fitted")  # optional fitted channels (group features)
        self.pop = None
        fit = self.cfg.get("jump_yaw_fitted")
        if fit:
            import json
            from pathlib import Path

            from ..paths import ROOT

            path = Path(fit["path"]) if Path(fit["path"]).is_absolute() else ROOT / fit["path"]
            w = json.loads(path.read_text())
            pidx = groups.c.index_of(w["neuron_ids"], strict=False)
            if len(pidx) != len(w["neuron_ids"]):
                raise ValueError("fitted readout neurons missing from connectome (wrong dataset?)")
            self.pop = {"idx": pidx, "coef": np.asarray(w["coef"]), "intercept": np.asarray(w["intercept"]),
                        "mean": np.asarray(w["feature_mean"]), "scale": np.asarray(w["feature_scale"])}
            self._pop_pos = np.full(n, -1, np.int64)
            self._pop_pos[pidx] = np.arange(len(pidx))
            self.pop_trace = np.zeros(len(pidx))

    def all_neurons(self) -> np.ndarray:
        parts = [self.idx[g] for g in self.names] + ([self.pop["idx"]] if self.pop is not None else [])
        return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)

    def observe(self, spikes: np.ndarray) -> None:
        self.trace *= self.decay
        if self.pop is not None:
            self.pop_trace *= self.decay
            if spikes.size:
                k = self._pop_pos[spikes]
                k = k[k >= 0]
                if k.size:
                    np.add.at(self.pop_trace, k, 1.0)
        if spikes.size:
            for i in spikes[self._is_member[spikes]]:
                for k in self._slots[int(i)]:
                    self.trace[k] += 1.0

    def population_rates(self) -> np.ndarray | None:
        return None if self.pop is None else self.pop_trace / (self.cfg["tau_ms"] * 1e-3)

    def rates(self) -> dict[str, float]:
        """Per-group mean firing rate (Hz) from the exponential trace."""
        r = self.trace / self.sizes / (self.cfg["tau_ms"] * 1e-3)
        return dict(zip(self.names, r))

    def _mean(self, r, names):
        return float(np.mean([r[n] for n in names])) if names else 0.0

    def command(self) -> MotorCommand:
        r, c = self.rates(), self.cfg
        cmd = MotorCommand()
        if c.get("takeoff"):
            cmd.takeoff = self._mean(r, c["takeoff"]["groups"]) > c["takeoff"]["threshold_hz"]
        if c.get("jump_yaw"):
            s = c["jump_yaw"]
            d = self._mean(r, s["left"]) - self._mean(r, s["right"])
            cmd.jump_yaw = float(s["sign"] * s["max_rad"] * np.tanh(d / s["scale_hz"]))
        if c.get("forward"):
            s = c["forward"]
            cmd.forward = float(np.tanh((self._mean(r, s["pos"]) - self._mean(r, s.get("neg", []))) / s["scale_hz"]))
        for ch in ("turn", "roll", "yaw"):
            s = c.get(ch)
            if s:
                d = self._mean(r, s["left"]) - self._mean(r, s["right"])
                setattr(cmd, ch, float(s.get("sign", 1.0) * np.tanh(d / s["scale_hz"])))
        if c.get("power"):
            s = c["power"]
            cmd.power = float(np.tanh((self._mean(r, s["pos"]) - s["baseline_hz"]) / s["scale_hz"]))
        if self.pop is not None:
            x = (self.pop_trace / (self.cfg["tau_ms"] * 1e-3) - self.pop["mean"]) / self.pop["scale"]
            y = self.pop["coef"] @ x + self.pop["intercept"]
            cmd.jump_yaw = float(np.arctan2(y[1], y[0]))
        if self.weights:
            for ch, w in self.weights.items():
                x = np.array([r[n] for n in w["features"]])
                setattr(cmd, ch, float(np.tanh(np.dot(w["coef"], x) + w["intercept"])) * w.get("scale", 1.0))
        return cmd
