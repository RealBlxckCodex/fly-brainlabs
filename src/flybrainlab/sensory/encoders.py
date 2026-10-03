"""Sensory front-end: world state -> Poisson rates of identified sensory/visual neurons.

All encoders here are *engineered* (hand-designed, documented in docs/assumptions.md);
they are not part of the connectome. Each maps an analytic stimulus onto real neuron
groups of the connectome:

vision/looming  LPLC2, LC4 (visual projection neurons; per-neuron random receptive
                field on the side of their optic lobe; rate from angular expansion)
photoreceptors  R1-R6 per eye; rate from luminance of light sources in the RF
olfaction       attractive-odour ORNs (DM1/DM2/DM4/VM2) per antenna; rate from
                concentration at the antenna (Michaelis-Menten)
mechanosensory  Johnston's organ JO-C / JO-E per antenna; rate from static arista
                deflection by gravity and by relative airflow
optic flow      HS (HSE/HSN/HSS) and VS lobula plate tangential cells per eye; rate from
                self-rotation (yaw for HS, roll for VS) and forward translation

FlyVis (TuragaLab) is the planned replacement for the analytic looming encoder; its
outputs stop at the medulla/lobula-plate cell types, so an LPLC2/LC4 adapter would be
needed (see docs/roadmap in README).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np

from ..connectome.groups import GroupResolver


@dataclass
class LoomingObject:
    name: str
    pos: np.ndarray  # world, cm
    radius: float  # cm


@dataclass
class LightSource:
    pos: np.ndarray
    intensity: float


@dataclass
class Percept:
    t: float
    R: np.ndarray  # body->world rotation
    eye_pos: dict  # side -> world pos
    antenna_pos: dict  # side -> world pos
    velocity: np.ndarray  # fly velocity (world), cm/s
    angvel: np.ndarray  # body angular velocity (body frame), rad/s
    gravity: np.ndarray  # world gravity vector, cm/s^2
    wind: np.ndarray = field(default_factory=lambda: np.zeros(3))
    objects: list = field(default_factory=list)
    lights: list = field(default_factory=list)
    ambient: float = 0.0
    odor: callable = None  # f(world pos) -> concentration


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _rf_directions(ids: np.ndarray, side: str, az_range, el_range, salt: int) -> np.ndarray:
    """Deterministic pseudo-random receptive-field centres (body frame unit vectors).

    Seeded by neuron id so every run/trial uses the same retinotopy.
    """
    out = np.zeros((len(ids), 3))
    for k, nid in enumerate(ids):
        r = np.random.default_rng((int(nid) * 2654435761 + salt) % (2**63))
        az = np.deg2rad(r.uniform(*az_range))
        # uniform on the sphere band: sample sin(el)
        s = r.uniform(np.sin(np.deg2rad(el_range[0])), np.sin(np.deg2rad(el_range[1])))
        el = np.arcsin(s)
        az = az if side == "L" else -az
        out[k] = [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
    return out


DEFAULTS = {
    "looming": {
        "groups": {"lplc2": ["lplc2_L", "lplc2_R"], "lc4": ["lc4_L", "lc4_R"]},
        "az_range_deg": [-15, 175],  # per eye; + = towards that eye's side
        "el_range_deg": [-60, 80],
        "lplc2": {"rf_deg": 30, "r_max": 150.0, "thr_dps": 60.0, "slope_dps": 25.0, "min_size_deg": 3.0},
        "lc4": {"rf_deg": 20, "r_max": 120.0, "sat_dps": 300.0, "min_size_deg": 1.0},
    },
    "photoreceptors": {"groups": ["r1r6_L", "r1r6_R"], "rf_deg": 45, "r_min": 0.0, "r_max": 60.0, "half_sat": 1.0,
                       "falloff_cm": 5.0},
    "olfaction": {"groups": ["orn_attr_L", "orn_attr_R"], "r_max": 50.0, "K": 0.2},
    "jo": {"groups": {"C": ["jo_c_L", "jo_c_R"], "E": ["jo_e_L", "jo_e_R"]},
           "axis": [0.45, 0.3, -0.84],  # arista load axis (body frame, mirrored in y for right side)
           "gain_g": 60.0, "gain_wind": 0.4, "base": 0.0},
    "optic_flow": {"hs": ["hs_L", "hs_R"], "vs": ["vs_L", "vs_R"], "r_max": 120.0, "omega_sat": 8.0,
                   "v_sat": 60.0, "trans_weight": 0.3, "base": 0.0},
    "tonic": {},  # {group: rate_hz} constant drive, e.g. locomotor drive onto DNp09
}


class SensoryEncoder:
    def __init__(self, groups: GroupResolver, cfg: dict | None = None, ablate: list[str] | None = None):
        from ..config import deep_merge

        self.cfg = deep_merge(DEFAULTS, cfg or {})
        self.ablate = set(ablate or [])
        self.g = groups
        ids = groups.c.neurons["id"].to_numpy()
        lc = self.cfg["looming"]
        self.loom = []  # (kind, side, idx, rf_dirs)
        for kind, names in lc["groups"].items():
            for name in names:
                side = name[-1]
                idx = groups.resolve(name)
                rf = _rf_directions(ids[idx], side, lc["az_range_deg"], lc["el_range_deg"], salt=zlib.crc32(kind.encode()))
                self.loom.append((kind, side, idx, rf))
        pc = self.cfg["photoreceptors"]
        self.photo = []
        for name in pc["groups"]:
            side = name[-1]
            idx = groups.resolve(name)
            self.photo.append((side, idx, _rf_directions(ids[idx], side, [-15, 175], [-70, 80], salt=11)))
        self.orn = [(n[-1], groups.resolve(n)) for n in self.cfg["olfaction"]["groups"]]
        self.jo = [(kind, n[-1], groups.resolve(n)) for kind, ns in self.cfg["jo"]["groups"].items() for n in ns]
        oc = self.cfg["optic_flow"]
        self.flow = [(kind, n[-1], groups.resolve(n)) for kind in ("hs", "vs") for n in oc[kind]]
        self.tonic = [(groups.resolve(n), float(r)) for n, r in self.cfg["tonic"].items()]
        self._prev_theta: dict = {}
        self.last_stats: dict = {}

    # ------------------------------------------------------------------
    def input_neurons(self) -> np.ndarray:
        parts = [i for _, _, i, _ in self.loom] + [i for _, i, _ in self.photo] + [i for _, i in self.orn]
        parts += [i for _, _, i in self.jo] + [i for _, _, i in self.flow] + [i for i, _ in self.tonic]
        return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)

    def encode(self, P: Percept, dt: float) -> tuple[np.ndarray, np.ndarray]:
        idx_parts, rate_parts = [], []

        def add(idx, rates):
            idx_parts.append(idx)
            rate_parts.append(np.broadcast_to(rates, idx.shape).astype(float))

        stats = {}
        RT = P.R.T
        if "vision" not in self.ablate and "looming" not in self.ablate:
            lc = self.cfg["looming"]
            for kind, side, idx, rf in self.loom:
                if f"vision_{side}" in self.ablate:
                    continue
                prm = lc[kind]
                rates = np.zeros(len(idx))
                for obj in P.objects:
                    rel = obj.pos - P.eye_pos[side]
                    dist = float(np.linalg.norm(rel))
                    theta = float(np.arcsin(min(1.0, obj.radius / max(dist, 1e-9))))
                    key = (obj.name, side)
                    dtheta = (theta - self._prev_theta.get(key, theta)) / dt
                    self._prev_theta[key] = theta
                    d_body = RT @ (rel / max(dist, 1e-9))
                    ang = np.arccos(np.clip(rf @ d_body, -1, 1))
                    inside = ang < np.deg2rad(prm["rf_deg"]) + theta
                    if np.rad2deg(2 * theta) < prm["min_size_deg"] or not inside.any():
                        continue
                    dps = np.rad2deg(2 * dtheta)  # angular size expansion, deg/s
                    if kind == "lplc2":
                        # sigmoid in expansion rate, shifted so that r(0 deg/s) = 0 exactly
                        sig = lambda x: 1.0 / (1.0 + np.exp(-(x - prm["thr_dps"]) / prm["slope_dps"]))  # noqa: E731
                        r = prm["r_max"] * max(sig(dps) - sig(0.0), 0.0) / (1.0 - sig(0.0))
                    else:
                        r = prm["r_max"] * np.clip(abs(dps) / prm["sat_dps"], 0, 1)
                    rates = np.maximum(rates, np.where(inside, r, 0.0))
                    stats[f"{kind}_{side}_dps"] = dps
                add(idx, rates)
                stats[f"{kind}_{side}_mean_hz"] = float(rates.mean()) if len(rates) else 0.0
        if "photoreceptors" not in self.ablate and "vision" not in self.ablate and P.lights:
            pc = self.cfg["photoreceptors"]
            for side, idx, rf in self.photo:
                if f"vision_{side}" in self.ablate:
                    continue
                lum = np.full(len(idx), P.ambient)
                for L in P.lights:
                    rel = L.pos - P.eye_pos[side]
                    dist = float(np.linalg.norm(rel))
                    d_body = RT @ (rel / max(dist, 1e-9))
                    ang = np.arccos(np.clip(rf @ d_body, -1, 1))
                    w = np.exp(-0.5 * (ang / np.deg2rad(pc["rf_deg"])) ** 2)
                    lum += L.intensity * w / (1 + (dist / pc["falloff_cm"]) ** 2)
                rates = pc["r_min"] + (pc["r_max"] - pc["r_min"]) * lum / (lum + pc["half_sat"])
                add(idx, rates)
                stats[f"photo_{side}_mean_hz"] = float(rates.mean())
        if "olfaction" not in self.ablate and P.odor is not None:
            oc = self.cfg["olfaction"]
            for side, idx in self.orn:
                c = float(P.odor(P.antenna_pos[side]))
                r = oc["r_max"] * c / (c + oc["K"])
                add(idx, r)
                stats[f"orn_{side}_hz"] = r
                stats[f"odor_{side}"] = c
        if "jo" not in self.ablate and "mechanosensory" not in self.ablate:
            jc = self.cfg["jo"]
            g_body = RT @ P.gravity
            air_body = RT @ (P.wind - P.velocity)
            for kind, side, idx in self.jo:
                ax = np.array(jc["axis"], float)
                if side == "R":
                    ax[1] = -ax[1]
                ax /= np.linalg.norm(ax)
                s = jc["gain_g"] * float(g_body @ ax) / 981.0 + jc["gain_wind"] * float(air_body @ ax)
                r = jc["base"] + (max(s, 0.0) if kind == "C" else max(-s, 0.0))
                add(idx, r)
                stats[f"jo_{kind}_{side}_hz"] = r
        if "optic_flow" not in self.ablate and "vision" not in self.ablate:
            oc = self.cfg["optic_flow"]
            w = np.asarray(P.angvel, float)
            v_fwd = float((RT @ P.velocity)[0])
            for kind, side, idx in self.flow:
                if f"vision_{side}" in self.ablate:
                    continue
                sgn = 1.0 if side == "L" else -1.0
                if kind == "hs":  # yaw to the left (+z) = front-to-back motion on the right eye -> HS_R
                    drive = -sgn * w[2] / oc["omega_sat"] + oc["trans_weight"] * max(v_fwd, 0) / oc["v_sat"]
                else:  # roll about +x lifts the left side: world moves down on the left eye -> VS_L
                    drive = sgn * w[0] / oc["omega_sat"]
                r = oc["base"] + oc["r_max"] * float(np.clip(drive, 0.0, 1.0))
                add(idx, r)
                stats[f"{kind}_{side}_hz"] = r
        for idx, r in self.tonic:
            add(idx, r)
        self.last_stats = stats
        if not idx_parts:
            return np.zeros(0, np.int64), np.zeros(0)
        idx = np.concatenate(idx_parts)
        rates = np.concatenate(rate_parts)
        # a neuron listed in several encoders gets the max rate
        order = np.lexsort((-rates, idx))
        idx, rates = idx[order], rates[order]
        first = np.r_[True, idx[1:] != idx[:-1]]
        return idx[first], rates[first]
