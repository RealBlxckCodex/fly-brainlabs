"""Scenarios: arena, scripted agents, stimuli, termination and metric computation.

Every metric is computed here from logged physical state (never judged by the LLM).
"""
from __future__ import annotations

import numpy as np

from ..config import deep_merge
from ..sensory.encoders import LightSource, LoomingObject


def _angle_between_2d(a, b) -> float:
    a, b = np.asarray(a[:2], float), np.asarray(b[:2], float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1))))


class Scenario:
    name = "base"
    defaults: dict = {}

    def __init__(self, cfg: dict | None, rng: np.random.Generator):
        self.cfg = deep_merge(self.defaults, cfg or {})
        self.rng = rng
        self.duration = float(self.cfg.get("duration", 1.0))

    # world description ---------------------------------------------------
    def world(self) -> dict:
        return {"floor": True}

    def init_pose(self):
        return np.array([0, 0, 0.075]), 0.0, "ground"

    def reset(self, model, data, body) -> None:
        self.model, self.data, self.body = model, data, body

    def update(self, t: float) -> None:
        pass

    def objects(self, t: float) -> list:
        return []

    def lights(self, t: float) -> list:
        return []

    def odor(self):
        return None

    def wind(self) -> np.ndarray:
        return np.zeros(3)

    def done(self, t: float) -> bool:
        return t >= self.duration

    def log_extra(self, t: float) -> dict:
        return {}

    def metrics(self, log: dict, events: list) -> dict:
        raise NotImplementedError

    @staticmethod
    def metric_info() -> dict:
        return {}

    def _set_mocap(self, name: str, pos) -> None:
        mid = self.model.body(name).mocapid[0]
        self.data.mocap_pos[mid] = pos


# --------------------------------------------------------------------- looming
class LoomingScenario(Scenario):
    """M3 reflex test: a dark sphere approaches the standing fly at constant speed."""

    name = "looming"
    defaults = {"duration": 0.6, "radius": 0.5, "speed": 25.0, "start_distance": 10.0,
                "azimuth_deg": [-180, 180], "elevation_deg": 20.0, "t_start": 0.02}

    def world(self):
        return {"floor": True, "mocaps": [{"name": "looming", "pos": [50, 0, 5],
                                           "geoms": [{"type": "sphere", "size": [self.cfg["radius"]]}]}]}

    def reset(self, model, data, body):
        super().reset(model, data, body)
        az = self.cfg["azimuth_deg"]
        self.az = np.deg2rad(self.rng.uniform(*az) if isinstance(az, list) else az)
        el = np.deg2rad(self.cfg["elevation_deg"])
        self.dir = np.array([np.cos(el) * np.cos(self.az), np.sin(self.az) * np.cos(el), np.sin(el)])
        self.target = body.site_pos("eye_left") * 0.5 + body.site_pos("eye_right") * 0.5
        self.t_collision = self.cfg["t_start"] + (self.cfg["start_distance"] - self.cfg["radius"]) / self.cfg["speed"]
        self.update(0.0)

    def _obj_pos(self, t):
        d = self.cfg["start_distance"] - self.cfg["speed"] * max(t - self.cfg["t_start"], 0)
        return self.target + self.dir * max(d, self.cfg["radius"])

    def update(self, t):
        self.pos = self._obj_pos(t)
        self._set_mocap("looming", self.pos)

    def objects(self, t):
        return [LoomingObject("looming", self.pos, self.cfg["radius"])]

    def done(self, t):
        return t >= min(self.duration, self.t_collision + 0.05)

    @staticmethod
    def metric_info():
        return {"takeoff": "1 if the fly jumped before collision", "time_before_collision_ms":
                "collision time minus takeoff time (ms, >0 = escaped in time)",
                "theta_at_takeoff_deg": "angular size of the stimulus at takeoff",
                "escape_dir_error_deg": "angle between jump direction and the direction away from the stimulus"}

    def metrics(self, log, events):
        t_take = next((t for t, e in events if e == "takeoff"), None)
        out = {"takeoff": float(t_take is not None and t_take < self.t_collision), "stim_azimuth_deg": np.degrees(self.az)}
        if t_take is not None:
            d = self.cfg["start_distance"] - self.cfg["speed"] * max(t_take - self.cfg["t_start"], 0)
            out["time_before_collision_ms"] = (self.t_collision - t_take) * 1e3
            out["theta_at_takeoff_deg"] = float(np.degrees(2 * np.arcsin(min(1, self.cfg["radius"] / max(d, 1e-6)))))
            out["escape_dir_error_deg"] = _angle_between_2d(self.body.jump_dir, -self.dir)
        else:
            out.update(time_before_collision_ms=np.nan, theta_at_takeoff_deg=np.nan, escape_dir_error_deg=np.nan)
        return out


# --------------------------------------------------------------------- spider
class SpiderScenario(Scenario):
    """M4: a scripted spider stalks and then rushes the fly on the ground."""

    name = "spider"
    defaults = {"duration": 1.5, "spider_radius": 0.4, "start_distance": [5.0, 7.0], "t_start": 0.05,
                "stalk_speed": 6.0, "rush_speed": 30.0, "rush_distance": 3.0, "capture_margin": 0.15,
                "reach_height": 0.8, "policy": "scripted"}

    def world(self):
        r = self.cfg["spider_radius"]
        geoms = [{"type": "ellipsoid", "size": [r, r * 0.8, r * 0.6], "rgba": [0.12, 0.08, 0.05, 1]},
                 {"type": "sphere", "size": [r * 0.45], "pos": [r * 0.9, 0, r * 0.1], "rgba": [0.1, 0.05, 0.03, 1]}]
        for k in range(4):
            for s in (-1, 1):
                a = np.deg2rad(-60 + 40 * k)
                geoms.append({"type": "capsule", "size": [0.03, r * 0.8],
                              "pos": [np.cos(a) * r * 0.9, s * (r + np.sin(abs(a)) * 0.1 + 0.2), -r * 0.3],
                              "rgba": [0.1, 0.06, 0.04, 1]})
        return {"floor": True, "mocaps": [{"name": "spider", "pos": [50, 0, r], "geoms": geoms}]}

    def init_pose(self):
        return np.array([0, 0, 0.075]), float(self.rng.uniform(-np.pi, np.pi)), "ground"

    def reset(self, model, data, body):
        super().reset(model, data, body)
        d0 = self.rng.uniform(*self.cfg["start_distance"])
        self.az = self.rng.uniform(-np.pi, np.pi)  # world azimuth of the spider start
        r = self.cfg["spider_radius"]
        self.spos = np.array([d0 * np.cos(self.az), d0 * np.sin(self.az), r * 0.6])
        self.t_last = 0.0
        self.captured_at = None
        self.takeoff_away = None
        self._set_mocap("spider", self.spos)

    def update(self, t):
        dt, self.t_last = t - self.t_last, t
        fly = self.body.pos
        rel = fly - self.spos
        rel[2] = 0
        dist = float(np.linalg.norm(rel))
        if t >= self.cfg["t_start"] and self.captured_at is None and dist > 1e-6:
            speed = self.cfg["rush_speed"] if dist < self.cfg["rush_distance"] else self.cfg["stalk_speed"]
            if fly[2] > self.cfg["reach_height"]:
                speed = 0.0  # cannot follow into the air
            self.spos = self.spos + rel / dist * min(speed * dt, dist)
        if self.captured_at is None and self.takeoff_away is None and self.body.mode != "ground":
            self.takeoff_away = (fly - self.spos).copy()
        full = float(np.linalg.norm(self.body.pos - self.spos))
        if self.captured_at is None and full < self.cfg["spider_radius"] + self.cfg["capture_margin"] + 0.1:
            self.captured_at = t
        self._set_mocap("spider", self.spos)

    def objects(self, t):
        return [LoomingObject("spider", self.spos, self.cfg["spider_radius"])]

    def done(self, t):
        return t >= self.duration or self.captured_at is not None

    def log_extra(self, t):
        return {"spider": self.spos.copy()}

    @staticmethod
    def metric_info():
        return {"survived": "1 if not captured within the trial", "survival_time_s": "time until capture (or trial end)",
                "escape_latency_ms": "takeoff time minus spider start (ms)",
                "escape_dir_error_deg": "angle between jump direction and direction away from spider (0 = straight away)",
                "takeoff": "1 if the fly jumped", "min_distance_cm": "closest approach of the spider"}

    def metrics(self, log, events):
        t_take = next((t for t, e in events if e == "takeoff"), None)
        out = {"survived": float(self.captured_at is None),
               "survival_time_s": float(self.captured_at if self.captured_at is not None else log["t"][-1]),
               "takeoff": float(t_take is not None),
               "escape_latency_ms": (t_take - self.cfg["t_start"]) * 1e3 if t_take is not None else np.nan,
               "escape_dir_error_deg": _angle_between_2d(self.body.jump_dir, self.takeoff_away)
               if (t_take is not None and self.takeoff_away is not None) else np.nan}
        sp = np.asarray(log["spider"])
        fp = np.asarray(log["pos"])
        out["min_distance_cm"] = float(np.min(np.linalg.norm(sp - fp, axis=1)))
        return out


# --------------------------------------------------------------------- foraging
class ForagingScenario(Scenario):
    """Walk to an odour source (Gaussian plume) around obstacles."""

    name = "foraging"
    defaults = {"duration": 4.0, "source": [4.0, 0.0], "source_radius": 0.4, "plume_sigma": 2.5,
                "strength": 1.0, "heading_jitter_deg": 90, "obstacles": [{"pos": [2.0, 0.6, 0.15], "size": [0.15, 0.5, 0.15]}],
                "start_in_flight": False}

    def world(self):
        s = self.cfg["source"]
        boxes = [{"name": f"obstacle{k}", **o} for k, o in enumerate(self.cfg["obstacles"])]
        return {"floor": True, "boxes": boxes,
                "markers": [{"name": "food", "pos": [s[0], s[1], 0.05], "size": self.cfg["source_radius"],
                             "rgba": [0.9, 0.5, 0.1, 0.5]}]}

    def init_pose(self):
        h = np.deg2rad(self.rng.uniform(-1, 1) * self.cfg["heading_jitter_deg"])
        mode = "flight" if self.cfg["start_in_flight"] else "ground"
        return np.array([0, 0, 0.075 if mode == "ground" else 1.0]), float(h), mode

    def reset(self, model, data, body):
        super().reset(model, data, body)
        self.src = np.array([*self.cfg["source"], 0.0])
        self.reached_at = None

    def odor(self):
        src, sig, S = self.src, self.cfg["plume_sigma"], self.cfg["strength"]
        return lambda p: S * np.exp(-np.sum((np.asarray(p)[:2] - src[:2]) ** 2) / (2 * sig ** 2))

    def update(self, t):
        if self.reached_at is None and np.linalg.norm(self.body.pos[:2] - self.src[:2]) < self.cfg["source_radius"]:
            self.reached_at = t

    def done(self, t):
        return t >= self.duration or self.reached_at is not None

    @staticmethod
    def metric_info():
        return {"reached": "1 if the fly reached the odour source", "time_to_goal_s": "time to reach (nan if not)",
                "path_efficiency": "straight-line distance / travelled path length (1 = optimal)",
                "final_distance_cm": "distance to source at trial end", "landed": "1 if it reached the source on the ground",
                "approach_cm": "start distance minus final distance"}

    def metrics(self, log, events):
        p = np.asarray(log["pos"])
        path = float(np.sum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)))
        d0 = float(np.linalg.norm(p[0, :2] - self.src[:2]))
        d1 = float(np.linalg.norm(p[-1, :2] - self.src[:2]))
        reached = self.reached_at is not None
        return {"reached": float(reached), "time_to_goal_s": self.reached_at if reached else np.nan,
                "path_efficiency": (d0 - self.cfg["source_radius"]) / path if (reached and path > 0) else np.nan,
                "final_distance_cm": d1, "approach_cm": d0 - d1,
                "landed": float(reached and self.body.mode == "ground")}


# --------------------------------------------------------------------- light choice
class LightChoiceScenario(Scenario):
    """T-maze: walk up the stem, choose the bright or the dark arm."""

    name = "lightchoice"
    defaults = {"duration": 3.0, "stem_length": 2.5, "arm_length": 2.5, "width": 0.8, "wall_h": 0.3,
                "bright": 5.0, "dark": 0.1, "ambient": 0.05, "choice_y": 1.0}

    def world(self):
        L, A, w, h = self.cfg["stem_length"], self.cfg["arm_length"], self.cfg["width"], self.cfg["wall_h"]
        t = 0.05
        boxes = [  # stem walls (along x), back wall, junction far wall, arm walls (along y)
            {"name": "stem_l", "pos": [L / 2, w / 2 + t, h], "size": [L / 2, t, h]},
            {"name": "stem_r", "pos": [L / 2, -w / 2 - t, h], "size": [L / 2, t, h]},
            {"name": "back", "pos": [-t, 0, h], "size": [t, w / 2, h]},
            {"name": "far", "pos": [L + w + t, 0, h], "size": [t, A + w / 2, h]},
            {"name": "arm_l_near", "pos": [L - t, w / 2 + A / 2, h], "size": [t, A / 2, h]},
            {"name": "arm_r_near", "pos": [L - t, -w / 2 - A / 2, h], "size": [t, A / 2, h]},
            {"name": "arm_l_end", "pos": [L + w / 2, w / 2 + A + t, h], "size": [w / 2, t, h]},
            {"name": "arm_r_end", "pos": [L + w / 2, -w / 2 - A - t, h], "size": [w / 2, t, h]},
        ]
        return {"floor": True, "boxes": boxes, "markers": [
            {"name": "light_left", "pos": self._light_pos(+1), "size": 0.25, "rgba": [1, 1, 0.6, 0.9]},
            {"name": "light_right", "pos": self._light_pos(-1), "size": 0.25, "rgba": [1, 1, 0.6, 0.9]}]}

    def _light_pos(self, side):
        L, A, w = self.cfg["stem_length"], self.cfg["arm_length"], self.cfg["width"]
        return [L + w / 2, side * (w / 2 + A - 0.2), 0.4]

    def init_pose(self):
        self.bright_side = 1 if self.rng.random() < 0.5 else -1  # +1 = left arm bright
        return np.array([0.3, 0, 0.075]), float(np.deg2rad(self.rng.uniform(-10, 10))), "ground"

    def lights(self, t):
        b, d = self.cfg["bright"], self.cfg["dark"]
        return [LightSource(np.array(self._light_pos(self.bright_side)), b),
                LightSource(np.array(self._light_pos(-self.bright_side)), d)]

    def reset(self, model, data, body):
        super().reset(model, data, body)
        self.choice = 0
        self.choice_t = None
        # recolour the dim marker
        gid = model.geom("light_left" if self.bright_side == -1 else "light_right").id
        model.geom_rgba[gid] = [0.25, 0.25, 0.2, 0.9]

    def update(self, t):
        y = self.body.pos[1]
        if self.choice == 0 and self.body.pos[0] > self.cfg["stem_length"] - 0.2 and abs(y) > self.cfg["choice_y"]:
            self.choice, self.choice_t = int(np.sign(y)), t

    def done(self, t):
        return t >= self.duration or self.choice != 0

    @staticmethod
    def metric_info():
        return {"chose": "1 if an arm was entered", "chose_bright": "1 bright arm, 0 dark arm, nan no choice",
                "choice_time_s": "time of the choice", "progress_x_cm": "how far the fly walked up the stem"}

    def metrics(self, log, events):
        p = np.asarray(log["pos"])
        return {"chose": float(self.choice != 0),
                "chose_bright": float(self.choice == self.bright_side) if self.choice != 0 else np.nan,
                "chose_left": float(self.choice == 1) if self.choice != 0 else np.nan,
                "choice_time_s": self.choice_t if self.choice_t is not None else np.nan,
                "progress_x_cm": float(p[:, 0].max() - p[0, 0]), "bright_side": float(self.bright_side)}


# --------------------------------------------------------------------- zero-g
class ZeroGScenario(Scenario):
    """Free flight in air with adjustable gravity (0 = weightless, air unchanged)."""

    name = "zerog"
    defaults = {"duration": 1.0, "start_height": 8.0, "init_angvel": 4.0, "init_speed": 0.0,
                "upright_deg": 30.0, "floor": False,
                # Ornstein-Uhlenbeck gust torque, in units of the torque that holds 1 rad/s against
                # flapping damping (so gust_rate=3 alone would spin the fly at ~3 rad/s)
                "gust_rate": 3.0, "gust_tau": 0.1}

    def world(self):
        return {"floor": self.cfg["floor"]}

    def init_pose(self):
        return np.array([0, 0, self.cfg["start_height"]]), float(self.rng.uniform(-np.pi, np.pi)), "flight"

    def reset(self, model, data, body):
        super().reset(model, data, body)
        w = self.rng.normal(size=3)
        w = w / np.linalg.norm(w) * self.cfg["init_angvel"]
        data.qvel[3:6] = w  # free joint angular velocity (body frame)
        v = self.rng.normal(size=3)
        data.qvel[0:3] = v / np.linalg.norm(v) * self.cfg["init_speed"]
        self.p0 = body.pos.copy()
        self.gust = np.zeros(3)
        self.t_last = 0.0

    def update(self, t):
        dt, self.t_last = t - self.t_last, t
        if dt <= 0 or self.cfg["gust_rate"] <= 0:
            return
        tau = self.cfg["gust_tau"]
        self.gust += -self.gust * dt / tau + np.sqrt(2 * dt / tau) * self.rng.normal(size=3)
        D = self.body.I_body / self.body.p.flapping_damping_tau
        self.body.external_torque = self.body.R @ (D @ (self.cfg["gust_rate"] * self.gust))

    def done(self, t):
        return t >= self.duration or (self.body.mode == "ground")

    @staticmethod
    def metric_info():
        return {"drift_cm": "distance from start position at trial end",
                "mean_tilt_deg": "mean angle between body up-axis and world up",
                "final_tilt_deg": "tilt at trial end", "upright_fraction": "fraction of time with tilt < upright_deg",
                "mean_angspeed_rad_s": "mean body angular speed", "control_effort": "mean |motor command| (sum over channels)",
                "corrective_yaw_corr": "corr(yaw command, -yaw rate): >0 means the brain turns against rotation",
                "corrective_roll_corr": "corr(roll command, -roll rate)",
                "tumbled": "1 if tilt ever exceeded 90 deg", "crashed": "1 if the fly hit the floor",
                "vertical_drift_cm": "z(end) - z(start)"}

    def metrics(self, log, events):
        p = np.asarray(log["pos"])
        up = np.asarray(log["up"])
        tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))
        return {"drift_cm": float(np.linalg.norm(p[-1] - p[0])), "vertical_drift_cm": float(p[-1, 2] - p[0, 2]),
                "mean_tilt_deg": float(tilt.mean()), "final_tilt_deg": float(tilt[-1]),
                "upright_fraction": float(np.mean(tilt < self.cfg["upright_deg"])),
                "mean_angspeed_rad_s": float(np.mean(np.linalg.norm(np.asarray(log["angvel"]), axis=1))),
                "control_effort": float(np.mean(log["effort"])), "tumbled": float(tilt.max() > 90),
                "crashed": float(p[:, 2].min() < 0.2),
                "corrective_yaw_corr": self._corr(np.asarray(log["cmd"])[:, 5], -np.asarray(log["angvel"])[:, 2]),
                "corrective_roll_corr": self._corr(np.asarray(log["cmd"])[:, 3], -np.asarray(log["angvel"])[:, 0])}

    @staticmethod
    def _corr(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])


SCENARIOS = {c.name: c for c in (LoomingScenario, SpiderScenario, ForagingScenario, LightChoiceScenario, ZeroGScenario)}


def make_scenario(name: str, cfg: dict | None, rng: np.random.Generator) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; choose from {list(SCENARIOS)}")
    return SCENARIOS[name](cfg, rng)
