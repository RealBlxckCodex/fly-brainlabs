"""Fly body in MuJoCo plus the *motor primitive* layer.

Two body models share one interface:

``simple``   assets/simple_fly.xml: ellipsoid body, rigid leg skids (fast; default).
``flybody``  the anatomically detailed fruit fly of Vaxenburg et al. (TuragaLab/flybody,
             Apache-2.0), loaded from ``data/external/flybody`` if downloaded, made rigid in
             its rest pose (internal joints removed).

Neither body is driven joint-by-joint by the connectome: flybody's joint controllers
are trained MLPs (not connectome). Instead, descending-neuron output is converted by
the readout (motor/readout.py) into a ``MotorCommand``, and this module turns that into
a net force/torque on the thorax ("motor primitives": walk, turn, jump, flight thrust,
flight torques). This is an explicit, documented abstraction of the VNC + muscles +
wing aerodynamics (see docs/assumptions.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..paths import ASSETS, DATA

G_EARTH = 981.0  # cm/s^2, reference for muscle/wing force scaling (independent of world gravity)
FLYBODY_XML = DATA / "external" / "flybody" / "flybody" / "fruitfly" / "assets" / "fruitfly.xml"


@dataclass
class MotorCommand:
    forward: float = 0.0  # walking drive, -1 (backward) .. 1 (forward)
    turn: float = 0.0  # walking yaw drive, -1 (right) .. 1 (left)
    takeoff: bool = False  # trigger a jump (only from the ground)
    jump_yaw: float = 0.0  # jump direction relative to heading, rad (+ = left)
    power: float = 0.0  # flight thrust modulation around baseline, -1..1
    roll: float = 0.0  # flight torques, -1..1 (body frame)
    pitch: float = 0.0
    yaw: float = 0.0

    def effort(self) -> float:
        return float(abs(self.forward) + abs(self.turn) + abs(self.power) + abs(self.roll) + abs(self.pitch) + abs(self.yaw))


@dataclass
class BodyParams:
    kind: str = "simple"
    walk_speed: float = 3.0  # cm/s at forward=1 (D. melanogaster walks ~1-3 cm/s)
    walk_tau: float = 0.01  # s, time constant of the walking velocity servo
    turn_rate: float = 6.0  # rad/s at turn=1
    turn_tau: float = 0.002  # s (stiff: must overcome foot friction when turning in place)
    jump_speed: float = 60.0  # cm/s takeoff speed (GF-mediated escapes ~0.5-1 m/s)
    jump_elevation: float = 0.9  # rad above horizontal
    jump_duration: float = 0.005  # s, TTM thrust phase
    baseline_thrust: float = 1.0  # flight thrust at power=0, in units of m*G_EARTH (hover on Earth)
    thrust_range: float = 0.6  # +- thrust modulation by 'power'
    flight_turn_rate: float = 3.0  # rad/s at command 1; loop gain < 1 so ~30 ms neural delay cannot cause oscillation
    flapping_damping_tau: float = 0.01  # s, flapping counter-torque (passive wing damping, Hedrick 2009)
    haltere_reflex_gain: float = 0.0  # optional VNC attitude reflex (off by default: not connectome)
    landing_speed: float = 15.0  # cm/s: below this, floor contact in flight counts as landing
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> "BodyParams":
        d = d or {}
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def build_fly_spec(kind: str = "simple") -> mujoco.MjSpec:
    if kind == "simple":
        return mujoco.MjSpec.from_file(str(ASSETS / "simple_fly.xml"))
    if kind == "flybody":
        if not FLYBODY_XML.exists():
            raise FileNotFoundError(f"flybody model not found at {FLYBODY_XML}; run scripts/download_data.py --flybody")
        spec = mujoco.MjSpec.from_file(str(FLYBODY_XML))
        # Rigid posture: the connectome drives motor primitives, not joints, so all internal
        # joints (and the actuators/tendons/sensors that reference them) are removed; geometry,
        # masses and fluid shapes of the anatomical model are kept.
        for coll in (spec.actuators, spec.sensors, spec.tendons, spec.equalities, spec.excludes):
            for el in list(coll):
                spec.delete(el)
        for j in list(spec.joints):
            if j.type != mujoco.mjtJoint.mjJNT_FREE:
                spec.delete(j)
        for k in list(spec.keys):
            spec.delete(k)
        for g in spec.geoms:  # MuJoCo takes the max friction of a contact pair: make legs skids
            g.friction = [0.02, 0.001, 0.0001]
        for b in spec.bodies:  # flybody has eye/antenna cameras; add sites for our sensors
            if b.name == "head":
                for nm, pos in (("eye_left", [0.02, 0.03, 0.0]), ("eye_right", [0.02, -0.03, 0.0]),
                                ("antenna_left", [0.03, 0.01, 0.02]), ("antenna_right", [0.03, -0.01, 0.02])):
                    b.add_site(name=nm, pos=pos, size=[0.004, 0, 0])
        return spec
    raise ValueError(f"unknown body kind {kind!r}")


class FlyBody:
    """Runtime handle: state readout + motor primitive -> thorax wrench."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, params: BodyParams, mode: str = "ground"):
        self.m, self.d, self.p = model, data, params
        self.bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "thorax")
        self.mass = float(model.body_subtreemass[self.bid])
        self.I_body = self._composite_inertia()  # whole fly (all child bodies), thorax frame
        self.inertia = np.diag(self.I_body).copy()
        self.mode = mode  # ground | jump | flight
        self.jump_t0 = None
        self.jump_dir = np.zeros(3)
        self.events: list[tuple[float, str]] = []
        self.external_torque = np.zeros(3)  # world frame, set by scenarios (e.g. gusts)
        self.floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    def _composite_inertia(self) -> np.ndarray:
        """Inertia tensor of the whole subtree about its centre of mass, in the thorax frame."""
        m, d = self.m, self.d
        mujoco.mj_forward(m, d)
        sub = [b for b in range(m.nbody) if self._is_descendant(b)]
        masses = m.body_mass[sub]
        com = (masses[:, None] * d.xipos[sub]).sum(0) / masses.sum()
        I = np.zeros((3, 3))
        for b, mb in zip(sub, masses):
            Rb = d.ximat[b].reshape(3, 3)
            I += Rb @ np.diag(m.body_inertia[b]) @ Rb.T
            r = d.xipos[b] - com
            I += mb * (r @ r * np.eye(3) - np.outer(r, r))
        R = d.xmat[self.bid].reshape(3, 3)
        return R.T @ I @ R

    def _is_descendant(self, b: int) -> bool:
        while b > 0:
            if b == self.bid:
                return True
            b = self.m.body_parentid[b]
        return False

    # ---------------------------------------------------------------- state
    @property
    def pos(self) -> np.ndarray:
        return self.d.xpos[self.bid].copy()

    @property
    def R(self) -> np.ndarray:
        """Rotation matrix body->world."""
        return self.d.xmat[self.bid].reshape(3, 3).copy()

    @property
    def vel(self) -> np.ndarray:
        """Linear velocity of the thorax frame in world coordinates."""
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.bid, v, 0)
        return v[3:].copy()

    @property
    def angvel_body(self) -> np.ndarray:
        """Angular velocity in the body (xmat) frame. (mj_objectVelocity's local frame for a
        body is the *inertial* frame, which can be rotated relative to the body frame.)"""
        return self.R.T @ self.d.cvel[self.bid][:3]

    @property
    def heading(self) -> float:
        fwd = self.R[:, 0]
        return float(np.arctan2(fwd[1], fwd[0]))

    def floor_contact(self) -> bool:
        if self.floor_gid < 0:
            return False
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.geom1 == self.floor_gid or c.geom2 == self.floor_gid:
                return True
        return False

    def site_pos(self, name: str) -> np.ndarray:
        return self.d.site(name).xpos.copy()

    # ---------------------------------------------------------------- control
    def apply(self, cmd: MotorCommand) -> None:
        p, t = self.p, self.d.time
        R = self.R
        force, torque = np.zeros(3), np.zeros(3)
        mg_ref = self.mass * G_EARTH
        if self.mode == "ground":
            if cmd.takeoff:
                self.mode, self.jump_t0 = "jump", t
                c, s = np.cos(cmd.jump_yaw), np.sin(cmd.jump_yaw)
                h = R[:, 0].copy()
                h[2] = 0
                h /= np.linalg.norm(h) + 1e-12
                left = np.array([-h[1], h[0], 0.0])
                horiz = c * h + s * left
                e = p.jump_elevation
                self.jump_dir = np.cos(e) * horiz + np.array([0, 0, np.sin(e)])
                self.events.append((t, "takeoff"))
            else:
                v = self.vel
                fwd = R[:, 0].copy()
                fwd[2] = 0
                fwd /= np.linalg.norm(fwd) + 1e-12
                v_target = np.clip(cmd.forward, -1, 1) * p.walk_speed
                v_fwd = float(v @ fwd)
                k = self.mass / p.walk_tau
                force += k * (v_target - v_fwd) * fwd
                lat = np.array([-fwd[1], fwd[0], 0.0])
                force += -k * float(v @ lat) * lat  # legs prevent side-slip
                wz = float(self.d.cvel[self.bid][2])
                torque[2] += self.inertia.max() / p.turn_tau * (np.clip(cmd.turn, -1, 1) * p.turn_rate - wz)
        if self.mode == "jump":
            # constant thrust that reaches jump_speed after jump_duration (plus gravity compensation)
            a = p.jump_speed / p.jump_duration
            force += self.mass * a * self.jump_dir - self.mass * self.m.opt.gravity
            if t - self.jump_t0 >= p.jump_duration:
                self.mode = "flight"
                self.events.append((t, "flight"))
        elif self.mode == "flight":
            thrust = mg_ref * (p.baseline_thrust + p.thrust_range * np.clip(cmd.power, -1, 1))
            force += thrust * R[:, 2]
            D = self.I_body / p.flapping_damping_tau
            w_cmd = p.flight_turn_rate * np.clip([cmd.roll, cmd.pitch, cmd.yaw], -1, 1)
            w = self.angvel_body
            # torque command drives body rates towards w_cmd; flapping counter-torque damps rotation
            tq_body = D @ w_cmd - D @ w * (1.0 + p.haltere_reflex_gain)
            torque += R @ tq_body
            if self.floor_contact() and np.linalg.norm(self.vel) < p.landing_speed:
                self.mode = "ground"
                self.events.append((t, "landing"))
        # xfrc_applied acts at the thorax body's own COM; shift the line of action to the
        # whole fly's COM so primitives do not create spurious torques (matters for flybody)
        lever = self.d.subtree_com[self.bid] - self.d.xipos[self.bid]
        self.d.xfrc_applied[self.bid, :3] = force
        self.d.xfrc_applied[self.bid, 3:] = torque + np.cross(lever, force) + self.external_torque

    def wings_active(self) -> bool:
        return self.mode in ("jump", "flight")
