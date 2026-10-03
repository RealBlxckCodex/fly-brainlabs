import mujoco
import numpy as np

from flybrainlab.body.fly import BodyParams, MotorCommand
from flybrainlab.env.world import build_world


def _run(world, mode, n, cmd_fn, pos=(0, 0, 8.0), qvel_w=None):
    m, d, b = build_world(world, BodyParams(), list(pos), 0.0, mode)
    if qvel_w is not None:
        d.qvel[3:6] = qvel_w
    for k in range(n):
        b.apply(cmd_fn(k))
        mujoco.mj_step(m, d)
    return b


def test_flight_rotation_is_damped():
    b = _run({"gravity": 0.0, "floor": False}, "flight", 5000, lambda k: MotorCommand(), qvel_w=[2, -3, 1.5])
    assert np.linalg.norm(b.angvel_body) < 1.5
    assert np.all(np.isfinite(b.pos))


def test_hover_at_earth_gravity_holds_height():
    b = _run({"gravity": 981.0, "floor": False}, "flight", 5000, lambda k: MotorCommand())
    assert abs(b.pos[2] - 8.0) < 0.5


def test_zero_g_keeps_air_drag():
    b = _run({"gravity": 0.0, "floor": False}, "flight", 10000, lambda k: MotorCommand())
    v = np.linalg.norm(b.vel)
    assert 50 < v < 500  # thrust balanced by air drag: terminal velocity, not free acceleration (981 cm/s)


def test_walking_and_turning():
    b = _run({"gravity": 981.0}, "ground", 10000, lambda k: MotorCommand(forward=1.0), pos=(0, 0, 0.075))
    assert 1.5 < np.linalg.norm(b.vel[:2]) < 3.5
    b = _run({"gravity": 981.0}, "ground", 3000, lambda k: MotorCommand(turn=1.0), pos=(0, 0, 0.075))
    assert b.heading > 0.5


def test_jump_takes_off():
    b = _run({"gravity": 981.0}, "ground", 2000, lambda k: MotorCommand(takeoff=(k == 5)), pos=(0, 0, 0.075))
    assert b.mode == "flight" and b.pos[2] > 1.0
    assert [e for _, e in b.events][:2] == ["takeoff", "flight"]


def test_flybody_rigid_backend():
    import pytest

    from flybrainlab.body.fly import FLYBODY_XML

    if not FLYBODY_XML.exists():
        pytest.skip("flybody not downloaded (flylab download --flybody)")
    bp = BodyParams(kind="flybody")
    m, d, b = build_world({"gravity": 981.0, "floor": False}, bp, [0, 0, 8.0], 0.0, "flight")
    assert m.nq == 7 and 0.0009 < b.mass < 0.0011  # rigid, ~1 mg
    for _ in range(5000):
        b.apply(MotorCommand())
        mujoco.mj_step(m, d)
    assert abs(b.pos[2] - 8.0) < 0.5 and b.R[2, 2] > 0.95
