"""Compose fly body + arena into one MuJoCo model (CGS units: cm, g, s).

Gravity and air are parameters: ``gravity`` is the magnitude in cm/s^2 (981 = Earth,
0 = weightless), ``air_density`` (g/cm^3) and ``air_viscosity`` (poise) drive MuJoCo's
fluid model. Zero-g keeps the air (zero-g is not vacuum).
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..body.fly import BodyParams, FlyBody, build_fly_spec

GEOM = {"sphere": mujoco.mjtGeom.mjGEOM_SPHERE, "box": mujoco.mjtGeom.mjGEOM_BOX,
        "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID, "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
        "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER}


def _size3(s):
    s = list(s) + [0.0] * (3 - len(s))
    return s[:3]


def build_world(world: dict, body_params: BodyParams, init_pos, init_heading: float, init_mode: str,
                timestep: float = 1e-4):
    spec = build_fly_spec(body_params.kind)
    spec.option.timestep = timestep
    spec.option.gravity = [0.0, 0.0, -float(world.get("gravity", 981.0))]
    spec.option.density = float(world.get("air_density", 0.00128))
    spec.option.viscosity = float(world.get("air_viscosity", 0.000185))
    wb = spec.worldbody
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.82, 0.82, 0.8], rgb2=[0.7, 0.7, 0.68], width=256, height=256)
    mat = spec.add_material(name="grid")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    mat.texrepeat = [20, 20]
    if world.get("floor", True):
        half = float(world.get("arena_half_size", 20.0))
        wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[half, half, 0.1], material="grid",
                    contype=1, conaffinity=1, friction=[float(world.get("floor_friction", 0.05)), 0.005, 0.0001])
        # MuJoCo uses the max friction of the two geoms; legs are skids, walking is the servo primitive
    wb.add_light(name="sun", pos=[0, 0, 30], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    for k, b in enumerate(world.get("boxes", [])):
        wb.add_geom(name=b.get("name", f"box{k}"), type=GEOM["box"], pos=b["pos"], size=_size3(b["size"]),
                    rgba=b.get("rgba", [0.45, 0.45, 0.5, 1]), contype=1, conaffinity=1)
    for m in world.get("mocaps", []):
        body = wb.add_body(name=m["name"], mocap=True, pos=m.get("pos", [0, 0, 0]))
        for k, g in enumerate(m["geoms"]):
            body.add_geom(name=f"{m['name']}_g{k}", type=GEOM[g["type"]], size=_size3(g["size"]),
                          pos=g.get("pos", [0, 0, 0]), rgba=g.get("rgba", [0.1, 0.1, 0.1, 1]),
                          contype=0, conaffinity=0)
    for k, mk in enumerate(world.get("markers", [])):
        wb.add_geom(name=mk.get("name", f"marker{k}"), type=GEOM["sphere"], pos=mk["pos"],
                    size=[mk.get("size", 0.2), 0, 0], rgba=mk.get("rgba", [1, 0.9, 0.2, 0.6]), contype=0,
                    conaffinity=0)
    model = spec.compile()
    data = mujoco.MjData(model)
    q = np.array([np.cos(init_heading / 2), 0, 0, np.sin(init_heading / 2)])
    data.qpos[0:3] = init_pos
    data.qpos[3:7] = q
    if init_mode == "flight":
        data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    body = FlyBody(model, data, body_params, mode=init_mode)
    return model, data, body
