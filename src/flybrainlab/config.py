"""YAML config loading with simple deep-merge of overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIGS


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def default_config() -> dict[str, Any]:
    return load_yaml(CONFIGS / "default.yaml")


def scenario_config(name: str) -> dict[str, Any]:
    p = CONFIGS / "scenarios" / f"{name}.yaml"
    if not p.exists():
        raise KeyError(f"unknown scenario '{name}' (no {p})")
    return load_yaml(p)


def groups_config(dataset: str) -> dict[str, Any]:
    return load_yaml(CONFIGS / "groups" / f"{dataset}.yaml")
