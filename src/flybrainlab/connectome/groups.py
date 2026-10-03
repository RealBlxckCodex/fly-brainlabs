"""Named neuron groups (e.g. 'giant_fiber', 'lplc2_L') resolved to simulation indices.

Group definitions live in ``configs/groups/<dataset>.yaml``. A selector is a dict
with any of:

    ids:        [flywire/body ids]
    type:       [exact cell types]
    type_regex: regex on cell type
    superclass: [..]   cls: [..]   subclass: [..]
    side:       L | R
    union:      [other group names]

Multiple keys are combined with AND; ``union`` is OR over named groups.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..config import groups_config
from .model import Connectome


def _types(df: pd.DataFrame) -> pd.Series:
    return df["type"].fillna("").astype(str)


class GroupResolver:
    def __init__(self, connectome: Connectome, definitions: dict | None = None):
        self.c = connectome
        self.defs = definitions if definitions is not None else groups_config(connectome.meta["dataset"])
        self._cache: dict[str, np.ndarray] = {}

    def names(self) -> list[str]:
        return sorted(self.defs)

    def resolve(self, spec) -> np.ndarray:
        """Resolve a group name, a selector dict or a list of ids to sorted indices."""
        if isinstance(spec, str):
            if spec not in self._cache:
                if spec not in self.defs:
                    raise KeyError(f"unknown neuron group {spec!r}")
                self._cache[spec] = self._select(self.defs[spec])
            return self._cache[spec]
        if isinstance(spec, dict):
            return self._select(spec)
        if isinstance(spec, (list, tuple, np.ndarray)):
            if len(spec) and isinstance(spec[0], str):
                return np.unique(np.concatenate([self.resolve(s) for s in spec]))
            return np.sort(self.c.index_of(spec, strict=False))
        raise TypeError(f"cannot resolve group spec {spec!r}")

    def _select(self, sel: dict) -> np.ndarray:
        n = self.c.neurons
        mask = pd.Series(True, index=n.index)
        if "union" in sel:
            idx = np.unique(np.concatenate([self.resolve(g) for g in sel["union"]]))
            m = pd.Series(False, index=n.index)
            m.iloc[idx] = True
            mask &= m
        if "ids" in sel:
            m = pd.Series(False, index=n.index)
            # ids absent from this release (e.g. proofreading changed a root id) are skipped
            m.iloc[self.c.index_of(sel["ids"], strict=False)] = True
            mask &= m
        for key, col in (("type", "type"), ("superclass", "superclass"), ("cls", "cls"), ("subclass", "subclass")):
            if key in sel:
                vals = sel[key] if isinstance(sel[key], list) else [sel[key]]
                mask &= n[col].isin(vals)
        if "type_regex" in sel:
            mask &= _types(n).str.contains(sel["type_regex"], regex=True).to_numpy()
        if "side" in sel:
            mask &= n["side"] == sel["side"]
        return np.flatnonzero(mask.to_numpy())

    def describe(self, name: str) -> dict:
        idx = self.resolve(name)
        sub = self.c.neurons.iloc[idx]
        return {
            "name": name,
            "definition": self.defs.get(name),
            "n_neurons": int(len(idx)),
            "types": _types(sub).value_counts().head(10).to_dict(),
            "sides": sub["side"].fillna("?").astype(str).value_counts().to_dict(),
        }

    def search(self, query: str, limit: int = 25) -> list[dict]:
        """Search group names and cell types (case-insensitive substring)."""
        q = query.lower()
        out = [{"kind": "group", **self.describe(g)} for g in self.names() if q in g.lower()]
        types = _types(self.c.neurons)
        hits = types[types != ""]
        hits = hits[hits.str.lower().str.contains(re.escape(q))].value_counts().head(limit)
        out += [{"kind": "cell_type", "type": t, "n_neurons": int(k),
                 "selector": {"type": [t]}} for t, k in hits.items()]
        return out[:limit]
