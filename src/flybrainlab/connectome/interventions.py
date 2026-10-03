"""Connectome interventions: lesion, rewiring and shuffle controls.

All functions return a *new* Connectome; the original is never mutated.
Every intervention is appended to ``meta['modifications']`` for the run log.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .groups import GroupResolver
from .model import Connectome


def lesion(c: Connectome, idx: np.ndarray, note: str = "") -> Connectome:
    """Silence neurons: remove all their incoming and outgoing synapses.

    (Shiu et al. silence by zeroing outgoing synapses only; additionally removing
    inputs makes the lesion independent of the neuron's own dynamics. The LIF engine
    also clamps silenced neurons so external Poisson input cannot drive them.)
    """
    idx = np.asarray(idx, dtype=np.int64)
    keep = np.ones(c.n_neurons, dtype=np.float32)
    keep[idx] = 0.0
    D = sp.diags(keep)
    W = D @ c.W @ D
    return c.with_weights(W, f"lesion {note or ''} n={len(idx)}".strip())


def _block_mask(W: sp.csr_matrix, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Boolean mask over W.data selecting edges pre->post."""
    coo = W.tocoo()
    in_pre = np.zeros(W.shape[1], bool)
    in_pre[pre] = True
    in_post = np.zeros(W.shape[0], bool)
    in_post[post] = True
    return in_pre[coo.col] & in_post[coo.row], coo


def rewire(c: Connectome, ops: list[dict], groups: GroupResolver, seed: int = 0) -> Connectome:
    """Apply a list of rewiring operations.

    Supported ops (``pre``/``post``/``to`` are group names, selectors or id lists):

    - ``{op: scale, pre, post, factor}``   multiply existing synapse counts
    - ``{op: cut, pre, post}``             remove connections (= scale 0)
    - ``{op: add, pre, post, synapses, sign?}``  create all-to-all connections
    - ``{op: shift, pre, post, to}``       move pre->post synapses onto random targets in ``to``
    - ``{op: flip_sign, pre}``             invert the sign of all outputs of ``pre``
    - ``{op: set_sign, pre, sign: 1|-1}``  force the sign of all outputs of ``pre``
    """
    rng = np.random.default_rng(seed)
    W = c.W.tocsr(copy=True)
    notes = []
    for op in ops:
        kind = op["op"]
        pre = groups.resolve(op["pre"])
        if kind in ("scale", "cut"):
            post = groups.resolve(op["post"])
            factor = 0.0 if kind == "cut" else float(op["factor"])
            m, coo = _block_mask(W, pre, post)
            data = coo.data.copy()
            data[m] *= factor
            W = sp.csr_matrix((data, (coo.row, coo.col)), shape=W.shape)
            notes.append(f"{kind} {op['pre']}->{op['post']} x{factor} ({int(m.sum())} edges)")
        elif kind == "add":
            post = groups.resolve(op["post"])
            n_syn = float(op.get("synapses", 5))
            sign = np.sign(op["sign"]) if "sign" in op else c.neurons["sign"].to_numpy()[pre].astype(np.float32)
            rows = np.repeat(post, len(pre))
            cols = np.tile(pre, len(post))
            vals = np.tile(np.broadcast_to(sign, len(pre)), len(post)) * n_syn
            W = W + sp.csr_matrix((vals.astype(np.float32), (rows, cols)), shape=W.shape)
            notes.append(f"add {op['pre']}->{op['post']} {n_syn} syn ({len(rows)} edges)")
        elif kind == "shift":
            post, to = groups.resolve(op["post"]), groups.resolve(op["to"])
            m, coo = _block_mask(W, pre, post)
            row = coo.row.copy()
            row[m] = rng.choice(to, size=int(m.sum()))
            W = sp.csr_matrix((coo.data, (row, coo.col)), shape=W.shape)
            notes.append(f"shift {op['pre']}->{op['post']} onto {op['to']} ({int(m.sum())} edges)")
        elif kind == "set_sign":
            m, coo = _block_mask(W, pre, np.arange(W.shape[0]))
            data = coo.data.copy()
            data[m] = np.abs(data[m]) * np.sign(op["sign"])
            W = sp.csr_matrix((data, (coo.row, coo.col)), shape=W.shape)
            notes.append(f"set_sign {op['pre']} -> {op['sign']} ({int(m.sum())} edges)")
        elif kind == "flip_sign":
            m, coo = _block_mask(W, pre, np.arange(W.shape[0]))
            data = coo.data.copy()
            data[m] *= -1
            W = sp.csr_matrix((data, (coo.row, coo.col)), shape=W.shape)
            notes.append(f"flip_sign {op['pre']} ({int(m.sum())} edges)")
        else:
            raise ValueError(f"unknown rewiring op {kind!r}")
    return c.with_weights(W, "rewire: " + "; ".join(notes))


def shuffle(c: Connectome, seed: int, mode: str = "targets") -> Connectome:
    """Shuffle baseline (control).

    ``targets`` (default): for every presynaptic neuron keep its number of output
    connections and their signed weights, but draw the postsynaptic partners
    uniformly at random (no self-connections). Destroys the real wiring while
    keeping out-degree, weight distribution, sign and total synapse count.

    ``degree``: globally permute the postsynaptic end of all edges. Preserves both
    the in-degree and out-degree (edge counts) of every neuron.
    """
    rng = np.random.default_rng(seed)
    coo = c.W.tocoo()
    pre, data = coo.col, coo.data
    n = c.n_neurons
    if mode == "targets":
        post = rng.integers(0, n - 1, size=len(pre))
        post = post + (post >= pre)  # skip self-connection
    elif mode == "degree":
        post = rng.permutation(coo.row)
    else:
        raise ValueError(f"unknown shuffle mode {mode!r}")
    W = sp.csr_matrix((data, (post, pre)), shape=c.W.shape)
    return c.with_weights(W, f"shuffle mode={mode} seed={seed}")


def apply_interventions(c: Connectome, groups: GroupResolver, lesion_spec=None, rewiring=None,
                        shuffle_spec=None, seed: int = 0, model_mods=None) -> tuple[Connectome, np.ndarray]:
    """Apply model mods -> shuffle -> rewiring -> lesion. Returns (connectome, silenced indices).

    ``model_mods`` are scenario-level modelling assumptions (e.g. photoreceptor sign), applied
    before the shuffle so that controls share them; ``rewiring`` is the experimental intervention.
    """
    if model_mods:
        c = rewire(c, model_mods, groups, seed=0)
    if shuffle_spec:
        mode = shuffle_spec if isinstance(shuffle_spec, str) else "targets"
        c = shuffle(c, seed=10_000 + seed, mode=mode)
    if rewiring:
        c = rewire(c, rewiring, groups, seed=seed)
    silenced = np.zeros(0, dtype=np.int64)
    if lesion_spec:
        specs = lesion_spec if isinstance(lesion_spec, list) else [lesion_spec]
        silenced = np.unique(np.concatenate([groups.resolve(s) for s in specs]))
        c = lesion(c, silenced, note=str(lesion_spec))
    return c, silenced
