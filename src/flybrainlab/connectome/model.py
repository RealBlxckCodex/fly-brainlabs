"""In-memory connectome representation shared by all datasets.

The weight matrix ``W`` is a sparse CSR matrix of shape (N_post, N_pre) holding
*signed synapse counts* (sign from the presynaptic neurotransmitter). The spiking
model multiplies it by ``w_syn`` (mV per synapse).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

NEURON_COLUMNS = ["id", "type", "superclass", "cls", "side", "nt", "sign"]


@dataclass
class Connectome:
    neurons: pd.DataFrame  # index 0..N-1 == simulation index
    W: sp.csr_matrix  # (post, pre) signed synapse counts, float32
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.neurons)
        if self.W.shape != (n, n):
            raise ValueError(f"W shape {self.W.shape} does not match {n} neurons")
        self._id2idx: dict[int, int] | None = None

    # ------------------------------------------------------------------ basics
    @property
    def n_neurons(self) -> int:
        return len(self.neurons)

    @property
    def n_edges(self) -> int:
        return int(self.W.nnz)

    @property
    def n_synapses(self) -> int:
        return int(np.abs(self.W.data).sum())

    def index_of(self, ids, strict: bool = True) -> np.ndarray:
        if self._id2idx is None:
            self._id2idx = {int(i): k for k, i in enumerate(self.neurons["id"].to_numpy())}
        out, missing = [], []
        for i in ids:
            k = self._id2idx.get(int(i))
            (missing if k is None else out).append(i if k is None else k)
        if missing and strict:
            raise KeyError(f"{len(missing)} ids not in connectome, e.g. {missing[:3]}")
        return np.asarray(out, dtype=np.int64)

    def with_weights(self, W: sp.spmatrix, note: str) -> "Connectome":
        W = sp.csr_matrix(W, dtype=np.float32)
        W.eliminate_zeros()
        meta = dict(self.meta)
        meta["modifications"] = list(meta.get("modifications", [])) + [note]
        return Connectome(self.neurons, W, meta)

    def summary(self) -> dict:
        return {
            "dataset": self.meta.get("dataset"),
            "n_neurons": self.n_neurons,
            "n_edges": self.n_edges,
            "n_synapses": self.n_synapses,
            "frac_inhibitory_synapses": float(np.abs(self.W.data[self.W.data < 0]).sum() / max(self.n_synapses, 1)),
            "modifications": self.meta.get("modifications", []),
        }

    # ------------------------------------------------------------------ cache
    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.neurons.to_parquet(folder / "neurons.parquet")
        sp.save_npz(folder / "weights.npz", self.W, compressed=False)
        (folder / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str))

    @classmethod
    def load(cls, folder: str | Path) -> "Connectome":
        folder = Path(folder)
        neurons = pd.read_parquet(folder / "neurons.parquet")
        W = sp.load_npz(folder / "weights.npz").tocsr().astype(np.float32)
        meta = json.loads((folder / "meta.json").read_text())
        return cls(neurons, W, meta)

    @classmethod
    def from_edges(cls, neurons: pd.DataFrame, pre_idx, post_idx, signed_count, meta: dict) -> "Connectome":
        n = len(neurons)
        W = sp.coo_matrix(
            (np.asarray(signed_count, dtype=np.float32), (np.asarray(post_idx), np.asarray(pre_idx))), shape=(n, n)
        ).tocsr()
        W.sum_duplicates()
        W.eliminate_zeros()
        neurons = neurons.reset_index(drop=True)
        for c in NEURON_COLUMNS:
            if c not in neurons:
                neurons[c] = None
        return cls(neurons, W, meta)
