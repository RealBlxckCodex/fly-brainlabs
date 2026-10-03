"""Loaders for the supported connectome releases.

Supported datasets
------------------
``flywire_v783``
    FlyWire / FAFB public release 783 (female brain), in the exact format used by
    Shiu et al. 2024 (``Completeness_783.csv`` + ``Connectivity_783.parquet``),
    enriched with the FlyWire annotations (Schlegel et al. 2024).
``malecns_v1.0``
    Janelia/Google MaleCNS v1.0 (male brain + VNC), flat-connectome feather files
    from ``gs://flyem-male-cns/v1.0/connectome-data/flat-connectome``.

Raw files are fetched by ``scripts/download_data.py``. Processed caches live in
``data/processed/<dataset>/``.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import PROCESSED, RAW
from .model import Connectome

# Neurotransmitter -> sign. ACh excitatory, GABA and glutamate inhibitory (as in
# Shiu et al. 2024). Histamine is inhibitory (photoreceptor -> LMC synapses).
# Modulatory/unknown transmitters default to excitatory (assumption, see docs).
NT_SIGN = {
    "acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1,
    "dopamine": 1, "serotonin": 1, "octopamine": 1, "tyramine": 1,
}

SIDE_MAP = {"left": "L", "right": "R", "L": "L", "R": "R", "center": "C", "M": "C"}

# Reference numbers from the releases, used by the sanity checks.
EXPECTED = {
    "flywire_v783": {"n_neurons": 138_639, "n_synapses": 54_492_922, "tol": 0.001,
                     "source": "Shiu et al. Completeness/Connectivity_783 (FlyWire v783 public, >=5 syn threshold)"},
    "malecns_v1.0": {"n_neurons": 166_700, "n_synapses": 125_000_000, "tol": 0.05,
                     "source": "MaleCNS v1.0 release: ~166,700 neurons, ~125 million synapses"},
}


def available_datasets() -> list[str]:
    return list(EXPECTED)


def _log(msg: str) -> None:
    print(f"[connectome] {msg}", flush=True)


# --------------------------------------------------------------------- FlyWire
def build_flywire_v783(raw: Path = RAW / "flywire_v783") -> Connectome:
    comp = pd.read_csv(raw / "Completeness_783.csv", index_col=0)
    con = pd.read_parquet(raw / "Connectivity_783.parquet")
    neurons = pd.DataFrame({"id": comp.index.astype(np.int64)})
    ann_path = raw / "Supplemental_file1_neuron_annotations.tsv"
    if ann_path.exists():
        ann = pd.read_csv(ann_path, sep="\t", low_memory=False,
                          usecols=["root_id", "cell_type", "hemibrain_type", "super_class", "cell_class",
                                   "cell_sub_class", "side", "top_nt"])
        ann = ann.drop_duplicates("root_id").set_index("root_id")
        a = ann.reindex(neurons["id"])
        neurons["type"] = a["cell_type"].fillna(a["hemibrain_type"]).to_numpy()
        neurons["superclass"] = a["super_class"].to_numpy()
        neurons["cls"] = a["cell_class"].to_numpy()
        neurons["subclass"] = a["cell_sub_class"].to_numpy()
        neurons["side"] = a["side"].map(SIDE_MAP).to_numpy()
        neurons["nt"] = a["top_nt"].to_numpy()
        neurons["hemibrain_type"] = a["hemibrain_type"].to_numpy()
    else:
        _log("annotation file missing: types/sides unavailable")
    # Sign is already encoded per edge by Shiu et al. ('Excitatory' column).
    pre, post = con["Presynaptic_Index"].to_numpy(), con["Postsynaptic_Index"].to_numpy()
    neurons["sign"] = 0
    first_edge = pd.Series(con["Excitatory"].to_numpy(), index=pre).groupby(level=0).first()
    neurons.loc[first_edge.index, "sign"] = first_edge.to_numpy()
    meta = {"dataset": "flywire_v783", "license": "CC BY-NC 4.0 (FlyWire)",
            "sources": [str(raw / "Completeness_783.csv"), str(raw / "Connectivity_783.parquet"), str(ann_path)],
            "sign_rule": "per-edge 'Excitatory' column from Shiu et al. 2024"}
    return Connectome.from_edges(neurons, pre, post, con["Excitatory x Connectivity"].to_numpy(), meta)


# --------------------------------------------------------------------- MaleCNS
MALECNS_FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
}


def build_malecns_v1(raw: Path = RAW / "malecns_v1.0") -> Connectome:
    ann = pd.read_feather(raw / MALECNS_FILES["annotations"])
    # A "neuron" = annotated body with a superclass (166,700 in v1.0; matches release).
    ann = ann[ann["superclass"].notna()].copy()
    nt = pd.read_feather(raw / MALECNS_FILES["neurotransmitters"], columns=["body", "consensus_nt", "predicted_nt"])
    nt = nt.drop_duplicates("body").set_index("body")
    neurons = pd.DataFrame({
        "id": ann["bodyId"].astype(np.int64).to_numpy(),
        "type": ann["type"].to_numpy(),
        "superclass": ann["superclass"].to_numpy(),
        "cls": ann["class"].to_numpy(),
        "subclass": ann["subclass"].to_numpy(),
        "side": ann["somaSide"].fillna(ann["rootSide"]).map(SIDE_MAP).to_numpy(),
        "flywire_type": ann["flywireType"].to_numpy(),
        "status": ann["statusLabel"].astype(str).to_numpy(),
    })
    n = nt.reindex(neurons["id"])
    neurons["nt"] = n["consensus_nt"].fillna(n["predicted_nt"]).to_numpy()
    neurons["sign"] = neurons["nt"].map(NT_SIGN).fillna(1).astype(np.int8).to_numpy()

    w = pd.read_feather(raw / MALECNS_FILES["weights"], columns=["body_pre", "body_post", "weight"])
    idx = pd.Series(np.arange(len(neurons)), index=neurons["id"].to_numpy())
    pre = idx.reindex(w["body_pre"].to_numpy()).to_numpy()
    post = idx.reindex(w["body_post"].to_numpy()).to_numpy()
    keep = ~(np.isnan(pre) | np.isnan(post))
    _log(f"MaleCNS: kept {keep.sum():,}/{len(w):,} edges between the {len(neurons):,} neurons")
    pre, post = pre[keep].astype(np.int64), post[keep].astype(np.int64)
    signed = w["weight"].to_numpy()[keep] * neurons["sign"].to_numpy()[pre]
    meta = {"dataset": "malecns_v1.0", "license": "CC BY 4.0 (Janelia / Google Research MaleCNS)",
            "sources": [str(raw / f) for f in MALECNS_FILES.values()],
            "raw_edges_total": int(len(w)), "raw_synapses_total": int(w["weight"].sum()),
            "sign_rule": "consensus_nt per presynaptic body: ACh +, GABA/Glu/His -, other/unknown +"}
    return Connectome.from_edges(neurons, pre, post, signed, meta)


BUILDERS = {"flywire_v783": build_flywire_v783, "malecns_v1.0": build_malecns_v1}


def load_connectome(dataset: str = "malecns_v1.0", rebuild: bool = False) -> Connectome:
    """Load a processed connectome (building and caching it on first use)."""
    if dataset not in BUILDERS:
        raise KeyError(f"unknown dataset {dataset!r}; choose from {list(BUILDERS)}")
    cache = PROCESSED / dataset
    if cache.exists() and not rebuild:
        return Connectome.load(cache)
    t0 = time.time()
    _log(f"building {dataset} from raw files ...")
    c = BUILDERS[dataset]()
    c.save(cache)
    _log(f"built {dataset} in {time.time() - t0:.0f}s: {c.n_neurons:,} neurons, {c.n_edges:,} edges")
    return c


def sanity_check(c: Connectome) -> dict:
    """Compare neuron/synapse counts with the numbers published for the release."""
    ds = c.meta.get("dataset")
    exp = EXPECTED.get(ds, {})
    res = {"dataset": ds, **c.summary(), "expected": exp, "checks": {}}
    syn = c.meta.get("raw_synapses_total", c.n_synapses)
    for key, val in (("n_neurons", c.n_neurons), ("n_synapses", syn)):
        if key in exp:
            rel = abs(val - exp[key]) / exp[key]
            res["checks"][key] = {"value": int(val), "expected": exp[key], "rel_error": rel, "ok": rel <= exp["tol"]}
    res["ok"] = all(v["ok"] for v in res["checks"].values()) if res["checks"] else None
    return res
