import numpy as np
import pandas as pd
import pytest

from flybrainlab.connectome import Connectome, GroupResolver
from flybrainlab.paths import PROCESSED, RAW

# Synthetic "toy fly": every group the encoders/readout use exists with a few neurons.
TOY_TYPES = {
    "LPLC2": 20, "LC4": 10, "DNp01": 2, "DNp02": 2, "DNp11": 2, "DNa02": 2, "DNa01": 2, "DNp09": 2, "MDN": 2,
    "DNg02_a": 4, "DNp20": 2, "DNb06": 2, "HSE": 2, "VS": 4, "JO-C1": 6, "JO-E1": 6, "R1-R6": 20, "ORN_DM1": 10, "MN9": 2, "SUGAR": 6,
    "OTHER": 200,
}

TOY_GROUPS = {
    "giant_fiber": {"type": ["DNp01"]}, "lplc2": {"type": ["LPLC2"]},
    "lplc2_L": {"type": ["LPLC2"], "side": "L"}, "lplc2_R": {"type": ["LPLC2"], "side": "R"},
    "lc4_L": {"type": ["LC4"], "side": "L"}, "lc4_R": {"type": ["LC4"], "side": "R"},
    "dnp02_L": {"type": ["DNp02"], "side": "L"}, "dnp02_R": {"type": ["DNp02"], "side": "R"},
    "dnp11_L": {"type": ["DNp11"], "side": "L"}, "dnp11_R": {"type": ["DNp11"], "side": "R"},
    "dna02_L": {"type": ["DNa02"], "side": "L"}, "dna02_R": {"type": ["DNa02"], "side": "R"},
    "dna01_L": {"type": ["DNa01"], "side": "L"}, "dna01_R": {"type": ["DNa01"], "side": "R"},
    "dnp09": {"type": ["DNp09"]}, "mdn": {"type": ["MDN"]},
    "dng02": {"type_regex": "^DNg02"}, "dng02_L": {"type_regex": "^DNg02", "side": "L"},
    "dng02_R": {"type_regex": "^DNg02", "side": "R"},
    "dnp20_L": {"type": ["DNp20"], "side": "L"}, "dnp20_R": {"type": ["DNp20"], "side": "R"},
    "dnb06_L": {"type": ["DNb06"], "side": "L"}, "dnb06_R": {"type": ["DNb06"], "side": "R"},
    "hs_L": {"type": ["HSE"], "side": "L"}, "hs_R": {"type": ["HSE"], "side": "R"},
    "vs_L": {"type": ["VS"], "side": "L"}, "vs_R": {"type": ["VS"], "side": "R"},
    "jo_c_L": {"type_regex": "^JO-C", "side": "L"}, "jo_c_R": {"type_regex": "^JO-C", "side": "R"},
    "jo_e_L": {"type_regex": "^JO-E", "side": "L"}, "jo_e_R": {"type_regex": "^JO-E", "side": "R"},
    "r1r6_L": {"type": ["R1-R6"], "side": "L"}, "r1r6_R": {"type": ["R1-R6"], "side": "R"},
    "orn_attr_L": {"type": ["ORN_DM1"], "side": "L"}, "orn_attr_R": {"type": ["ORN_DM1"], "side": "R"},
    "sugar_grn": {"type": ["SUGAR"]}, "mn9": {"type": ["MN9"]},
}


def make_toy(seed: int = 0) -> Connectome:
    rng = np.random.default_rng(seed)
    rows = []
    for t, n in TOY_TYPES.items():
        for k in range(n):
            rows.append({"id": 1000 + len(rows), "type": t, "superclass": "toy", "cls": None,
                         "side": "L" if k % 2 == 0 else "R", "nt": "acetylcholine", "sign": 1})
    neurons = pd.DataFrame(rows)
    idx = {t: np.flatnonzero(neurons["type"].to_numpy() == t) for t in TOY_TYPES}
    pre, post, w = [], [], []

    def connect(a, b, syn):
        for i in idx[a]:
            for j in idx[b]:
                pre.append(i)
                post.append(j)
                w.append(syn)

    connect("LPLC2", "DNp01", 30)  # looming -> giant fiber: strong
    connect("LC4", "DNp01", 30)
    connect("SUGAR", "MN9", 40)
    # sparse random background wiring among OTHER neurons
    o = idx["OTHER"]
    sign = {int(i): (-1 if k % 4 == 0 else 1) for k, i in enumerate(o)}  # one sign per neuron (Dale)
    for _ in range(600):
        i = int(rng.choice(o))
        pre.append(i)
        post.append(rng.choice(o))
        w.append(rng.integers(1, 5) * sign[i])
    return Connectome.from_edges(neurons, pre, post, w, {"dataset": "toy"})


@pytest.fixture
def toy():
    c = make_toy()
    return c, GroupResolver(c, TOY_GROUPS)


def have_data(ds: str) -> bool:
    return (PROCESSED / ds / "weights.npz").exists() or (RAW / ds).exists()
