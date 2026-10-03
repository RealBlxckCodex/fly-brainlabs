"""Scan w_syn for a dataset: how many neurons respond to a sparse stimulus, and does the
known pathway (sugar -> MN9, LPLC2 -> DNp01) respond? Used to pick the MaleCNS value.

Usage: python scripts/calibrate_wsyn.py --dataset malecns_v1.0
"""
from __future__ import annotations

import argparse
import json


from flybrainlab.brain import LIFNetwork, LIFParams
from flybrainlab.connectome import GroupResolver, load_connectome
from flybrainlab.paths import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="malecns_v1.0")
    ap.add_argument("--values", type=float, nargs="*", default=[0.08, 0.1, 0.12, 0.14, 0.15, 0.16, 0.2, 0.275])
    ap.add_argument("--ms", type=float, default=500)
    a = ap.parse_args()
    c = load_connectome(a.dataset)
    g = GroupResolver(c)
    rows = []
    for w in a.values:
        p = LIFParams(w_syn=w, f_poi=0.275 * 250 / w)  # constant 68.75 mV Poisson kick
        row = {"w_syn": w}
        for src, rate, tgt in (("sugar_grn", 150.0, "mn9"), ("lplc2", 100.0, "giant_fiber")):
            net = LIFNetwork(c.W, p, seed=1)
            net.set_poisson_rates(g.resolve(src), rate)
            net.run(a.ms)
            r = net.rates_hz()
            row[f"{src}_n_active"] = int((r > 0).sum())
            row[f"{src}->{tgt}_hz"] = float(r[g.resolve(tgt)].mean())
        rows.append(row)
        print(row, flush=True)
    (ROOT / "docs" / "validation" / f"wsyn_scan_{a.dataset}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
