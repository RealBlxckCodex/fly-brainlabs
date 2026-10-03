"""Brian2 reference implementation (Shiu et al. 2024 model.py, adapted to our Connectome).

Used only for validation of the step-wise numpy engine (``lif.py``); far too slow to
build per closed-loop step. Equations, constants and update semantics are copied from
https://github.com/philshiu/Drosophila_brain_model (MIT license).
"""
from __future__ import annotations

from textwrap import dedent

import numpy as np

from .lif import LIFParams


def run_brian2_trial(W, exc_idx, rate_hz: float, duration_ms: float, params: LIFParams | None = None,
                     seed: int = 0, silenced=None) -> np.ndarray:
    import brian2 as b2
    from brian2 import Hz, mV, ms

    p = params or LIFParams()
    b2.seed(seed)
    b2.defaultclock.dt = p.dt * ms
    ns = {
        "v_0": p.v_0 * mV, "v_rst": p.v_rst * mV, "v_th": p.v_th * mV,
        "t_mbr": p.t_mbr * ms, "tau": p.tau * ms,
    }
    eqs = dedent("""
        dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau               : volt (unless refractory)
        rfc                            : second
        """)
    n = W.shape[0]
    neu = b2.NeuronGroup(n, eqs, method="linear", threshold="v > v_th", reset="v = v_rst; g = 0 * mV",
                         refractory="rfc", namespace=ns, name="neurons")
    neu.v = p.v_0 * mV
    neu.g = 0 * mV
    neu.rfc = p.t_rfc * ms
    coo = W.tocoo()
    syn = b2.Synapses(neu, neu, "w : volt", on_pre="g += w", delay=p.t_dly * ms, name="synapses")
    syn.connect(i=coo.col.astype(np.int64), j=coo.row.astype(np.int64))
    syn.w = coo.data.astype(np.float64) * p.w_syn * mV
    if silenced is not None and len(silenced):
        sil = np.zeros(n, bool)
        sil[np.asarray(silenced)] = True
        syn.w[sil[coo.col]] = 0 * mV
    pois = []
    for i in exc_idx:
        pois.append(b2.PoissonInput(target=neu[int(i):int(i) + 1], target_var="v", N=1, rate=rate_hz * Hz,
                                    weight=p.w_syn * p.f_poi * mV))
        neu[int(i)].rfc = 0 * ms
    mon = b2.SpikeMonitor(neu)
    net = b2.Network(neu, syn, mon, *pois)
    net.run(duration_ms * ms)
    counts = np.bincount(np.asarray(mon.i), minlength=n)
    return counts / (duration_ms * 1e-3)
