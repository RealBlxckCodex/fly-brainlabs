"""Step-by-step equivalence with the Brian2 reference model (Shiu et al. semantics)."""
from textwrap import dedent

import numpy as np
import pytest
import scipy.sparse as sp

from flybrainlab.brain import LIFNetwork, LIFParams

b2 = pytest.importorskip("brian2")


def test_replayed_input_train_matches_brian2():
    from brian2 import mV, ms

    p = LIFParams()
    rng = np.random.default_rng(0)
    steps = np.sort(rng.choice(np.arange(5, 3000), size=60, replace=False))
    b2.start_scope()
    b2.defaultclock.dt = p.dt * ms
    ns = {"v_0": p.v_0 * mV, "v_rst": p.v_rst * mV, "v_th": p.v_th * mV, "t_mbr": p.t_mbr * ms, "tau": p.tau * ms}
    eqs = dedent("""
        dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau : volt (unless refractory)
        rfc : second""")
    neu = b2.NeuronGroup(3, eqs, method="linear", threshold="v > v_th", reset="v = v_rst; g = 0 * mV",
                         refractory="rfc", namespace=ns)
    neu.v = p.v_0 * mV
    neu.rfc = p.t_rfc * ms
    neu.rfc[0] = 0 * ms
    gen = b2.SpikeGeneratorGroup(1, np.zeros(len(steps), int), (steps - 1) * p.dt * ms)
    kick = b2.Synapses(gen, neu, on_pre="v_post += 68.75*mV")
    kick.connect(i=0, j=0)
    W = sp.csr_matrix(([200.0, 30.0], ([1, 2], [0, 0])), shape=(3, 3))
    coo = W.tocoo()
    syn = b2.Synapses(neu, neu, "w : volt", on_pre="g += w", delay=p.t_dly * ms)
    syn.connect(i=coo.col, j=coo.row)
    syn.w = coo.data * p.w_syn * mV
    sm = b2.SpikeMonitor(neu)
    b2.Network(neu, syn, gen, kick, sm).run(300 * ms)
    brian = sorted((int(round(float(t / ms) / p.dt)), int(i)) for t, i in zip(sm.t, sm.i))

    net = LIFNetwork(W, p, use_numba=False)
    net.rfc_steps[0] = 0
    kicks, ours = set(steps - 1), []
    for k in range(3000):
        ours += [(k, int(i)) for i in net.step()]
        if k in kicks:
            net.v[0] += 68.75
    assert ours == brian
