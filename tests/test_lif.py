import numpy as np
import scipy.sparse as sp

from flybrainlab.brain.lif import HAVE_NUMBA, LIFNetwork, LIFParams


def test_exact_propagator_matches_ode():
    """Free decay of (v, g) after a kick equals the analytic solution."""
    p = LIFParams()
    net = LIFNetwork(sp.csr_matrix((1, 1), dtype=np.float32), p, use_numba=False)
    net.v[0], net.g[0] = p.v_0, 5.0
    for _ in range(100):  # 10 ms
        net.step()
    t = 10.0
    A = 5.0 * p.tau / (p.tau - p.t_mbr)
    x = -A * np.exp(-t / p.t_mbr) + A * np.exp(-t / p.tau)
    assert abs((net.v[0] - p.v_0) - x) < 1e-4
    assert abs(net.g[0] - 5.0 * np.exp(-t / p.tau)) < 1e-4


def test_synaptic_delay_and_sign():
    p = LIFParams()
    W = sp.csr_matrix(([10.0, -10.0], ([1, 2], [0, 0])), shape=(3, 3))
    net = LIFNetwork(W, p, use_numba=False)
    net.v[0] = -40.0  # spikes in the first step
    g1 = []
    for _ in range(25):
        net.step()
        g1.append((net.g[1], net.g[2]))
    d = net.delay_steps
    assert g1[d - 1][0] == 0.0 and g1[d][0] > 0  # arrives exactly t_dly later
    assert g1[d][1] < 0  # inhibitory


def test_silenced_neuron_never_spikes():
    W = sp.csr_matrix((2, 2), dtype=np.float32)
    net = LIFNetwork(W, LIFParams(), silenced=np.array([0]), use_numba=False)
    net.set_poisson_rates(np.array([0, 1]), 500.0)
    net.run(200)
    assert net.spike_counts[0] == 0 and net.spike_counts[1] > 50


def test_numba_and_numpy_identical(toy):
    if not HAVE_NUMBA:
        return
    c, g = toy
    rates = []
    for nb in (True, False):
        net = LIFNetwork(c.W, LIFParams(), seed=3, use_numba=nb)
        net.set_poisson_rates(g.resolve("lplc2"), 120.0)
        net.run(300)
        rates.append(net.spike_counts.copy())
    assert np.array_equal(rates[0], rates[1])


def test_pathway_activation(toy):
    c, g = toy
    net = LIFNetwork(c.W, LIFParams(), seed=0)
    net.set_poisson_rates(g.resolve("sugar_grn"), 150.0)
    net.run(500)
    r = net.rates_hz()
    assert r[g.resolve("mn9")].mean() > 20 and r[g.resolve("giant_fiber")].sum() == 0
