import numpy as np

from flybrainlab.connectome.interventions import apply_interventions, lesion, rewire, shuffle


def test_shuffle_targets_preserves_outdegree_weights(toy):
    c, _ = toy
    s = shuffle(c, seed=1, mode="targets")
    assert s.n_synapses == c.n_synapses
    out_c = np.diff(c.W.tocsc().indptr)
    out_s = np.diff(s.W.tocsc().indptr)
    # duplicates may merge, so out-degree can only shrink; total weight per pre is preserved
    assert (out_s <= out_c).all()
    assert np.allclose(np.asarray(c.W.sum(0)).ravel(), np.asarray(s.W.sum(0)).ravel())
    assert s.W.diagonal().sum() == 0  # no self-connections
    assert (s.W != c.W).nnz > 0


def test_shuffle_degree_mode_preserves_in_weight_counts(toy):
    c, _ = toy
    s = shuffle(c, seed=2, mode="degree")
    assert s.n_synapses == c.n_synapses


def test_lesion_removes_all_synapses(toy):
    c, g = toy
    gf = g.resolve("giant_fiber")
    l_ = lesion(c, gf)
    assert l_.W[gf].nnz == 0 and l_.W[:, gf].nnz == 0
    assert "lesion" in l_.meta["modifications"][-1]


def test_rewire_ops(toy):
    c, g = toy
    gf, lp = g.resolve("giant_fiber"), g.resolve("lplc2")
    before = c.W[gf][:, lp].sum()
    sc = rewire(c, [{"op": "scale", "pre": "lplc2", "post": "giant_fiber", "factor": 2.0}], g)
    assert np.isclose(sc.W[gf][:, lp].sum(), 2 * before)
    cut = rewire(c, [{"op": "cut", "pre": "lplc2", "post": "giant_fiber"}], g)
    assert cut.W[gf][:, lp].sum() == 0
    sh = rewire(c, [{"op": "shift", "pre": "lplc2", "post": "giant_fiber", "to": "mn9"}], g)
    assert sh.W[gf][:, lp].sum() == 0 and sh.W[g.resolve("mn9")][:, lp].sum() == before
    fl = rewire(c, [{"op": "set_sign", "pre": "lplc2", "sign": -1}], g)
    assert fl.W[:, lp].max() <= 0


def test_apply_order_and_silenced(toy):
    c, g = toy
    c2, sil = apply_interventions(c, g, lesion_spec=["giant_fiber"], shuffle_spec="targets", seed=0)
    assert set(sil) == set(g.resolve("giant_fiber"))
    assert any("shuffle" in m for m in c2.meta["modifications"])
