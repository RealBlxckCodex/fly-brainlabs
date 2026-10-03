"""Vectorised leaky integrate-and-fire network (numpy), step-compatible with Shiu et al. 2024.

Model (identical equations and constants to ``philshiu/Drosophila_brain_model``):

    dv/dt = (v_0 - v + g) / t_mbr      (unless refractory)
    dg/dt = -g / tau                    (unless refractory)
    spike if v > v_th  ->  v = v_rst, g = 0, refractory for t_rfc
    presynaptic spike  ->  g_post += w_syn * signed_synapse_count   after t_dly
                           (input arriving while the target is refractory is lost, as in Brian2)
    Poisson input      ->  v += w_syn * f_poi  per event

Why not Brian2 directly? The closed loop needs to exchange data with MuJoCo every
millisecond. This engine advances the network one ``dt`` at a time with the same
update order as Brian2 (state update -> threshold -> synaptic delivery / Poisson ->
reset) and uses the exact (``method='linear'``) propagator. Two Brian2 details matter
for quantitative agreement and are reproduced: (1) the refractory flag used by the state
update is the one computed in the previous step's threshold stage; (2) synaptic input
that arrives at a refractory neuron does not persist in g (verified step-by-step against
Brian2 2.9 in tests/test_brian2_equivalence.py). ``validate_m1.py``
compares it against the Brian2 reference implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import scipy.sparse as sp

try:  # optional JIT acceleration (~5x); the numpy path is the reference
    import numba

    @numba.njit(cache=True, nogil=True)
    def _nb_update(v, g, not_refr, lastspike, rfc_steps, silenced, n, Pxx, Pxg, Pgg, v0, vth, spk_buf):
        k = 0
        for i in range(v.shape[0]):
            if not_refr[i]:  # flag from the previous step's threshold stage (Brian2 semantics)
                x = v[i] - v0
                v[i] = v0 + Pxx * x + Pxg * g[i]
                g[i] = Pgg * g[i]
            nr = n - lastspike[i] >= rfc_steps[i]
            not_refr[i] = nr
            if nr and v[i] > vth and not silenced[i]:
                spk_buf[k] = i
                k += 1
        return k

    @numba.njit(cache=True, nogil=True)
    def _nb_deliver(pre, indptr, post, w, g, not_refr):
        for j in pre:
            for e in range(indptr[j], indptr[j + 1]):
                if not_refr[post[e]]:
                    g[post[e]] += w[e]

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False


@dataclass
class LIFParams:
    dt: float = 0.1  # ms
    v_0: float = -52.0  # mV resting potential
    v_rst: float = -52.0  # mV reset
    v_th: float = -45.0  # mV threshold
    t_mbr: float = 20.0  # ms membrane time constant
    tau: float = 5.0  # ms synaptic time constant
    t_rfc: float = 2.2  # ms refractory period
    t_dly: float = 1.8  # ms synaptic delay
    w_syn: float = 0.275  # mV per synapse (free parameter in Shiu et al.)
    f_poi: float = 250.0  # Poisson input weight factor (w_syn * f_poi per event)
    poisson_no_refractory: bool = True  # Shiu et al. set rfc=0 for Poisson-driven neurons

    @classmethod
    def from_dict(cls, d: dict | None) -> "LIFParams":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


class LIFNetwork:
    def __init__(self, W: sp.spmatrix, params: LIFParams | None = None, seed: int = 0,
                 silenced: np.ndarray | None = None, record: np.ndarray | None = None,
                 use_numba: bool | None = None):
        self.p = p = params or LIFParams()
        self.use_numba = HAVE_NUMBA if use_numba is None else (use_numba and HAVE_NUMBA)
        self.n = n = W.shape[0]
        csc = sp.csc_matrix(W, dtype=np.float32)  # column j = outputs of presynaptic neuron j
        self._indptr = csc.indptr.astype(np.int64)
        self._post = csc.indices.astype(np.int32)
        self._w = (csc.data * p.w_syn).astype(np.float32)
        self.rng = np.random.default_rng(seed)

        # exact propagator for x = v - v_0 and g
        a, b = np.exp(-p.dt / p.t_mbr), np.exp(-p.dt / p.tau)
        self._Pxx, self._Pgg = np.float32(a), np.float32(b)
        self._Pxg = np.float32(p.tau / (p.tau - p.t_mbr) * (b - a))

        self.delay_steps = max(1, int(round(p.t_dly / p.dt)))
        self.rfc_steps = np.full(n, int(round(p.t_rfc / p.dt)), dtype=np.int32)
        self.silenced = np.zeros(n, bool)
        if silenced is not None and len(silenced):
            self.silenced[np.asarray(silenced)] = True
        self.record_mask = np.zeros(n, bool)
        if record is not None:
            self.record_mask[np.asarray(record)] = True
        self.reset_state()

    # ----------------------------------------------------------------- state
    def reset_state(self) -> None:
        p = self.p
        self.v = np.full(self.n, p.v_0, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.lastspike = np.full(self.n, -10**9, dtype=np.int64)
        self.not_refr = np.ones(self.n, dtype=bool)
        self.step_count = 0
        self._queue: list[np.ndarray] = [np.zeros(0, np.int64) for _ in range(max(self.delay_steps, 1))]
        self._poi_idx = np.zeros(0, np.int64)
        self._poi_p = np.zeros(0, np.float64)
        self.spike_counts = np.zeros(self.n, dtype=np.int32)
        self.recorded: list[tuple[int, np.ndarray]] = []
        self._spk_buf = np.zeros(self.n, dtype=np.int64)

    @property
    def t_ms(self) -> float:
        return self.step_count * self.p.dt

    # ----------------------------------------------------------------- input
    def set_poisson_rates(self, idx: np.ndarray, rates_hz: np.ndarray | float) -> None:
        """Set the Poisson input rate (Hz) of neurons ``idx`` (replaces previous input)."""
        idx = np.asarray(idx, dtype=np.int64)
        rates = np.broadcast_to(np.asarray(rates_hz, dtype=np.float64), idx.shape)
        keep = (rates > 0) & ~self.silenced[idx]
        self._poi_idx, rates = idx[keep], rates[keep]
        self._poi_p = 1.0 - np.exp(-rates * self.p.dt * 1e-3)  # P(>=1 event in dt); N=1 input
        if self.p.poisson_no_refractory:
            self.rfc_steps[self._poi_idx] = 0

    # ----------------------------------------------------------------- dynamics
    def step(self) -> np.ndarray:
        """Advance one dt. Returns indices of neurons that spiked in this step."""
        p, n = self.p, self.step_count
        v, g = self.v, self.g
        slot = n % len(self._queue)
        pre = self._queue[slot]
        if self.use_numba:
            k = _nb_update(v, g, self.not_refr, self.lastspike, self.rfc_steps, self.silenced, n, self._Pxx, self._Pxg,
                           self._Pgg, np.float32(p.v_0), np.float32(p.v_th), self._spk_buf)
            spk = self._spk_buf[:k].copy()
            if pre.size:
                _nb_deliver(pre, self._indptr, self._post, self._w, g, self.not_refr)
        else:
            # 1) state update (exact), frozen while refractory. As in Brian2, the
            #    refractory flag used here was computed in the previous step's threshold stage.
            x = v - p.v_0
            v_new = p.v_0 + self._Pxx * x + self._Pxg * g
            g_new = self._Pgg * g
            np.copyto(v, v_new, where=self.not_refr)
            np.copyto(g, g_new, where=self.not_refr)
            # 2) threshold (refractory flag updated first)
            self.not_refr = (n - self.lastspike) >= self.rfc_steps
            spk = np.flatnonzero((v > p.v_th) & self.not_refr & ~self.silenced)
        # 3) delayed synaptic delivery of spikes emitted delay_steps ago
        if pre.size and not self.use_numba:
            starts, ends = self._indptr[pre], self._indptr[pre + 1]
            lens = ends - starts
            tot = int(lens.sum())
            if tot:
                offs = np.repeat(ends - np.cumsum(lens), lens) + np.arange(tot)
                dg = np.bincount(self._post[offs], weights=self._w[offs], minlength=self.n).astype(np.float32)
                g += np.where(self.not_refr, dg, np.float32(0))
        #    Poisson input onto v
        if self._poi_idx.size:
            hit = self._poi_idx[self.rng.random(self._poi_idx.size) < self._poi_p]
            if hit.size:
                v[hit] += p.w_syn * p.f_poi
        # 4) reset
        if spk.size:
            v[spk] = p.v_rst
            g[spk] = 0.0
            self.lastspike[spk] = n
            self.not_refr[spk] = False
            self.spike_counts[spk] += 1
            rec = spk[self.record_mask[spk]]
            if rec.size:
                self.recorded.append((n, rec))
        self._queue[slot] = spk  # delivered delay_steps later (delay_steps >= 1)
        self.step_count += 1
        return spk

    def run(self, duration_ms: float) -> None:
        for _ in range(int(round(duration_ms / self.p.dt))):
            self.step()

    # ----------------------------------------------------------------- output
    def rates_hz(self, duration_ms: float | None = None) -> np.ndarray:
        dur = duration_ms if duration_ms is not None else self.t_ms
        return self.spike_counts / max(dur, 1e-9) * 1e3

    def recorded_spikes(self) -> tuple[np.ndarray, np.ndarray]:
        """(times_ms, neuron_idx) of recorded neurons."""
        if not self.recorded:
            return np.zeros(0), np.zeros(0, np.int64)
        t = np.concatenate([np.full(len(i), s * self.p.dt) for s, i in self.recorded])
        i = np.concatenate([i for _, i in self.recorded])
        return t, i
