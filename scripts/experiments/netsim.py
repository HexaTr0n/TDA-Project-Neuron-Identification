"""Small spiking network simulator with known wiring, numpy only.

Leaky integrate-and-fire neurons, each with private noise. A spike of neuron j changes the
voltage of neuron i by W[i, j] millivolts after neuron j's delay. So W is the ground truth.
"""
import numpy as np

DT = 0.0005  # seconds


def simulate(W, seconds, seed=0, mu=14.0, sigma=4.5, tau=0.02, v_th=20.0, v_reset=10.0,
             t_ref=0.002, common=0.0, tau_common=0.02):
    """Returns a list of spike time arrays. common: size in mV of a slow input shared by all."""
    rng = np.random.default_rng(seed)
    n = len(W)
    steps = int(seconds / DT)
    delay = rng.integers(2, 7, n)                 # 1 to 3 ms, one per sending neuron
    buf = np.zeros((8, n))
    v = rng.uniform(v_reset, v_th, n)
    ref = np.zeros(n, dtype=np.int64)
    ref_steps = int(t_ref / DT)
    a, b = DT / tau, sigma * np.sqrt(2 * DT / tau)
    ou, rho = 0.0, np.exp(-DT / tau_common)
    out_t, out_i = [], []
    Wt = np.ascontiguousarray(W.T)
    CH = 20000
    for start in range(0, steps, CH):
        xi = rng.standard_normal((min(CH, steps - start), n))
        zc = rng.standard_normal(len(xi)) * np.sqrt(1 - rho ** 2) if common else None
        for k in range(len(xi)):
            t = start + k
            slot = t % 8
            if common:
                ou = rho * ou + zc[k]
            v += a * (mu + common * ou - v) + b * xi[k] + buf[slot]
            buf[slot] = 0.0
            ref -= 1
            v[ref > 0] = v_reset
            fired = np.flatnonzero(v >= v_th)
            if len(fired):
                v[fired] = v_reset
                ref[fired] = ref_steps
                for j in fired:
                    buf[(t + delay[j]) % 8] += Wt[j]
                out_t.append(np.full(len(fired), t)); out_i.append(fired)
    t = np.concatenate(out_t) * DT; i = np.concatenate(out_i)
    return [t[i == k] for k in range(n)]


def random_ei(n_exc=80, n_inh=20, p=0.1, w=0.8, g=4.0, seed=0, p_ee=None):
    """The textbook cortical model: 80% excitatory, 20% inhibitory, sparse random wiring.
    p_ee: optional (n_exc, n_exc) matrix of connection probabilities between excitatory cells."""
    rng = np.random.default_rng(seed)
    n = n_exc + n_inh
    P = np.full((n, n), p)
    if p_ee is not None:
        P[:n_exc, :n_exc] = p_ee
    W = (rng.random((n, n)) < P).astype(float)
    np.fill_diagonal(W, 0)
    W[:, :n_exc] *= w
    W[:, n_exc:] *= -g * w
    return W


def correlograms(trains, seconds, max_lag=0.02):
    """C[a, b, k]: number of spikes of b at lag (k - L) * DT after a spike of a."""
    from scipy.sparse import csr_matrix
    n, nb, L = len(trains), int(seconds / DT) + 1, int(max_lag / DT)
    rows = np.concatenate([np.full(len(t), i) for i, t in enumerate(trains)])
    cols = np.concatenate([(t / DT).astype(int) for t in trains])
    S = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, nb))
    C = np.zeros((n, n, 2 * L + 1), dtype=np.float32)
    for l in range(L + 1):
        c = (S[:, :nb - l] @ S[:, l:].T).toarray()
        C[:, :, L + l] = c
        C[:, :, L - l] = c.T
    return C
