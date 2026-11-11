"""Pilots on simulated networks with known wiring. Run: python exp19_connectivity_pilots.py [A B C D]

A  Is neuron a connected to neuron b? Persistence of the cross-correlogram peak.
B  Which neurons form groups? H0 (clustering) of the correlation distance.
C  Which group drives which, and is there a loop? Lead-lag flow between groups.
D  Does the wiring live on a ring? H1 of the correlation distance.
"""
import sys
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, roc_auc_score

import netsim as ns
from tda import rips_matrix

which = sys.argv[1:] or list("ABCD")
L = int(0.02 / ns.DT)                                  # correlogram half-width in bins
WIN = slice(L + 2, L + 8)                              # lags 1 to 3.5 ms after the sender


# ------------------------------------------------------------------ A: single connections
def prominence_in_window(c):
    """Largest prominence of a peak of c inside the window: its H0 persistence."""
    peaks = find_peaks(c)[0]
    peaks = peaks[(peaks >= WIN.start) & (peaks < WIN.stop)]
    return peak_prominences(c, peaks)[0].max() if len(peaks) else 0.0


def detectors(C):
    n = len(C)
    out = {k: np.zeros((n, n)) for k in ("height", "baseline", "persistence")}
    dip = {k: np.zeros((n, n)) for k in out}
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            c = C[a, b].astype(float)
            noise = np.sqrt(max(c.mean(), 1e-9))
            slow = gaussian_filter1d(c, 10, mode="nearest")             # 5 ms
            out["height"][a, b] = (c[WIN].max() - c.mean()) / noise
            out["baseline"][a, b] = ((c - slow)[WIN]).max() / noise
            out["persistence"][a, b] = prominence_in_window(c) / noise
            dip["height"][a, b] = (c.mean() - c[WIN].min()) / noise
            dip["baseline"][a, b] = ((slow - c)[WIN]).max() / noise
            dip["persistence"][a, b] = prominence_in_window(-c) / noise
    return out, dip


def pilot_a():
    print("=== A. single connections, 80 excitatory + 20 inhibitory neurons, 10% wiring")
    W = ns.random_ei(seed=1)
    truth = W.T                                          # truth[a, b]: from a to b
    off = ~np.eye(len(W), dtype=bool)
    for common in (0.0, 3.0):
        for seconds in (60, 300, 1200):
            trains = ns.simulate(W, seconds, seed=2, common=common)
            pos, neg = detectors(ns.correlograms(trains, seconds))
            none = (truth == 0) & off
            row = []
            for k in pos:
                e = roc_auc_score(np.r_[np.ones((truth > 0).sum()), np.zeros(none.sum())],
                                  np.r_[pos[k][truth > 0], pos[k][none]])
                i = roc_auc_score(np.r_[np.ones((truth < 0).sum()), np.zeros(none.sum())],
                                  np.r_[neg[k][truth < 0], neg[k][none]])
                # share of true excitatory connections found when 1% of non-connections are flagged
                cut = np.quantile(pos[k][none], 0.99)
                row.append(f"{k}: exc {e:.3f} inh {i:.3f} found@1% {np.mean(pos[k][truth > 0] > cut):.2f}")
            print(f"  shared input {common:g} mV, {seconds:4d} s | " + " | ".join(row), flush=True)


# ------------------------------------------------------------------ groups and flow
def grouped(n_groups=4, size=24, n_inh=24, p_in=0.25, p_out=0.02, forward=0.0, loop=False, seed=0):
    """Excitatory neurons in groups. forward: extra wiring from group k to group k+1."""
    n_exc = n_groups * size
    g = np.repeat(np.arange(n_groups), size)
    P = np.where(g[:, None] == g[None, :], p_in, p_out)   # P[to, from]
    if forward:
        P[(g[:, None] - g[None, :]) == 1] = forward
        if loop:
            P[(g[:, None] == 0) & (g[None, :] == n_groups - 1)] = forward
    return ns.random_ei(n_exc, n_inh, seed=seed, p_ee=P), g


def corr_distance(trains, seconds, bin_s=0.005, n=None):
    edges = np.arange(0, seconds + bin_s, bin_s)
    X = gaussian_filter1d(np.stack([np.histogram(t, edges)[0] for t in trains[:n]]).astype(float), 2, axis=1)
    D = 1 - np.corrcoef(X)
    np.fill_diagonal(D, 0)
    return (D + D.T) / 2


def pilot_b():
    print("=== B. groups: 4 groups of 24 excitatory neurons (25% wiring inside, 2% between), 24 inhibitory")
    W, g = grouped(seed=3)
    for seconds in (30, 120, 600):
        trains = ns.simulate(W, seconds, seed=4)
        D = corr_distance(trains, seconds, n=len(g))
        Z = linkage(squareform(D, checks=False), "average")
        ari = adjusted_rand_score(g, fcluster(Z, 4, "maxclust"))
        h = Z[:, 2][::-1]                                 # merge heights, last merges first
        gaps = h[:8] / h[1:9]
        print(f"  {seconds:4d} s | groups recovered (ARI, 1 = perfect) {ari:.2f} | "
              f"number of groups suggested by the largest jump in merge height: {int(np.argmax(gaps)) + 2}", flush=True)
    return


def lead(C):
    """Y[a, b] > 0 when b tends to fire in the 5 ms after a more than in the 5 ms before."""
    after = C[:, :, L + 1:L + 11].sum(2); before = C[:, :, L - 10:L].sum(2)
    return (after - before) / np.sqrt(np.maximum(after + before, 1))


def group_flow(Y, labels):
    """F[a, b] > 0: neurons of group b tend to fire just after neurons of group a."""
    k = labels.max() + 1
    F = np.array([[Y[np.ix_(labels == a, labels == b)].mean() for b in range(k)] for a in range(k)])
    return (F - F.T) / 2


def has_loop(edges, k):
    """Is there a directed cycle? Remove groups with no incoming edge until none are left."""
    left = set(range(k)); edges = set(edges)
    while True:
        free = [v for v in left if not any(b == v and a in left for a, b in edges)]
        if not free:
            return bool(left)
        left -= set(free)


def pilot_c():
    print("=== C. which group drives which: chain 0>1>2>3 against loop 0>1>2>3>0")
    for seconds in (120, 600):
        for loop in (False, True):
            W, g = grouped(forward=0.10, loop=loop, seed=5)
            trains = ns.simulate(W, seconds, seed=6)
            Y = lead(ns.correlograms(trains, seconds)[:len(g), :len(g)])
            rng = np.random.default_rng(0)
            shuf = [np.sort((t + rng.uniform(0, seconds)) % seconds) for t in trains]
            Ys = lead(ns.correlograms(shuf, seconds)[:len(g), :len(g)])
            D = corr_distance(trains, seconds, n=len(g))
            found = fcluster(linkage(squareform(D, checks=False), "average"), 4, "maxclust") - 1
            truth = [(a, a + 1) for a in range(3)] + ([(3, 0)] if loop else [])
            print(f"  {'loop ' if loop else 'chain'}, {seconds} s. True edges {truth}. Groups recovered without labels: ARI {adjusted_rand_score(g, found):.2f}")
            for name, lab in (("true groups", g), ("recovered groups", found)):
                F = group_flow(Y, lab); k = lab.max() + 1
                cut = 5 * np.abs(group_flow(Ys, lab)).max()        # 5 times the largest flow in the control
                edges = [(a, b) for a in range(k) for b in range(k) if F[a, b] > cut]
                print(f"      {name:16s}: edges found {edges} | loop found: {has_loop(edges, k)}")
            if not loop:
                print(f"      per-neuron order score against true group: Spearman {spearmanr(Y.mean(0), g)[0]:.2f}", flush=True)


def pilot_d():
    print("=== D. wiring on a ring, a line, or at random: 100 excitatory + 20 inhibitory, 300 s")
    n = 100
    x = np.arange(n)
    ring = np.minimum(np.abs(x[:, None] - x[None, :]), n - np.abs(x[:, None] - x[None, :]))
    line = np.abs(x[:, None] - x[None, :])
    P_ring = 0.5 * np.exp(-ring ** 2 / (2 * 6.0 ** 2))
    P_line = 0.5 * np.exp(-line ** 2 / (2 * 6.0 ** 2))
    P_rand = np.full((n, n), P_ring.mean())
    for name, P in (("ring", P_ring), ("line", P_line), ("random", P_rand)):
        res = []
        for seed in range(3):
            W = ns.random_ei(n, 20, seed=10 + seed, p_ee=P)
            trains = ns.simulate(W, 300, seed=20 + seed)
            D = corr_distance(trains, 300, n=n)
            _, b, d = rips_matrix(D)
            life = np.sort(d - b)[::-1] / D.max()
            res.append((life[0], life[1], np.mean([len(t) for t in trains]) / 300))
        r = np.array(res)
        print(f"  {name:6s} | longest H1 bar {np.round(r[:, 0], 3)} | second longest {np.round(r[:, 1], 3)} | rate {r[:, 2].mean():.1f} Hz", flush=True)


for k in which:
    {"A": pilot_a, "B": pilot_b, "C": pilot_c, "D": pilot_d}[k]()
