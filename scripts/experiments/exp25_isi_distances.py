"""W3: can the stimulus-free ISI signature be sharpened? Retina, 60 neurons, frozen trials.

Run from the repo root: python scripts/experiments/exp25_isi_distances.py
Same ISI fractions curve under other distances, quantile curves, and a two-parameter version:
the share of consecutive ISI pairs with ISI_k > a and ISI_(k+1) > b.
Scores: 1-NN, and k-means with mean curves (mean of 3 seeds, +-2.5 points seed noise).
"""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import run_pipeline as rp
from load_dataset import load_dataset

csv = REPO / "data" / "isi_frozennoise.csv"
df = load_dataset(csv if csv.exists() else rp.DEFAULT_PATH, sessions=["20180126_sr_le_sp"])
blocks = rp.pick_neurons(rp.make_blocks(df, rp.Config()), 60, 0)
y = np.unique([b[0] for b in blocks], return_inverse=True)[1]; k = y.max() + 1
logs = [np.log(np.maximum(np.diff(np.asarray(b[1][0].spike_times, float)), 1e-4)) for b in blocks]
rng = np.random.default_rng(0)
LG = np.linspace(np.log(1e-4), np.log(10), 230, endpoint=False)
G2 = np.linspace(np.log(1e-3), np.log(3), 24)
Q = (np.arange(64) + .5) / 64


def k_means(X, metric, seed, restarts=3, iters=25):
    r = np.random.default_rng(seed); best, cost = None, np.inf
    for _ in range(restarts):
        C = X[r.choice(len(X), k, replace=False)].copy()
        for _ in range(iters):
            a = cdist(X, C, metric).argmin(1)
            new = np.stack([X[a == c].mean(0) if (a == c).any() else X[r.integers(len(X))] for c in range(k)])
            if np.allclose(new, C):
                break
            C = new
        D = cdist(X, C, metric)
        if D.min(1).sum() < cost:
            best, cost = D.argmin(1), D.min(1).sum()
    return best


def report(name, X, metric="cityblock"):
    D = cdist(X, X, metric); np.fill_diagonal(D, np.inf)
    nn = np.mean(y[D.argmin(1)] == y)
    km = [rp.matched_accuracy(y, k_means(X, metric, s)) for s in range(3)]
    print(f"  {name:58s} 1-NN {nn:.3f}   k-means {np.mean(km):.3f} ({min(km):.3f}-{max(km):.3f})", flush=True)


frac = lambda l: (l[None, :] > LG[:, None]).mean(1)
joint = lambda l: ((l[:-1][None, None, :] > G2[:, None, None]) & (l[1:][None, None, :] > G2[None, :, None])).mean(2).ravel()
F = np.stack([frac(l) for l in logs]).astype(np.float32)
print("ISI fractions under other distances")
report("L1 (reference)", F); report("L2", F, "euclidean"); report("largest gap (sup norm)", F, "chebyshev")
report("square root of the curve, L2", np.sqrt(F), "euclidean")
print("quantile curves (64 quantiles of log ISI)")
Qm = np.stack([np.quantile(l, Q) for l in logs]).astype(np.float32)
report("L1 (same as L1 on fractions, coarser)", Qm); report("L2 (Wasserstein-2)", Qm, "euclidean")
report("L2, centred on the trial's mean log ISI (shape only)", Qm - Qm.mean(1, keepdims=True), "euclidean")
print("two-parameter: consecutive ISI pairs")
J = np.stack([joint(l) for l in logs]).astype(np.float32)
Js = np.stack([joint(rng.permutation(l)) for l in logs]).astype(np.float32)
report("joint fractions", J); report("joint fractions, ISIs shuffled", Js)
# what order adds: joint minus the product of its margins
m = lambda l: (l[None, :] > G2[:, None]).mean(1)
E = np.stack([joint(l) - np.outer(m(l[:-1]), m(l[1:])).ravel() for l in logs]).astype(np.float32)
Es = np.stack([joint(p) - np.outer(m(p[:-1]), m(p[1:])).ravel() for p in (rng.permutation(l) for l in logs)]).astype(np.float32)
report("joint minus product of margins (order only)", E); report("same, ISIs shuffled", Es)
sc = lambda X: X / np.median(cdist(X[::7], X[::7], "cityblock"))
report("ISI fractions + order-only part", np.hstack([sc(F), .5 * sc(E)]))
