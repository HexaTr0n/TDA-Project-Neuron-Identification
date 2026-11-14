"""W14: a mixture of floor(sqrt(w)) Gaussians fitted to the LAST w ISIs, one pdf per spike (user's idea).

Run from the repo root: python scripts/pilots/window_mixture.py
Retina, 60 neurons, frozen trials; numpy and scipy only (no stablebear).
The H1 readout needs the small Rips library next to this file; build it once with
    gcc -O3 -march=native -fopenmp -shared -fPIC scripts/pilots/rips.c -o scripts/pilots/librips.so
For every spike with at least w ISIs behind it: fit a mixture of K = floor(sqrt(w)) Gaussians to
the last w log ISIs (own EM, all windows at once, 40 iterations, sd at least 0.1; means start at
the window's quantiles, so nothing is random). The point of that spike is the fitted pdf as its
survival curve, so L1 between points is Wasserstein-1 between fitted pdfs. A trial becomes a
path of pdfs that follows the local ISI distribution. Control: ISIs shuffled before fitting.
Readouts: mean point, cloud without homology, path at fixed times, Rips H0 (whole cloud, and in
8 regions), Rips H1 on 40 landmarks (farthest-point sampling). Scores: 1-NN and k-medoids.
"""
import sys
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import norm
from sklearn.cluster import KMeans
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pipeline as rp
import tda
from load_dataset import load_dataset

LG = np.linspace(np.log(1e-4), np.log(10), 230, endpoint=False)
GRID = LG[::5]; STEP = GRID[1] - GRID[0]
RADII = np.concatenate([[0], np.geomspace(0.002, 3, 39)])
LAND = 40


def fit_windows(X, K, iters=40, chunk=20000):
    """X: (B, w) windows of log ISIs. Returns survival curves (B, G) of the fitted mixtures."""
    out = np.zeros((len(X), len(GRID)), np.float32)
    q = (np.arange(K) + .5) / K
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk]
        mu = np.quantile(x, q, axis=1).T
        sd = np.maximum(x.std(1, keepdims=True) / np.sqrt(K), 0.1) * np.ones((1, K)); wt = np.full((len(x), K), 1.0 / K)
        for _ in range(iters):
            lp = -0.5 * ((x[:, :, None] - mu[:, None, :]) / sd[:, None, :]) ** 2 - np.log(sd[:, None, :]) + np.log(wt[:, None, :] + 1e-300)
            r = np.exp(lp - lp.max(2, keepdims=True)); r /= r.sum(2, keepdims=True)
            n = r.sum(1) + 1e-9
            mu = (r * x[:, :, None]).sum(1) / n
            sd = np.maximum(np.sqrt((r * (x[:, :, None] - mu[:, None, :]) ** 2).sum(1) / n), 0.1)
            wt = n / n.sum(1, keepdims=True)
        out[s:s + chunk] = (wt[:, :, None] * norm.sf(GRID[None, None, :], mu[:, :, None], sd[:, :, None])).sum(1)
    return out


def h0_curve(P):
    if len(P) < 2:
        return np.zeros(len(RADII))
    e = minimum_spanning_tree(cdist(P, P, "cityblock") * STEP + 1e-9).data
    return (e[None, :] > RADII[:, None]).sum(1).astype(float)


def landmarks(P):
    """Farthest-point sampling under L1, starting from the first point."""
    if len(P) <= LAND:
        return P
    idx = [0]; d = np.abs(P - P[0]).sum(1)
    for _ in range(LAND - 1):
        j = int(d.argmax()); idx.append(j); d = np.minimum(d, np.abs(P - P[j]).sum(1))
    return P[idx]


def h1_curve(P):
    L = landmarks(P)
    if len(L) < 4:
        return np.zeros(len(RADII))
    _, b, d = tda.rips_matrix(cdist(L, L, "cityblock") * STEP)
    return ((d - b)[None, :] > RADII[:, None]).sum(1).astype(float)


def signatures(logs, ends, marks, w, anchors=None, rng=None):
    """ends[i][j]: time at which ISI j of sample i ends, from the sample's start."""
    K = int(np.sqrt(w)); sizes = [max(len(l) - w + 1, 0) for l in logs]
    X = np.concatenate([sliding_window_view(l, w) for l in logs if len(l) >= w])
    C = fit_windows(X, K); off = np.r_[0, np.cumsum(sizes)]
    whole = fit_windows(np.stack([np.resize(l, w) if len(l) else np.zeros(w) for l in logs]), K)   # for samples shorter than w
    clouds = [C[off[i]:off[i + 1]] if sizes[i] else whole[i:i + 1] for i in range(len(logs))]
    out = {}
    out["mean point of the path"] = np.stack([c.mean(0) for c in clouds]) * STEP
    out["cloud, no homology: quantiles over points"] = np.stack([np.quantile(c, [.1, .3, .5, .7, .9], axis=0).ravel() for c in clouds]) * STEP / 5
    rows = []
    for c, e, n in zip(clouds, ends, sizes):
        t = e[w - 1:] if n else np.array([0.0])
        rows.append(np.concatenate([c[min(max(np.searchsorted(t, m, "right") - 1, 0), len(c) - 1)] for m in marks]))
    out["path read at fixed times"] = np.stack(rows) * STEP / len(marks)
    out["spread of the path (sd per grid point)"] = np.stack([c.std(0) for c in clouds]) * STEP
    H = np.stack([h0_curve(c) for c in clouds])
    out["Rips H0 of the path, whole cloud"] = H
    out["Rips H0, divided by point count"] = H / np.array([len(c) for c in clouds])[:, None]
    out["Rips H1, 40 landmarks"] = np.stack([h1_curve(c) for c in clouds])
    if anchors is None:
        pool = np.concatenate(clouds); pool = pool[rng.choice(len(pool), min(20000, len(pool)), replace=False)]
        anchors = KMeans(8, n_init=3, random_state=0).fit(pool).cluster_centers_
    reg, cnt = [], []
    for c in clouds:
        lab = cdist(c, anchors).argmin(1)
        reg.append(np.concatenate([h0_curve(c[lab == j]) + (lab == j).sum() * (RADII == 0) for j in range(8)]))
        cnt.append(np.bincount(lab, minlength=8).astype(float))
    out["Rips H0 in 8 regions (whole cloud)"] = np.stack(reg)
    out["region counts (twin without bars)"] = np.stack(cnt)
    return out, anchors, np.median(sizes)


if __name__ == "__main__":
    csv = REPO / "data" / "isi_frozennoise.csv"
    df = load_dataset(csv if csv.exists() else rp.DEFAULT_PATH, sessions=["20180126_sr_le_sp"])
    blocks = rp.pick_neurons(rp.make_blocks(df, rp.Config()), 60, 0)
    y = np.unique([b[0] for b in blocks], return_inverse=True)[1]; k = y.max() + 1
    times = [np.asarray(b[1][0].spike_times, float) for b in blocks]
    HALF = 9.94 / 2; rng = np.random.default_rng(0)
    cut = {"full": lambda t: t, "A": lambda t: t[t < HALF], "B": lambda t: t[t >= HALF] - HALF}
    lg = lambda t: np.log(np.maximum(np.diff(t), 1e-4))
    MARKS = {"full": [2.485, 4.97, 7.455, 9.94], "A": [1.2425, 2.485, 3.7275, 4.97], "B": [1.2425, 2.485, 3.7275, 4.97]}

    def make(name, shuffle):
        logs, ends = [], []
        for t in times:
            s = cut[name](t); l = lg(s)
            if shuffle and len(l):
                l = rng.permutation(l); s = s[0] + np.concatenate([[0], np.cumsum(np.exp(l))])
            logs.append(l); ends.append(s[1:] if len(s) > 1 else np.zeros(0))
        return logs, ends

    def nn(X):
        D = cdist(X, X, "cityblock"); np.fill_diagonal(D, np.inf)
        return 100 * np.mean(y[D.argmin(1)] == y)
    def kmed(X):
        D = cdist(X, X, "cityblock")
        return np.mean([100 * rp.matched_accuracy(y, rp.k_medoids(D, k, seed=s)) for s in range(3)])
    def across(XA, XB):
        D = cdist(XA, XB, "cityblock"); np.fill_diagonal(D, np.inf)
        return 100 * np.mean(y[D.argmin(1)] == y)

    F = {s: np.stack([(l[None, :] > LG[:, None]).mean(1) if len(l) else np.zeros(len(LG)) for l in make(s, False)[0]]) for s in cut}
    print(f"{len(times)} trials, {k} neurons; chance 1.7%")
    print(f"reference: ISI fractions   full 1-NN {nn(F['full']):.1f}  k-medoids {kmed(F['full']):.1f}   halves A->B {across(F['A'], F['B']):.1f}", flush=True)
    sc = lambda X: X / max(np.median(cdist(X[::7], X[::7], "cityblock")), 1e-12)
    for w in (16, 32, 64):
        real, anc, med = signatures(*make("full", False), MARKS["full"], w, rng=rng)
        shuf, _, _ = signatures(*make("full", True), MARKS["full"], w, anc)
        halves = w <= 32
        if halves:
            ra, _, mh = signatures(*make("A", False), MARKS["A"], w, anc); rb, _, _ = signatures(*make("B", False), MARKS["B"], w, anc)
            sa, _, _ = signatures(*make("A", True), MARKS["A"], w, anc); sb_, _, _ = signatures(*make("B", True), MARKS["B"], w, anc)
        print(f"\n=== last {w} ISIs, {int(np.sqrt(w))} Gaussians; median points per trial {int(med)}" + (f", per half {int(mh)}" if halves else "; halves too short"))
        print(f"{'signature':50s} | full trial 1-NN (shuffled)  k-medoids (shuffled) | halves A->B 1-NN (shuffled)")
        hv = lambda a, b, c, d: f"{across(a, b):5.1f} ({across(c, d):5.1f})" if halves else "    -"
        for name in real:
            print(f"{name:50s} | {nn(real[name]):5.1f} ({nn(shuf[name]):5.1f})   {kmed(real[name]):5.1f} ({kmed(shuf[name]):5.1f}) | "
                  + (hv(ra[name], rb[name], sa[name], sb_[name]) if halves else "    -"), flush=True)
        for name in ("path read at fixed times", "Rips H0 in 8 regions (whole cloud)", "Rips H0 of the path, whole cloud"):
            j = lambda f, d: np.hstack([sc(f), sc(d[name])])
            print(f"{'ISI fractions + ' + name[:34]:50s} | {nn(j(F['full'], real)):5.1f} ({nn(j(F['full'], shuf)):5.1f})   "
                  f"{kmed(j(F['full'], real)):5.1f} ({kmed(j(F['full'], shuf)):5.1f}) | "
                  + (hv(j(F['A'], ra), j(F['B'], rb), j(F['A'], sa), j(F['B'], sb_)) if halves else "    -"), flush=True)
