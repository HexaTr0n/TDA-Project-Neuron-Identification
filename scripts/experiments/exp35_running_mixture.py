"""W13: a mixture of floor(sqrt(n)) Gaussians fitted to the first n ISIs, one pdf per spike (user's idea).

Run from the repo root: python scripts/experiments/exp35_running_mixture.py
Retina, 60 neurons, frozen trials; numpy, scipy, scikit-learn only (no stablebear).
For every 4th spike from n = 8 on: fit a mixture of floor(sqrt(n)) Gaussians to the log ISIs
seen so far (EM, sd at least 0.1). The point of that spike is the fitted pdf, stored as its
survival curve, so L1 between points is Wasserstein-1 between fitted pdfs. A trial becomes a
path of pdfs. Control: ISIs shuffled before fitting (same distribution, no order, no drift).
"""
import sys, warnings
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import run_pipeline as rp
from load_dataset import load_dataset
warnings.filterwarnings("ignore")

LG = np.linspace(np.log(1e-4), np.log(10), 230, endpoint=False)
GRID = LG[::5]; STEP = GRID[1] - GRID[0]
RADII = np.concatenate([[0], np.geomspace(0.002, 3, 39)])
EVERY, FIRST = 4, 8


def path(l, seed):
    """Fitted survival curves for the prefixes n = 8, 12, ... and the full train; and the n values."""
    warnings.filterwarnings("ignore")
    ns = sorted(set(list(range(FIRST, len(l) + 1, EVERY)) + ([len(l)] if len(l) >= 2 else [])))
    out = []
    for n in ns:
        K = max(1, min(int(np.sqrt(n)), n // 2))
        g = GaussianMixture(K, reg_covar=1e-2, n_init=1, max_iter=60, random_state=seed).fit(l[:n, None])
        out.append((g.weights_ * norm.sf(GRID[:, None], g.means_[:, 0], np.sqrt(g.covariances_[:, 0, 0]))).sum(1))
    if not out:
        return np.zeros((1, len(GRID)), np.float32), np.array([0])
    return np.array(out, np.float32), np.array(ns)


def h0_curve(P):
    if len(P) < 2:
        return np.zeros(len(RADII))
    e = minimum_spanning_tree(cdist(P, P, "cityblock") * STEP + 1e-9).data
    return (e[None, :] > RADII[:, None]).sum(1).astype(float)


def signatures(logs, ends, marks, anchors=None, rng=None):
    """ends[i][j]: time at which ISI j of sample i ends, from the sample's start. marks: times at which the path is read."""
    res = Parallel(n_jobs=2)(delayed(path)(l, 0) for l in logs)
    clouds = [r[0] for r in res]; out = {}
    out["last point: fit to the whole sample"] = np.stack([c[-1] for c in clouds]) * STEP
    out["mean point of the path"] = np.stack([c.mean(0) for c in clouds]) * STEP
    out["cloud, no homology: quantiles over points"] = np.stack([np.quantile(c, [.1, .3, .5, .7, .9], axis=0).ravel() for c in clouds]) * STEP / 5
    rows = []
    for (c, ns), e in zip(res, ends):
        t = e[ns - 1] if len(e) and ns[0] > 0 else np.array([0.0])
        rows.append(np.concatenate([c[max(np.searchsorted(t, m, "right") - 1, 0)] for m in marks]))
    out["path read at fixed times"] = np.stack(rows) * STEP / len(marks)
    out["path with its end point subtracted (drift only)"] = np.stack([r.reshape(len(marks), -1)[:-1].ravel() - np.tile(r.reshape(len(marks), -1)[-1], len(marks) - 1) for r in rows]) * STEP
    H = np.stack([h0_curve(c) for c in clouds])
    out["Rips H0 of the path, whole cloud"] = H
    out["Rips H0, divided by point count"] = H / np.array([len(c) for c in clouds])[:, None]
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
    return out, anchors


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

    def k_means(X, seed, restarts=3, iters=25):
        r = np.random.default_rng(seed); best, cost = None, np.inf
        for _ in range(restarts):
            C = X[r.choice(len(X), k, replace=False)].copy()
            for _ in range(iters):
                a = cdist(X, C, "cityblock").argmin(1)
                new = np.stack([X[a == c].mean(0) if (a == c).any() else X[r.integers(len(X))] for c in range(k)])
                if np.allclose(new, C):
                    break
                C = new
            D = cdist(X, C, "cityblock")
            if D.min(1).sum() < cost:
                best, cost = D.argmin(1), D.min(1).sum()
        return best

    def nn(X):
        D = cdist(X, X, "cityblock"); np.fill_diagonal(D, np.inf)
        return 100 * np.mean(y[D.argmin(1)] == y)
    km = lambda X: np.mean([100 * rp.matched_accuracy(y, k_means(X.astype(np.float32), s)) for s in range(3)])
    def across(XA, XB):
        D = cdist(XA, XB, "cityblock"); np.fill_diagonal(D, np.inf)
        return 100 * np.mean(y[D.argmin(1)] == y)

    F = {s: np.stack([(l[None, :] > LG[:, None]).mean(1) if len(l) else np.zeros(len(LG)) for l in make(s, False)[0]]) for s in cut}
    print(f"{len(times)} trials, {k} neurons; chance 1.7%")
    print(f"reference: ISI fractions   full 1-NN {nn(F['full']):.1f}  k-means {km(F['full']):.1f}   halves A->B {across(F['A'], F['B']):.1f}\n", flush=True)
    real, anc = signatures(*make("full", False), MARKS["full"], rng=rng); print("full done", flush=True)
    shuf, _ = signatures(*make("full", True), MARKS["full"], anc); print("full shuffled done", flush=True)
    ra, _ = signatures(*make("A", False), MARKS["A"], anc); rb, _ = signatures(*make("B", False), MARKS["B"], anc); print("halves done", flush=True)
    sa, _ = signatures(*make("A", True), MARKS["A"], anc); sb_, _ = signatures(*make("B", True), MARKS["B"], anc)
    sc = lambda X: X / np.median(cdist(X[::7], X[::7], "cityblock"))
    print(f"{'signature':50s} | full trial 1-NN (shuffled)  k-means (shuffled) | halves A->B 1-NN (shuffled)")
    for name in real:
        print(f"{name:50s} | {nn(real[name]):5.1f} ({nn(shuf[name]):5.1f})   {km(real[name]):5.1f} ({km(shuf[name]):5.1f}) | "
              f"{across(ra[name], rb[name]):5.1f} ({across(sa[name], sb_[name]):5.1f})", flush=True)
    for name in ("path read at fixed times", "Rips H0 in 8 regions (whole cloud)"):
        j = lambda f, d: np.hstack([sc(f), sc(d[name])])
        print(f"{'ISI fractions + ' + name[:34]:50s} | {nn(j(F['full'], real)):5.1f} ({nn(j(F['full'], shuf)):5.1f})   "
              f"{km(j(F['full'], real)):5.1f} ({km(j(F['full'], shuf)):5.1f}) | "
              f"{across(j(F['A'], ra), j(F['B'], rb)):5.1f} ({across(j(F['A'], sa), j(F['B'], sb_)):5.1f})", flush=True)
