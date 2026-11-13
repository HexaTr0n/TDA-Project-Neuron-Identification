"""W12: one mixture-of-Gaussians pdf per spike, conditioned on the entire history (user's idea).

Run from the repo root: python scripts/experiments/exp34_hmm_pdf_per_spike.py
Retina, 60 neurons, frozen trials; numpy, scipy, scikit-learn only (no stablebear).
Model, fitted to each trial separately by EM (Baum-Welch): a hidden Markov model on
x = log ISI with K hidden states, a Gaussian per state, and a K x K transition table.
After ISI number i the filter gives P(next state | ISI_1..ISI_i), so the predicted pdf of the
next ISI is a mixture of the K Gaussians whose weights depend on the whole past.
Every spike becomes that pdf, stored as its survival curve on a grid, so L1 between two
points is Wasserstein-1 between two predicted pdfs. Control: ISIs shuffled before fitting.
"""
import sys, warnings
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import norm
from sklearn.cluster import KMeans
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import run_pipeline as rp
from load_dataset import load_dataset
warnings.filterwarnings("ignore")

csv = REPO / "data" / "isi_frozennoise.csv"
df = load_dataset(csv if csv.exists() else rp.DEFAULT_PATH, sessions=["20180126_sr_le_sp"])
blocks = rp.pick_neurons(rp.make_blocks(df, rp.Config()), 60, 0)
y = np.unique([b[0] for b in blocks], return_inverse=True)[1]; k = y.max() + 1
times = [np.asarray(b[1][0].spike_times, float) for b in blocks]
lg = lambda t: np.log(np.maximum(np.diff(t), 1e-4))
HALF = 9.94 / 2
SETS = {"full": [lg(t) for t in times], "A": [lg(t[t < HALF]) for t in times], "B": [lg(t[t >= HALF]) for t in times]}
LG = np.linspace(np.log(1e-4), np.log(10), 230, endpoint=False)
GRID = LG[::5]; STEP = GRID[1] - GRID[0]
RADII = np.concatenate([[0], np.geomspace(0.005, 3, 39)])
rng = np.random.default_rng(0)


def pad(logs):
    T = max(max(len(l) for l in logs), 2); X = np.zeros((len(logs), T)); M = np.zeros((len(logs), T), bool)
    for i, l in enumerate(logs):
        X[i, :len(l)] = l; M[i, :len(l)] = True
    return X, M


def fit_hmm(X, M, K, iters=60):
    """EM for all trials at once. Returns mu, sd, A (states sorted by mean) and the predictive weights W (N, T, K):
    W[:, t] = P(state of ISI t+1 | ISIs 0..t)."""
    N, T = X.shape; Xn = np.where(M, X, np.nan); n = M.sum(1)
    edges = np.nanquantile(np.where(n[:, None] > 0, Xn, 0.0), np.linspace(0, 1, K + 1), axis=1).T      # (N, K+1)
    mu = np.zeros((N, K)); sd = np.ones((N, K))
    for j in range(K):
        g = np.where((Xn >= edges[:, [j]]) & (Xn <= edges[:, [j + 1]]), Xn, np.nan)
        mu[:, j] = np.nan_to_num(np.nanmean(g, 1)); sd[:, j] = np.maximum(np.nan_to_num(np.nanstd(g, 1), nan=1.0), 0.1)
    A = np.full((N, K, K), 1.0 / K); pi = np.full((N, K), 1.0 / K)

    def forward():
        B = np.where(M[:, :, None], norm.pdf(X[:, :, None], mu[:, None, :], sd[:, None, :]) + 1e-300, 1.0)
        a = np.zeros((N, T, K)); c = np.ones((N, T))
        v = pi * B[:, 0]; c[:, 0] = v.sum(1); a[:, 0] = v / c[:, [0]]
        for t in range(1, T):
            v = np.einsum("ni,nij->nj", a[:, t - 1], A) * B[:, t]
            v = np.where(M[:, [t]], v, a[:, t - 1]); c[:, t] = v.sum(1); a[:, t] = v / c[:, [t]]
        return B, a, c

    for _ in range(iters):
        B, a, c = forward()
        b = np.ones((N, K)); gam = np.zeros((N, T, K)); xi = np.zeros((N, K, K))
        gam[:, T - 1] = a[:, T - 1]
        for t in range(T - 2, -1, -1):
            u = B[:, t + 1] * b / c[:, [t + 1]]
            x = a[:, t][:, :, None] * A * u[:, None, :]
            xi += np.where(M[:, t + 1][:, None, None], x, 0.0)
            b = np.where(M[:, [t + 1]], np.einsum("nij,nj->ni", A, u), b)
            gam[:, t] = a[:, t] * b
        gam = np.where(M[:, :, None], gam / np.maximum(gam.sum(2, keepdims=True), 1e-300), 0.0)
        w = gam.sum(1) + 1e-9
        mu = (gam * X[:, :, None]).sum(1) / w
        sd = np.maximum(np.sqrt((gam * (X[:, :, None] - mu[:, None, :]) ** 2).sum(1) / w), 0.1)
        A = xi + 0.5; A /= A.sum(2, keepdims=True)                      # half a pseudo-count per transition
        pi = gam[:, 0] + 0.5; pi /= pi.sum(1, keepdims=True)
    o = np.argsort(mu, 1); r = np.arange(N)[:, None]
    mu, sd, pi = mu[r, o], sd[r, o], pi[r, o]; A = A[r[:, :, None], o[:, :, None], o[:, None, :]]
    _, a, _ = forward()
    W = np.einsum("nti,nij->ntj", a, A)
    stat = np.where(M[:, :, None], a, 0).sum(1) / np.maximum(M.sum(1), 1)[:, None]
    return mu, sd, A, W, stat


def h0_curve(P):
    """Stable rank of the H0 bars of one cloud under L1 between curves: bars longer than each radius."""
    if len(P) < 2:
        return np.zeros(len(RADII))
    D = cdist(P, P, "cityblock") * STEP + 1e-9
    e = minimum_spanning_tree(D).data
    return (e[None, :] > RADII[:, None]).sum(1).astype(float)


def signatures(logs, K, shuffle=False, anchors=None):
    if shuffle:
        logs = [rng.permutation(l) for l in logs]
    X, M = pad(logs); N = len(logs)
    mu, sd, A, W, stat = fit_hmm(X, M, K)
    S = norm.sf(GRID[None, None, :], mu[:, :, None], sd[:, :, None])                 # (N, K, G) survival per state
    P = np.einsum("ntk,nkg->ntg", W, S).astype(np.float32)                           # one curve per spike
    Pn = np.where(M[:, :, None], P, np.nan)
    out = {}
    out["mean predicted pdf (no history left)"] = np.nan_to_num(np.nanmean(Pn, 1)) * STEP
    out["model: next-ISI pdf after each state, weighted"] = (np.einsum("nkj,njg->nkg", A, S) * stat[:, :, None]).reshape(N, -1) * STEP
    out["cloud, no homology: quantiles over spikes"] = np.nan_to_num(np.nanquantile(Pn, [.1, .3, .5, .7, .9], axis=1)).transpose(1, 0, 2).reshape(N, -1) * STEP / 5
    out["how much the prediction moves (spread only)"] = np.nan_to_num(np.nanstd(Pn, 1)) * STEP
    clouds = [P[i, M[i]] for i in range(N)]
    H = np.stack([h0_curve(c) for c in clouds])
    out["cloud, Rips H0, whole cloud"] = H
    out["cloud, Rips H0, divided by point count"] = H / np.maximum(M.sum(1), 1)[:, None]
    if anchors is None:
        pool = np.concatenate(clouds); pool = pool[rng.choice(len(pool), min(20000, len(pool)), replace=False)]
        anchors = KMeans(8, n_init=3, random_state=0).fit(pool).cluster_centers_
    reg, cnt = [], []
    for c in clouds:
        lab = cdist(c, anchors).argmin(1) if len(c) else np.zeros(0, int)
        reg.append(np.concatenate([h0_curve(c[lab == j]) + (lab == j).sum() * (RADII == 0) for j in range(8)]))
        cnt.append(np.bincount(lab, minlength=8).astype(float))
    out["cloud, Rips H0 in 8 regions (whole cloud)"] = np.stack(reg)
    out["cloud, region counts (twin without bars)"] = np.stack(cnt)
    return out, anchors, dict(A=A, mu=mu, sd=sd)


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

def km(X):
    return np.mean([100 * rp.matched_accuracy(y, k_means(X.astype(np.float32), s)) for s in range(3)])

def across(XA, XB):
    D = cdist(XA, XB, "cityblock"); np.fill_diagonal(D, np.inf)
    return 100 * np.mean(y[D.argmin(1)] == y)


F = {s: np.stack([(l[None, :] > LG[:, None]).mean(1) if len(l) else np.zeros(len(LG)) for l in SETS[s]]) for s in SETS}
print(f"{len(times)} trials, {k} neurons, median ISIs per trial {int(np.median([len(l) for l in SETS['full']]))}; chance 1.7%")
print(f"reference: ISI fractions   full 1-NN {nn(F['full']):.1f}  k-means {km(F['full']):.1f}   halves A->B {across(F['A'], F['B']):.1f}\n", flush=True)
sc = lambda X, R=None: X / np.median(cdist((X if R is None else R)[::7], (X if R is None else R)[::7], "cityblock"))
for K in (2, 3):
    real, anc, par = signatures(SETS["full"], K)
    shuf, _, _ = signatures(SETS["full"], K, shuffle=True, anchors=anc)
    ra, _, _ = signatures(SETS["A"], K, anchors=anc); rb, _, _ = signatures(SETS["B"], K, anchors=anc)
    sa, _, _ = signatures(SETS["A"], K, shuffle=True, anchors=anc); sb_, _, _ = signatures(SETS["B"], K, shuffle=True, anchors=anc)
    stick = np.mean([np.trace(a) / K for a in par["A"]])
    print(f"=== {K} hidden states (mean probability of staying in a state {stick:.2f}; {1 / K:.2f} would mean no memory)")
    print(f"{'signature':50s} | full trial 1-NN (shuffled)  k-means (shuffled) | halves A->B 1-NN (shuffled)")
    for name in real:
        print(f"{name:50s} | {nn(real[name]):5.1f} ({nn(shuf[name]):5.1f})   {km(real[name]):5.1f} ({km(shuf[name]):5.1f}) | "
              f"{across(ra[name], rb[name]):5.1f} ({across(sa[name], sb_[name]):5.1f})", flush=True)
    for name in ("cloud, no homology: quantiles over spikes", "cloud, Rips H0 in 8 regions (whole cloud)"):
        j = lambda f, d: np.hstack([sc(f), sc(d[name])])
        print(f"{'ISI fractions + ' + name[:34]:50s} | {nn(j(F['full'], real)):5.1f} ({nn(j(F['full'], shuf)):5.1f})   "
              f"{km(j(F['full'], real)):5.1f} ({km(j(F['full'], shuf)):5.1f}) | "
              f"{across(j(F['A'], ra), j(F['B'], rb)):5.1f} ({across(j(F['A'], sa), j(F['B'], sb_)):5.1f})", flush=True)
    print(flush=True)
