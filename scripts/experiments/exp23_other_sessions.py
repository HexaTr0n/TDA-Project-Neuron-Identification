"""The best signatures from exp22 on every recording session, 60 random neurons each.

Run from the repo root: python scripts/experiments/exp23_other_sessions.py
Each session is a different retina. Columns: clustering with k-medoids and with mean curves on
full trials, 1-NN and clustering across movie halves.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import run_pipeline as rp
from load_dataset import load_dataset

T = 9.94
LG = np.linspace(np.log(1e-4), np.log(10), 230, endpoint=False)
csv = REPO / "data" / "isi_frozennoise.csv"
ALL = load_dataset(csv if csv.exists() else rp.DEFAULT_PATH)


def cut(t, h):
    return t if h is None else (t[t < T / 2] if h == 0 else t[t >= T / 2] - T / 2)


def window(t, s, dur):
    g = np.arange(s / 40, dur + s, s / 20)
    return (np.searchsorted(t, g, "right") - np.searchsorted(t, g - s, "right")).astype(np.float32)


def features(times, dur, rng):
    S = lambda rows: np.stack(rows).astype(np.float32)
    isi = S([(np.log(np.maximum(np.diff(t), 1e-4))[None, :] > LG[:, None]).mean(1) for t in times])
    w = {s: S([window(t, s, dur) for t in times]) for s in (.02, 2 / 30, .2)}
    def mix(parts, weights):
        out = []
        for x, wt in zip(parts, weights):
            i = rng.choice(len(x), min(300, len(x)), replace=False)
            out.append(x / np.median(cdist(x[i], x[i], "cityblock")) * wt)
        return np.hstack(out)
    return {"time bars 67 ms": w[2 / 30], "square-root time bars 67 ms": np.sqrt(w[2 / 30]),
            "multi-scale, square root": mix([np.sqrt(x) for x in w.values()], [1, 1, 1]),
            "ISI fractions": isi,
            "square-root time bars + ISI fractions": mix([np.sqrt(w[2 / 30]), isi], [1, .5])}


def nn(D, y, allowed=None):
    D = D.copy(); np.fill_diagonal(D, np.inf)
    if allowed is not None:
        D[~allowed] = np.inf
    return np.mean(y[D.argmin(1)] == y)


def k_means_l1(X, k, restarts=3, iters=25, seed=0):
    r = np.random.default_rng(seed); best, cost = None, np.inf
    for _ in range(restarts):
        C = X[r.choice(len(X), k, replace=False)].copy()
        for _ in range(iters):
            a = cdist(X, C, "cityblock").argmin(1)
            new = np.stack([X[a == c].mean(0) if (a == c).any() else X[r.integers(len(X))] for c in range(k)])
            if np.allclose(new, C):
                break
            C = new
        D = cdist(X, C, "cityblock"); c_ = D.min(1).sum()
        if c_ < cost:
            best, cost = D.argmin(1), c_
    return best


def transfer(D, y, half):
    out = []
    for h in (0, 1):
        A, B = np.flatnonzero(half == h), np.flatnonzero(half != h); k = y.max() + 1
        cl = rp.k_medoids(D[np.ix_(A, A)], k)
        med = np.array([A[cl == c][D[np.ix_(A[cl == c], A[cl == c])].sum(0).argmin()] if (cl == c).any() else A[0] for c in range(k)])
        table = np.zeros((k, k)); np.add.at(table, (y[A], cl), 1)
        r, c = linear_sum_assignment(-table); name = np.full(k, -1); name[c] = r
        out.append(np.mean(name[D[np.ix_(B, med)].argmin(1)] == y[B]))
    return float(np.mean(out))


def run(session, seed, drop_first=0):
    rng = np.random.default_rng(seed)
    df = ALL[ALL.session == session]
    blocks = rp.make_blocks(df, rp.Config())
    blocks = [b for b in blocks if b[1][0].trial >= drop_first]
    names = sorted({b[0] for b in blocks})
    if len(names) > 60:
        blocks = rp.pick_neurons(blocks, 60, seed)
    times = [np.asarray(b[1][0].spike_times, float) for b in blocks]
    y = np.unique([b[0] for b in blocks], return_inverse=True)[1]; k = y.max() + 1
    rows = {}
    for name, X in features(times, T, rng).items():
        D = cdist(X, X, "cityblock")
        rows[name] = dict(nn=nn(D, y), medoids=rp.matched_accuracy(y, rp.k_medoids(D, k)), means=rp.matched_accuracy(y, k_means_l1(X, k)))
    ok = [i for i, t in enumerate(times) if min(len(cut(t, 0)), len(cut(t, 1))) > 21]
    cnt = pd.Series(y[ok]).value_counts(); ok = [i for i in ok if cnt[y[i]] >= 2]
    ht = [cut(times[i], h) for h in (0, 1) for i in ok]
    hy = np.unique(np.r_[y[ok], y[ok]], return_inverse=True)[1]; half = np.repeat([0, 1], len(ok)); tid = np.r_[ok, ok]
    same = half[:, None] == half[None, :]; cross = ~same & (tid[:, None] != tid[None, :])
    for name, X in features(ht, T / 2, rng).items():
        D = cdist(X, X, "cityblock")
        rows[name].update(cross_nn=nn(D, hy, cross), cross_cluster=transfer(D, hy, half))
    return k, len(times), hy.max() + 1, pd.DataFrame(rows).T


out = []
jobs = [(s, 0, 0) for s in sorted(ALL.session.unique())] + [("20180126_sr_le_sp", 1, 0), ("20180126_sr_le_sp", 2, 0), ("20180126_sr_le_sp", 0, 5)]
for session, seed, drop in jobs:
    k, n, kh, tab = run(session, seed, drop)
    label = f"{session} seed {seed}" + (", first 5 trials dropped" if drop else "")
    print(f"\n{label}: {k} neurons, {n} trials; {kh} neurons in the half-trial test"); print(tab.round(3).to_string(), flush=True)
    out.append(tab.assign(run=label, neurons=k).reset_index(names="signature"))
pd.concat(out).to_csv(Path(__file__).with_name("other_sessions.csv"), index=False)
