"""How should a delay cloud be cut to size before Rips? Retina, 60 neurons, H0 only.

Run from the repo root: python scripts/experiments/exp30_cloud_sampling.py
Three ways to prepare the cloud of a trial:
  random     25 points drawn at random, 20 draws averaged (what we did so far)
  stretch    25 CONSECUTIVE points: a random number of spikes cut off at the start and the end,
             20 draws averaged (the user's suggestion: keeps neighbouring spikes together)
  whole      every point, once. The stable rank then starts at the number of points, so the
             spike count leaks in; "whole, as fractions" divides by the number of points.
For plain delay Rips (m = 3) and delay regions (8 anchors). H0 bars are the edges of the
minimum spanning tree. Scores: 1-NN and k-medoids on full trials; and across movie halves.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import run_pipeline as rp
from load_dataset import load_dataset

T, M, K, DRAWS = 9.94, 3, 8, 20
csv = REPO / "data" / "isi_frozennoise.csv"
df = load_dataset(csv if csv.exists() else rp.DEFAULT_PATH, sessions=["20180126_sr_le_sp"])
blocks = rp.pick_neurons(rp.make_blocks(df, rp.Config()), 60, 0)
rng = np.random.default_rng(0)
used = {b[0] for b in blocks}
GRID = np.linspace(0, 3.0, 256, endpoint=False)
LG = np.linspace(np.log(1e-4), np.log(10), 230, endpoint=False)


def delay(t):
    x = np.log(np.maximum(np.diff(t), 1e-4))
    return np.stack([x[i:len(x) - M + 1 + i] for i in range(M)], axis=1)


pool = df[~df.neuron_id.isin(used) & (df.isis.map(len) >= 30)]
P = np.concatenate([delay(np.asarray(r.spike_times, float)) for r in pool.sample(300, random_state=0).itertuples()])
ANCH = KMeans(K, n_init=3, random_state=0).fit(P[rng.choice(len(P), 8000, replace=False)]).cluster_centers_


def h0(c):
    if len(c) < 2:
        return np.zeros(0)
    return minimum_spanning_tree(cdist(c, c)).data


def rank(bars):
    return (bars[None, :] > GRID[:, None]).sum(1).astype(np.float32)


def signature(c, regions):
    if not regions:
        return rank(h0(c))
    lab = np.linalg.norm(c[:, None] - ANCH[None], axis=2).argmin(1)
    return np.concatenate([rank(h0(c[lab == r])) for r in range(K)])


def features(times, n, mode, regions, shuffle):
    out = []
    for t in times:
        if shuffle:
            t = t[0] + np.r_[0, np.cumsum(rng.permutation(np.diff(t)))]
        c = delay(t)
        if mode == "random":
            f = np.mean([signature(c[rng.choice(len(c), n, replace=False)], regions) for _ in range(DRAWS)], 0)
        elif mode == "stretch":
            f = np.mean([signature(c[s:s + n], regions) for s in rng.integers(0, len(c) - n + 1, DRAWS)], 0)
        else:
            f = signature(c, regions)
            if mode == "whole, as fractions":
                f = f / len(c)
        out.append(f)
    return np.stack(out)


def cut(t, h):
    return t[t < T / 2] if h == 0 else t[t >= T / 2] - T / 2


def nn(D, y, allowed=None):
    D = D.copy(); np.fill_diagonal(D, np.inf)
    if allowed is not None:
        D[~allowed] = np.inf
    return np.mean(y[D.argmin(1)] == y)


def transfer(D, y, half):
    out = []
    for h in (0, 1):
        A, B = np.flatnonzero(half == h), np.flatnonzero(half != h); k = y.max() + 1
        cl = rp.k_medoids(D[np.ix_(A, A)], k)
        med = np.array([A[cl == c][D[np.ix_(A[cl == c], A[cl == c])].sum(0).argmin()] if (cl == c).any() else A[0] for c in range(k)])
        tab = np.zeros((k, k)); np.add.at(tab, (y[A], cl), 1)
        r, c = linear_sum_assignment(-tab); name = np.full(k, -1); name[c] = r
        out.append(np.mean(name[D[np.ix_(B, med)].argmin(1)] == y[B]))
    return float(np.mean(out))


full = [np.asarray(b[1][0].spike_times, float) for b in blocks]
y = np.unique([b[0] for b in blocks], return_inverse=True)[1]
ok = [i for i, t in enumerate(full) if min(len(cut(t, 0)), len(cut(t, 1))) > 21]
halves = [cut(full[i], h) for h in (0, 1) for i in ok]
hy = np.unique(np.r_[y[ok], y[ok]], return_inverse=True)[1]; half = np.repeat([0, 1], len(ok)); tid = np.r_[ok, ok]
cross = (half[:, None] != half[None, :]) & (tid[:, None] != tid[None, :])

rows = []
def score(name, F, Fh):
    D = cdist(F, F, "cityblock"); Dh = cdist(Fh, Fh, "cityblock")
    rows.append(dict(method=name, full_1nn=nn(D, y), full_cluster=rp.matched_accuracy(y, rp.k_medoids(D, y.max() + 1)),
                     cross_1nn=nn(Dh, hy, cross), cross_cluster=transfer(Dh, hy, half)))
    print({k: (round(float(v), 3) if k != "method" else v) for k, v in rows[-1].items()}, flush=True)

c = np.array([len(t) for t in full], float)[:, None]; ch = np.array([len(t) for t in halves], float)[:, None]
score("reference: spike count", c, ch)
frac = lambda ts: np.stack([(np.log(np.maximum(np.diff(t), 1e-4))[None, :] > LG[:, None]).mean(1) for t in ts])
score("reference: ISI fractions", frac(full), frac(halves))
for regions in (False, True):
    for mode in ("random", "stretch", "whole", "whole, as fractions"):
        for shuffle in (False, True):
            name = f"{'delay regions' if regions else 'delay Rips'} H0, {mode}" + (", ISIs shuffled" if shuffle else "")
            score(name, features(full, 25, mode, regions, shuffle), features(halves, 15, mode, regions, shuffle))
tab = pd.DataFrame(rows).round(3); print(tab.to_string(index=False)); tab.to_csv(Path(__file__).with_name("cloud_sampling.csv"), index=False)
