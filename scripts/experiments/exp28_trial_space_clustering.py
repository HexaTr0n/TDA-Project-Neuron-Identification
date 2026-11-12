"""Is the clustering step the bottleneck? Topological clustering of the trial space.

Run from the repo root: python scripts/experiments/exp28_trial_space_clustering.py
Points are trials, distances come from a signature. Compared on the retina (60 neurons):
  k-means with mean curves, k-medoids, average linkage,
  ToMATo (Chazal et al.): density peaks merged by persistence, on a nearest-neighbour graph.
ToMATo also gives a barcode of the trial space, whose long bars count the neurons.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist, squareform
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import run_pipeline as rp
from load_dataset import load_dataset

T = 9.94
LG = np.linspace(np.log(1e-4), np.log(10), 230, endpoint=False)
csv = REPO / "data" / "isi_frozennoise.csv"
ALL = load_dataset(csv if csv.exists() else rp.DEFAULT_PATH)


def window(t, s):
    g = np.arange(s / 8, T + s, s / 4)      # coarse grid: enough here, and much faster
    return np.sqrt(np.searchsorted(t, g, "right") - np.searchsorted(t, g - s, "right")).astype(np.float32)


def scaled(x, rng):
    i = rng.choice(len(x), min(300, len(x)), replace=False)
    return x / np.median(cdist(x[i], x[i], "cityblock"))


def k_means_l1(X, k, seed, restarts=3, iters=25):
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


def tomato(D, k_density=8, k_graph=8, tau=np.inf):
    """Returns labels and the prominence of every density peak (the trial space's H0 bars)."""
    n = len(D); order_nb = np.argsort(D, axis=1)[:, 1:max(k_density, k_graph) + 1]
    logf = -np.log(np.take_along_axis(D, order_nb[:, :k_density], 1).mean(1))
    nb = [set(order_nb[i, :k_graph]) for i in range(n)]
    for i in range(n):
        for j in order_nb[i, :k_graph]:
            nb[j].add(i)
    parent = np.arange(n); root_f = {}; prom = {}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    done = np.zeros(n, bool)
    for i in np.argsort(-logf):
        up = [j for j in nb[i] if done[j]]
        done[i] = True
        if not up:
            root_f[i] = logf[i]; continue
        roots = {find(j) for j in up}
        main = max(roots, key=lambda r: root_f[r]); parent[i] = main
        for r in roots - {main}:
            if root_f[r] - logf[i] < tau:
                prom[r] = root_f[r] - logf[i]; parent[r] = main
    lab = np.array([find(i) for i in range(n)])
    for r in set(lab):
        prom.setdefault(r, np.inf)
    return np.unique(lab, return_inverse=True)[1], np.sort([p for p in prom.values()])[::-1]


def tomato_k(D, k, **kw):
    _, prom = tomato(D, **kw)
    fin = prom[np.isfinite(prom)]
    if len(fin) + np.isinf(prom).sum() < k:
        return tomato(D, tau=0.0, **kw)[0], prom
    need = k - np.isinf(prom).sum()
    tau = (fin[need - 1] + (fin[need] if need < len(fin) else 0)) / 2 if need > 0 else np.inf
    return tomato(D, tau=tau, **kw)[0], prom


def run(session, seed=0):
    rng = np.random.default_rng(seed)
    blocks = rp.make_blocks(ALL[ALL.session == session], rp.Config())
    if len({b[0] for b in blocks}) > 60:
        blocks = rp.pick_neurons(blocks, 60, seed)
    times = [np.asarray(b[1][0].spike_times, float) for b in blocks]
    y = np.unique([b[0] for b in blocks], return_inverse=True)[1]; k = y.max() + 1
    isi = np.stack([(np.log(np.maximum(np.diff(t), 1e-4))[None, :] > LG[:, None]).mean(1) for t in times]).astype(np.float32)
    ms = np.hstack([scaled(np.stack([window(t, s) for t in times]), rng) for s in (.02, 2 / 30, .2)])
    feats = {"multi-scale sqrt time bars": ms, "ISI fractions": isi,
             "multi-scale sqrt time bars + ISI fractions": np.hstack([scaled(ms, rng), .5 * scaled(isi, rng)])}
    rows = []
    for name, X in feats.items():
        D = cdist(X, X, "cityblock")
        r = dict(session=session, signature=name, neurons=k,
                 kmeans=np.mean([rp.matched_accuracy(y, k_means_l1(X, k, s)) for s in range(3)]),
                 kmedoids=rp.matched_accuracy(y, rp.k_medoids(D, k)),
                 average_linkage=rp.matched_accuracy(y, fcluster(linkage(squareform(D, checks=False), "average"), k, "maxclust") - 1))
        best = None
        for kd in (5, 8, 12):
            lab, prom = tomato_k(D, k, k_density=kd, k_graph=kd)
            r[f"tomato_k{kd}"] = rp.matched_accuracy(y, lab)
            fin = prom[np.isfinite(prom)]
            gaps = fin[:-1] - fin[1:]                      # largest drop in prominence = suggested number of clusters
            lo = 5; r[f"suggested_k{kd}"] = int(np.argmax(gaps[lo:min(len(gaps), 200)]) + lo + 1 + np.isinf(prom).sum())
        rows.append(r)
    return rows


out = []
for session in ("20180126_sr_le_sp", "20171215_sr_le_sp", "20180405_sr_le"):
    out += run(session)
    print(pd.DataFrame(out[-3:]).drop(columns="session").round(3).to_string(index=False), flush=True)
pd.DataFrame(out).to_csv(Path(__file__).with_name("trial_space_clustering.csv"), index=False)
