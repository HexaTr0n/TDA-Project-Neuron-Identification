"""Shared harness: same trials, same scores, one results file.

Scores, all computed from a trial-by-trial distance matrix:
  acc   leave-one-out 1-NN: is the closest other trial from the same neuron?
  ami   average-linkage clustering into the true number of neurons vs the labels
  auc   direct test: for trial t of neuron A, how often is A on another trial closer
        than another neuron B on trial t?  0.5 is chance.
acc and auc get 95% bootstrap intervals over neurons.
"""
import os, sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "legacy_scripts"), str(HERE.parent), str(HERE)]  # versions these runs used
import run_pipeline as rp
import spike_distance as sd
from tda import rips, stable_ranks

REPO = HERE.parent.parent
SESSION = "20180126_sr_le_sp"
OUT = Path(os.environ.get("LAB_OUT", REPO / "results" / "experiments"))
Rec = namedtuple("Rec", "neuron trial isis spike_times")


def load_data(csv=None):
    path = csv or (REPO / "data" / "isi_frozennoise.csv")
    return rp.load_dataset(str(path), sessions=[SESSION])


def records(df, n_neurons=60):
    """Same trials, in the same order, as spike_distance.py --neurons 60."""
    blocks = rp.make_blocks(df, rp.Config())
    if n_neurons:
        blocks = sd.pick_neurons(blocks, n_neurons, 0)
    return [Rec(n, t[0].trial, t[0].isis, t[0].spike_times) for n, t in blocks]


def score(D, recs, seed=0, boot=1000):
    D = np.array(D, dtype=np.float64)
    names, y = np.unique([r.neuron for r in recs], return_inverse=True)
    tr = np.array([r.trial for r in recs])
    n = len(y)
    pos = D[D > 0]
    scale = np.median(pos) if len(pos) else 1.0
    rng = np.random.default_rng(seed)
    jitter = rng.random((n, n)) * 1e-9 * scale
    N = D + jitter + jitter.T
    np.fill_diagonal(N, np.inf)
    hit = (y[N.argmin(1)] == y).astype(float)
    auc = np.full(n, np.nan)
    for i in range(n):
        same = N[i, (y == y[i]) & (np.arange(n) != i)]
        other = N[i, (tr == tr[i]) & (y != y[i])]
        if len(same) and len(other):
            order = np.sort(other)
            lo = np.searchsorted(order, same, side="left")
            hi = np.searchsorted(order, same, side="right")
            auc[i] = 1 - (lo + hi).mean() / (2 * len(other))
    pn_hit = np.array([hit[y == k].mean() for k in range(len(names))])
    pn_auc = np.array([np.nanmean(auc[y == k]) for k in range(len(names))])
    idx = rng.integers(0, len(names), (boot, len(names)))
    ci = lambda v: np.percentile(np.nanmean(v[idx], axis=1), [2.5, 97.5])
    a_lo, a_hi = ci(pn_hit)
    u_lo, u_hi = ci(pn_auc)
    pred = AgglomerativeClustering(n_clusters=len(names), metric="precomputed",
                                   linkage="average").fit_predict(D)
    return dict(n_trials=n, n_neurons=len(names), acc=hit.mean(), acc_lo=a_lo, acc_hi=a_hi,
                ami=adjusted_mutual_info_score(y, pred), auc=np.nanmean(pn_auc),
                auc_lo=u_lo, auc_hi=u_hi, chance=float(np.mean((np.bincount(y)[y] - 1) / (n - 1))))


def ms(D):
    """Scale a distance matrix by its median so that matrices can be added."""
    return D / np.median(D[D > 0])


def subsample(clouds, n, S, seed=0):
    """clouds -> list of S clouds of n points each, for every cloud."""
    rng = np.random.default_rng(seed)
    out = []
    for c in clouds:
        for _ in range(S):
            out.append(c[rng.choice(len(c), n, replace=False)] if len(c) > n else c)
    return out


def topo_distances(clouds, n=25, S=20, seed=0, grid_size=512, max_dim=1):
    """Averaged stable ranks and their L1 distances. Returns {'H0': D, 'H1': D}."""
    sub = subsample(clouds, n, S, seed)
    H0, H1, _ = rips(sub)
    out = {}
    for name, bars in (("H0", H0), ("H1", H1))[:max_dim + 1]:
        all_len = np.concatenate([b for b in bars if len(b)] or [np.zeros(1)])
        top = np.percentile(all_len, 99.9) if len(all_len) else 1.0
        grid = np.linspace(0, top * 1.0001, grid_size, endpoint=False)
        F = stable_ranks(bars, grid).reshape(len(clouds), S, grid_size).mean(1)
        out[name] = cdist(F, F, "cityblock") * (grid[1] - grid[0])
    return out


def sliced_wasserstein(clouds, n_dirs=40, n_levels=32, seed=0):
    """Compares whole clouds, position included, with no topology."""
    d = clouds[0].shape[1]
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n_dirs, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    levels = (np.arange(n_levels) + 0.5) / n_levels
    Q = np.stack([np.quantile(c @ U.T, levels, axis=0).ravel() for c in clouds])
    return cdist(Q, Q) / np.sqrt(n_dirs * n_levels)


def cloud_rows(name, clouds, recs, n=25, S=20, seed=0, extra=None, topo=True):
    """Score a cloud-valued embedding on every view of it."""
    keep = [i for i, c in enumerate(clouds) if c is not None and len(c) >= 3]
    clouds = [clouds[i] for i in keep]
    rs = [recs[i] for i in keep]
    M = np.stack([c.mean(0) for c in clouds])
    Ds = {"position": cdist(M, M), "whole cloud": sliced_wasserstein(clouds, seed=seed)}
    if topo:
        T = topo_distances(clouds, n, S, seed)
        Ds.update(T)
        Ds["H0+H1"] = ms(T["H0"]) + ms(T["H1"]) if T["H1"].max() > 0 else ms(T["H0"])
        Ds["position+H0+H1"] = ms(Ds["position"]) + Ds["H0+H1"]
        Ds["whole cloud+H0+H1"] = ms(Ds["whole cloud"]) + Ds["H0+H1"]
    rows = []
    for view, D in Ds.items():
        rows.append(dict(embedding=name, view=view, **score(D, rs, seed)))
    if extra is not None:
        idx = np.array(keep)
        for label, D in extra.items():
            sub = D[np.ix_(idx, idx)]
            if topo:
                rows.append(dict(embedding=name, view=f"{label}+H0+H1",
                                 **score(ms(sub) + Ds["H0+H1"], rs, seed)))
    return rows


def save(rows, file):
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / file, mode="a", header=not (OUT / file).exists(), index=False)


def show(rows):
    df = pd.DataFrame(rows)[["embedding", "view", "n_trials", "acc", "acc_lo", "acc_hi",
                             "ami", "auc", "auc_lo", "auc_hi"]]
    print(df.round(3).to_string(index=False), flush=True)
