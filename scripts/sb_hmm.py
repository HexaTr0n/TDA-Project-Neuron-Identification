"""Task L25: per-spike mixture pdf with the entire history (hidden Markov model), in stablebear.

Port of scripts/experiments/exp34_hmm_pdf_per_spike.py (W12). Per sample, a hidden Markov model on
x = log ISI with K states (one Gaussian each, a K x K transition table) is fitted by EM. After
every ISI the forward filter gives the weights of the next ISI's mixture, so each spike becomes a
pdf that depends on the whole past; it is stored as its survival curve on 46 log-ISI points
(L1 between two curves = Wasserstein-1 between the pdfs).

Readouts (as W12):
  mean predicted pdf             L1
  cloud, no homology             5 quantiles over spikes of the curves, L1
  spread                         sd over spikes, L1
  cloud, Rips H0 (whole)         stablebear Rips on the whole cloud (Euclidean between curves),
                                 stable rank, L1
  cloud, Rips H0 in 8 regions    k-means anchors fitted on curves of outside neurons
  + ISI fractions                each of the above added to ISI fractions, median-scaled
References on the same samples: ISI fractions; whole-cloud delay regions k = 32 H0 (L22).
Mean probability of staying in a state (trace(A) / K; 1/K = no memory) is reported.
Control: ISIs shuffled before fitting. Scores as L17 (1-NN, k-medoids; k-means off).

Run from the repo root:
    python scripts/sb_hmm.py --dataset openscope124 --states 2 3
Writes results/hmm_<dataset>.csv.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from barcodes import shuffle_isis
from sb_cross import isi_fraction_pcfs
from sb_feature_cloud import evaluate_one
from sb_full_table import delay_cloud, evaluate, rips_parts, split_regions
from sb_headline import load

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
LG = np.linspace(np.log(1e-4), np.log(10), 230, endpoint=False)
GRID = LG[::5]
STEP = GRID[1] - GRID[0]


def fit_hmm(X, M, K, iters=60):
    """EM on a batch of padded trials (as exp34). Returns mu, sd, A, W (predictive weights), stat."""
    N, T = X.shape
    Xn = np.where(M, X, np.nan)
    n = M.sum(1)
    edges = np.nanquantile(np.where(n[:, None] > 0, Xn, 0.0), np.linspace(0, 1, K + 1), axis=1).T
    mu, sd = np.zeros((N, K)), np.ones((N, K))
    for j in range(K):
        g = np.where((Xn >= edges[:, [j]]) & (Xn <= edges[:, [j + 1]]), Xn, np.nan)
        mu[:, j] = np.nan_to_num(np.nanmean(g, 1))
        sd[:, j] = np.maximum(np.nan_to_num(np.nanstd(g, 1), nan=1.0), 0.1)
    A = np.full((N, K, K), 1.0 / K)
    pi = np.full((N, K), 1.0 / K)

    def forward():
        B = np.where(M[:, :, None], norm.pdf(X[:, :, None], mu[:, None, :], sd[:, None, :]) + 1e-300, 1.0)
        a, c = np.zeros((N, T, K)), np.ones((N, T))
        v = pi * B[:, 0]
        c[:, 0] = v.sum(1)
        a[:, 0] = v / c[:, [0]]
        for t in range(1, T):
            v = np.einsum("ni,nij->nj", a[:, t - 1], A) * B[:, t]
            v = np.where(M[:, [t]], v, a[:, t - 1])
            c[:, t] = v.sum(1)
            a[:, t] = v / c[:, [t]]
        return B, a, c

    for _ in range(iters):
        B, a, c = forward()
        b = np.ones((N, K))
        gam = np.zeros((N, T, K))
        xi = np.zeros((N, K, K))
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
        A = xi + 0.5
        A /= A.sum(2, keepdims=True)
        pi = gam[:, 0] + 0.5
        pi /= pi.sum(1, keepdims=True)
    o = np.argsort(mu, 1)
    r = np.arange(N)[:, None]
    mu, sd = mu[r, o], sd[r, o]
    A = A[r[:, :, None], o[:, :, None], o[:, None, :]]
    _, a, _ = forward()
    return mu, sd, A, np.einsum("nti,nij->ntj", a, A)


def curves(trains, K, batch=40):
    """Per sample: the cloud of predicted survival curves (n_isi x 46) and the stay probability."""
    logs = [np.log(np.maximum(np.diff(t), 1e-4)) for t in trains]
    order = np.argsort([len(l) for l in logs])
    clouds, stay = [None] * len(logs), np.zeros(len(logs))
    for s0 in range(0, len(order), batch):
        ids = order[s0:s0 + batch]
        T = max(2, max(len(logs[i]) for i in ids))
        X, M = np.zeros((len(ids), T)), np.zeros((len(ids), T), bool)
        for a, i in enumerate(ids):
            X[a, :len(logs[i])] = logs[i]
            M[a, :len(logs[i])] = True
        mu, sd, A, W = fit_hmm(X, M, K)
        S = norm.sf(GRID[None, None, :], mu[:, :, None], sd[:, :, None])
        for a, i in enumerate(ids):
            clouds[i] = (W[a, :len(logs[i])] @ S[a]).astype(np.float64)
            stay[i] = np.trace(A[a]) / K
    return clouds, stay


def signatures(clouds, anchors):
    mean = np.array([c.mean(0) for c in clouds]) * STEP
    quant = np.array([np.quantile(c, [.1, .3, .5, .7, .9], axis=0).ravel() for c in clouds]) * STEP / 5
    spread = np.array([c.std(0) for c in clouds]) * STEP
    pad = lambda c: c if len(c) >= 2 else np.zeros((2, c.shape[1]))
    whole = rips_parts([[[pad(c)]] for c in clouds], 0)[0][0]
    reg = rips_parts([[split_regions(pad(c), anchors)] for c in clouds], 0)[0]
    return {"mean predicted pdf": [("vec", mean, "l1")],
            "cloud, no homology (quantiles)": [("vec", quant, "l1")],
            "spread": [("vec", spread, "l1")],
            "cloud, Rips H0 (whole)": [("pcf", whole)],
            "cloud, Rips H0 in 8 regions": [("pcf", p) for p in reg]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--drop-piece", type=int)
    ap.add_argument("--states", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    A, B, pool, _ = load(args.dataset, args)
    one = B is None
    both = A if one else pd.concat([A, B]).reset_index(drop=True)
    names = sorted(set(both["neuron_id"]))
    y = both["neuron_id"].map({n: i for i, n in enumerate(names)}).to_numpy()
    k = len(names)
    ia, ib = np.arange(len(A)), np.arange(len(A), len(both))
    trains = [np.sort(np.asarray(t, float)) for t in both["spike_times"]]
    rng = np.random.default_rng(args.seed)
    pool = pool[pool["spike_times"].map(len) > 5]
    pool = pool.iloc[rng.choice(len(pool), min(200, len(pool)), replace=False)]
    ptrains = [np.sort(np.asarray(t, float)) for t in pool["spike_times"]]
    rs = np.random.default_rng(args.seed + 1)
    conds = {"real": trains, "ISIs shuffled": [shuffle_isis(t, rs) for t in trains]}
    print(f"{args.dataset}: {k} neurons, {len(ia)} + {len(ib)} samples, median {np.median([len(t) for t in trains]):.0f} "
          f"spikes; chance {100 / k:.1f}%", flush=True)
    out = REPO / "results" / f"hmm_{args.dataset.replace(':', '_')}.csv"
    rows = []

    def score(name, groups, cond, K, extra=None):
        t1 = time.time()
        r = (evaluate_one(groups, False, y, k, args.seed, False) if one
             else evaluate(groups, False, y, ia, ib, k, args.seed, False))
        rows.append(dict(dataset=args.dataset, states=K, readout=name, control=cond, neurons=k, **(extra or {}), **r))
        f = lambda key: f"{100 * r[key]:5.1f}" if key in r else "    -"
        print(f"  K={K} {name:<46} {cond:<13} | A {f('nn_A')} {f('kmedoids_A')} | B {f('nn_B')} {f('kmedoids_B')}"
              f" | A->B {f('nn_Ato')} {f('kmedoids_Ato')} | B->A {f('nn_Bto')}  ({time.time() - t1:.0f} s)", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)

    # references on the same samples
    pts = np.concatenate([delay_cloud(t) for t in ptrains])
    anc32 = KMeans(32, n_init=3, random_state=args.seed).fit(pts[rng.choice(len(pts), min(8000, len(pts)), replace=False)]).cluster_centers_
    for cond, tr in conds.items():
        score("ref: ISI fractions", [([("pcf", isi_fraction_pcfs(tr))], 1.0)], cond, 0)
        reg = rips_parts([[split_regions(delay_cloud(t) if len(t) >= 5 else np.zeros((2, 3)), anc32)] for t in tr], 0)[0]
        score("ref: delay regions k=32, H0 (whole cloud)", [([("pcf", p) for p in reg], 1.0)], cond, 0)

    for K in args.states:
        t0 = time.time()
        pc, _ = curves(ptrains, K)
        P = np.concatenate(pc)
        anchors = KMeans(8, n_init=3, random_state=args.seed).fit(P[rng.choice(len(P), min(20000, len(P)), replace=False)]).cluster_centers_
        for cond, tr in conds.items():
            cl, stay = curves(tr, K)
            print(f"  K={K} {cond}: HMM fits in {time.time() - t0:.0f} s; mean stay probability {stay.mean():.3f} "
                  f"(no memory: {1 / K:.3f})", flush=True)
            sig = signatures(cl, anchors)
            isi = isi_fraction_pcfs(tr)
            for name, parts in sig.items():
                score(name, [(parts, 1.0)], cond, K, dict(stay=stay.mean()))
                score("ISI fractions + " + name, [([("pcf", isi)], 1.0), (parts, 1.0)], cond, K, dict(stay=stay.mean()))
            t0 = time.time()
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
