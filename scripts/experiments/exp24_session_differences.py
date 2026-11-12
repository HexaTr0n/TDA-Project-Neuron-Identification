"""W1: why do some retina sessions cluster much better than others?

Run from the repo root: python scripts/experiments/exp24_session_differences.py
Square-root time bars, k-means with mean curves, 60 random neurons per session (as exp23).
Per neuron: share of its trials in its main cluster, against rate and response reliability.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import run_pipeline as rp
from load_dataset import load_dataset

T, S = 9.94, 2 / 30
csv = REPO / "data" / "isi_frozennoise.csv"
ALL = load_dataset(csv if csv.exists() else rp.DEFAULT_PATH)
G = np.arange(S / 40, T + S, S / 20)
curve = lambda t: np.sqrt(np.searchsorted(t, G, "right") - np.searchsorted(t, G - S, "right")).astype(np.float32)


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


rows, per_neuron = [], []
for session in sorted(ALL.session.unique()):
    blocks = rp.make_blocks(ALL[ALL.session == session], rp.Config())
    if len({b[0] for b in blocks}) > 60:
        blocks = rp.pick_neurons(blocks, 60, 0)
    times = [np.asarray(b[1][0].spike_times, float) for b in blocks]
    names, y = np.unique([b[0] for b in blocks], return_inverse=True); k = len(names)
    X = np.stack([curve(t) for t in times])
    acc, kept = [], np.zeros(k)
    for seed in range(3):
        c = k_means_l1(X, k, seed); acc.append(rp.matched_accuracy(y, c))
        tab = np.zeros((k, k)); np.add.at(tab, (y, c), 1); kept += tab.max(1) / tab.sum(1) / 3
    count = np.array([len(t) for t in times])
    # reliability: correlation between the mean curve of odd and even trials of a neuron
    tr = np.array([b[1][0].trial for b in blocks]); rel = np.full(k, np.nan); snr = np.full(k, np.nan)
    for j in range(k):
        m = y == j
        a, b = X[m & (tr % 2 == 0)], X[m & (tr % 2 == 1)]
        if len(a) >= 2 and len(b) >= 2:
            rel[j] = np.corrcoef(a.mean(0), b.mean(0))[0, 1]
        mu = X[m].mean(0)
        within = np.abs(X[m] - mu).sum(1).mean()                      # spread of trials around their own mean
        snr[j] = within
    centres = np.stack([X[y == j].mean(0) for j in range(k)])
    Dc = cdist(centres, centres, "cityblock"); np.fill_diagonal(Dc, np.inf)
    margin = Dc.min(1) / snr                                           # distance to nearest other neuron / own spread
    rate = np.array([count[y == j].mean() for j in range(k)]) / T
    ntr = np.bincount(y)
    rows.append(dict(session=session, neurons=k, clustering=np.mean(acc), rate_hz=np.median(rate),
                     reliability=np.nanmedian(rel), margin=np.median(margin), trials_per_neuron=np.median(ntr),
                     trials_total=len(times)))
    per_neuron.append(pd.DataFrame(dict(session=session, kept=kept, rate=rate, reliability=rel, margin=margin, trials=ntr)))
S_ = pd.DataFrame(rows).sort_values("clustering", ascending=False)
print(S_.round(3).to_string(index=False))
print("\nacross sessions, Spearman with clustering:")
for c in ("rate_hz", "reliability", "margin", "trials_per_neuron", "neurons"):
    print(f"  {c:18s} {spearmanr(S_.clustering, S_[c])[0]:+.2f}")
P = pd.concat(per_neuron)
print("\nper neuron, all sessions pooled (n = %d): Spearman of 'share of trials in main cluster' with" % len(P))
for c in ("rate", "reliability", "margin", "trials"):
    ok = P[c].notna(); print(f"  {c:12s} {spearmanr(P.kept[ok], P[c][ok])[0]:+.2f}   within sessions (mean): "
                              f"{np.nanmean([spearmanr(g.kept[g[c].notna()], g[c][g[c].notna()])[0] for _, g in P.groupby('session')]):+.2f}")
q = pd.qcut(P.reliability, 4); print("\nshare of trials in main cluster by reliability quartile:"); print(P.groupby(q, observed=True).kept.agg(["mean", "size"]).round(3).to_string())
S_.to_csv(Path(__file__).with_name("session_differences.csv"), index=False)
