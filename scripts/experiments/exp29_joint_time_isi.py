"""Does a joint "when x how" signature beat adding the two up? Retina, one stimulus.

Run from the repo root: python scripts/experiments/exp29_joint_time_isi.py
Two-parameter time bars: N(t, a) = spikes in the last s seconds whose preceding ISI exceeds a.
a = 0 gives ordinary time bars; summing over t gives the ISI stable rank. So it contains both
winners and also says WHICH kind of spike (burst spike or isolated spike) occurs when.
Scores: 1-NN and k-means with mean curves (mean of 3 seeds), 60 neurons per session.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import run_pipeline as rp
from load_dataset import load_dataset

T = 9.94
LG = np.linspace(np.log(1e-4), np.log(10), 230, endpoint=False)
csv = REPO / "data" / "isi_frozennoise.csv"
ALL = load_dataset(csv if csv.exists() else rp.DEFAULT_PATH)
THRESH = (0.0, 0.005, 0.02, 0.08)


def window(t, s):
    g = np.arange(s / 8, T + s, s / 4)
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


def joint(t, s, kind):
    prev = np.r_[np.inf, np.diff(t)]                       # the first spike counts as isolated
    parts = []
    for a in THRESH:
        keep = prev > a if kind == "above" else prev <= a
        if kind == "below" and a == 0:
            continue
        parts.append(window(t[keep], s))
    return parts


def run(session, seed=0):
    rng = np.random.default_rng(seed)
    blocks = rp.make_blocks(ALL[ALL.session == session], rp.Config())
    if len({b[0] for b in blocks}) > 60:
        blocks = rp.pick_neurons(blocks, 60, seed)
    times = [np.asarray(b[1][0].spike_times, float) for b in blocks]
    shuf = [t[0] + np.r_[0, np.cumsum(rng.permutation(np.diff(t)))] for t in times]
    y = np.unique([b[0] for b in blocks], return_inverse=True)[1]; k = y.max() + 1
    S = lambda rows: np.stack(rows)
    isi = S([(np.log(np.maximum(np.diff(t), 1e-4))[None, :] > LG[:, None]).mean(1) for t in times]).astype(np.float32)
    tb = S([window(t, 2 / 30) for t in times])
    def J(ts, s, kind="above"):
        cols = list(zip(*[joint(t, s, kind) for t in ts]))
        return np.hstack([scaled(S(c), rng) for c in cols])
    feats = {
        "sqrt time bars 67 ms": tb,
        "sqrt time bars + ISI fractions (sum)": np.hstack([scaled(tb, rng), .5 * scaled(isi, rng)]),
        "multi-scale sqrt time bars": np.hstack([scaled(S([window(t, s) for t in times]), rng) for s in (.02, 2 / 30, .2)]),
        "joint: time x ISI threshold, 67 ms": J(times, 2 / 30),
        "joint, ISIs shuffled": J(shuf, 2 / 30),
        "joint: burst spikes and isolated spikes separately (20 ms)": np.hstack([scaled(S([window(t[np.r_[np.inf, np.diff(t)] > .02], 2 / 30) for t in times]), rng),
                                                                                  scaled(S([window(t[np.r_[np.inf, np.diff(t)] <= .02], 2 / 30) for t in times]) + 1e-6, rng)]),
        "joint, multi-scale (20 + 67 + 200 ms)": np.hstack([J(times, s) for s in (.02, 2 / 30, .2)]),
        "joint multi-scale + ISI fractions": None,
    }
    feats["joint multi-scale + ISI fractions"] = np.hstack([scaled(feats["joint, multi-scale (20 + 67 + 200 ms)"], rng), .5 * scaled(isi, rng)])
    rows = []
    for name, X in feats.items():
        D = cdist(X, X, "cityblock"); np.fill_diagonal(D, np.inf)
        km = [rp.matched_accuracy(y, k_means_l1(X, k, s)) for s in range(3)]
        rows.append(dict(session=session, signature=name, nn=np.mean(y[D.argmin(1)] == y), kmeans=np.mean(km), kmeans_min=min(km), kmeans_max=max(km)))
    return rows


out = []
for session in ("20180126_sr_le_sp", "20171215_sr_le_sp", "20180405_sr_le"):
    out += run(session)
    print(pd.DataFrame(out[-8:]).round(3).to_string(index=False), flush=True)
pd.DataFrame(out).to_csv(Path(__file__).with_name("joint_time_isi.csv"), index=False)
