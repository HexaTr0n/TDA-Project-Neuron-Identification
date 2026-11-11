"""Seven more piecewise constant signatures, and k-means with mean curves as centres.

Run from the repo root: python scripts/experiments/exp22_pcf_ideas.py
Every signature is a function sampled on a grid, compared by L1, as a PCF would be in stablebear.
Scores: clustering on full trials (k-medoids, and k-means with mean curves), and half-trials
matched across movie halves (different stimulus).
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
csv = REPO / "data" / "isi_frozennoise.csv"
df = load_dataset(csv if csv.exists() else rp.DEFAULT_PATH, sessions=["20180126_sr_le_sp"])
blocks = rp.pick_neurons(rp.make_blocks(df, rp.Config()), 60, 0)
by_trial = {tr: {r.neuron_id: np.asarray(r.spike_times, float) for r in g.itertuples()} for tr, g in df.groupby("trial")}
LAST = max(by_trial)
rng = np.random.default_rng(0)
LG = np.linspace(np.log(1e-4), np.log(10), 230, endpoint=False)
LOG = lambda x: np.log(np.maximum(x, 1e-4))


def cut(t, h):
    return t if h is None else (t[t < T / 2] if h == 0 else t[t >= T / 2] - T / 2)


def frac(values, grid=LG):
    v = np.asarray(values)
    return (v[None, :] > grid[:, None]).sum(1).astype(np.float32) / max(len(v), 1)


def window(t, s, dur):
    g = np.arange(s / 40, dur + s, s / 20)
    return (np.searchsorted(t, g, "right") - np.searchsorted(t, g - s, "right")).astype(np.float32)


def current_isi(t, dur, dt=0.005):
    """At each moment, the log length of the gap between spikes that contains it."""
    g = np.arange(dt / 2, dur, dt); e = np.r_[0.0, t, dur]
    return LOG(np.diff(e))[np.clip(np.searchsorted(e, g, "right") - 1, 0, len(e) - 2)].astype(np.float32)


def burst_size(t, s, dur, dt=0.005):
    """At each moment, the number of spikes in the burst alive then (0 between bursts)."""
    g = np.arange(dt / 2, dur + s, dt); c = np.flatnonzero(np.diff(t) > s)
    first, last = t[np.r_[0, c + 1]], t[np.r_[c, len(t) - 1]] + s
    size = np.diff(np.r_[0, c + 1, len(t)])
    k = np.searchsorted(first, g, "right") - 1
    return np.where((k >= 0) & (g < last[np.maximum(k, 0)]), size[np.maximum(k, 0)], 0).astype(np.float32)


def others(neuron, trial, h):
    return np.sort(np.concatenate([cut(s, h) for n, s in by_trial[trial].items() if n != neuron]))


S_GRID = np.geomspace(0.001, 0.1, 14)
def coincidences(t, pop, s):
    return np.searchsorted(pop, t + s, "right") - np.searchsorted(pop, t - s, "left")


def features(items, dur):
    """items: (neuron, trial, half, times). Returns {name: feature matrix}, compared by L1."""
    F = {}; S = lambda rows: np.stack(rows).astype(np.float32)
    times = [it[3] for it in items]; isis = [np.diff(t) for t in times]
    F["ref: time bars 67 ms"] = S([window(t, 2 / 30, dur) for t in times])
    F["ref: ISI fractions"] = S([frac(LOG(i)) for i in isis])
    F["1. square-root time bars 67 ms"] = np.sqrt(F["ref: time bars 67 ms"])
    F["1. square-root time bars 200 ms"] = S([np.sqrt(window(t, .2, dur)) for t in times])
    F["2. multi-scale time bars 20+67+200 ms"] = [S([window(t, s, dur) for t in times]) for s in (.02, 2 / 30, .2)]
    F["2. multi-scale, square root"] = [np.sqrt(x) for x in F["2. multi-scale time bars 20+67+200 ms"]]
    F["3. current-ISI function"] = S([current_isi(t, dur) for t in times])
    F["4. burst-size function 67 ms (square root)"] = S([np.sqrt(burst_size(t, 2 / 30, dur)) for t in times])
    def cond(i, key):
        a, b = i[1:][key], i[1:][~key]
        return np.r_[frac(LOG(a)), frac(LOG(b))]
    F["5. ISI fractions after a short / after a long ISI"] = S([cond(i, i[:-1] < np.median(i)) for i in isis])
    F["5. control: ISIs shuffled"] = S([cond(j, j[:-1] < np.median(j)) for j in (rng.permutation(i) for i in isis)])
    pops = [others(n, tr, h) for n, tr, h, t in items]
    nxt = [others(n, (tr + 1 if tr < LAST else tr - 1), h) for n, tr, h, t in items]
    busy = [coincidences(t[:-1], p, .05) for t, p in zip(times, pops)]
    F["6. ISI fractions in busy / quiet moments of the population"] = S([cond(np.r_[0, i], b >= np.median(b)) for i, b in zip(isis, busy)])
    F["6. control: moments split at random"] = S([cond(np.r_[0, i], rng.permutation(b >= np.median(b))) for i, b in zip(isis, busy)])
    # 7. coincidences with the rest of the population
    def over_time(t, p, s=0.01):
        w = coincidences(t, p, s); g = np.arange(.0025, dur + .067, .005)
        cs = np.r_[0, np.cumsum(w)]
        return (cs[np.searchsorted(t, g, "right")] - cs[np.searchsorted(t, g - 2 / 30, "right")]).astype(np.float32)
    F["7a. coincidence function over time (10 ms)"] = S([np.sqrt(over_time(t, p)) for t, p in zip(times, pops)])
    grow = lambda t, p: np.array([coincidences(t, p, s).sum() for s in S_GRID]) / (len(t) * len(p) * 2 * S_GRID / dur)
    same = S([grow(t, p) for t, p in zip(times, pops)]); other = S([grow(t, p) for t, p in zip(times, nxt)])
    F["7b. coincidences as the radius grows (1 = chance)"] = same
    F["7c. same, minus the next trial's population"] = same - other
    exc = [coincidences(t, p, .01) - coincidences(t, q, .01) for t, p, q in zip(times, pops, nxt)]
    eg = np.linspace(-30, 30, 121)
    F["7d. excess coincidences per spike, as fractions"] = S([frac(e, eg) for e in exc])
    F["ref: time bars + ISI fractions"] = [F["ref: time bars 67 ms"], F["ref: ISI fractions"]]
    F["1+ISI: square-root time bars + ISI fractions"] = [F["1. square-root time bars 67 ms"], F["ref: ISI fractions"]]
    F["3+ISI: current-ISI function + ISI fractions"] = [F["3. current-ISI function"], F["ref: ISI fractions"]]
    F["5+7c+ISI: stimulus-free mix"] = [F["ref: ISI fractions"], F["5. ISI fractions after a short / after a long ISI"], F["7c. same, minus the next trial's population"]]
    out = {}
    for k, v in F.items():                       # parts of a mix are scaled so that L1 adds them fairly
        if isinstance(v, list):
            parts = []
            for j, x in enumerate(v):
                idx = rng.choice(len(x), min(300, len(x)), replace=False)
                parts.append(x / np.median(cdist(x[idx], x[idx], "cityblock")) * (1.0 if j == 0 or len(v) > 2 else 0.5))
            v = np.hstack(parts)
        out[k] = v
    return out


def nn(D, y, allowed=None):
    D = D.copy(); np.fill_diagonal(D, np.inf)
    if allowed is not None:
        D[~allowed] = np.inf
    return np.mean(y[D.argmin(1)] == y)


def k_means_l1(X, k, restarts=3, iters=25, seed=0):
    """Centres are mean curves (what stablebear's mean of PCFs gives), assignment by L1."""
    r = np.random.default_rng(seed); best, cost = None, np.inf
    for _ in range(restarts):
        C = X[r.choice(len(X), k, replace=False)].copy()
        for _ in range(iters):
            D = cdist(X, C, "cityblock"); a = D.argmin(1)
            new = np.stack([X[a == c].mean(0) if (a == c).any() else X[r.integers(len(X))] for c in range(k)])
            if np.allclose(new, C):
                break
            C = new
        D = cdist(X, C, "cityblock"); a = D.argmin(1); c_ = D.min(1).sum()
        if c_ < cost:
            best, cost = a, c_
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


rows = {}
items = [(n, tr[0].trial, None, np.asarray(tr[0].spike_times, float)) for n, tr in blocks]
y = np.unique([it[0] for it in items], return_inverse=True)[1]; k = y.max() + 1
for name, X in features(items, T).items():
    D = cdist(X, X, "cityblock")
    rows[name] = dict(full_1nn=nn(D, y), cluster_medoids=rp.matched_accuracy(y, rp.k_medoids(D, k)),
                      cluster_means=rp.matched_accuracy(y, k_means_l1(X, k)))
    print(name, {a: round(float(b), 3) for a, b in rows[name].items()}, flush=True)
ok = [(n, tr[0]) for n, tr in blocks if min(len(cut(np.asarray(tr[0].spike_times), 0)), len(cut(np.asarray(tr[0].spike_times), 1))) > 21]
items = [(n, r.trial, h, cut(np.asarray(r.spike_times, float), h)) for h in (0, 1) for n, r in ok]
y = np.unique([it[0] for it in items], return_inverse=True)[1]; half = np.array([it[2] for it in items])
tid = np.array([hash((it[0], it[1])) for it in items])
same = half[:, None] == half[None, :]; cross = ~same & (tid[:, None] != tid[None, :])
for name, X in features(items, T / 2).items():
    D = cdist(X, X, "cityblock")
    r = dict(cross_1nn=nn(D, y, cross), cross_cluster=transfer(D, y, half))
    rows[name].update(r); print(name, {a: round(float(b), 3) for a, b in r.items()}, flush=True)
tab = pd.DataFrame(rows).T.round(3)
print(f"\n{len(blocks)} trials, {len(items)} half-trials, 60 neurons"); print(tab.to_string())
tab.to_csv(Path(__file__).with_name("pcf_ideas.csv"))
