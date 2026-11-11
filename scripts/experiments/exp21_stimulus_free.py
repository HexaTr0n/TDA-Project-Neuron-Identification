"""Three signatures that do not use movie time, scored by clustering and across stimuli.

Run from the repo root: python scripts/experiments/exp21_stimulus_free.py

1. ISI bars in rescaled time: the clock is the firing of all other neurons on the same trial,
   so speed-ups that the whole population shares cancel.
2. Burst structure without times: at each scale s, how many bursts hold at least m spikes.
3. Persistence of the autocorrelogram: its peaks and dips as bars.
Scores as in exp18: clustering on full trials, and half-trials matched across movie halves.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, peak_prominences
from scipy.spatial.distance import cdist
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import run_pipeline as rp
from load_dataset import load_dataset
from spike_distance import matching_matrix

T = 9.94
csv = REPO / "data" / "isi_frozennoise.csv"
df = load_dataset(csv if csv.exists() else rp.DEFAULT_PATH, sessions=["20180126_sr_le_sp"])
blocks = rp.pick_neurons(rp.make_blocks(df, rp.Config()), 60, 0)
by_trial = {tr: {r.neuron_id: np.asarray(r.spike_times, float) for r in g.itertuples()} for tr, g in df.groupby("trial")}
rng = np.random.default_rng(0)


def cut(t, h):
    return t if h is None else (t[t < T / 2] if h == 0 else t[t >= T / 2] - T / 2)


def stable_rank(values, grid, normalise=False):
    s = (np.asarray(values)[None, :] > grid[:, None]).sum(1).astype(np.float32)
    return s / max(len(values), 1) if normalise else s


LG = np.linspace(np.log(1e-4), np.log(10), 460, endpoint=False)
LOG = lambda x: np.log(np.maximum(x, 1e-4))


def rescaled(t, neuron, trial, h, dur):
    """Spike times on the population's clock: equal steps = equal numbers of other neurons' spikes."""
    pop = np.sort(np.concatenate([cut(s, h) for n, s in by_trial[trial].items() if n != neuron]))
    return np.interp(t, pop, np.arange(len(pop))) / len(pop) * dur


SCALES = np.geomspace(0.002, 1.0, 20); SIZES = np.array([1, 2, 3, 4, 6, 8, 12, 16])
def burst_table(t):
    """B[s, m]: number of bursts at scale s (gaps of at most s) with at least m spikes."""
    d = np.diff(t); out = np.zeros((len(SCALES), len(SIZES)), np.float32)
    for i, s in enumerate(SCALES):
        edges = np.r_[0, np.flatnonzero(d > s) + 1, len(t)]
        out[i] = (np.diff(edges)[None, :] >= SIZES[:, None]).sum(1)
    return out.ravel()


def acg(t, dur, max_lag=0.2, dt=0.001):
    """Autocorrelogram, 1 for a Poisson train of the same rate."""
    j = np.searchsorted(t, t + max_lag)
    lags = np.concatenate([t[i + 1:j[i]] - t[i] for i in range(len(t))]) if len(t) > 1 else np.zeros(0)
    c = np.histogram(lags, np.arange(0, max_lag + dt / 2, dt))[0].astype(float)
    return gaussian_filter1d(c, 2, mode="nearest") / max(len(t) ** 2 * dt / dur, 1e-9)


PG = np.linspace(0, 6, 120, endpoint=False)
def acg_bars(a):
    pk = find_peaks(a)[0]; dp = find_peaks(-a)[0]
    pp = peak_prominences(a, pk)[0] if len(pk) else np.zeros(0)
    dd = peak_prominences(-a, dp)[0] if len(dp) else np.zeros(0)
    return np.r_[stable_rank(pp, PG), stable_rank(dd, PG)], (pk[pp >= .3] * 0.001, pp[pp >= .3])


def views(items, dur):
    """items: (neuron, trial, half, times)."""
    V = {}; L1 = lambda F: cdist(np.stack(F), np.stack(F), "cityblock")
    times = [it[3] for it in items]; isis = [np.diff(t) for t in times]
    V["ref: spike count"] = L1([[len(t)] for t in times])
    V["ref: log ISI bars"] = L1([stable_rank(LOG(i), LG) for i in isis])
    V["log ISI bars, as fractions"] = L1([stable_rank(LOG(i), LG, True) for i in isis])
    V["log ISI bars, ISI / mean ISI, fractions"] = L1([stable_rank(LOG(i / i.mean() * .1), LG, True) for i in isis])
    res = [np.diff(rescaled(t, n, tr, h, dur)) for n, tr, h, t in items]
    V["1. rescaled-time ISI bars"] = L1([stable_rank(LOG(i), LG) for i in res])
    V["1. rescaled-time ISI bars, as fractions"] = L1([stable_rank(LOG(i), LG, True) for i in res])
    V["2. burst table"] = L1([burst_table(t) for t in times])
    V["2. burst table, as fractions"] = L1([burst_table(t) / len(t) for t in times])
    sh = [t[0] + np.r_[0, np.cumsum(rng.permutation(np.diff(t)))] for t in times]
    V["2. burst table, as fractions, ISIs shuffled"] = L1([burst_table(t) / len(t) for t in sh])
    A = [acg(t, dur) for t in times]; B = [acg_bars(a) for a in A]
    V["3. autocorrelogram curve (no topology)"] = cdist(np.stack(A), np.stack(A))
    V["3. autocorrelogram bars, lags dropped"] = L1([b[0] for b in B])
    V["3. autocorrelogram peaks matched in lag"] = matching_matrix([b[1] for b in B], 50.0, jobs=2)
    f = lambda k: rp.scaled(V[k])
    V["1 + 2 + 3 (fractions, bars)"] = f("1. rescaled-time ISI bars, as fractions") + f("2. burst table, as fractions") + f("3. autocorrelogram bars, lags dropped")
    V["fractions + burst table fractions"] = f("log ISI bars, as fractions") + f("2. burst table, as fractions")
    return V


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
        table = np.zeros((k, k)); np.add.at(table, (y[A], cl), 1)
        r, c = linear_sum_assignment(-table); name = np.full(k, -1); name[c] = r
        out.append(np.mean(name[D[np.ix_(B, med)].argmin(1)] == y[B]))
    return float(np.mean(out))


rows = {}
items = [(n, tr[0].trial, None, np.asarray(tr[0].spike_times, float)) for n, tr in blocks]
y = np.unique([it[0] for it in items], return_inverse=True)[1]
for name, D in views(items, T).items():
    rows[name] = dict(full_1nn=nn(D, y), full_cluster=rp.matched_accuracy(y, rp.k_medoids(D, y.max() + 1)))
    print(name, rows[name], flush=True)
ok = [(n, tr[0]) for n, tr in blocks if min(len(cut(np.asarray(tr[0].spike_times), 0)), len(cut(np.asarray(tr[0].spike_times), 1))) > 21]
items = [(n, r.trial, h, cut(np.asarray(r.spike_times, float), h)) for h in (0, 1) for n, r in ok]
y = np.unique([it[0] for it in items], return_inverse=True)[1]; half = np.array([it[2] for it in items])
tid = np.array([hash((it[0], it[1])) for it in items])
same = half[:, None] == half[None, :]; cross = ~same & (tid[:, None] != tid[None, :])
for name, D in views(items, T / 2).items():
    r = dict(same_1nn=nn(D, y, same), cross_1nn=nn(D, y, cross), cross_cluster=transfer(D, y, half))
    rows[name].update(r); print(name, r, flush=True)
tab = pd.DataFrame(rows).T.round(3)
print(f"\n{len(blocks)} trials, {len(items)} half-trials, 60 neurons"); print(tab.to_string())
tab.to_csv(Path(__file__).with_name("stimulus_free.csv"))
