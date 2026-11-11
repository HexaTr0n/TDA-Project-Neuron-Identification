"""Pearson correlation of two Gaussian-smoothed trains, computed exactly from spike pairs
with no binning, against the same correlation after 2 ms binning. Each trial becomes a
profile: its correlation with every partner. The target's own entry, and the other trial's
neuron entry, are left out of each pairwise distance, so no blank row can leak identity."""
from lab import *
from scipy.ndimage import gaussian_filter1d
df = load_data(); recs = records(df)
neurons = sorted(df.neuron_id.unique()); pos = {n: i for i, n in enumerate(neurons)}
trials = {tr: {t.neuron_id: t.spike_times for t in g.itertuples()} for tr, g in df.groupby("trial")}
last = max(trials); T = 9.94
SIG = (0.002, 0.005, 0.02, 0.05, 0.2)

def pair_sums(a, b):
    d2 = ((b[:, None] - a[None, :]) ** 2).astype(np.float32)
    return np.array([np.exp(-d2 / (4 * s ** 2)).sum() / (2 * s * np.sqrt(np.pi)) for s in SIG])

_self = {}
def self_sum(tr, n, t):
    k = (tr, n)
    if k not in _self:
        _self[k] = pair_sums(t, t)
    return _self[k]

def exact_profile(t_target, tr, skip, key=None):
    n = len(neurons)
    A = np.zeros((len(SIG), n))
    ta = t_target
    na = len(ta)
    if na < 2:
        return A
    sa = pair_sums(ta, ta)
    va = sa / T - (na / T) ** 2
    for name, tb in trials[tr].items():
        j = pos[name]
        if j == skip or len(tb) < 20:
            continue
        cross = pair_sums(ta, tb) / T - na * len(tb) / T ** 2
        vb = self_sum(tr, name, tb) / T - (len(tb) / T) ** 2
        A[:, j] = cross / np.sqrt(np.maximum(va * vb, 1e-12))
    return A

def binned_profile(t_target, tr, skip, dt=0.002):
    n = int(round(T / dt)); out = np.zeros((len(SIG), len(neurons)))
    def smooth(t, s):
        b = np.zeros(n); np.add.at(b, np.minimum((np.asarray(t) / dt).astype(int), n - 1), 1)
        return gaussian_filter1d(b, s / dt)
    for k, s in enumerate(SIG):
        fa = smooth(t_target, s); fa = fa - fa.mean(); na = np.sqrt((fa ** 2).sum())
        for name, tb in trials[tr].items():
            j = pos[name]
            if j == skip or len(tb) < 20 or na == 0:
                continue
            fb = smooth(tb, s); fb = fb - fb.mean()
            out[k, j] = (fa * fb).sum() / max(na * np.sqrt((fb ** 2).sum()), 1e-12)
    return out

def dist(A, idx):
    D2 = cdist(A, A, "sqeuclidean")
    N = len(A); S = A[np.arange(N)[:, None], idx[None, :]]
    d = np.diag(S)
    t1 = (d[:, None] - S.T) ** 2
    t2 = (S - d[None, :]) ** 2
    same = idx[:, None] == idx[None, :]
    return np.sqrt(np.maximum(D2 - t1 - np.where(same, 0, t2), 0))

idx = np.array([pos[r.neuron] for r in recs])
rng = np.random.default_rng(0)
def shuf(r):
    return r.spike_times[0] + np.r_[0, np.cumsum(rng.permutation(r.isis))]
rows = []
for name, tar, trf in (("real", lambda r: r.spike_times, lambda r: r.trial),
                       ("target ISIs shuffled", shuf, lambda r: r.trial),
                       ("partners from next trial", lambda r: r.spike_times, lambda r: r.trial + 1 if r.trial < last else r.trial - 1)):
    P = np.stack([exact_profile(tar(r), trf(r), pos[r.neuron]) for r in recs])
    print(name, "exact done", flush=True)
    for k, s in enumerate(SIG):
        rows.append(dict(embedding=f"partner profile, exact, {name}", view=f"sigma={s * 1000:g} ms", **score(dist(P[:, k], idx), recs)))
    if name == "real":
        Bp = np.stack([binned_profile(r.spike_times, r.trial, pos[r.neuron]) for r in recs])
        for k, s in enumerate(SIG):
            rows.append(dict(embedding="partner profile, 2 ms bins, real", view=f"sigma={s * 1000:g} ms", **score(dist(Bp[:, k], idx), recs)))
            print(" exact vs binned correlation of entries, sigma", s, round(np.corrcoef(P[:, k].ravel(), Bp[:, k].ravel())[0, 1], 4), flush=True)
    show(rows[-len(SIG) * (2 if name == "real" else 1):])
save(rows, "results_correlation.csv")
