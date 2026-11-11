"""Sublevel-set persistence of one trial's firing rate.
r(t) is the Gaussian-smoothed train. Every peak is a bar: (time, prominence). The mass
m = prominence * sigma * sqrt(2 pi) is the peak's size in spikes, so an isolated spike has
mass close to 1. Trials are compared by time-ordered matching, which is the spike distance
with spikes replaced by events: with unit masses the two are identical."""
from lab import *
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences
from tda import ordered_match
DT = 0.001; T = 9.94; MIN_MASS = 0.2
df = load_data(); recs = records(df)
base = np.load(OUT / "baselines.npz")
hist = base["ISI_histogram"]; vp = base["spike_distance_q10"]

def rate(t, sigma):
    n = int(round(T / DT)); b = np.zeros(n)
    np.add.at(b, np.minimum((np.asarray(t) / DT).astype(int), n - 1), 1 / DT)
    return gaussian_filter1d(b, sigma / DT, mode="constant")

def diagram(t, sigma):
    r = rate(t, sigma)
    pk = find_peaks(np.r_[0, r, 0])[0] - 1
    if len(pk) == 0:
        return np.zeros(0), np.zeros(0)
    pr = peak_prominences(np.r_[0, r, 0], pk + 1)[0]
    pr[np.argmax(r[pk])] = r[pk].max() - r.min()  # the highest peak never dies
    m = pr * sigma * np.sqrt(2 * np.pi)
    k = m >= MIN_MASS
    return pk[k] * DT, m[k]

def shuf(r, rng):
    return r.spike_times[0] + np.r_[0, np.cumsum(rng.permutation(r.isis))]

rows = []
rng = np.random.default_rng(0)
times = {"": [r.spike_times for r in recs], " shuffled": [shuf(r, rng) for r in recs]}
for sigma in (0.005, 0.01, 0.02, 0.05, 0.1):
    for lab_, ts in times.items():
        ev = [diagram(t, sigma) for t in ts]
        name = f"rate sigma={sigma * 1000:g} ms{lab_}"
        F = np.stack([rate(t, sigma)[::max(1, int(sigma / DT / 2))] for t in ts])
        vr = cdist(F, F)
        rows.append(dict(embedding=name, view="van Rossum, no topology", n_events=np.median([len(e[0]) for e in ev]), **score(vr, recs)))
        bars = [e[1] for e in ev]
        al = np.concatenate([b for b in bars if len(b)])
        g = np.linspace(0, np.percentile(al, 99.9), 512, endpoint=False)
        S = stable_ranks(bars, g); sr = cdist(S, S, "cityblock") * (g[1] - g[0])
        rows.append(dict(embedding=name, view="stable rank, time lost", n_events=np.median([len(e[0]) for e in ev]), **score(sr, recs)))
        for q in (3, 10, 30):
            D = ordered_match(ev, q)
            rows.append(dict(embedding=name, view=f"peak matching q={q}", n_events=np.median([len(e[0]) for e in ev]), **score(D, recs)))
            if not lab_:
                rows.append(dict(embedding=name, view=f"peak matching q={q}+0.5 histogram", n_events=np.median([len(e[0]) for e in ev]), **score(ms(D) + 0.5 * ms(hist), recs)))
                np.save(OUT / f"peak_sigma{sigma * 1000:g}_q{q}.npy", D.astype(np.float32))
        show(rows[-(5 if lab_ else 8):])
save(rows, "results_sublevel.csv")
