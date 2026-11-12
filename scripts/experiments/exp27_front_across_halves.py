"""Fill the gaps in the comparison: the homology-heavy methods scored by clustering and across
movie halves (retina, session 20180126_sr_le_sp, 60 neurons). Uses the experiment Rips code.

Run from scripts/experiments: python exp27_front_across_halves.py
Columns: 1-NN and k-medoids clustering on full trials; half-trials matched to the other half
of other trials (1-NN), and clustered on one half then assigned on the other (clustering).
"""
import importlib.util
from lab import *
import embeddings as em
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from barcodes import burst_bars, peak_events
spec = importlib.util.spec_from_file_location("sd_new", HERE.parent / "spike_distance.py")
sd = importlib.util.module_from_spec(spec); spec.loader.exec_module(sd)

T = 9.94
df = load_data(); recs = records(df)
rng = np.random.default_rng(0)
others = sorted(set(df.neuron_id) - {r.neuron for r in recs})
pool = [np.asarray(t.isis, float) for t in df[df.neuron_id.isin(others)].itertuples() if len(t.isis) >= 30]
pool = [pool[i] for i in rng.choice(len(pool), 300, replace=False)]
P = np.concatenate([em.isi_delay(i, m=3) for i in pool])
ANCH = KMeans(8, n_init=3, random_state=0).fit(P[rng.choice(len(P), 8000, replace=False)]).cluster_centers_


def cut(t, h):
    return t if h is None else (t[t < T / 2] if h == 0 else t[t >= T / 2] - T / 2)


def k_medoids(D, k, seed=0, restarts=5, iters=100):
    r = np.random.default_rng(seed); n = len(D); best, cost = None, np.inf
    for _ in range(restarts):
        med = [int(r.integers(n))]; near = D[med[0]].copy()
        for _ in range(k - 1):
            med.append(int(r.choice(n, p=near ** 2 / (near ** 2).sum()))); near = np.minimum(near, D[med[-1]])
        med = np.array(med)
        for _ in range(iters):
            a = D[:, med].argmin(1); new = med.copy()
            for c in range(k):
                m = np.flatnonzero(a == c)
                if len(m):
                    new[c] = m[D[np.ix_(m, m)].sum(0).argmin()]
            if np.array_equal(new, med):
                break
            med = new
        a = D[:, med].argmin(1); c_ = D[np.arange(n), med[a]].sum()
        if c_ < cost:
            best, cost = (a, med), c_
    return best


def matched(y, c):
    tab = np.zeros((y.max() + 1, c.max() + 1)); np.add.at(tab, (y, c), 1)
    r, cc = linear_sum_assignment(-tab); return tab[r, cc].sum() / len(y)


def regional(clouds, n):
    sub = subsample(clouds, n, 20); k = len(ANCH); out = {}
    lab_ = [np.linalg.norm(s[:, None] - ANCH[None], axis=2).argmin(1) for s in sub]
    for hom in (0, 1):
        regs = [s[l == r] if (l == r).sum() else np.zeros((0, s.shape[1])) for s, l in zip(sub, lab_) for r in range(k)]
        bars = rips(regs)[hom]
        al = np.concatenate([b for b in bars if len(b)] or [np.zeros(1)])
        g = np.linspace(0, np.percentile(al, 99.9) * 1.0001, 256, endpoint=False)
        F = stable_ranks(bars, g).reshape(len(clouds), 20, k, 256).mean(1).reshape(len(clouds), -1)
        out[hom] = cdist(F, F, "cityblock") * (g[1] - g[0])
    return out


def views(times, n, dur):
    V = {}; isis = [np.diff(t) for t in times]
    for tag, sh in (("", False), (", ISIs shuffled", True)):
        cl = [em.isi_delay(i, m=3, shuffle=sh) for i in isis]
        Tp = topo_distances(cl, n, 20)
        V["Delay Rips H0" + tag] = Tp["H0"]; V["Delay Rips H0+H1" + tag] = ms(Tp["H0"]) + ms(Tp["H1"])
        R = regional(cl, n)
        V["Delay regions H0" + tag] = R[0]; V["Delay regions H0+H1" + tag] = ms(R[0]) + ms(R[1])
    g = np.arange(.0025, dur + .07, .005)
    def bb(t):
        b = burst_bars(t, 2 / 30); return ((b[:, 0][None] <= g[:, None]) & (g[:, None] < b[:, 1][None])).sum(1).astype(np.float32)
    B = np.stack([bb(t) for t in times]); V["Burst bars 67 ms"] = cdist(B, B, "cityblock")
    V["Peak matching, sigma 20 ms, q 10"] = sd.matching_matrix([peak_events(t, 0.02) for t in times], 10.0, jobs=2)
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
        cl, med = k_medoids(D[np.ix_(A, A)], k)
        tab = np.zeros((k, k)); np.add.at(tab, (y[A], cl), 1)
        r, c = linear_sum_assignment(-tab); name = np.full(k, -1); name[c] = r
        out.append(np.mean(name[D[np.ix_(B, A[med])].argmin(1)] == y[B]))
    return float(np.mean(out))


rows = {}
times = [np.asarray(r.spike_times, float) for r in recs]
y = np.unique([r.neuron for r in recs], return_inverse=True)[1]
for name, D in views(times, 25, T).items():
    rows[name] = dict(full_1nn=nn(D, y), full_cluster=matched(y, k_medoids(D, y.max() + 1)[0])); print(name, rows[name], flush=True)
ok = [r for r in recs if min(len(cut(np.asarray(r.spike_times), 0)), len(cut(np.asarray(r.spike_times), 1))) > 21]
times = [cut(np.asarray(r.spike_times, float), h) for h in (0, 1) for r in ok]
y = np.unique([r.neuron for r in ok] * 2, return_inverse=True)[1]; half = np.repeat([0, 1], len(ok)); tid = np.r_[np.arange(len(ok)), np.arange(len(ok))]
cross = (half[:, None] != half[None, :]) & (tid[:, None] != tid[None, :])
for name, D in views(times, 15, T / 2).items():
    r = dict(cross_1nn=nn(D, y, cross), cross_cluster=transfer(D, y, half)); rows[name].update(r); print(name, r, flush=True)
tab = pd.DataFrame(rows).T.round(3); print(tab.to_string()); tab.to_csv(HERE / "front_across_halves.csv")
