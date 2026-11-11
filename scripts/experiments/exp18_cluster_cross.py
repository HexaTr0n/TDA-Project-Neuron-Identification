"""The exploratory embeddings again, scored by clustering and across stimuli.

full trials   cluster: k-medoids with k = neurons, share of trials in the right cluster
half trials   each trial cut in two, the halves show different parts of the movie
              cross cluster: cluster one half without labels, name each cluster after a
              neuron, then give the other half's trials to their nearest cluster centre.
              both halves cluster: all half-trials clustered together. Not a clean test:
              a method locked to movie time can still reach 50% by splitting neurons.
              cross 1-NN: nearest half-trial among the other half of the other trials.
"""
from lab import *
from extra_embeddings import *
import embeddings as em
from scipy.ndimage import gaussian_filter1d

from scipy.optimize import linear_sum_assignment


# Same two functions as in scripts/run_py.
def k_medoids(D, k, seed=0, restarts=5, iters=100):
    """k-means for a distance matrix: centres are trials, not averages. No labels used."""
    rng = np.random.default_rng(seed)
    n = len(D)
    best, best_cost = None, np.inf
    for _ in range(restarts):
        medoids = [int(rng.integers(n))]  # k-means++ style start
        near = D[medoids[0]].copy()
        for _ in range(k - 1):
            medoids.append(int(rng.choice(n, p=near ** 2 / (near ** 2).sum())))
            near = np.minimum(near, D[medoids[-1]])
        medoids = np.array(medoids)
        for _ in range(iters):
            assign = D[:, medoids].argmin(1)
            new = medoids.copy()
            for c in range(k):
                members = np.flatnonzero(assign == c)
                if len(members):
                    new[c] = members[D[np.ix_(members, members)].sum(0).argmin()]
            if np.array_equal(new, medoids):
                break
            medoids = new
        assign = D[:, medoids].argmin(1)
        cost = D[np.arange(n), medoids[assign]].sum()
        if cost < best_cost:
            best, best_cost = assign, cost
    return best

def matched_accuracy(y, clusters):
    """Share of trials in the right cluster, after pairing each cluster with one neuron
    so that as many trials as possible agree (Hungarian method)."""
    table = np.zeros((y.max() + 1, clusters.max() + 1))
    np.add.at(table, (y, clusters), 1)
    rows, cols = linear_sum_assignment(-table)
    return float(table[rows, cols].sum() / len(y))


from barcodes import peak_events

T = 9.94
df = load_data(); recs = records(df)
neurons = sorted(df.neuron_id.unique()); pos = {n: i for i, n in enumerate(neurons)}


def cut(t, h):
    t = np.asarray(t, dtype=np.float64)
    return t if h is None else (t[t < T / 2] if h == 0 else t[t >= T / 2] - T / 2)


def population(h):
    """Per trial: rfft of binned trains for the CCG, and unit-norm smoothed trains."""
    n_bins = binned(np.zeros(1)).shape[0]; size = ccg_size(n_bins)
    n_use = n_bins if h is None else n_bins // 2   # a half-trial fills half the bins
    F, cnt, S = {}, {}, {}
    for tr, g in df.groupby("trial"):
        B = np.zeros((len(neurons), n_bins), dtype=np.float32)
        for r in g.itertuples():
            B[pos[r.neuron_id]] = binned(cut(r.spike_times, h))
        F[tr] = np.fft.rfft(B, size, axis=1); cnt[tr] = B.sum(1)
        G = gaussian_filter1d(B[:, :n_use], 10, axis=1)[:, ::5]  # sigma 20 ms
        G = G - G.mean(1, keepdims=True)                         # Pearson correlation
        S[tr] = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-9)
    return F, cnt, S


def cloud_views(name, clouds, n):
    keep = np.array([i for i, c in enumerate(clouds) if c is not None and len(c) >= n])
    cl = [clouds[i] for i in keep]
    M = np.stack([c.mean(0) for c in cl])
    Tp = topo_distances(cl, n, 20)
    out = {f"{name}: position": cdist(M, M), f"{name}: whole cloud": sliced_wasserstein(cl),
           f"{name}: H0": Tp["H0"]}
    if Tp["H1"].max() > 0:
        out[f"{name}: H0+H1"] = ms(Tp["H0"]) + ms(Tp["H1"])
    return {k: (v, keep) for k, v in out.items()}


def views(items, pop, n_cloud):
    """items: (neuron, trial, half, times). Returns {name: (D, kept indices)}."""
    F, cnt, S = pop
    every = np.arange(len(items))
    times = [it[3] for it in items]; isis = [np.diff(t) for t in times]
    V = {}
    # references
    c = np.array([len(t) for t in times], float); V["ref: spike count"] = (np.abs(c[:, None] - c[None]), every)
    lg = np.linspace(np.log(1e-3), np.log(10), 400, endpoint=False)
    SR = np.stack([(logx(i)[None, :] > lg[:, None]).sum(1) for i in isis]).astype(np.float32)
    V["ref: ISI bars, log"] = (cdist(SR, SR, "cityblock"), every)
    s = 2 / 30; g = np.arange(s / 40, T + s, s / 20)
    W = np.stack([np.searchsorted(t, g, "right") - np.searchsorted(t, g - s, "right") for t in times]).astype(np.float32)
    V["ref: time bars s=67 ms"] = (cdist(W, W, "cityblock"), every)
    V["ref: time bars + log ISI bars"] = (ms(V["ref: time bars s=67 ms"][0]) + .5 * ms(V["ref: ISI bars, log"][0]), every)
    # clouds
    V.update(cloud_views("delay m=3", [em.isi_delay(i, m=3) for i in isis], n_cloud))
    V.update(cloud_views("pdf per spike", [em.predictive_state(i, L=2, bandwidth=1.) for i in isis], n_cloud))
    V.update(cloud_views("fitted lognormal", [fitted_lognormal(i, W=15) for i in isis], min(n_cloud, 5)))
    V.update(cloud_views("fitted gamma", [fitted_gamma(i, W=15) for i in isis], min(n_cloud, 5)))
    sp = [windowed_spectrum(t, sigma=0.5) for t in times]
    V.update(cloud_views("spectrum windows", sp, 4))
    A = np.stack([np.nan_to_num(windowed_spectrum(t, sigma=0.5, aligned=True, min_eff=2.0)).ravel() for t in times])
    V["spectrogram, aligned to the movie"] = (cdist(A, A), every)
    # peaks of the smoothed rate with their times removed
    pk = [np.sort(peak_events(t, 0.02)[1]) for t in times]
    gm = np.linspace(0, 6, 600, endpoint=False)
    P = np.stack([(m[None, :] > gm[:, None]).sum(1) for m in pk]).astype(np.float32)
    V["rate peaks, time removed (stable rank)"] = (cdist(P, P, "cityblock"), every)
    # relation to the other neurons of the same trial
    conc, prof = [], []
    for nrn, tr, h, t in items:
        keep = cnt[tr] >= 10; keep[pos[nrn]] = False
        cm = ccg_matrix(binned(t), F[tr], cnt[tr])
        conc.append(np.where(keep[:, None], cm, 0.0).ravel().astype(np.float32))
        p = S[tr][pos[nrn]] @ S[tr].T; p[pos[nrn]] = 0; prof.append(p)
    C = np.stack(conc); V["cross-correlograms with partners"] = (cdist(C, C), every)
    Pm = np.stack(prof); me = np.array([pos[it[0]] for it in items])
    D2 = cdist(Pm, Pm, "sqeuclidean")
    a = Pm[np.arange(len(items))[:, None], me[None, :]] ** 2    # a[x, y] = P_x[neuron of y]
    D2 = D2 - np.where(me[:, None] != me[None, :], a + a.T, 0)   # drop both self entries
    V["correlation with partners, 20 ms"] = (np.sqrt(np.maximum(D2, 0)), every)
    V["partners + log ISI bars"] = (ms(V["correlation with partners, 20 ms"][0]) + .5 * ms(V["ref: ISI bars, log"][0]), every)
    return V


def transfer(D, y, half):
    """Cluster one half without labels, name each cluster after one neuron, then put the
    other half's trials in the cluster of their nearest centre. Share that end up right."""
    out = []
    for h in (0, 1):
        A, B = np.flatnonzero(half == h), np.flatnonzero(half != h)
        k = y.max() + 1
        cl = k_medoids(D[np.ix_(A, A)], k)
        med = np.array([A[cl == c][D[np.ix_(A[cl == c], A[cl == c])].sum(0).argmin()] if (cl == c).any() else A[0]
                        for c in range(k)])
        table = np.zeros((k, k)); np.add.at(table, (y[A], cl), 1)
        rows_, cols_ = linear_sum_assignment(-table)
        name = np.full(k, -1); name[cols_] = rows_
        out.append(np.mean(name[D[np.ix_(B, med)].argmin(1)] == y[B]))
    return float(np.mean(out))


def nn(D, y):
    D = D.copy(); np.fill_diagonal(D, np.inf); return np.mean(y[D.argmin(1)] == y)


rows = {}
# ---- full trials
items = [(r.neuron, r.trial, None, np.asarray(r.spike_times, float)) for r in recs]
y = np.unique([it[0] for it in items], return_inverse=True)[1]
for name, (D, keep) in views(items, population(None), 25).items():
    yy = np.unique(y[keep], return_inverse=True)[1]
    rows[name] = dict(n=len(keep), full_1nn=nn(D, yy), full_cluster=matched_accuracy(yy, k_medoids(D, yy.max() + 1)))
    print(name, rows[name], flush=True)
# ---- half trials
ok = [r for r in recs if min(len(cut(r.spike_times, 0)), len(cut(r.spike_times, 1))) > 20]
items = [(r.neuron, r.trial, h, cut(r.spike_times, h)) for h in (0, 1) for r in ok]
y = np.unique([it[0] for it in items], return_inverse=True)[1]; half = np.array([it[2] for it in items])
trial_id = np.array([hash((it[0], it[1])) for it in items])
F0, c0, S0 = population(0); F1, c1, S1 = population(1)
pop = ({(tr, 0): v for tr, v in F0.items()} | {(tr, 1): v for tr, v in F1.items()},
       {(tr, 0): v for tr, v in c0.items()} | {(tr, 1): v for tr, v in c1.items()},
       {(tr, 0): v for tr, v in S0.items()} | {(tr, 1): v for tr, v in S1.items()})
items_k = [(it[0], (it[1], it[2]), it[2], it[3]) for it in items]
for name, (D, keep) in views(items_k, pop, 15).items():
    yy = np.unique(y[keep], return_inverse=True)[1]; hh = half[keep]
    X = D.copy(); X[hh[:, None] == hh[None, :]] = np.inf          # only the other half ...
    X[trial_id[keep][:, None] == trial_id[keep][None, :]] = np.inf  # ... of other trials
    Sm = D.copy(); Sm[hh[:, None] != hh[None, :]] = np.inf; np.fill_diagonal(Sm, np.inf)
    cl = k_medoids(D, yy.max() + 1)
    r = dict(n_half=len(keep), same_1nn=np.mean(yy[Sm.argmin(1)] == yy), cross_1nn=np.mean(yy[X.argmin(1)] == yy),
             both_halves_cluster=matched_accuracy(yy, cl), cross_cluster=transfer(D, yy, hh))
    rows.setdefault(name, {}).update(r); print(name, r, flush=True)
tab = pd.DataFrame(rows).T.round(3)
OUT.mkdir(parents=True, exist_ok=True); tab.to_csv(OUT / "cluster_cross.csv")
print(tab.to_string())
