"""Controls for the fixes that looked useful.
1. Shuffled versions of the regional and anchored stable ranks.
2. Does topology add anything to a baseline at its best weight? The weight is chosen on
   half of the neurons and scored on the other half, over 20 splits."""
from lab import *
import embeddings as em
from sklearn.cluster import KMeans
df = load_data(); recs = records(df)
base = np.load(OUT / "baselines.npz")
hist, vp = base["ISI_histogram"], base["spike_distance_q10"]
rng = np.random.default_rng(1)
others = sorted(set(df.neuron_id) - {r.neuron for r in recs})
pool = [t.isis for t in df[df.neuron_id.isin(others)].itertuples() if len(t.isis) >= 30]
pool = [pool[i] for i in rng.choice(len(pool), 300, replace=False)]
S, K = 20, 8

def sr_dist(bars, shape, size=256):
    al = np.concatenate([b for b in bars if len(b)] or [np.zeros(1)])
    g = np.linspace(0, np.percentile(al, 99.9) * 1.0001, size, endpoint=False)
    F = stable_ranks(bars, g).reshape(*shape, size).mean(1).reshape(shape[0], -1)
    return cdist(F, F, "cityblock") * (g[1] - g[0])

def regional(sub, n, A):
    lab_ = [np.linalg.norm(s[:, None] - A[None], axis=2).argmin(1) for s in sub]
    regs = [s[l == r] if (l == r).sum() else np.zeros((0, s.shape[1])) for s, l in zip(sub, lab_) for r in range(len(A))]
    H0, H1, _ = rips(regs)
    return sr_dist(H0, (n, S, len(A))), sr_dist(H1, (n, S, len(A)))

def weighted(topo, baseD, label):
    """Paired held-out gain in 1-NN accuracy from adding topo to baseD."""
    y = np.array([r.neuron for r in recs]); names = np.unique(y)
    lams = [0, .02, .05, .1, .2, .5, 1]
    b, t = ms(baseD), ms(topo)
    def acc(lam, mask):
        D = (b + lam * t)[np.ix_(mask, mask)].copy(); np.fill_diagonal(D, np.inf)
        return (y[mask][D.argmin(1)] == y[mask]).mean()
    gains = []; chosen = []
    r = np.random.default_rng(0)
    for _ in range(20):
        p = r.permutation(names); A, B = np.isin(y, p[:30]), np.isin(y, p[30:])
        best = max(lams, key=lambda l: acc(l, A))
        gains.append(acc(best, B) - acc(0, B)); chosen.append(best)
    return dict(pair=label, gain=np.mean(gains), sd=np.std(gains), median_weight=np.median(chosen))

out = []; wrows = []
shuf = [rng.permutation(r.isis) for r in recs]
for nm, f in (("delay m=3", lambda x: em.isi_delay(x, m=3)),
              ("pdf L=2 W2 b=1", lambda x: em.predictive_state(x, L=2, bandwidth=1.))):
    pts = np.concatenate([f(x) for x in pool])
    A = KMeans(K, n_init=3, random_state=0).fit(pts[rng.choice(len(pts), 8000, replace=False)]).cluster_centers_
    for label, xs in (("", [r.isis for r in recs]), (" shuffled", shuf)):
        cl = [f(x) for x in xs]; sub = subsample(cl, 25, S)
        R0, R1 = regional(sub, len(cl), A)
        a0, a1, _ = rips([np.vstack([c, A]) for c in sub])
        An0 = sr_dist(a0, (len(cl), S)); An1 = sr_dist(a1, (len(cl), S))
        for v, D in (("regional H0", R0), ("regional H1", R1), ("regional H0+H1", ms(R0) + ms(R1)),
                     ("anchored H0", An0), ("anchored H1", An1)):
            out.append(dict(embedding=f"{nm}{label} k={K}", view=v, **score(D, recs)))
        if not label:
            T = topo_distances(cl, 25, S)
            for tn, D in (("H0", T["H0"]), ("H0+H1", ms(T["H0"]) + ms(T["H1"])), ("regional H0", R0), ("anchored H0", An0)):
                for bn, B in (("histogram", hist), ("spike distance", vp)):
                    wrows.append(weighted(D, B, f"{bn} + {nm} {tn}"))
        show(out[-5:])
save(out, "results.csv")
w = pd.DataFrame(wrows); w.to_csv(OUT / "weighted.csv", index=False)
print(w.round(4).to_string(index=False))
