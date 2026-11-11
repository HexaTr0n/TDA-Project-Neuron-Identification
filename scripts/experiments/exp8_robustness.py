"""Is the small gain from topology real structure, or the ISI distribution again?
Also repeats the key comparisons with other subsampling seeds and on all neurons."""
from lab import *
import embeddings as em
df = load_data(); recs = records(df)
base = np.load(OUT / "baselines.npz")
hist, vp = base["ISI_histogram"], base["spike_distance_q10"]
y = np.array([r.neuron for r in recs]); names = np.unique(y)
rng = np.random.default_rng(5)
shuf = [rng.permutation(r.isis) for r in recs]

def gain(topo, baseD):
    lams = [0, .02, .05, .1, .2, .5, 1]
    b, t = ms(baseD), ms(topo)
    def acc(lam, mask):
        D = (b + lam * t)[np.ix_(mask, mask)].copy(); np.fill_diagonal(D, np.inf)
        return (y[mask][D.argmin(1)] == y[mask]).mean()
    r = np.random.default_rng(0); g = []; c = []
    for _ in range(20):
        p = r.permutation(names); A, B = np.isin(y, p[:30]), np.isin(y, p[30:])
        best = max(lams, key=lambda l: acc(l, A)); g.append(acc(best, B) - acc(0, B)); c.append(best)
    return round(np.mean(g), 4), round(np.std(g), 4), np.median(c)

T = topo_distances([em.isi_delay(r.isis, m=3) for r in recs], 25, 20)
Ts = topo_distances([em.isi_delay(x, m=3) for x in shuf], 25, 20)
print("spike distance + histogram", gain(hist, vp), flush=True)
print("histogram + spike distance", gain(vp, hist), flush=True)
print("spike distance + real H0", gain(T["H0"], vp), flush=True)
print("spike distance + shuffled H0", gain(Ts["H0"], vp), flush=True)
print("histogram + real H0", gain(T["H0"], hist), flush=True)
print("histogram + shuffled H0", gain(Ts["H0"], hist), flush=True)
D = ms(vp) + 0.5 * ms(hist)
print("spike distance + 0.5 histogram, then real H0", gain(T["H0"], D), flush=True)
print("spike distance + 0.5 histogram, then shuffled H0", gain(Ts["H0"], D), flush=True)
rows = [dict(embedding="combined", view="spike distance+0.5 histogram", **score(D, recs))]
for seed in (1, 2, 3):
    for lab_, cl in (("real", [em.isi_delay(r.isis, m=3) for r in recs]), ("shuffled", [em.isi_delay(x, m=3) for x in shuf])):
        t = topo_distances(cl, 25, 20, seed=seed)
        for k, d in (("H0", t["H0"]), ("H1", t["H1"]), ("H0+H1", ms(t["H0"]) + ms(t["H1"]))):
            rows.append(dict(embedding=f"delay m=3 {lab_} seed={seed}", view=k, **score(d, recs)))
show(rows); save(rows, "results.csv")
# All neurons
allr = records(df, n_neurons=None)
print("all neurons", len(allr), len({r.neuron for r in allr}), flush=True)
bl = [[r] for r in allr]
rows = [dict(embedding="all neurons", view="ISI histogram", **score(rp.baseline_isi_histogram(bl), allr)),
        dict(embedding="all neurons", view="spike count", **score(rp.baseline_spike_count(bl), allr))]
for nm, f in (("delay m=3", lambda r: em.isi_delay(r, m=3)),
              ("pdf L=2 Hellinger b=1", lambda r: em.predictive_state(r, L=2, bandwidth=1., distance="hellinger"))):
    r2 = np.random.default_rng(6)
    for lab_, xs in (("", [r.isis for r in allr]), (" shuffled", [r2.permutation(r.isis) for r in allr])):
        rows += cloud_rows(f"all neurons {nm}{lab_}", [f(x) for x in xs], allr)
        show(rows[-7:])
save(rows, "results.csv")
