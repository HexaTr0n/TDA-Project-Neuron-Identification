"""One change at a time to the pdf embedding, then fixes for global information loss."""
from lab import *
import embeddings as em
from sklearn.cluster import KMeans
tag = sys.argv[1]
df = load_data(); recs = records(df)
base = np.load(OUT / "baselines.npz")
hist = base["ISI_histogram"]
EX = {"histogram": hist, "spike distance": base["spike_distance_q10"]}

def pdf(rs=recs, **k):
    return [em.predictive_state(r.isis, **k) for r in rs]

def run(name, clouds, **k):
    rows = cloud_rows(name, clouds, recs, **k); save(rows, "results.csv"); show(rows); return rows

if tag == "bandwidth":
    for b in (0.25, 0.5, 1.0, 2.0):
        run(f"pdf L=2 W2 b={b}", pdf(L=2, bandwidth=b), extra=EX)
    run("pdf L=2 W2 b=0.5 shuffled", pdf(L=2, shuffle=True), extra=EX)
    run("pdf L=2 W2 b=2 shuffled", pdf(L=2, bandwidth=2., shuffle=True), extra=EX)
    run("pdf L=1 W2 b=1", pdf(L=1, bandwidth=1.), extra=EX)
    run("pdf L=1 W2 b=1 shuffled", pdf(L=1, bandwidth=1., shuffle=True), extra=EX)
    run("pdf L=2 Hellinger b=1", pdf(L=2, bandwidth=1., distance="hellinger"), extra=EX)
    run("pdf L=2 Hellinger b=1 shuffled", pdf(L=2, bandwidth=1., distance="hellinger", shuffle=True), extra=EX)

if tag == "size":
    keep = [i for i, r in enumerate(recs) if len(r.isis) >= 110]
    rs = [recs[i] for i in keep]
    names = pd.Series([r.neuron for r in rs]).value_counts()
    keep = [i for i in keep if names[recs[i].neuron] >= 2]
    rs = [recs[i] for i in keep]
    print("trials", len(rs), "neurons", len(set(r.neuron for r in rs)))
    idx = np.array(keep); h = hist[np.ix_(idx, idx)]
    rows = [dict(embedding="same trials", view="ISI histogram", **score(h, rs))]
    for nm, cl in (("pdf L=2 W2 b=1", [em.predictive_state(r.isis, L=2, bandwidth=1.) for r in rs]),
                   ("delay m=3", [em.isi_delay(r.isis, m=3) for r in rs])):
        for n, S in ((25, 20), (50, 20), (100, 8)):
            T = topo_distances(cl, n, S)
            for k, D in T.items():
                rows.append(dict(embedding=f"{nm} n={n}", view=k, **score(D, rs)))
            if T["H1"].max() > 0:
                rows.append(dict(embedding=f"{nm} n={n}", view="H0+H1", **score(ms(T["H0"]) + ms(T["H1"]), rs)))
        show(rows[-9:])
    save(rows, "results.csv")

if tag == "global":
    rng = np.random.default_rng(1)
    others = sorted(set(df.neuron_id) - {r.neuron for r in recs})
    pool_recs = [Rec(n, t.trial, t.isis, t.spike_times) for t in df[df.neuron_id.isin(others)].itertuples() for n in [t.neuron_id] if len(t.isis) >= 30]
    pool_recs = [pool_recs[i] for i in rng.choice(len(pool_recs), 300, replace=False)]
    for nm, f in (("delay m=3", lambda r: em.isi_delay(r.isis, m=3)),
                  ("pdf L=2 W2 b=1", lambda r: em.predictive_state(r.isis, L=2, bandwidth=1.))):
        pts = np.concatenate([f(r) for r in pool_recs])
        clouds = [f(r) for r in recs]
        rows = []
        for k in (4, 8):
            A = KMeans(k, n_init=3, random_state=0).fit(pts[rng.choice(len(pts), 8000, replace=False)]).cluster_centers_
            # Anchored: fixed reference points join every cloud, so position changes the bars.
            sub = subsample(clouds, 25, 20)
            H0, H1, _ = rips([np.vstack([c, A]) for c in sub])
            Dd = {}
            for name, bars in (("H0", H0), ("H1", H1)):
                al = np.concatenate([b for b in bars if len(b)] or [np.zeros(1)])
                g = np.linspace(0, np.percentile(al, 99.9) * 1.0001, 512, endpoint=False)
                F = stable_ranks(bars, g).reshape(len(clouds), 20, 512).mean(1)
                Dd[name] = cdist(F, F, "cityblock") * (g[1] - g[0])
            for a, D in (("H0", Dd["H0"]), ("H1", Dd["H1"]), ("H0+H1", ms(Dd["H0"]) + ms(Dd["H1"]))):
                rows.append(dict(embedding=f"{nm} anchored k={k}", view=a, **score(D, recs)))
            # Regional: one stable rank per region around the anchors, so we keep where.
            lab_ = [np.linalg.norm(s[:, None] - A[None], axis=2).argmin(1) for s in sub]
            for hom in (0, 1):
                regs = [s[l == r] if (l == r).sum() else np.zeros((0, s.shape[1])) for s, l in zip(sub, lab_) for r in range(k)]
                bars = rips(regs)[hom]
                al = np.concatenate([b for b in bars if len(b)] or [np.zeros(1)])
                g = np.linspace(0, np.percentile(al, 99.9) * 1.0001, 256, endpoint=False)
                F = stable_ranks(bars, g).reshape(len(clouds), 20, k, 256).mean(1).reshape(len(clouds), -1)
                Dd[f"R{hom}"] = cdist(F, F, "cityblock") * (g[1] - g[0])
            rows.append(dict(embedding=f"{nm} regional k={k}", view="H0", **score(Dd["R0"], recs)))
            rows.append(dict(embedding=f"{nm} regional k={k}", view="H1", **score(Dd["R1"], recs)))
            rows.append(dict(embedding=f"{nm} regional k={k}", view="H0+H1", **score(ms(Dd["R0"]) + ms(Dd["R1"]), recs)))
            show(rows[-6:])
        save(rows, "results.csv")
