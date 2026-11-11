"""Spectral and cross-correlation embeddings, each with its controls."""
from lab import *
from extra_embeddings import *
tag = sys.argv[1]
df = load_data(); recs = records(df)
base = np.load(OUT / "baselines.npz")
EX = {"histogram": base["ISI_histogram"], "spike distance": base["spike_distance_q10"]}

def shuffled_times(r, rng):
    if len(r.isis) == 0:
        return r.spike_times
    return r.spike_times[0] + np.r_[0, np.cumsum(rng.permutation(r.isis))]

if tag == "spectrum":
    done = set(pd.read_csv(OUT / "results.csv").embedding) if (OUT / "results.csv").exists() else set()
    for sigma, n in ((0.25, 25), (0.5, 12), (1.0, 6)):
        for shuf in (False, True):
            if f"spectrum sigma={sigma}" + (" shuffled" if shuf else "") in done:
                continue
            rng = np.random.default_rng(0)
            times = [shuffled_times(r, rng) if shuf else r.spike_times for r in recs]
            cl = [windowed_spectrum(t, sigma=sigma) for t in times]
            cl = [c if c is not None and len(c) >= n else None for c in cl]
            name = f"spectrum sigma={sigma}" + (" shuffled" if shuf else "")
            print(name, "usable", sum(c is not None for c in cl), flush=True)
            rows = cloud_rows(name, cl, recs, n=n, S=20, extra=EX)
            A = np.stack([np.nan_to_num(windowed_spectrum(t, sigma=sigma, aligned=True, min_eff=2.0) if len(t) >= 5 else np.zeros((len(spectrum_grid(sigma)[0]), 24))).ravel() for t in times])
            rows.append(dict(embedding=name, view="time-aligned spectrogram, no topology", **score(cdist(A, A), recs)))
            save(rows, "results.csv"); show(rows)

if tag == "ccg":
    all_rows = df
    neurons = sorted(all_rows.neuron_id.unique()); pos = {n: i for i, n in enumerate(neurons)}
    n_bins = binned(np.zeros(1)).shape[0]; size = ccg_size(n_bins)
    F, cnt = {}, {}
    for tr, g in all_rows.groupby("trial"):
        B = np.zeros((len(neurons), n_bins), dtype=np.float32)
        for t in g.itertuples():
            B[pos[t.neuron_id]] = binned(t.spike_times)
        F[tr] = np.fft.rfft(B, size, axis=1); cnt[tr] = B.sum(1)
    print("population", len(neurons), "trials", len(F), flush=True)
    last = max(F)
    rng = np.random.default_rng(0)
    variants = {"ccg": lambda r: (binned(r.spike_times), r.trial),
                "ccg, target ISIs shuffled": lambda r: (binned(shuffled_times(r, rng)), r.trial),
                "ccg, partners from next trial": lambda r: (binned(r.spike_times), r.trial + 1 if r.trial < last else r.trial - 1)}
    for name, f in variants.items():
        vec, summ, conc = [], [], []
        for r in recs:
            b, tr = f(r)
            keep = (cnt[tr] >= 20); keep[pos[r.neuron]] = False
            c = ccg_matrix(b, F[tr], cnt[tr])
            full = np.where(keep[:, None], c, 0.0); conc.append(full.ravel().astype(np.float32))
            vec.append(c[keep].astype(np.float64)); summ.append(ccg_summary(c[keep]))
        print(name, "partners median", int(np.median([len(v) for v in vec])), flush=True)
        A = np.stack(conc)
        rows = [dict(embedding=name, view="partner-by-partner, no topology", **score(cdist(A, A), recs))]
        rows += cloud_rows(f"{name} vector cloud", vec, recs, n=25, S=20, extra=EX)
        rows += cloud_rows(f"{name} summary cloud", summ, recs, n=25, S=20, extra=EX)
        save(rows, "results.csv"); show(rows)
