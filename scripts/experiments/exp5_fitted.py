"""Topology on clouds of fitted pdfs, plus time-tagged clouds."""
from lab import *
import pickle
import embeddings as em
from extra_embeddings import *
df = load_data(); recs = records(df)
base = np.load(OUT / "baselines.npz")
EX = {"histogram": base["ISI_histogram"], "spike distance": base["spike_distance_q10"]}
hmms = pickle.load(open(OUT / "hmms.pkl", "rb"))
rng = np.random.default_rng(0)
shuf = [rng.permutation(r.isis) for r in recs]
def go(name, f):
    for label, xs in (("", [r.isis for r in recs]), (" shuffled", shuf)):
        cl = [f(x) for x in xs]
        cl = [c if c is not None and len(c) >= 25 else None for c in cl]
        rows = cloud_rows(name + label, cl, recs, extra=EX); save(rows, "results.csv"); show(rows)
go("fit lognormal W=20", lambda x: fitted_lognormal(x, 20))
go("fit gamma W=20", lambda x: fitted_gamma(x, 20))
go("fit 2-mixture W=40", lambda x: fitted_mixture(x, 40) if len(x) >= 65 else None)
for K in (3, 4, 6):
    if K in hmms:
        go(f"shared HMM K={K}", lambda x: hmm_states(x, hmms[K]))
# Baseline delay clouds with H1, for comparison
for m in (1, 2, 3, 5):
    go(f"delay m={m}", lambda x: em.isi_delay(x, m=m))
# Time-tagged: keeps the movie moment inside the cloud
for a in (0.5, 2.0, 8.0):
    rows = cloud_rows(f"time-tagged delay m=1 alpha={a}", [time_tagged(r.isis, r.spike_times, 1, a) for r in recs], recs, extra=EX)
    save(rows, "results.csv"); show(rows)
