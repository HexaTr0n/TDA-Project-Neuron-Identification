"""Non-topological baselines on the shared trials."""
from lab import *
VP = sys.argv[1] if len(sys.argv) > 1 else REPO / "results"  # folder with vp_q*.npy
df = load_data(); recs = records(df)
blocks = [(r.neuron, [r]) for r in recs]
bl = [[r] for r in recs]
Ds = {"spike count": rp.baseline_spike_count(bl), "ISI histogram": rp.baseline_isi_histogram(bl),
      "ISI histogram, rate removed": rp.baseline_isi_histogram(bl, center=True)}
for q in (1, 10, 100, 1000):
    Ds[f"spike distance q={q}"] = np.load(f"{VP}/vp_q{q}.npy")
rows = [dict(embedding="baseline", view=k, **score(D, recs)) for k, D in Ds.items()]
save(rows, "results.csv"); show(rows)
np.savez_compressed(OUT / "baselines.npz", **{k.replace(" ", "_").replace(",", "").replace("=", ""): v for k, v in Ds.items()})
