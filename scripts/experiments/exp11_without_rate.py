"""Could firing rate be hiding families? Drop the axis that tracks it and test the rest."""
from exp9_families import *
Z = np.load(OUT / "neuron_spaces.npz")
for label, key, rs in (("histogram, 60", "hist60", recs60), ("spike distance, 60", "vp60", recs60),
                       ("combined, 60", "combined60", recs60), ("histogram, 146", "hist146", recs146)):
    E = Z[key]
    lam, V = spectrum(E)
    y = np.array([r.neuron for r in rs])
    f = neuron_features(rs).loc[np.unique(y)]
    rate = np.log(f.n.values)
    corr = [abs(spearmanr(V[:, i], rate)[0]) for i in range(len(lam))]
    drop = int(np.argmax(corr[:5]))
    X = V * np.sqrt(lam)
    keep = [i for i in range(len(lam)) if i != drop]
    E2 = squareform(pdist(X[:, keep]))
    lam2 = lam[keep]
    obs = stats(E2)
    rng = np.random.default_rng(0)
    null = pd.DataFrame([stats(squareform(pdist(rng.normal(size=(len(E2), len(lam2))) * np.sqrt(lam2 / (len(E2) - 1))))) for _ in range(300)])
    print(f"\n=== {label}: dropped axis {drop + 1} (rho with log rate {corr[drop]:.2f}, {lam[drop] / lam.sum():.2f} of variance); next axis holds {lam2[0] / lam2.sum():.2f} of what is left", flush=True)
    for k in obs:
        p = (1 + (null[k] >= obs[k]).sum()) / (len(null) + 1)
        print(f"  {k:10s} observed {obs[k]:.3f}  null median {null[k].median():.3f}  null 95% {null[k].quantile(.95):.3f}  p {p:.3f}", flush=True)
