"""Topology on the space of neurons, built from trial-to-trial distances.
Do neurons fall into families, or is this one continuous blob?

Each trial-by-trial matrix becomes a neuron-by-neuron energy distance, which removes
single-trial noise. The barcode is compared with a one-blob null: Gaussian clouds with
the same spectrum as the real neuron space, so same size and shape but no clusters."""
from lab import *
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, silhouette_score, adjusted_mutual_info_score
from tda import rips_matrix

df = load_data()
recs60 = records(df); recs146 = records(df, n_neurons=None)
base = np.load(OUT / "baselines.npz")


def energy(D, y, mask=None):
    names = np.unique(y)
    Z = (y[:, None] == names[None, :]).astype(float)
    if mask is not None:
        Z = Z * mask[:, None]
    c = Z.sum(0)
    S = Z.T @ D @ Z
    M = S / np.outer(c, c)
    w = np.diag(S) / (c * (c - 1))
    E = M - 0.5 * (w[:, None] + w[None, :])
    np.fill_diagonal(E, 0)
    return np.sqrt(np.maximum(E, 0))


def spectrum(E):
    n = len(E)
    J = np.eye(n) - 1 / n
    lam, V = np.linalg.eigh(-0.5 * J @ E ** 2 @ J)
    o = np.argsort(lam)[::-1]
    lam, V = lam[o], V[:, o]
    keep = lam > 1e-12
    return lam[keep], V[:, keep]


def family_gap(E, m=3):
    """Largest merge height in single linkage where both sides have at least m neurons."""
    Z = linkage(squareform(E, checks=False), "single")
    n = len(E)
    size = np.r_[np.ones(n), Z[:, 3]]
    g = 0.0
    for a, b, h, _ in Z:
        if size[int(a)] >= m and size[int(b)] >= m:
            g = max(g, h)
    return g


def best_silhouette(E):
    Z = linkage(squareform(E, checks=False), "average")
    return max(silhouette_score(E, fcluster(Z, k, "maxclust"), metric="precomputed")
               for k in range(2, 9) if len(set(fcluster(Z, k, "maxclust"))) > 1)


def stats(E):
    h0, b, d = rips_matrix(E)
    s = np.sort(d - b)[::-1] if len(b) else np.zeros(1)
    sc = np.median(E[E > 0])
    return dict(gap=family_gap(E) / sc, h0_top=np.sort(h0)[-1] / sc, h1_top=s[0] / sc,
                h1_second=(s[1] if len(s) > 1 else 0) / sc, sil=best_silhouette(E))


def analyse(label, D, recs, sims):
    y = np.array([r.neuron for r in recs]); tr = np.array([r.trial for r in recs])
    E = energy(D, y)
    odd, even = tr % 2 == 1, tr % 2 == 0
    # The split-half part needs at least 3 trials of a neuron in each half.
    names = np.unique(y)
    ok = np.array([min((y[odd] == n).sum(), (y[even] == n).sum()) >= 3 for n in names])
    use = np.isin(y, names[ok])
    Eo = energy(D[np.ix_(use, use)], y[use], odd[use])
    Ee = energy(D[np.ix_(use, use)], y[use], even[use])
    iu = np.triu_indices(len(Eo), 1)
    rel = spearmanr(Eo[iu], Ee[iu])[0]
    lam, _ = spectrum(E)
    obs = stats(E)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(sims):
        X = rng.normal(size=(len(E), len(lam))) * np.sqrt(lam / (len(E) - 1))
        null.append(stats(squareform(pdist(X))))
    null = pd.DataFrame(null)
    Zo, Ze = linkage(squareform(Eo, checks=False), "average"), linkage(squareform(Ee, checks=False), "average")
    ari = {k: adjusted_rand_score(fcluster(Zo, k, "maxclust"), fcluster(Ze, k, "maxclust")) for k in (2, 3, 4, 6, 8)}
    print(f"\n=== {label}: {len(E)} neurons, split-half reliability {rel:.3f} on {ok.sum()} neurons, top 3 axes hold {lam[:3].sum() / lam.sum():.2f} of variance", flush=True)
    for k in obs:
        p = (1 + (null[k] >= obs[k]).sum()) / (len(null) + 1)
        print(f"  {k:10s} observed {obs[k]:.3f}  null median {null[k].median():.3f}  null 95% {null[k].quantile(.95):.3f}  p {p:.3f}", flush=True)
    print("  split-half cluster agreement (ARI):", {k: round(v, 3) for k, v in ari.items()}, flush=True)
    return E, y, obs, null


def neuron_features(recs):
    t = pd.DataFrame([dict(neuron=r.neuron, n=len(r.isis), cv=r.isis.std() / r.isis.mean(),
                           burst=(r.isis < 0.01).mean(), med=np.median(np.log(r.isis))) for r in recs])
    return t.groupby("neuron").mean()


def explain(label, E, y, recs):
    f = neuron_features(recs)
    names = np.unique(y); f = f.loc[names]
    feats = pd.DataFrame(dict(log_rate=np.log(f.n), cv=f.cv, burst=f.burst, median_isi=f.med))
    iu = np.triu_indices(len(E), 1)
    lam, V = spectrum(E)
    print(f"  what the {label} axes track:")
    for c in feats:
        d = np.abs(feats[c].values[:, None] - feats[c].values[None, :])[iu]
        ax = [abs(spearmanr(V[:, i], feats[c])[0]) for i in range(3)]
        print(f"    {c:11s} distance corr {spearmanr(E[iu], d)[0]:.3f}  |rho| with axes 1-3 {np.round(ax, 2)}", flush=True)
    ch = np.array([int(n.split("_C")[-1].split("_")[0]) for n in names])
    same = (ch[:, None] == ch[None, :])[iu]
    print(f"    same electrode pairs {same.sum()}: median distance {np.median(E[iu][same]) if same.sum() else float('nan'):.3f} vs others {np.median(E[iu][~same]):.3f}", flush=True)
    return feats


if __name__ == "__main__":
    hist60 = base["ISI_histogram"]; vp = base["spike_distance_q10"]
    combo = ms(vp) + 0.5 * ms(hist60)
    sims = 500
    res = {}
    for lab_, D in (("histogram, 60 neurons", hist60), ("spike distance q=10, 60 neurons", vp), ("combined, 60 neurons", combo)):
        E, y, o, n = analyse(lab_, D, recs60, sims); res[lab_] = E
        explain(lab_, E, y, recs60)
    iu = np.triu_indices(60, 1)
    a, b = res["histogram, 60 neurons"], res["spike distance q=10, 60 neurons"]
    print("\nagreement between histogram and spike-distance neuron spaces:", round(spearmanr(a[iu], b[iu])[0], 3), flush=True)
    bl = [[r] for r in recs146]
    H = rp.baseline_isi_histogram(bl)
    E, y, o, n = analyse("histogram, 146 neurons", H, recs146, 200)
    explain("histogram 146", E, y, recs146)
    np.savez_compressed(OUT / "neuron_spaces.npz", hist146=E, hist60=a, vp60=b, combined60=res["combined, 60 neurons"])
    y60 = np.array([r.neuron for r in recs60])
    for link in ("single", "average"):
        pred = AgglomerativeClustering(n_clusters=60, metric="precomputed", linkage=link).fit_predict(combo)
        print("trial-level clustering", link, "AMI", round(adjusted_mutual_info_score(y60, pred), 3), flush=True)
