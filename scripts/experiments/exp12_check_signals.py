"""Two signals appeared once rate was removed. Are they real?
A. The long H0 bar in spike distance: is it a family, or a few odd neurons?
B. The H1 loop in the histogram space: does it replicate in independent trials and in
   subsets of neurons, and does a loop-free one-parameter neuron model produce it too?"""
from exp9_families import *
Z = np.load(OUT / "neuron_spaces.npz")

def no_rate(E, rate):
    lam, V = spectrum(E)
    c = [abs(spearmanr(V[:, i], rate)[0]) for i in range(min(5, len(lam)))]
    d = int(np.argmax(c))
    keep = [i for i in range(len(lam)) if i != d]
    return squareform(pdist((V * np.sqrt(lam))[:, keep])), lam[keep]

def null_p(E2, lam2, key, sims, seed=0):
    r = np.random.default_rng(seed)
    n = np.array([stats(squareform(pdist(r.normal(size=(len(E2), len(lam2))) * np.sqrt(lam2 / (len(E2) - 1)))))[key] for _ in range(sims)])
    o = stats(E2)[key]
    return o, np.median(n), np.quantile(n, .95), (1 + (n >= o).sum()) / (sims + 1)

def rates(rs):
    y = np.array([r.neuron for r in rs]); f = neuron_features(rs).loc[np.unique(y)]
    return y, np.log(f.n.values), f, pd.Series(y).value_counts().loc[np.unique(y)].values

print("=== A. spike distance after removing rate", flush=True)
y, rate, f, ntr = rates(recs60)
E2, lam2 = no_rate(Z["vp60"], rate)
L = linkage(squareform(E2, checks=False), "single")
size = np.r_[np.ones(60), L[:, 3]]
for a, b, h, _ in L[::-1][:5]:
    print(f"  merge at {h / np.median(E2[E2 > 0]):.3f}: sizes {int(size[int(a)])} and {int(size[int(b)])}", flush=True)
d2 = np.sort(E2, axis=1)[:, 1]
iso = np.argsort(d2)[::-1][:4]
print("  most isolated neurons:", [(np.unique(y)[i][-12:], f"nn {d2[i] / np.median(E2[E2 > 0]):.2f}", f"trials {ntr[i]}", f"spikes/trial {f.n.values[i]:.0f}") for i in iso], flush=True)
print("  trials per neuron: min", ntr.min(), "median", int(np.median(ntr)), flush=True)
for thr in (2, 4):
    drop = np.argsort(d2)[::-1][:thr]
    k = np.setdiff1d(np.arange(60), drop)
    Es = Z["vp60"][np.ix_(k, k)]; E4, l4 = no_rate(Es, rate[k])
    print(f"  without the {thr} most isolated neurons:", flush=True)
    for key in ("h0_top", "sil", "gap"):
        o, m, q, p = null_p(E4, l4, key, 300)
        print(f"    {key:7s} observed {o:.3f}  null median {m:.3f}  null 95% {q:.3f}  p {p:.3f}", flush=True)

print("\n=== B. histogram loop on 146 neurons", flush=True)
y, rate, f, ntr = rates(recs146)
tr = np.array([r.trial for r in recs146])
H = rp.baseline_isi_histogram([[r] for r in recs146])
names = np.unique(y)
ok = np.array([min(((y == n) & (tr % 2 == 1)).sum(), ((y == n) & (tr % 2 == 0)).sum()) >= 3 for n in names])
use = np.isin(y, names[ok])
for lab_, m in (("odd trials", tr % 2 == 1), ("even trials", tr % 2 == 0)):
    Em = energy(H[np.ix_(use, use)], y[use], m[use])
    E2, l2 = no_rate(Em, rate[ok])
    o, md, q, p = null_p(E2, l2, "h1_top", 300)
    print(f"  {lab_}: h1_top {o:.3f}  null median {md:.3f}  null 95% {q:.3f}  p {p:.3f}", flush=True)
E0 = Z["hist146"]
r = np.random.default_rng(1); hits = 0; vals = []
for b in range(40):
    k = np.sort(r.choice(146, 117, replace=False))
    E2, l2 = no_rate(E0[np.ix_(k, k)], rate[k])
    o, md, q, p = null_p(E2, l2, "h1_top", 30, seed=b)
    hits += o > q; vals.append(o)
print(f"  80% subsets of neurons above their own null 95%: {hits} of 40, h1_top median {np.median(vals):.3f}", flush=True)
o_real = stats(no_rate(E0, rate)[0])["h1_top"]
print(f"  real h1_top {o_real:.3f}", flush=True)

def simulate(params, seed):
    g = np.random.default_rng(seed)
    bl = []
    for r in recs146:
        m, s = params[r.neuron]
        x = g.normal(m, s, len(r.isis))
        bl.append([Rec(r.neuron, r.trial, np.exp(x), None)])
    Es = energy(rp.baseline_isi_histogram(bl), y)
    return stats(no_rate(Es, rate)[0])

logs = {}
for r in recs146:
    logs.setdefault(r.neuron, []).append(np.log(np.maximum(r.isis, 1e-4)))
mu = {n: np.concatenate(v).mean() for n, v in logs.items()}
sd = {n: np.concatenate(v).std() for n, v in logs.items()}
gsd = np.mean(list(sd.values()))
for lab_, p in (("one parameter: lognormal, each neuron's mean, shared sd", {n: (mu[n], gsd) for n in mu}),
                ("two parameters: lognormal, each neuron's mean and sd", {n: (mu[n], sd[n]) for n in mu})):
    v = [simulate(p, s)["h1_top"] for s in range(15)]
    print(f"  {lab_}: h1_top median {np.median(v):.3f}  range {min(v):.3f}-{max(v):.3f}", flush=True)
# Empirical model: draw every neuron's ISIs from its own pooled ISIs. Keeps real histogram
# shapes, drops trial structure, so any loop that survives lives in histogram shape.
def simulate_emp(seed):
    g = np.random.default_rng(seed)
    pool = {n: np.exp(np.concatenate(v)) for n, v in logs.items()}
    bl = [[Rec(r.neuron, r.trial, g.choice(pool[r.neuron], len(r.isis)), None)] for r in recs146]
    return stats(no_rate(energy(rp.baseline_isi_histogram(bl), y), rate)[0])["h1_top"]
v = [simulate_emp(s) for s in range(15)]
print(f"  resampled real histogram shapes: h1_top median {np.median(v):.3f}  range {min(v):.3f}-{max(v):.3f}", flush=True)
