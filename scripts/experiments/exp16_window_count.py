import time
from lab import *
from scipy.signal import lfilter
from scipy.stats import spearmanr
import spike_distance as sd
df = load_data(); recs = records(df)
tr = [np.sort(np.asarray(r.spike_times, float)) for r in recs]
T = max(x[-1] for x in tr if len(x))
base = np.load(OUT / "baselines.npz"); hist = base["ISI_histogram"]
iu = np.triu_indices(len(tr), 1)

def box(q, dt=None):
    s = 2 / q; dt = dt or s / 20
    g = np.arange(dt / 2, T + s, dt)
    return np.stack([np.searchsorted(x, g, "right") - np.searchsorted(x, g - s, "right") for x in tr]).astype(np.float32), dt / s

def exact_box(x, y, s):
    """L1 of box counts, exact: integrate between all change points."""
    e = np.sort(np.r_[x, x + s, y, y + s]); m = (e[1:] + e[:-1]) / 2; w = np.diff(e)
    a = np.searchsorted(x, m, "right") - np.searchsorted(x, m - s, "right")
    b = np.searchsorted(y, m, "right") - np.searchsorted(y, m - s, "right")
    return (np.abs(a - b) * w).sum() / s

# 1. single edits on one random train: the claimed identities
rng = np.random.default_rng(0); q = 10.; s = 2 / q
x = np.sort(rng.uniform(0, 10, 30))
print("shift delta   VP   box-L1   claimed min(q d,2)")
for dl in (.001, .01, .05, .1, .15, .2, .3, 1.):
    y = x.copy(); i = 15; y[i] += dl; y = np.sort(y)
    print(f"  {dl:5.3f}  {sd.vp_one_to_many(x, y[None], np.array([len(y)]), q)[0]:.4f}  {exact_box(x, y, s):.4f}   {min(q*dl, 2):.4f}")
y = np.delete(x, 15); print("  one deletion: VP", sd.vp_one_to_many(x, y[None], np.array([len(y)]), q)[0], "box", round(exact_box(x, y, s), 4))

# 2. lower bound on real pairs, exact
idx = rng.choice(len(tr), (4000, 2)); V10 = base["spike_distance_q10"]
ex = np.array([exact_box(tr[i], tr[j], s) for i, j in idx]); vp = V10[idx[:, 0], idx[:, 1]]
ok = idx[:, 0] != idx[:, 1]; ex, vp = ex[ok], vp[ok]
print("exact box / VP on 4000 real pairs: min", ex.min() / 1, "ratio median", round(np.median(ex / vp), 3), "p1", round(np.percentile(ex / vp, 1), 3), "p99", round(np.percentile(ex / vp, 99), 3), "max", round((ex / vp).max(), 3), "violations", int((ex > vp * (1 + 1e-9)).sum()))

rows = []
def report(name, D, V=None):
    r = dict(name=name, **score(D, recs))
    if V is not None:
        r["pearson"] = round(np.corrcoef(D[iu], V[iu])[0, 1], 4); r["spearman"] = round(spearmanr(D[iu][::37], V[iu][::37])[0], 4)
        r["ratio"] = round(np.median(D[iu] / np.maximum(V[iu], 1e-9)), 3)
    rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("name", "acc", "ami", "auc", "pearson", "spearman", "ratio")}, flush=True)

report("VP q=10 exact", V10)
for q, dt in ((1., None), (10., None), (100., .001)):
    V = base[f"spike_distance_q{int(q)}"]
    t0 = time.time(); F, sc = box(q, dt); D = cdist(F, F, "cityblock") * sc
    print(f"q={q}: features {F.shape}, {time.time()-t0:.1f} s", flush=True)
    report(f"box L1 s=2/q q={q:g}", D, V)
    if q == 10:
        Db = D
        for dt2 in (s / 4, s / 2):
            F2, sc2 = box(q, dt2); report(f"box L1 q=10 coarse step {dt2*1000:.0f} ms ({F2.shape[1]} dims)", cdist(F2, F2, "cityblock") * sc2, V)
    # squared van Rossum, tau = 2/q, so shift cost 2(1-exp(-q d/2)) and insertion 1
    tau = 2 / q; d = dt or tau / 20; n = int((T + 8 * tau) / d) + 1
    B = np.zeros((len(tr), n), np.float32)
    for i, x in enumerate(tr):
        np.add.at(B[i], np.minimum((x / d).astype(int), n - 1), 1)
    a = np.exp(-d / tau); E = lfilter([1], [1, -a], B, axis=1)
    Dv = cdist(E, E, "sqeuclidean") * d / tau * 2 * (1 - a) ** 2 * (tau / d) ** 2
    report(f"squared van Rossum tau=2/q q={q:g}", Dv, V)
    if q == 10: Dvr = Dv
print("--- combinations")
report("box L1 q=10 + 0.5 histogram", ms(Db) + .5 * ms(hist))
report("VP q=10 + 0.5 histogram", ms(V10) + .5 * ms(hist))
print("--- landmarks: spike distance to m random trials as the features")
for m in (20, 50, 100, 200):
    L = rng.choice(len(tr), m, replace=False); keep = np.setdiff1d(np.arange(len(tr)), L)
    F = V10[np.ix_(keep, L)]; Dl = cdist(F, F, "cityblock") / m
    rr = [recs[i] for i in keep]; r = score(Dl, rr)
    print(f"  m={m}: acc {r['acc']:.3f} ami {r['ami']:.3f} auc {r['auc']:.3f}  pearson with full {np.corrcoef(Dl[np.triu_indices(len(keep),1)], V10[np.ix_(keep, keep)][np.triu_indices(len(keep),1)])[0,1]:.3f}  distance work {m/len(tr):.1%} of full", flush=True)
pd.DataFrame(rows).to_csv(OUT / "boxtest.csv", index=False)
