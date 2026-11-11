from lab import *
from scipy.ndimage import gaussian_filter1d
df = load_data(); recs = records(df)
tr = [np.sort(np.asarray(r.spike_times, float)) for r in recs]
T = max(x[-1] for x in tr if len(x)); n = len(tr)
base = np.load(OUT / "baselines.npz"); hist = base["ISI_histogram"]
SIG = (5, 10, 20, 50, 100); Q = (3, 10, 30)
g = int(T * 1000) + 500
B = np.zeros((n, g), np.float32)
for i, x in enumerate(tr):
    np.add.at(B[i], np.minimum((x * 1000).astype(int), g - 1), 1)
fam = {"peak": {}, "vR": {}, "VP": {}, "box": {}}
for s in SIG:
    for q in Q:
        fam["peak"][(s, q)] = np.load(OUT / f"peak_sigma{s}_q{q}.npy")
    F = gaussian_filter1d(B, s, axis=1)[:, ::max(1, s // 4)]
    fam["vR"][s] = cdist(F, F, "euclidean"); print("vR", s, flush=True)
for q in (1, 10, 100):
    fam["VP"][q] = base[f"spike_distance_q{q}"]
for q in (3, 10, 30, 100):
    s = 2 / q; dt = s / 10; grid = np.arange(dt / 2, T + s, dt)
    F = np.stack([np.searchsorted(x, grid, "right") - np.searchsorted(x, grid - s, "right") for x in tr]).astype(np.float32)
    fam["box"][q] = cdist(F, F, "cityblock"); print("box", q, flush=True)
y = np.unique([r.neuron for r in recs], return_inverse=True)[1]
def nn(D, idx):
    S = D[np.ix_(idx, idx)].copy(); np.fill_diagonal(S, np.inf)
    return (y[idx][S.argmin(1)] == y[idx]).mean()
mix = {f: {k: (ms(v), ms(v) + .5 * ms(hist)) for k, v in d.items()} for f, d in fam.items()}
print("all neurons, best setting")
for f, d in mix.items():
    al = {k: nn(v[0], np.arange(n)) for k, v in d.items()}; co = {k: nn(v[1], np.arange(n)) for k, v in d.items()}
    print(f"  {f:5s} alone {max(al.values()):.3f} at {max(al, key=al.get)}   + histogram {max(co.values()):.3f} at {max(co, key=co.get)}")
print("held out: choose setting on half the neurons, score on the other half, 50 splits")
rng = np.random.default_rng(0); res = {f: [[], []] for f in mix}; picks = {f: [] for f in mix}
for it in range(50):
    p = rng.permutation(y.max() + 1); A, Bb = p[:30], p[30:]
    ia, ib = np.flatnonzero(np.isin(y, A)), np.flatnonzero(np.isin(y, Bb))
    for f, d in mix.items():
        for w in (0, 1):
            k = max(d, key=lambda k: nn(d[k][w], ia)); res[f][w].append(nn(d[k][w], ib))
            if w: picks[f].append(k)
ref = None
for f in mix:
    print(f"  {f:5s} alone {np.mean(res[f][0]):.3f}  + histogram {np.mean(res[f][1]):.3f} (sd over splits {np.std(res[f][1]):.3f})  most picked {pd.Series(picks[f]).value_counts().index[0]}")
hp = np.array(res["peak"][1])
for f in ("VP", "vR", "box"):
    dd = hp - np.array(res[f][1]); print(f"  peak minus {f}, same splits, with histogram: {dd.mean():+.3f}, peak ahead in {(dd>0).mean():.0%} of splits")
# per-neuron bootstrap at the most picked settings
def per_neuron_hits(D):
    S = D.copy(); np.fill_diagonal(S, np.inf); h = (y[S.argmin(1)] == y).astype(float)
    return np.array([h[y == k].mean() for k in range(y.max() + 1)])
a = per_neuron_hits(mix["peak"][pd.Series(picks["peak"]).value_counts().index[0]][1]); b = per_neuron_hits(mix["VP"][10][1]); c = per_neuron_hits(mix["vR"][pd.Series(picks["vR"]).value_counts().index[0]][1])
br = np.random.default_rng(1); ix = br.integers(0, len(a), (5000, len(a)))
for nm, o in (("VP", b), ("vR", c)):
    d = (a - o)[ix].mean(1); print(f"  neuron bootstrap peak - {nm}: mean {(a-o).mean():+.3f}  95% [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]")
