"""Stacked cross-correlation: C_i(y, tau) as one matrix per trial, partners by lags.
Partners get one fixed order for everyone, so nearby rows are similar neurons."""
from lab import *
from extra_embeddings import *
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.ndimage import gaussian_filter
from tda import cubical
df = load_data(); recs = records(df)
base = np.load(OUT / "baselines.npz")
EX = {"histogram": base["ISI_histogram"], "spike distance": base["spike_distance_q10"]}
neurons = sorted(df.neuron_id.unique()); pos = {n: i for i, n in enumerate(neurons)}
n_bins = binned(np.zeros(1)).shape[0]; size = ccg_size(n_bins)
F, cnt, slow = {}, {}, []
for tr, g in df.groupby("trial"):
    B = np.zeros((len(neurons), n_bins), dtype=np.float32)
    for t in g.itertuples():
        B[pos[t.neuron_id]] = binned(t.spike_times)
    F[tr] = np.fft.rfft(B, size, axis=1); cnt[tr] = B.sum(1)
    slow.append(B[:, :n_bins // 10 * 10].reshape(len(neurons), -1, 10).sum(2))
S = np.concatenate(slow, 1)
C = np.corrcoef(S + 1e-3 * np.random.default_rng(0).random(S.shape))
order = leaves_list(linkage(S / (S.std(1, keepdims=True) + 1e-9), "average", metric="correlation", optimal_ordering=True))
rand_order = np.random.default_rng(1).permutation(len(neurons))
print("partners", len(neurons), "adjacent-row correlation: fitted order", round(np.mean(np.diag(C[np.ix_(order, order)], 1)), 3),
      "random order", round(np.mean(np.diag(C[np.ix_(rand_order, rand_order)], 1)), 3), flush=True)
del S, C, slow
last = max(F); rng = np.random.default_rng(0)

def shuffled_times(r):
    return r.spike_times[0] + np.r_[0, np.cumsum(rng.permutation(r.isis))] if len(r.isis) else r.spike_times

def image(r, mode, ordr):
    tr = r.trial
    if mode == "shuffled":
        b = binned(shuffled_times(r))
    else:
        b = binned(r.spike_times)
    if mode == "next trial":
        tr = tr + 1 if tr < last else tr - 1
    keep = cnt[tr] >= 20; keep[pos[r.neuron]] = False
    c = np.where(keep[:, None], ccg_matrix(b, F[tr], cnt[tr]), 0.0)
    return c[ordr].astype(np.float64)

def peaks_cloud(I, M=40):
    idx = np.argsort(I.ravel())[-M:]
    y, l = np.unravel_index(idx, I.shape)
    return np.column_stack([l.astype(float), y / 4.0])

def sr(bars, N, size=512):
    al = np.concatenate([b for b in bars if len(b)] or [np.zeros(1)])
    g = np.linspace(0, np.percentile(al, 99.9) * 1.0001, size, endpoint=False)
    Fv = stable_ranks(bars, g)
    return cdist(Fv, Fv, "cityblock") * (g[1] - g[0])

for mode, ordr, name in (("real", order, "stacked ccg"), ("real", rand_order, "stacked ccg, random partner order"),
                         ("shuffled", order, "stacked ccg, target ISIs shuffled"), ("next trial", order, "stacked ccg, partners from next trial")):
    imgs = np.stack([image(r, mode, ordr) for r in recs])
    rows = []
    A = imgs.reshape(len(recs), -1)
    rows.append(dict(embedding=name, view="raw matrix, no topology", **score(cdist(A, A), recs)))
    for sy in (1.0, 3.0):
        Bl = gaussian_filter(imgs, sigma=(0, sy, 0.5))
        A = Bl.reshape(len(recs), -1)
        rows.append(dict(embedding=name, view=f"blurred across partners sy={sy}, no topology", **score(cdist(A, A), recs)))
        c0 = sr(cubical(Bl), len(recs)); c1 = sr(cubical(-Bl), len(recs))
        rows.append(dict(embedding=name, view=f"image valleys sy={sy}", **score(c0, recs)))
        rows.append(dict(embedding=name, view=f"image peaks sy={sy}", **score(c1, recs)))
        rows.append(dict(embedding=name, view=f"image peaks+valleys sy={sy}", **score(ms(c0) + ms(c1), recs)))
    rows += cloud_rows(f"{name}, 2D peak cloud", [peaks_cloud(gaussian_filter(I, (1.0, 0.5))) for I in imgs], recs, n=25, S=20, extra=EX)
    save(rows, "results.csv"); show(rows)
