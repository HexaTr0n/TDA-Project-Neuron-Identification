"""Does history predict the next ISI within a single trial? Bits per spike vs the
trial's own ISI distribution, all held out."""
from lab import *
from concurrent.futures import ProcessPoolExecutor
LOG2 = np.log(2)
SMOOTH = (0.15, 0.3, 0.5)
BW = (0.25, 0.5, 1.0, 2.0)
LS = (1, 2, 3)


def gauss(z, s):
    return np.exp(-z ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi))


def gains(x, Lmax=3):
    """Mean log2 p(next ISI | history) minus the marginal, on the same held-out spikes."""
    N = len(x) - Lmax
    y = x[Lmax:]
    idx = np.arange(N)
    far = np.abs(idx[:, None] - idx[None, :]) > Lmax  # own neighbourhood is held out
    res = {}
    for s in SMOOTH:
        K = gauss(y[:, None] - y[None, :], s)
        uni = far / far.sum(1, keepdims=True)
        marg = (uni * K).sum(1)
        res[("marginal", 0, 0.0, s)] = np.log(marg).mean() / LOG2
        for L in LS:
            h = np.stack([x[i + Lmax - L:i + Lmax] for i in range(N)])
            d2 = ((h[:, None, :] - h[None, :, :]) ** 2).mean(-1)
            d2 = np.where(far, d2, np.inf)
            d2 = d2 - d2.min(1, keepdims=True)
            for b in BW:
                w = np.exp(-d2 / (2 * b ** 2))
                w /= w.sum(1, keepdims=True)
                p = 0.9 * (w * K).sum(1) + 0.1 * marg
                res[("cond", L, b, s)] = np.log(p).mean() / LOG2
    return res


def local_shuffle(x, rng, block=16):
    x = x.copy()
    for s in range(0, len(x), block):
        x[s:s + block] = rng.permutation(x[s:s + block])
    return x


def one(a):
    i, isis, seed = a
    x = np.log(np.maximum(isis, 1e-4))
    rng = np.random.default_rng(seed)
    if len(x) < 30 or len(x) > 500:   # 500 caps memory; few trials exceed it
        return None
    out = []
    for kind, arr in (("real", x), ("shuffled", rng.permutation(x)), ("local shuffle", local_shuffle(x, rng))):
        for k, v in gains(arr).items():
            out.append((i, kind) + k + (v,))
    return out


if __name__ == "__main__":
    df = load_data(); recs = records(df)
    with ProcessPoolExecutor(2) as ex:
        res = [r for r in ex.map(one, [(i, r.isis, i) for i, r in enumerate(recs)], chunksize=8) if r]
    t = pd.DataFrame([x for r in res for x in r], columns="rec kind model L b s bits".split())
    t["neuron"] = [recs[i].neuron for i in t.rec]
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "memory_raw.csv.gz", index=False)
    m = t[t.model == "marginal"].groupby(["kind", "s", "rec"]).bits.first().rename("m")
    c = t[t.model == "cond"].join(m, on=["kind", "s", "rec"]); c["gain"] = c.bits - c.m
    tab = c.groupby(["kind", "L", "b", "s"]).gain.mean().unstack("kind")
    print(tab.round(4).to_string())
    print('best overall for real:', tab.real.idxmax(), tab.real.max().round(4))
    print('marginal bits by s:', t[t.model == "marginal"].groupby(["kind", "s"]).bits.mean().round(4).to_string())
