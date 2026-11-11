"""Held-out next-ISI likelihood for every pdf model, in bits per spike above the trial's
own ISI distribution, on the same spikes (spike 41 onward of each trial)."""
from lab import *
import pickle
from fits import *
from numpy.lib.stride_tricks import sliding_window_view
from concurrent.futures import ProcessPoolExecutor
W0 = 40
LOG2 = np.log(2)

def gauss(z, s): return np.exp(-z ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi))

def kde_marginal(x, s=0.3):
    y = x[W0:]; K = gauss(y[:, None] - x[None, :], s)
    i = np.arange(len(y)); j = i + W0
    far = np.abs(j[:, None] - np.arange(len(x))[None, :]) > 3
    return np.log((K * far).sum(1) / far.sum(1))

def kernel_cond(x, L=1, b=1.0, s=0.5):
    N = len(x); idx = np.arange(L, N)
    h = np.stack([x[i - L:i] for i in idx]); y = x[idx]
    d2 = ((h[:, None] - h[None, :]) ** 2).mean(-1)
    far = np.abs(idx[:, None] - idx[None, :]) > 3
    d2 = np.where(far, d2, np.inf); d2 = d2 - d2.min(1, keepdims=True)
    w = np.exp(-d2 / (2 * b ** 2)); w /= w.sum(1, keepdims=True)
    K = gauss(y[:, None] - y[None, :], s)
    marg = ((far / far.sum(1, keepdims=True)) * K).sum(1)
    return np.log(0.9 * (w * K).sum(1) + 0.1 * marg)[W0 - L:]

def run_one(a):
    x, others, stim, hmms, seed = a
    out = {}
    base = kde_marginal(x)
    y = x[W0:]
    def add(k, lp): out[k] = (lp - base).mean() / LOG2
    add("kernel L=1 b=1 s=.5", kernel_cond(x))
    add("kernel L=2 b=.5 s=.3 (as embedded)", kernel_cond(x, 2, .5, .3))
    for W in (10, 20, 40):
        m, s = local_lognormal(x, W); add(f"local lognormal W={W}", norm_logpdf(y, m[W0 - W:], s[W0 - W:]))
        k, th = local_gamma(x, W); add(f"local gamma W={W}", gamma_logpdf_log(y, k[W0 - W:], th[W0 - W:]))
    add("local 2-mixture W=40", mixture_logpdf(y, *local_mixture(x, 40)))
    for K, h in hmms.items():
        add(f"shared HMM K={K}", h.logpdf(x)[W0:])
    # Diagnostics only: these use the neuron's OTHER trials, so they break the single-trial rule.
    ox, ot = others
    if len(ox) > 50:
        for s in (0.15, 0.3):
            pm = np.log(gauss(y[:, None] - ox[:, -1][None, :], s).mean(1))
            add(f"DIAG other trials marginal s={s}", pm)
        K = gauss(y[:, None] - ox[:, -1][None, :], 0.3); pm = K.mean(1)
        for L, b in ((1, .25), (1, .5), (2, .25), (2, .5), (3, .5)):
            h = np.stack([x[i - L:i] for i in range(W0, len(x))])
            d2 = ((h[:, None, :] - ox[None, :, -1 - L:-1]) ** 2).mean(-1)
            w = np.exp(-d2 / (2 * b ** 2)) + 1e-12; w /= w.sum(1, keepdims=True)
            add(f"DIAG other trials history L={L} b={b}", np.log(.9 * (w * K).sum(1) + .1 * pm))
        tn = stim
        for sig in (.02, .05, .1, .3):
            d2 = (tn[:, None] - ot[None, :]) ** 2
            w = np.exp(-d2 / (2 * sig ** 2)) + 1e-12; w /= w.sum(1, keepdims=True)
            add(f"DIAG other trials movie time sigma={sig}", np.log(.9 * (w * K).sum(1) + .1 * pm))
    return out

if __name__ == "__main__":
    allrec = pd.read_csv(REPO / "data/isi_frozennoise.csv", usecols=["session", "isis"])
    rng = np.random.default_rng(0)
    pool = allrec[(allrec.session != SESSION) & allrec.isis.notna()]
    seqs = []
    for s in pool.isis.sample(frac=1, random_state=0):
        v = np.fromstring(s, sep=";")
        if 30 <= len(v):
            seqs.append(np.log(np.maximum(v, 1e-4))[:300])
        if len(seqs) == 400: break
    pooled = np.concatenate(seqs)
    hmms = {K: HMM(K, pooled).fit(seqs) for K in (2, 3, 4, 6)}
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(hmms, open(OUT / "hmms.pkl", "wb"))
    for K, h in hmms.items():
        print(K, 'means(ms)', np.round(np.exp(h.mu) * 1000, 1), 'sd', h.sd.round(2), 'stay', np.diag(h.A).round(2), flush=True)
    df = load_data(); recs = records(df)
    byn = {}
    for r in recs: byn.setdefault(r.neuron, []).append(r)
    jobs = []
    for i, r in enumerate(recs):
        x = np.log(np.maximum(r.isis, 1e-4))
        if len(x) < W0 + 10 or len(x) > 900: continue
        ox, ot = [], []
        for o in byn[r.neuron]:
            if o.trial != r.trial and len(o.isis) > 6:
                z = np.log(np.maximum(o.isis, 1e-4))
                ox.append(sliding_window_view(z, 4)); ot.append(o.spike_times[3:len(z)])
        ox = np.concatenate(ox); ot = np.concatenate(ot)
        k = np.random.default_rng(i).choice(len(ox), min(len(ox), 1500), replace=False)
        jobs.append((x, (ox[k], ot[k]), r.spike_times[W0:len(x)], hmms, i))
    print('jobs', len(jobs), flush=True)
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(run_one, jobs, chunksize=6))
    t = pd.DataFrame(res); t.to_csv(OUT / "fits_raw.csv", index=False)
    s = pd.DataFrame(dict(bits=t.mean(), se=t.std() / np.sqrt(t.count()), n=t.count())).sort_values("bits", ascending=False)
    print(s.round(4).to_string())
