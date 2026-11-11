"""How many peaks does one trial's ISI distribution need? Five-fold held-out likelihood
inside each trial, in bits per ISI above a single lognormal. Also a dip-style bimodality
count: a 2-Gaussian model must win by more than the fold noise."""
from lab import *
from fits import norm_logpdf, gamma_logpdf_log, SD_FLOOR
from scipy.special import logsumexp
LOG2 = np.log(2)

def fit_mix(x, K, iters=40):
    mu = np.quantile(x, (np.arange(K) + .5) / K); sd = np.full(K, max(x.std() / K, SD_FLOOR)); pi = np.ones(K) / K
    for _ in range(iters):
        lp = np.log(pi) + norm_logpdf(x[:, None], mu, sd)
        r = np.exp(lp - logsumexp(lp, axis=1, keepdims=True)); nk = r.sum(0) + 1e-6
        pi = nk / len(x); mu = (r * x[:, None]).sum(0) / nk
        sd = np.maximum(np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / nk), SD_FLOOR)
    return pi, mu, sd

def heldout(x, seed):
    idx = np.random.default_rng(seed).permutation(len(x)); out = {k: [] for k in ("lognormal", "gamma", "mix2", "mix3", "mix4", "kde")}
    for f in np.array_split(idx, 5):
        tr = np.setdiff1d(idx, f); a, b = x[tr], x[f]
        out["lognormal"].append(norm_logpdf(b, a.mean(), max(a.std(), .05)))
        m = np.exp(a).mean(); s = max(np.log(m) - a.mean(), 1e-3); k = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
        out["gamma"].append(gamma_logpdf_log(b, k, m / k))
        for K in (2, 3, 4):
            pi, mu, sd = fit_mix(a, K); out[f"mix{K}"].append(logsumexp(np.log(pi) + norm_logpdf(b[:, None], mu, sd), axis=1))
        z = (b[:, None] - a[None, :]) / .3
        out["kde"].append(np.log(np.exp(-z ** 2 / 2).mean(1) / (.3 * np.sqrt(2 * np.pi))))
    return {k: np.concatenate(v).mean() / LOG2 for k, v in out.items()}

df = load_data(); recs = records(df)
t = pd.DataFrame([dict(neuron=r.neuron, **heldout(np.log(np.maximum(r.isis, 1e-4)), i)) for i, r in enumerate(recs)])
g = t.drop(columns="lognormal").set_index("neuron").sub(t.set_index("neuron").lognormal, axis=0)
print("bits per ISI above a single lognormal, per trial"); print(pd.DataFrame(dict(mean=g.mean(), se=g.std() / np.sqrt(len(g)))).round(4).to_string())
print("share of trials where mix2 beats lognormal", round((g.mix2 > 0).mean(), 3), "by more than 0.05 bits", round((g.mix2 > .05).mean(), 3))
print("share of trials where mix2 beats gamma", round((g.mix2 > g.gamma).mean(), 3))
pn = g.groupby("neuron").mean()
print("neurons (trial-averaged) with mix2 gain above 0.05 bits", int((pn.mix2 > .05).sum()), "of", len(pn))
t.to_csv(OUT / "shape_raw.csv", index=False)
