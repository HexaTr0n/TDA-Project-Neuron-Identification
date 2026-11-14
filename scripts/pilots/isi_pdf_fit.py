"""W11: fit a pdf to each trial's ISIs and compare the fitted pdfs (the user's idea).

Run from the repo root: python scripts/pilots/isi_pdf_fit.py
Retina, 60 neurons, frozen trials; numpy, scipy, scikit-learn only (no stablebear).
Models on x = log ISI: one Gaussian (ISI lognormal); a gamma on the ISI itself; mixtures of
K = 2, 3, 4 Gaussians; K chosen per trial by BIC; a kernel estimate (one Gaussian per ISI).
Every fitted pdf is turned into its survival curve on the grid of the ISI fractions, so the
L1 distance is Wasserstein-1 between fitted pdfs and k-means with mean curves works as before.
Also: Hellinger distance between the fitted densities, and the across-halves test (first half
of a trial matched among second halves of other trials).
"""
import sys, warnings
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm, gamma
from sklearn.mixture import GaussianMixture
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import run_pipeline as rp
from load_dataset import load_dataset
warnings.filterwarnings("ignore")

csv = REPO / "data" / "isi_frozennoise.csv"
df = load_dataset(csv if csv.exists() else rp.DEFAULT_PATH, sessions=["20180126_sr_le_sp"])
blocks = rp.pick_neurons(rp.make_blocks(df, rp.Config()), 60, 0)
y = np.unique([b[0] for b in blocks], return_inverse=True)[1]; k = y.max() + 1
times = [np.asarray(b[1][0].spike_times, float) for b in blocks]
lg = lambda t: np.log(np.maximum(np.diff(t), 1e-4))
HALF = 9.94 / 2
SETS = {"full": [lg(t) for t in times], "A": [lg(t[t < HALF]) for t in times], "B": [lg(t[t >= HALF]) for t in times]}
LG = np.linspace(np.log(1e-4), np.log(10), 230, endpoint=False)
STEP = LG[1] - LG[0]


def empirical(l):
    return (l[None, :] > LG[:, None]).mean(1), None

def mixture(K):
    def f(l):
        if len(l) < max(2 * K, 3):
            return empirical(l)
        g = GaussianMixture(K, reg_covar=1e-3, n_init=2, random_state=0).fit(l[:, None])
        w, m, s = g.weights_, g.means_[:, 0], np.sqrt(g.covariances_[:, 0, 0])
        return (w * norm.sf(LG[:, None], m, s)).sum(1), (w * norm.pdf(LG[:, None], m, s)).sum(1)
    return f

def by_bic(l):
    if len(l) < 8:
        return empirical(l)
    fits = [GaussianMixture(K, reg_covar=1e-3, n_init=2, random_state=0).fit(l[:, None]) for K in (1, 2, 3, 4)]
    g = min(fits, key=lambda g: g.bic(l[:, None])); by_bic.chosen.append(g.n_components)
    w, m, s = g.weights_, g.means_[:, 0], np.sqrt(g.covariances_[:, 0, 0])
    return (w * norm.sf(LG[:, None], m, s)).sum(1), (w * norm.pdf(LG[:, None], m, s)).sum(1)
by_bic.chosen = []

def gamma_fit(l):
    x = np.exp(l)
    if len(x) < 3 or x.std() == 0:
        return empirical(l)
    a, _, sc = gamma.fit(x, floc=0)
    e = np.exp(LG)
    return gamma.sf(e, a, scale=sc), gamma.pdf(e, a, scale=sc) * e

def kernel(l):
    if len(l) < 3:
        return empirical(l)
    h = max(1.06 * min(l.std(), np.subtract(*np.percentile(l, [75, 25])) / 1.34) * len(l) ** -0.2, 0.05)
    return norm.sf(LG[:, None], l[None, :], h).mean(1), norm.pdf(LG[:, None], l[None, :], h).mean(1)

MODELS = {"empirical (ISI fractions, reference)": empirical, "one Gaussian on log ISI (lognormal)": mixture(1),
          "gamma on the ISI": gamma_fit, "mixture of 2 Gaussians": mixture(2), "mixture of 3 Gaussians": mixture(3),
          "mixture of 4 Gaussians": mixture(4), "mixture, K by BIC per trial": by_bic, "kernel estimate (one Gaussian per ISI)": kernel}


def k_means(X, seed, restarts=3, iters=25):
    r = np.random.default_rng(seed); best, cost = None, np.inf
    for _ in range(restarts):
        C = X[r.choice(len(X), k, replace=False)].copy()
        for _ in range(iters):
            a = cdist(X, C, "cityblock").argmin(1)
            new = np.stack([X[a == c].mean(0) if (a == c).any() else X[r.integers(len(X))] for c in range(k)])
            if np.allclose(new, C):
                break
            C = new
        D = cdist(X, C, "cityblock")
        if D.min(1).sum() < cost:
            best, cost = D.argmin(1), D.min(1).sum()
    return best

def nn(X, metric):
    D = cdist(X, X, metric); np.fill_diagonal(D, np.inf)
    return 100 * np.mean(y[D.argmin(1)] == y)

def across(XA, XB, metric):
    D = cdist(XA, XB, metric); np.fill_diagonal(D, np.inf)   # a trial's own other half is left out
    return 100 * np.mean(y[D.argmin(1)] == y)


print(f"{len(times)} trials, {k} neurons; median ISIs per full trial {int(np.median([len(l) for l in SETS['full']]))}, per half {int(np.median([len(l) for l in SETS['A']]))}")
print(f"{'model':44s} | full trial: 1-NN W1, 1-NN Hellinger, k-means W1 (range) | halves A->B: 1-NN W1, Hellinger")
for name, f in MODELS.items():
    out = {s: [f(l) if len(l) else (np.zeros(len(LG)), None) for l in SETS[s]] for s in SETS}
    S = {s: np.stack([o[0] for o in out[s]]).astype(np.float32) for s in SETS}
    has = out["full"][0][1] is not None or name.startswith("mix") or name.startswith("one") or name.startswith("gamma") or name.startswith("kernel")
    if name.startswith("empirical"):   # histogram density on the same grid for a Hellinger reference
        H = {s: np.stack([np.sqrt(np.histogram(l, bins=np.append(LG, LG[-1] + STEP)[::5])[0] / max(len(l), 1)) for l in SETS[s]]) for s in SETS}
    else:
        H = {s: np.stack([np.sqrt(np.maximum(o[1], 0) * STEP) if o[1] is not None else np.zeros(len(LG)) for o in out[s]]) for s in SETS}
    km = [100 * rp.matched_accuracy(y, k_means(S["full"], s)) for s in range(3)]
    print(f"{name:44s} | {nn(S['full'], 'cityblock'):5.1f}  {nn(H['full'], 'euclidean'):5.1f}  {np.mean(km):5.1f} ({min(km):.1f}-{max(km):.1f}) | "
          f"{across(S['A'], S['B'], 'cityblock'):5.1f}  {across(H['A'], H['B'], 'euclidean'):5.1f}", flush=True)
c = np.bincount(by_bic.chosen, minlength=5)[1:]
print("K chosen by BIC (all fits, full trials and halves): " + ", ".join(f"K={i + 1}: {100 * v / c.sum():.0f}%" for i, v in enumerate(c)))
