"""Shared functions for the TDA analysis of retinal spike trains.

The pipeline, for every trial (one neuron, one 10 s repeat of the frozen noise movie):

    spike train --> point cloud --> Vietoris-Rips barcodes (H0, H1)
                --> stable ranks (piecewise constant functions)
                --> L1 distances between stable ranks
                --> "which neuron is this trial from?" (1-NN and nearest mean)

The heavy lifting (Rips complex, persistent homology, stable ranks, L1 distances) is done by
`stablebear`. This file only builds the point clouds, calls stablebear and scores the results.

Two point clouds are used:

* the LINE cloud: the spike times as points on the real line (1-D). Two neighbouring spikes
  are joined by an edge at scale t = their interspike interval (ISI), so every H0 bar is
  [0, ISI). The H0 stable rank at t is therefore "the number of ISIs longer than t".
* the DELAY cloud: one 2-D point (log ISI_i, log ISI_{i+1}) per pair of consecutive ISIs.
  It shows which ISI follows which (bursts, pauses), not only the ISI distribution.
"""
import sys
from pathlib import Path

import numpy as np
import stablebear as sb
from stablebear import persistence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from load_dataset import load_dataset, neuron_summary  # noqa: E402

FIG_DIR = ROOT / "figures" / "analysis"

# Figures are shrunk to the text width of an A4 report, so use large fonts.
import matplotlib  # noqa: E402
matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                            "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11})
RESULTS_DIR = ROOT / "results"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_trials(n_neurons=40, min_rate=1.0, n_trials=30, seed=0, block=1):
    """Pick `n_neurons` random neurons and return their trials as samples.

    Only neurons firing at least `min_rate` Hz on average and with at least `n_trials`
    trials are eligible. We keep the first `n_trials` trials of each neuron, so every
    neuron has the same number of samples.

    `block` > 1 glues `block` consecutive trials into one longer sample (e.g. block=3 gives
    30 s samples). The ISIs of the trials are simply concatenated, so no fake ISI is
    created at the junction between two trials.

    Returns a dict with
        isis    list of 1-D arrays, the ISIs (s) of each sample
        spikes  list of 1-D arrays, the spike times (s) of each sample, for plotting
                (with block > 1, trial j of the block is shifted by j * 10 s)
        labels  int array, the neuron index (0..n_neurons-1) of each sample
        neurons list of neuron ids, neurons[label] is the id of that neuron
    """
    df = load_dataset()
    summ = neuron_summary(df)
    ok = summ[(summ.rate_hz >= min_rate) & (summ.n_trials >= n_trials)].index.to_numpy()
    rng = np.random.default_rng(seed)
    neurons = sorted(rng.choice(ok, n_neurons, replace=False))

    isis, spikes, labels = [], [], []
    by_neuron = {nid: g.sort_values("trial") for nid, g in df[df.neuron_id.isin(neurons)].groupby("neuron_id")}
    for label, nid in enumerate(neurons):
        g = by_neuron[nid]
        trials = list(g["isis"])[:n_trials]
        times = list(g["spike_times"])[:n_trials]
        for start in range(0, n_trials - block + 1, block):
            isis.append(np.concatenate(trials[start:start + block]))
            spikes.append(np.concatenate([times[start + j] + 10.0 * j for j in range(block)]))
            labels.append(label)
    return dict(isis=isis, spikes=spikes, labels=np.array(labels), neurons=neurons)


def shuffle_isis(isis, seed=0):
    """Control: randomly permute the ISIs inside every sample.

    The ISI distribution (and so the firing rate and the LINE barcode) stays exactly the
    same, but the order of the ISIs is destroyed. Anything the DELAY cloud still sees after
    this shuffle comes from the distribution alone.
    """
    rng = np.random.default_rng(seed)
    return [rng.permutation(x) for x in isis]


# ---------------------------------------------------------------------------
# Point clouds
# ---------------------------------------------------------------------------

def cloud_line(isis):
    """Spike times as a 1-D point cloud, shape (n_spikes, 1).

    We rebuild the spike times from the ISIs (cumulative sum starting at 0). Only the gaps
    between points matter for the Rips complex, so the absolute start time is irrelevant.
    """
    return np.concatenate([[0.0], np.cumsum(isis)])[:, None]


def cloud_delay(isis):
    """Delay embedding of the log ISIs, shape (n_isis - 1, 2).

    Point i is (log10 ISI_i, log10 ISI_{i+1}). We use log10 because ISIs range from about
    1 ms (inside a burst) to several seconds (pauses): on a linear scale all the short ISIs
    would collapse into one point near 0.
    With fewer than 2 ISIs we return a single point at the origin: one point has no finite
    bar, so its stable rank is the zero function.
    """
    x = np.log10(isis)
    if len(x) < 2:
        return np.zeros((1, 2))
    return np.column_stack([x[:-1], x[1:]])


# ---------------------------------------------------------------------------
# Homology and stable ranks (stablebear)
# ---------------------------------------------------------------------------

def barcodes(clouds, max_dim):
    """Vietoris-Rips barcodes of a list of point clouds (computed with Ripser in stablebear).

    `reduced=True` removes the one infinite H0 bar (the final connected component), so all
    bars are finite and every stable rank goes to 0 for large t.
    Returns a BarcodeTensor of shape (n, max_dim + 1): one barcode per cloud and degree.
    """
    tensor = sb.zeros((len(clouds),), dtype=sb.pcloud64)
    for i, c in enumerate(clouds):
        tensor[i] = np.ascontiguousarray(c, dtype=np.float64)
    return persistence.compute_persistent_homology(tensor, max_dim=max_dim, reduced=True)


def bars_of(cloud, max_dim):
    """Barcodes of ONE cloud as numpy arrays [(birth, death) rows for H0, for H1, ...].

    Used for plotting. (stablebear drops the degree axis when given a single cloud, so we
    pass the cloud twice and keep the first copy.)
    """
    B = barcodes([cloud, cloud], max_dim)
    return [np.asarray(B[0, d].to_numpy()).reshape(-1, 2) for d in range(max_dim + 1)]


def stable_ranks(clouds, max_dim):
    """Stable rank of each homology degree, as a list [H0 PcfTensor, H1 PcfTensor, ...].

    The stable rank of a barcode is the function t -> number of bars longer than t.
    It is a piecewise constant function (PCF) and it is stable: moving the points by
    at most e changes it by a shift of at most 2e (see the report, Section 1).
    """
    pcfs = persistence.barcode_to_stable_rank(barcodes(clouds, max_dim))  # shape (n, max_dim + 1)
    return [pcfs[:, d] for d in range(max_dim + 1)]


def mean_pcf(pcfs):
    """Pointwise mean of a list of PCFs (the "average stable rank"), as a 1-element PcfTensor."""
    return sb.mean(sb.PcfTensor(list(pcfs)))


def pcf_to_xy(pcf):
    """Breakpoints of a PCF: arrays (t, value), value[k] holds on [t[k], t[k+1])."""
    a = np.asarray(pcf.to_numpy())
    return a[:, 0], a[:, 1]


def pcf_on_grid(pcf, grid):
    """Values of a PCF at the points of `grid` (to plot means and standard deviations)."""
    t, v = pcf_to_xy(pcf)
    k = np.searchsorted(t, grid, side="right") - 1
    return np.where(k >= 0, v[np.clip(k, 0, None)], 0.0)


# ---------------------------------------------------------------------------
# Distances
#
# Every representation gives two matrices:
#   D[i, j] = distance between sample i and sample j                  (n x n)
#   M[i, c] = distance between sample i and the mean of neuron c      (n x C)
# M is computed leave-one-out: when sample i is the test sample, it is removed from the data,
# so it is not part of any neuron mean.
# ---------------------------------------------------------------------------

def _leave_one_out(M, labels):
    """Leave-one-out means without recomputing them: a speed-up, not a use of the answer.

    In leave-one-out, sample i is removed from the data and every neuron mean is recomputed
    from the remaining samples. Only the mean of i's own neuron contains f_i, so it is the
    only one that changes; all other means stay the same. We therefore compute the means
    once and correct this single entry. The label of i is used only to find which entry
    to correct; the prediction equals the one from recomputing all means without i
    (checked by brute force on 150 test trials: 150/150 identical).

    If neuron c has n samples with mean m, removing f_i gives the mean
    m' = (n m - f_i) / (n - 1), and m' - f_i = n / (n - 1) * (m - f_i).
    So for any norm, ||m' - f_i|| = n / (n - 1) * ||m - f_i||: we only rescale one entry.
    """
    counts = np.bincount(labels)
    M = M.copy()
    own = np.arange(len(labels))
    M[own, labels] *= counts[labels] / (counts[labels] - 1)
    return M


def pcf_distances(pcfs, labels):
    """L1 distances between stable ranks: D = int |f_i - f_j| dt, and M to the neuron means."""
    D = np.asarray(sb.pdist(pcfs, p=1).to_dense(), dtype=float)
    means = sb.concatenate([mean_pcf([pcfs[i] for i in np.flatnonzero(labels == c)])
                            for c in range(labels.max() + 1)])
    M = np.asarray(sb.cdist(pcfs, means, p=1), dtype=float).reshape(len(labels), -1)
    return D, _leave_one_out(M, labels)


def vector_distances(X, labels):
    """L1 distances between plain feature vectors (used by the non-TDA baselines)."""
    X = np.asarray(X, dtype=float).reshape(len(labels), -1)
    D = np.abs(X[:, None, :] - X[None, :, :]).sum(-1)
    means = np.stack([X[labels == c].mean(0) for c in range(labels.max() + 1)])
    M = np.abs(X[:, None, :] - means[None, :, :]).sum(-1)
    return D, _leave_one_out(M, labels)


def combine(parts):
    """Add several (D, M) pairs after dividing each by the median of its D.

    H0 and H1 stable ranks live on very different scales (there are many more H0 bars),
    so a plain sum would be dominated by H0. Dividing by the median distance gives each
    part the same typical size.
    """
    D_tot, M_tot = 0, 0
    for D, M in parts:
        s = np.median(D[np.triu_indices(len(D), 1)])
        D_tot, M_tot = D_tot + D / s, M_tot + M / s
    return D_tot, M_tot


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def score_1nn(D, labels):
    """Leave-one-out 1-nearest-neighbour accuracy.

    For every sample, find the closest OTHER sample; it is a hit if both come from the
    same neuron.
    """
    D = D.astype(float).copy()
    np.fill_diagonal(D, np.inf)
    return float(np.mean(labels[D.argmin(1)] == labels))


def score_nearest_mean(M, labels):
    """Leave-one-out nearest-mean accuracy: assign each sample to the closest neuron mean."""
    return float(np.mean(M.argmin(1) == labels))


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def classical_mds(D, dim=2):
    """Classical multidimensional scaling: 2-D points whose distances approximate D.

    Square the distances, double-centre them (B = -1/2 J D^2 J), and keep the top
    eigenvectors scaled by the square root of their eigenvalues.
    """
    n = len(D)
    J = np.eye(n) - 1 / n
    B = -0.5 * J @ (D ** 2) @ J
    w, v = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:dim]
    return v[:, idx] * np.sqrt(np.maximum(w[idx], 0))


def savefig(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved {path.relative_to(ROOT)}")
