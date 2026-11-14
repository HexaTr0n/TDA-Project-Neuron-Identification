"""The pdf-per-spike embedding (`predictive_state`) used by sb_full_table.py, and the first
version of the ISI delay embedding. Pure numpy, no stablebear.

Kept under its own name because scripts/embeddings.py holds the newer delay embedding.

Every embedding has the same signature:

    embed(isis, spike_times, **params) -> (n_points, dim) float64 array, or None

None means the trial is too short for this embedding. Add a new embedding by
writing a function and registering it in EMBEDDINGS.

Anything that is a set of points under Euclidean distance fits this interface
directly. A non-Euclidean embedding should instead produce a distance matrix
(stablebear accepts a DistanceMatrixTensor in place of point clouds).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MIN_ISI = 1e-4  # seconds; guards log(0) for spikes sorted to the same sample
LOG_ISI_RANGE = (np.log(1e-3), np.log(10.0))  # 1 ms to 10 s


_shuffle_rng = np.random.default_rng(0)


def isi_delay(isis, spike_times=None, m=3, stride=1, log=True, center=False, shuffle=False):
    """Delay embedding of the ISI sequence, one point per window of m ISIs.

    m        embedding dimension. m=1 is the ISI histogram in disguise.
    stride   1 = overlapping windows (one point per spike), m = disjoint tokens.
    log      use log ISIs. Raw ISIs are dominated by the few longest gaps. This also
             makes the cloud blind to a global rate change: scaling all ISIs by c
             only translates the cloud, and Rips persistence ignores translations.
    center   subtract the trial's mean (log) ISI. For the same reason this changes
             nothing on a single-trial cloud. It only matters when trials are pooled.
    shuffle  permute the ISIs first. Control: keeps the ISI histogram, destroys the
             sequence. If m>1 beats this, the cloud sees more than the histogram.

    Coordinates are divided by sqrt(m) so that scales stay comparable across m.
    Rips persistence cannot see time reversal in this embedding: reversing the
    train flips the coordinate axes, which is an isometry of the cloud.
    """
    x = np.asarray(isis, dtype=np.float64)
    if len(x) < m:
        return None
    if shuffle:
        x = _shuffle_rng.permutation(x)
    if log:
        x = np.log(np.maximum(x, MIN_ISI))
    if center:
        x = x - x.mean()
    return sliding_window_view(x, m)[::stride].astype(np.float64) / np.sqrt(m)


def _quantile_coords(weights, y, k):
    """Euclidean distance between rows = Wasserstein-2 distance between the pdfs."""
    order = np.argsort(y)
    w = weights[:, order]
    cdf = np.cumsum(w, axis=1) - 0.5 * w
    levels = (np.arange(k) + 0.5) / k
    return np.stack([np.interp(levels, c, y[order]) for c in cdf]) / np.sqrt(k)


def _sqrt_density_coords(weights, y, k, smoothing):
    """Euclidean distance between rows = Hellinger distance between the pdfs."""
    grid = np.linspace(*LOG_ISI_RANGE, k)
    q = weights @ np.exp(-(grid[None, :] - y[:, None]) ** 2 / (2 * smoothing ** 2))
    return np.sqrt(q / q.sum(axis=1, keepdims=True) / 2)


def predictive_state(isis, spike_times=None, L=2, bandwidth=0.5, distance="w2",
                     n_coords=32, smoothing=0.3, shuffle=False):
    """One point per spike: the pdf of the next log ISI given the last L ISIs.

    Each pdf is a kernel average of what followed similar histories in the same trial,
    so nothing is fitted and several peaks are allowed. If ISIs are independent every
    pdf is the same and the cloud collapses, up to estimation noise, so its shape
    reflects history dependence. Keep the shuffled control: estimation noise still gets
    larger with fewer ISIs, even at a fixed cloud size.

    L          history length. L=1 puts the cloud on a curve, loops need L >= 2.
    bandwidth  how similar a history must be, as an RMS log-ISI difference per ISI.
    distance   "w2" (Wasserstein-2, in log-ISI units) or "hellinger".
    n_coords   quantile levels for w2, grid points for hellinger.
    smoothing  width of each next-ISI bump, hellinger only.
    shuffle    permute the ISIs first. Control, see isi_delay.

    A spike's own next ISI is left out, otherwise the point would mostly encode that
    ISI and the cloud would turn back into a delay embedding.
    """
    x = np.asarray(isis, dtype=np.float64)
    if len(x) < L + 5:
        return None
    if shuffle:
        x = _shuffle_rng.permutation(x)
    x = np.log(np.maximum(x, MIN_ISI))

    history = sliding_window_view(x[:-1], L)
    next_isi = x[L:]
    d2 = ((history[:, None, :] - history[None, :, :]) ** 2).mean(axis=-1)
    np.fill_diagonal(d2, np.inf)
    # Measured from the nearest history, so isolated spikes still get usable weights.
    d2 -= d2.min(axis=1, keepdims=True)
    weights = np.exp(-d2 / (2 * bandwidth ** 2))
    weights /= weights.sum(axis=1, keepdims=True)

    if distance == "w2":
        return _quantile_coords(weights, next_isi, n_coords)
    if distance == "hellinger":
        return _sqrt_density_coords(weights, next_isi, n_coords, smoothing)
    raise ValueError(f"unknown distance {distance!r}")


EMBEDDINGS = {
    "isi_delay": isi_delay,
    "predictive_state": predictive_state,
    # "kernel_delay": ...,
    # "windowed_spectrum": ...,
}
