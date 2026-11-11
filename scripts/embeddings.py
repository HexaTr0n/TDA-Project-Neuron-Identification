"""Spike train -> point cloud embeddings. Pure numpy, no stablebear.

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


EMBEDDINGS = {
    "isi_delay": isi_delay,
}
