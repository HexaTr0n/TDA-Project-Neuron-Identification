"""Spike train -> barcode, for barcodes that can be written down directly.

These sit next to embeddings.py. There a train becomes a point cloud and Rips computes
the barcode. Here the bars are known in closed form, so stablebear only has to turn them
into piecewise constant functions and compare them.

    time_bars    one bar [t, t + s) per spike. The filtration parameter is movie time, so
                 unlike Rips this keeps WHEN things happen. The Betti curve is the number
                 of spikes in the last s seconds, which is exactly the sliding window count.
    isi_bars     one bar per ISI. With log=False these are the H0 bars of the spike times
                 as a 1D Rips cloud. The stable rank says how many ISIs exceed each length.
    burst_bars   one bar per burst at scale s: H0 of the spike train thickened by s.
    peak_events  H0 persistence of the smoothed rate: one event per peak, with its time.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences

MIN_ISI = 1e-4     # seconds; same guard as embeddings.py
DURATION = 9.94    # seconds in one frozen noise trial
MIN_MASS = 0.2     # peaks smaller than this many spikes are dropped


def shuffle_isis(spike_times, rng):
    """Control: same ISIs in random order. Keeps the histogram, destroys stimulus locking."""
    t = np.asarray(spike_times, dtype=np.float64)
    return t[0] + np.r_[0, np.cumsum(rng.permutation(np.diff(t)))]


def time_bars(spike_times, s):
    t = np.asarray(spike_times, dtype=np.float64)
    return np.c_[t, t + s]


def burst_bars(spike_times, s):
    """One bar [first spike, last spike + s) per burst, where a burst is a run of spikes
    with gaps of at most s.

    Thicken every spike t to the interval [t, t + s) and take the union. Intervals overlap
    when the gap is below s, so the pieces of the union are the bursts. These are the H0
    classes of the thickened train. The Betti curve is 1 inside a burst and 0 outside, so
    unlike time_bars it ignores how many spikes a burst has.
    """
    t = np.asarray(spike_times, dtype=np.float64)
    cut = np.flatnonzero(np.diff(t) > s)
    return np.c_[t[np.r_[0, cut + 1]], t[np.r_[cut, len(t) - 1]] + s]


def isi_bars(spike_times, log=True):
    """log=True stretches short ISIs. Bars measure log(ISI / MIN_ISI), so they stay positive."""
    lengths = np.diff(np.asarray(spike_times, dtype=np.float64))
    if log:
        lengths = np.log(np.maximum(lengths, MIN_ISI) / MIN_ISI)
    return np.c_[np.zeros(len(lengths)), lengths]


def smoothed_rate(spike_times, sigma, dt=0.001):
    n = int(round(DURATION / dt))
    binned = np.zeros(n)
    np.add.at(binned, np.minimum((np.asarray(spike_times) / dt).astype(int), n - 1), 1 / dt)
    return gaussian_filter1d(binned, sigma / dt, mode="constant")


def peak_events(spike_times, sigma, dt=0.001):
    """(times, masses) of the peaks of the rate, smoothed with a Gaussian of width sigma.

    Sweep a level down from the top of r(t). A component of {r > level} is born at a peak
    and dies when it merges with a higher one, so each peak gets one H0 bar, and the bar's
    length is the peak's prominence. Its mass is prominence * sigma * sqrt(2 pi), the
    size of the peak in spikes: an isolated spike has mass close to 1.
    """
    r = smoothed_rate(spike_times, sigma, dt)
    # Two spikes give two exactly equal peaks, and scipy would call both the taller one.
    # A negligible ramp breaks the tie, so that one dies at the saddle between them.
    r = r + r.max() * 1e-9 * np.arange(len(r)) / len(r)
    padded = np.r_[0, r, 0]
    peaks = find_peaks(padded)[0]
    if len(peaks) == 0:
        return np.zeros(0), np.zeros(0)
    prominence = peak_prominences(padded, peaks)[0]
    peaks = peaks - 1
    prominence[np.argmax(r[peaks])] = r[peaks].max() - r.min()  # the highest peak never dies
    mass = prominence * sigma * np.sqrt(2 * np.pi)
    keep = mass >= MIN_MASS
    return peaks[keep] * dt, mass[keep]
