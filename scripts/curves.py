"""Spike train -> curve on a grid (a piecewise constant function), compared by L1.

Ported from the web-chat handoff. Every signature is sampled on a regular grid and the
values are multiplied by the grid step, so the L1 distance between two vectors is the L1
integral between the two functions. Means of these vectors are the means of the PCFs
(what sb.mean computes), so k-means with mean curves works directly on them.

    sqrt_time_bars     sqrt(spikes in (t - s, t]): the square root of the time-bar Betti curve
    time_bars          the same without the square root (as run_pipeline's time_bars)
    isi_fractions      share of ISIs with log ISI > x: the ISI-bar stable rank / ISI count
    current_isi        log length of the gap that contains t
    burst_bars         1 inside a burst, 0 outside: Betti curve of barcodes.burst_bars
    burst_size         sqrt(size of the burst alive at t); a burst lives [first, last + s)
    coincidence        sqrt(sum over own spikes in (t - s, t] of other neurons' spikes
                       within +-10 ms); the weights come from curve_eval.py
    cond_isi_fractions isi_fractions of ISIs after a short / after a long ISI, concatenated
    burst_table        bursts with >= m spikes at 20 scales, divided by the spike count

Long trials would give very long vectors, so a curve never has more than MAX_POINTS
samples: past that the step grows. On the retina (9.94 s) no curve reaches the cap.
"""
import numpy as np

S = 2 / 30                 # window of the time bars, seconds (q = 30)
MIN_ISI = 1e-4             # seconds
COINCIDENCE = 0.01         # +- seconds
CURVE_STEP = 0.005         # grid step of current_isi, burst_size and coincidence
MAX_POINTS = 15000         # per curve; RAM guard for long trials
ISI_GRID = np.linspace(np.log(MIN_ISI), np.log(10.0), 230)
BURST_SCALES = np.geomspace(0.002, 1.0, 20)
BURST_SIZES = np.array([1, 2, 3, 4, 6, 8, 12, 16])


def grid(stop, step, max_points=MAX_POINTS):
    """Sample points on [0, stop) and the step actually used."""
    n = int(np.ceil(stop / step))
    if n > max_points:
        n, step = max_points, stop / max_points
    return np.arange(n) * step, step


def count_in_window(t, g, s):
    """Number of spikes in (g - s, g] for every grid point g. t must be sorted."""
    return np.searchsorted(t, g, "right") - np.searchsorted(t, g - s, "right")


def time_bars(t, dur, s=S, sqrt=False):
    """The Betti curve of the bars [t_k, t_k + s) is the count of spikes in (g - s, g]."""
    g, step = grid(dur + s, s / 20)
    c = count_in_window(t, g, s).astype(np.float64)
    return (np.sqrt(c) if sqrt else c) * step


def sqrt_time_bars(t, dur, s=S):
    return time_bars(t, dur, s, sqrt=True)


def fractions(isis):
    """Share of ISIs whose log length exceeds each grid value (stable rank / count)."""
    step = ISI_GRID[1] - ISI_GRID[0]
    if len(isis) == 0:
        return np.zeros(len(ISI_GRID))
    L = np.sort(np.log(np.maximum(isis, MIN_ISI)))
    return (1 - np.searchsorted(L, ISI_GRID, "right") / len(L)) * step


def isi_fractions(t, dur=None):
    return fractions(np.diff(t))


def cond_isi_fractions(t, dur=None):
    """ISIs split by whether the ISI before them is below the trial's median ISI."""
    isis = np.diff(t)
    if len(isis) < 2:
        return np.zeros(2 * len(ISI_GRID))
    prev, cur = isis[:-1], isis[1:]
    short = prev < np.median(isis)
    return np.r_[fractions(cur[short]), fractions(cur[~short])]


def current_isi(t, dur):
    """log(max(gap containing g, MIN_ISI)), gaps between consecutive points of [0, t, dur]."""
    g, step = grid(dur, CURVE_STEP)
    pts = np.r_[0.0, t[(t > 0) & (t < dur)], dur]
    gaps = np.log(np.maximum(np.diff(pts), MIN_ISI))
    i = np.clip(np.searchsorted(pts, g, "right") - 1, 0, len(gaps) - 1)
    return gaps[i] * step


def bursts(t, s):
    """(first spike, last spike, size) of the runs of spikes whose gaps are <= s."""
    if len(t) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, int)
    cut = np.flatnonzero(np.diff(t) > s) + 1
    first = np.r_[0, cut]
    last = np.r_[cut - 1, len(t) - 1]
    return t[first], t[last], last - first + 1


def burst_bars(t, dur, s=S):
    """Betti curve of the bars [first, last + s): 1 while a burst is alive. Grid step s/20."""
    g, step = grid(dur + s, s / 20)
    start, end, _ = bursts(t, s)
    f = np.zeros(len(g))
    if len(start):
        i = np.searchsorted(start, g, "right") - 1
        f[(i >= 0) & (g < end[np.maximum(i, 0)] + s)] = 1.0
    return f * step


def burst_size(t, dur, s=S):
    """sqrt(spikes in the burst alive at g), where a burst lives on [first, last + s)."""
    g, step = grid(dur + s, CURVE_STEP)
    start, end, size = bursts(t, s)
    f = np.zeros(len(g))
    if len(start):
        i = np.searchsorted(start, g, "right") - 1
        alive = (i >= 0) & (g < end[np.maximum(i, 0)] + s)
        f[alive] = np.sqrt(size[i[alive]])
    return f * step


def coincidence(t, w, dur, s=S):
    """sqrt(sum of w over own spikes in (g - s, g]); w[k] belongs to spike t[k]."""
    g, step = grid(dur + s, CURVE_STEP)
    cw = np.r_[0.0, np.cumsum(w)]
    total = cw[np.searchsorted(t, g, "right")] - cw[np.searchsorted(t, g - s, "right")]
    return np.sqrt(np.maximum(total, 0)) * step


def burst_table(t, dur=None):
    """Bursts at scale s with at least m spikes, divided by the spike count; 20 x 8, flat."""
    out = np.zeros((len(BURST_SCALES), len(BURST_SIZES)))
    if len(t):
        for i, s in enumerate(BURST_SCALES):
            size = bursts(t, s)[2]
            out[i] = (size[:, None] >= BURST_SIZES[None, :]).sum(0) / len(t)
    return out.ravel()
