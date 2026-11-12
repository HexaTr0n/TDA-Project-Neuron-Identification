"""Matching distances between trains of events. No embedding, no stablebear.

Two trains are compared by the cheapest way of turning one into the other. Events can only
be matched in time order, so this is a dynamic program over both trains.

    delete or insert an event of mass m      costs m
    match events of masses a and b, dt apart costs q * |dt| * min(a, b) + |a - b|

With all masses 1 this is the Victor-Purpura spike distance: insert or delete costs 1,
shifting costs q per second, and a shift is only worth it when |dt| < 2/q, so 1/q is the
timescale. q=0 gives |n_a - n_b|. With peaks of the smoothed rate as events (barcodes.py)
it is sublevel peak matching.

Cost grows with (total events)^2. For a fast approximation see window_count in
run_pipeline.py, which equals the spike distance for single edits and never exceeds it.
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

CHUNK = 256  # trains compared at once. Similar lengths, so little padding is wasted.


def match_one_to_many(tx, mx, tY, mY, lengths, q):
    """Distance from one train to each row of the zero-padded matrices tY, mY.

    The standard recursion is D[i, j] = min(D[i-1, j] + a, D[i, j-1] + b, D[i-1, j-1] + c).
    The D[i, j-1] term is solved in one step per row: with C the running sum of b,
    D[i, j] = C[j] + cummin(A - C), where A is the minimum of the other two terms.
    Padding has mass 0 and sits to the right of each train, so it never affects a result.
    """
    m = tY.shape[0]
    C = np.c_[np.zeros(m), np.cumsum(mY, axis=1)]
    D = C.copy()
    A = np.empty_like(D)
    deleted = 0.0
    for t, a in zip(tx, mx):
        deleted += a
        A[:, 0] = deleted
        match = q * np.abs(t - tY) * np.minimum(a, mY) + np.abs(a - mY)
        A[:, 1:] = np.minimum(D[:, 1:] + a, D[:, :-1] + match)
        D = np.minimum.accumulate(A - C, axis=1) + C
    return D[np.arange(m), lengths]


def _pad(arrays):
    Y = np.zeros((len(arrays), max(len(a) for a in arrays)))
    for i, a in enumerate(arrays):
        Y[i, :len(a)] = a
    return Y


_times, _masses, _lengths, _q = None, None, None, None


def _init(times, masses, q):
    global _times, _masses, _lengths, _q
    _times, _masses, _q = times, masses, q
    _lengths = np.array([len(t) for t in times])


def _row(i):
    """Distances from train i to every longer train (trains are sorted by length)."""
    n = len(_times)
    if _lengths[i] == 0:  # only insertions
        return i, np.array([m.sum() for m in _masses[i + 1:]])
    out = np.empty(n - i - 1)
    for s in range(i + 1, n, CHUNK):
        e = min(s + CHUNK, n)
        out[s - i - 1:e - i - 1] = match_one_to_many(
            _times[i], _masses[i], _pad(_times[s:e]), _pad(_masses[s:e]), _lengths[s:e], _q)
    return i, out


def matching_matrix(events, q, jobs=1):
    """Pairwise distances. events: list of (sorted times, masses)."""
    n = len(events)
    order = np.argsort([len(e[0]) for e in events], kind="stable")
    times = [np.asarray(events[i][0], dtype=np.float64) for i in order]
    masses = [np.asarray(events[i][1], dtype=np.float64) for i in order]
    D = np.zeros((n, n))
    with ProcessPoolExecutor(jobs, initializer=_init, initargs=(times, masses, q)) as pool:
        for i, row in pool.map(_row, range(n - 1), chunksize=8):
            D[order[i], order[i + 1:]] = row
            D[order[i + 1:], order[i]] = row
    return D


def vp_distance_matrix(trains, q, jobs=1):
    """Pairwise Victor-Purpura distances. trains: list of sorted spike time arrays."""
    return matching_matrix([(t, np.ones(len(t))) for t in trains], q, jobs)


# ------------------------------------------------------------------ fast exact version (numba)
# The same Victor-Purpura recursion, one pair at a time, compiled. Gives the same matrix as
# vp_distance_matrix (checked to 1e-13) about 100x faster.

def _numba_kernels():
    from numba import njit, prange

    @njit(cache=True)
    def pair(a, b, q, D):
        m = len(b)
        for j in range(m + 1):
            D[j] = j
        for i in range(1, len(a) + 1):
            prev = D[0]
            D[0] = i
            for j in range(1, m + 1):
                cur = D[j]
                D[j] = min(cur + 1.0, D[j - 1] + 1.0, prev + q * abs(a[i - 1] - b[j - 1]))
                prev = cur
        return D[m]

    @njit(parallel=True, cache=True)
    def all_pairs(flat, start, q):
        n = len(start) - 1
        out = np.zeros((n, n))
        longest = 0
        for i in range(n):
            longest = max(longest, start[i + 1] - start[i])
        for i in prange(n):
            D = np.empty(longest + 1)
            a = flat[start[i]:start[i + 1]]
            for j in range(i + 1, n):
                v = pair(a, flat[start[j]:start[j + 1]], q, D)
                out[i, j] = v
                out[j, i] = v
        return out

    return all_pairs


_ALL_PAIRS = None


def vp_distance_matrix_fast(trains, q):
    """Same as vp_distance_matrix, compiled with numba (threads: NUMBA_NUM_THREADS)."""
    global _ALL_PAIRS
    if _ALL_PAIRS is None:
        _ALL_PAIRS = _numba_kernels()
    flat = np.concatenate([np.asarray(t, np.float64) for t in trains]) if trains else np.zeros(0)
    start = np.r_[0, np.cumsum([len(t) for t in trains])].astype(np.int64)
    return _ALL_PAIRS(flat, start, float(q))
