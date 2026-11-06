"""Checks on the maths. Run from the repo root: python scripts/test_methods.py
The stablebear checks are skipped if it is not installed."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from barcodes import isi_bars, peak_events, smoothed_rate, time_bars
from run_pipeline import isi_bars_distance, time_bars_distance, window_count
from spike_distance import match_one_to_many, matching_matrix, vp_distance_matrix

rng = np.random.default_rng(0)


def slow_match(ta, ma, tb, mb, q):
    """The recursion written out, to check the vectorised version."""
    D = np.zeros((len(ta) + 1, len(tb) + 1))
    D[:, 0] = np.r_[0, np.cumsum(ma)]
    D[0, :] = np.r_[0, np.cumsum(mb)]
    for i in range(1, len(ta) + 1):
        for j in range(1, len(tb) + 1):
            a, b = ma[i - 1], mb[j - 1]
            D[i, j] = min(D[i - 1, j] + a, D[i, j - 1] + b, D[i - 1, j - 1]
                          + q * abs(ta[i - 1] - tb[j - 1]) * min(a, b) + abs(a - b))
    return D[-1, -1]


def slow_merge_persistence(r):
    """H0 of the superlevel sets of r by union-find: {peak index: bar length}."""
    parent, top, bars = {}, {}, {}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in np.argsort(-r, kind="stable"):
        i = int(i)
        parent[i], top[i] = i, i
        for j in (i - 1, i + 1):
            if j in parent:
                a, b = find(i), find(j)
                if a != b:
                    hi, lo = (a, b) if r[top[a]] >= r[top[b]] else (b, a)
                    if top[lo] != i:
                        bars[top[lo]] = r[top[lo]] - r[i]
                    parent[lo] = hi
    bars[top[find(int(np.argmax(r)))]] = r.max() - r.min()
    return bars


def test_matching_recursion():
    for _ in range(100):
        ta, tb = np.sort(rng.uniform(0, 5, rng.integers(1, 20))), np.sort(rng.uniform(0, 5, rng.integers(1, 20)))
        ma, mb = rng.uniform(.1, 3, len(ta)), rng.uniform(.1, 3, len(tb))
        q = rng.choice([.5, 3, 30])
        fast = match_one_to_many(ta, ma, tb[None], mb[None], np.array([len(tb)]), q)[0]
        assert abs(fast - slow_match(ta, ma, tb, mb, q)) < 1e-9
    print("matching: agrees with the written-out recursion")


def test_padding_and_matrix():
    events = [(np.sort(rng.uniform(0, 5, rng.integers(0, 15))), None) for _ in range(12)]
    events = [(t, rng.uniform(.2, 2, len(t))) for t, _ in events]
    D = matching_matrix(events, 3.0, jobs=1)
    for i in range(12):
        for j in range(12):
            want = slow_match(*events[i], *events[j], 3.0) if len(events[i][0]) and len(events[j][0]) \
                else max(events[i][1].sum(), events[j][1].sum())
            assert abs(D[i, j] - want) < 1e-9 or i == j
    assert np.allclose(D, D.T)
    print("matching: padded matrix agrees, with empty trains")


def exact_window_l1(x, y, s):
    """L1 distance between window counts, integrated exactly between change points."""
    e = np.sort(np.r_[x, x + s, y, y + s])
    mid, width = (e[1:] + e[:-1]) / 2, np.diff(e)
    a = np.searchsorted(x, mid, "right") - np.searchsorted(x, mid - s, "right")
    b = np.searchsorted(y, mid, "right") - np.searchsorted(y, mid - s, "right")
    return (np.abs(a - b) * width).sum() / s


def test_single_edits():
    q, s = 10.0, 0.2
    x = np.r_[1.0, 5.0, 9.0]
    for shift in (.005, .05, .1, .19, .25, 1.0):
        y = np.r_[1.0, 5.0 + shift, 9.0]
        vp = vp_distance_matrix([x, y], q)[0, 1]
        assert abs(vp - min(q * shift, 2)) < 1e-9
        assert abs(exact_window_l1(x, y, s) - vp) < 1e-9
        assert abs(window_count([x, y], s)[0, 1] - vp) <= 0.1 + 1e-9  # the grid step is s/20
    y = np.r_[1.0, 9.0]
    assert abs(vp_distance_matrix([x, y], q)[0, 1] - 1) < 1e-9 and abs(exact_window_l1(x, y, s) - 1) < 1e-9
    print("window count: single shift costs min(q d, 2) and one deletion costs 1, as in the spike distance")
    a = [np.sort(rng.uniform(0, 10, 40)) for _ in range(30)]
    V = vp_distance_matrix(a, q)
    W = np.array([[exact_window_l1(p, r, s) for r in a] for p in a])
    assert (W <= V + 1e-9).all()
    print(f"window count: never above the spike distance, median ratio {np.median(W[V > 0] / V[V > 0]):.2f}")


def test_betti_is_window_count():
    t = np.sort(rng.uniform(0, 10, 60)); s = 0.1
    b = time_bars(t, s)
    g = np.linspace(0, 10.2, 5001)
    alive = ((b[:, 0][None] <= g[:, None]) & (g[:, None] < b[:, 1][None])).sum(1)
    assert np.array_equal(alive, np.searchsorted(t, g, "right") - np.searchsorted(t, g - s, "right"))
    print("time bars: Betti curve equals the sliding window count")


def test_peaks_are_h0():
    c = 0.02 * np.sqrt(2 * np.pi)
    for trial in range(40):
        # 40 ms apart at sigma 20 ms gives two equal peaks with a shallow saddle
        t = np.r_[2.0, 2.043, np.sort(rng.uniform(3, 9.9, 30))]
        _, masses = peak_events(t, 0.02)
        bars = slow_merge_persistence(smoothed_rate(t, 0.02))
        want = np.sort([v * c for v in bars.values() if v * c >= 0.2])
        assert len(want) == len(masses) and np.allclose(want, np.sort(masses), atol=1e-6)
    print("peaks: masses equal H0 persistence of the rate, including tied peaks")


def test_stablebear():
    try:
        import stablebear as sb
        from stablebear import persistence
    except ImportError:
        print("stablebear not installed: skipped its checks")
        return
    a = [np.sort(rng.uniform(0, 10, 40)) for _ in range(8)]
    D1, D2 = time_bars_distance(a, 0.2), window_count(a, 0.2)
    assert np.abs(D1 - D2).max() < 0.05 * D2.max(), np.abs(D1 - D2).max()
    print("stablebear: time bars match the numpy window count")
    D3 = isi_bars_distance(a, False)
    d = np.abs(np.sort(np.diff(a[0]))[None] - np.sort(np.diff(a[1]))[None]).sum()
    assert abs(D3[0, 1] - d) < 1e-3 * d
    print("stablebear: stable rank L1 of ISI bars is the sorted ISI difference")
    tensor = sb.zeros((1,), dtype=sb.pcloud64)
    tensor[0] = a[0][:, None]
    bars = np.asarray(persistence.compute_persistent_homology(tensor, max_dim=0, reduced=True)[0, 0])
    assert np.allclose(np.sort(bars[:, 1] - bars[:, 0]), np.sort(np.diff(a[0])))
    print("stablebear: H0 Rips of the spike times gives the ISIs")


if __name__ == "__main__":
    test_matching_recursion()
    test_padding_and_matrix()
    test_single_edits()
    test_betti_is_window_count()
    test_peaks_are_h0()
    test_stablebear()
    print("all passed")
