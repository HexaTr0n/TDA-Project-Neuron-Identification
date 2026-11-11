"""Python wrapper for rips.c, and stable ranks on a common grid."""
import ctypes
from pathlib import Path
import numpy as np
_lib = ctypes.CDLL(str(Path(__file__).resolve().parent / "librips.so"))
_P = ctypes.POINTER
_d, _i = ctypes.c_double, ctypes.c_int
_lib.rips_batch.argtypes = [_P(_d), _P(ctypes.c_int64), _i, _i, _P(_d), _P(_d), _P(_d), _i, _P(_i)]
_lib.rips_batch.restype = None


def rips(clouds, cap=400):
    """clouds: list of (n_i, d) arrays. Returns h0 and h1 lists of bar lengths."""
    dim = clouds[0].shape[1]
    sizes = np.array([len(c) for c in clouds], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(sizes)]
    X = np.concatenate(clouds).astype(np.float64)
    h0 = np.zeros(offsets[-1]); b = np.zeros((len(clouds), cap)); d = np.zeros((len(clouds), cap))
    nh = np.zeros(len(clouds), dtype=np.int32)
    _lib.rips_batch(X.ctypes.data_as(_P(_d)), offsets.ctypes.data_as(_P(ctypes.c_int64)),
                    len(clouds), dim, h0.ctypes.data_as(_P(_d)), b.ctypes.data_as(_P(_d)),
                    d.ctypes.data_as(_P(_d)), cap, nh.ctypes.data_as(_P(_i)))
    H0 = [h0[offsets[i]:offsets[i] + max(sizes[i] - 1, 0)] for i in range(len(clouds))]
    H1 = [d[i, :nh[i]] - b[i, :nh[i]] for i in range(len(clouds))]
    B1 = [(b[i, :nh[i]], d[i, :nh[i]]) for i in range(len(clouds))]
    return H0, H1, B1


_lib.rips_matrix.argtypes = [_P(_d), _i, _P(_d), _P(_d), _P(_d), _i]
_lib.rips_matrix.restype = _i
_lib.cubical_batch.argtypes = [_P(_d), _i, _i, _i, _P(_d), _i, _P(_i)]
_lib.cubical_batch.restype = None


def rips_matrix(D, cap=4000):
    """H0 lengths and H1 (birth, death) pairs from a dissimilarity matrix."""
    D = np.array(D, dtype=np.float64)
    n = len(D)
    h0, b, d = np.zeros(n), np.zeros(cap), np.zeros(cap)
    nb = _lib.rips_matrix(D.ctypes.data_as(_P(_d)), n, h0.ctypes.data_as(_P(_d)),
                          b.ctypes.data_as(_P(_d)), d.ctypes.data_as(_P(_d)), cap)
    return h0[:n - 1], b[:nb], d[:nb]


def cubical(images, cap=2000):
    """Sublevel-set H0 bar lengths of each image in an (N, h, w) array."""
    X = np.array(images, dtype=np.float64)
    N, h, w = X.shape
    bars = np.zeros((N, cap)); nb = np.zeros(N, dtype=np.int32)
    _lib.cubical_batch(X.ctypes.data_as(_P(_d)), N, h, w, bars.ctypes.data_as(_P(_d)), cap,
                       nb.ctypes.data_as(_P(_i)))
    return [bars[i, :nb[i]] for i in range(N)]


_lib.ordered_match.argtypes = [_P(_d), _P(_d), _P(ctypes.c_int64), _i, _d, _P(_d)]
_lib.ordered_match.restype = None


def ordered_match(events, q):
    """events: list of (times, masses) per trial, times sorted. Returns the N x N matrix."""
    sizes = np.array([len(e[0]) for e in events], dtype=np.int64)
    off = np.r_[0, np.cumsum(sizes)]
    t = np.concatenate([e[0] for e in events]).astype(np.float64)
    m = np.concatenate([e[1] for e in events]).astype(np.float64)
    D = np.zeros((len(events), len(events)))
    _lib.ordered_match(t.ctypes.data_as(_P(_d)), m.ctypes.data_as(_P(_d)),
                       off.ctypes.data_as(_P(ctypes.c_int64)), len(events), q, D.ctypes.data_as(_P(_d)))
    return D


def stable_ranks(lengths, grid):
    """Number of bars longer than t, for each t in grid. One row per cloud."""
    out = np.zeros((len(lengths), len(grid)), dtype=np.float32)
    for i, x in enumerate(lengths):
        if len(x):
            out[i] = len(x) - np.searchsorted(np.sort(x), grid, side="right")
    return out
