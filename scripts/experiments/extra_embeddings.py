"""Embeddings beyond embeddings.py. All are computed from single trials."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter1d
from fits import local_gamma, local_lognormal, local_mixture, mixture_quantiles

TRIAL_S = 9.94


def logx(isis):
    return np.log(np.maximum(np.asarray(isis, dtype=np.float64), 1e-4))


def fitted_lognormal(isis, W=20):
    """Point: (mean, sd) of the last W log ISIs. Euclidean distance here equals the
    Wasserstein-2 distance between the two fitted lognormals in log-ISI space."""
    x = logx(isis)
    if len(x) < W + 5:
        return None
    return np.column_stack(local_lognormal(x, W))


def fitted_gamma(isis, W=20):
    """Point: (log mean, log shape) of a gamma fitted to the last W ISIs."""
    x = logx(isis)
    if len(x) < W + 5:
        return None
    k, th = local_gamma(x, W)
    return np.column_stack([np.log(k * th), np.log(k)])


def fitted_mixture(isis, W=40, n_levels=32):
    """Point: quantiles of a two-Gaussian mixture fitted to the last W log ISIs."""
    x = logx(isis)
    if len(x) < W + 5:
        return None
    levels = (np.arange(n_levels) + 0.5) / n_levels
    return mixture_quantiles(*local_mixture(x, W), levels) / np.sqrt(n_levels)


def hmm_states(isis, hmm):
    """Point: sqrt of the predicted state weights. Shared states, so one trial alone
    decides each point. Euclidean distance is a Hellinger distance on the weights."""
    x = logx(isis)
    if len(x) < 10:
        return None
    return np.sqrt(hmm.predict(x)[1:] / 2)


def time_tagged(isis, spike_times, m=1, alpha=2.0):
    """Delay embedding with the spike's time attached, so the movie moment survives.
    alpha: log-ISI units per second, so 1/alpha seconds costs as much as a factor e."""
    x = logx(isis)
    if len(x) < m:
        return None
    v = sliding_window_view(x, m) / np.sqrt(m)
    return np.column_stack([alpha * np.asarray(spike_times)[:len(v)], v])


def spectrum_grid(sigma=0.5, fmin=1.0, fmax=100.0, nf=24):
    return np.arange(sigma, TRIAL_S - sigma / 2, sigma), np.geomspace(fmin, fmax, nf)


def windowed_spectrum(spike_times, sigma=0.5, fmin=1.0, fmax=100.0, nf=24, min_eff=4.0, aligned=False):
    """Point per window: log P_t(f), with P=1 for a Poisson train, as on the slide.
    aligned=True keeps every window in place (NaN when empty) for a time-aligned baseline."""
    t = np.asarray(spike_times, dtype=np.float64)
    centres, freqs = spectrum_grid(sigma, fmin, fmax, nf)
    if len(t) < 5:
        return None
    w = np.exp(-(t[None, :] - centres[:, None]) ** 2 / (2 * sigma ** 2))
    phase = np.exp(-2j * np.pi * freqs[:, None] * t[None, :])
    S = np.abs(w @ phase.T) ** 2
    w2 = (w ** 2).sum(1)
    eff = w.sum(1) ** 2 / np.maximum(w2, 1e-12)
    logP = np.log(np.maximum(S / np.maximum(w2, 1e-12)[:, None], 1e-3))
    good = eff >= min_eff
    if aligned:
        logP[~good] = np.nan
        return logP
    return logP[good] if good.sum() >= 3 else None


def binned(spike_times, dt=0.002, T=TRIAL_S):
    n = int(round(T / dt))
    b = np.zeros(n, dtype=np.float32)
    idx = np.minimum((np.asarray(spike_times) / dt).astype(int), n - 1)
    np.add.at(b, idx, 1)
    return b


def ccg_size(n, dt=0.002, maxlag=0.05):
    return 2 ** int(np.ceil(np.log2(n + int(round(maxlag / dt)) + 1)))


def ccg_matrix(target, F_partners, n_partners, dt=0.002, maxlag=0.05, smooth=2.0, step=3):
    """Single-trial cross-correlograms of one binned train against each partner.
    F_partners: rfft of the binned partners. Returns (partners, lags): coincidences per
    sqrt(n_i n_j), smoothed, minus the mean over lags."""
    L = int(round(maxlag / dt))
    size = ccg_size(len(target), dt, maxlag)
    c = np.fft.irfft(np.conj(np.fft.rfft(target, size))[None, :] * F_partners, size, axis=1)
    c = np.concatenate([c[:, -L:], c[:, :L + 1]], axis=1)
    norm = np.sqrt(np.maximum(target.sum() * n_partners, 1.0))[:, None]
    c = gaussian_filter1d(c / norm, smooth, axis=1)
    return (c - c.mean(1, keepdims=True))[:, ::step]


def ccg_summary(c, dt=0.006):
    """Point per partner: (lag, signed peak, width), each scaled to unit-ish spread."""
    i = np.abs(c).argmax(1)
    peak = c[np.arange(len(c)), i]
    lag = (i - c.shape[1] // 2) * dt
    width = (np.abs(c) >= 0.5 * np.abs(peak)[:, None]).sum(1) * dt
    return np.column_stack([lag / 0.02, peak / (np.std(peak) + 1e-9), width / 0.02])
