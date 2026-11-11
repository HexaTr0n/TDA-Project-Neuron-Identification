"""Fitted next-ISI models. Each uses only one trial's spikes, except shared HMM
parameters, which come from other recording sessions and never see the target neuron.
Every function works on x, a trial's log ISIs."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import gammaln, logsumexp

SD_FLOOR = 0.2


def norm_logpdf(y, m, s):
    return -0.5 * ((y - m) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)


def local_lognormal(x, W):
    """(mean, sd) of the last W log ISIs, for predicting ISI W, W+1, ..."""
    win = sliding_window_view(x[:-1], W)
    return win.mean(1), np.maximum(win.std(1), SD_FLOOR)


def local_gamma(x, W):
    """Gamma shape k and scale from the last W ISIs (Minka's approximation)."""
    win = sliding_window_view(x[:-1], W)
    m = np.exp(win).mean(1)
    s = np.maximum(np.log(m) - win.mean(1), 1e-3)
    k = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
    return k, m / k


def gamma_logpdf_log(y, k, th):
    """Density of log ISI when the ISI is gamma(k, th)."""
    return k * (y - np.log(th)) - np.exp(y) / th - gammaln(k)


def local_mixture(x, W=40, iters=15):
    """Two-Gaussian EM on each window of the last W log ISIs, fitted per spike."""
    X = sliding_window_view(x[:-1], W)
    q = np.quantile(X, [0.25, 0.75], axis=1)
    mu = q.T.copy()
    sd = np.full(mu.shape, max(0.3, SD_FLOOR))
    pi = np.full(mu.shape, 0.5)
    for _ in range(iters):
        lp = np.log(pi)[:, None, :] + norm_logpdf(X[:, :, None], mu[:, None, :], sd[:, None, :])
        r = np.exp(lp - logsumexp(lp, axis=2, keepdims=True))
        nk = r.sum(1) + 1e-6
        pi = nk / W
        mu = (r * X[:, :, None]).sum(1) / nk
        sd = np.maximum(np.sqrt((r * (X[:, :, None] - mu[:, None, :]) ** 2).sum(1) / nk), SD_FLOOR)
    return pi, mu, sd


def mixture_logpdf(y, pi, mu, sd):
    return logsumexp(np.log(pi) + norm_logpdf(y[:, None], mu, sd), axis=1)


def mixture_quantiles(pi, mu, sd, levels, grid=None):
    """Quantile function of each row's mixture, for Wasserstein-2 coordinates."""
    from scipy.special import ndtr
    grid = np.linspace(-9, 4, 400) if grid is None else grid
    cdf = (pi[:, None, :] * ndtr((grid[None, :, None] - mu[:, None, :]) / sd[:, None, :])).sum(2)
    return np.stack([np.interp(levels, c, grid) for c in cdf])


class HMM:
    """Gaussian HMM on log ISIs. The states and transitions are shared by everyone."""

    def __init__(self, K, pooled):
        self.K = K
        self.mu = np.quantile(pooled, (np.arange(K) + 0.5) / K)
        self.sd = np.full(K, np.std(pooled) / K + 0.2)
        self.A = np.full((K, K), 0.1 / K) + 0.9 * np.eye(K)
        self.A /= self.A.sum(1, keepdims=True)
        self.pi = np.ones(K) / K

    def emissions(self, x):
        return norm_logpdf(x[:, None], self.mu, self.sd)

    def fit(self, seqs, iters=15):
        for it in range(iters):
            xi = np.zeros((self.K, self.K)); gam0 = np.zeros(self.K)
            sw = np.zeros(self.K); swx = np.zeros(self.K); swxx = np.zeros(self.K)
            for x in seqs:
                b = np.exp(self.emissions(x) - self.emissions(x).max(1, keepdims=True)) + 1e-300
                T = len(x)
                al = np.zeros((T, self.K)); c = np.zeros(T)
                al[0] = self.pi * b[0]; c[0] = al[0].sum(); al[0] /= c[0]
                for t in range(1, T):
                    al[t] = (al[t - 1] @ self.A) * b[t]; c[t] = al[t].sum(); al[t] /= c[t]
                be = np.ones((T, self.K))
                for t in range(T - 2, -1, -1):
                    be[t] = self.A @ (b[t + 1] * be[t + 1]) / c[t + 1]
                g = al * be; g /= g.sum(1, keepdims=True)
                for t in range(T - 1):
                    v = al[t, :, None] * self.A * (b[t + 1] * be[t + 1])[None, :]
                    xi += v / v.sum()
                gam0 += g[0]; sw += g.sum(0); swx += (g * x[:, None]).sum(0)
                swxx += (g * x[:, None] ** 2).sum(0)
            self.A = (xi + 1e-3) / (xi + 1e-3).sum(1, keepdims=True)
            self.pi = gam0 / gam0.sum()
            self.mu = swx / sw
            self.sd = np.maximum(np.sqrt(np.maximum(swxx / sw - self.mu ** 2, 0)), SD_FLOOR)
        return self

    def predict(self, x):
        """State weights for each next ISI given the ISIs so far. Row t predicts x[t]."""
        T = len(x)
        b = np.exp(self.emissions(x) - self.emissions(x).max(1, keepdims=True)) + 1e-300
        P = np.zeros((T, self.K)); p = self.pi.copy()
        for t in range(T):
            P[t] = p
            f = p * b[t]; f /= f.sum()
            p = f @ self.A
        return P

    def logpdf(self, x):
        P = self.predict(x)
        return logsumexp(np.log(P + 1e-300) + self.emissions(x), axis=1)
