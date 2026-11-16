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

def load_trials(n_neurons=40, min_rate=1.0, n_trials=30, seed=0, block=1,
                path=None, session=None, query=None, trial_s=9.94, exclude=()):
    """Pick `n_neurons` random neurons and return their trials as samples.

    With the default arguments this is the retina frozen-noise dataset, all sessions (the
    first report). `path`, `session`, `query` and `trial_s` select another table with the same
    columns (for example the Neuropixels movie repeats, 30 s each). `exclude` removes neuron
    ids from the draw (to pick a parameter on other neurons than the ones that are scored).

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
    df = load_dataset(path) if path is not None else load_dataset()
    if session is not None:
        df = df[df.session == session]
    if query:
        df = df.query(query)
    summ = neuron_summary(df, trial_s)
    ok = summ[(summ.rate_hz >= min_rate) & (summ.n_trials >= n_trials)].index.to_numpy()
    if len(exclude):
        ok = ok[~np.isin(ok, list(exclude))]
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
            shift = 10.0 if trial_s == 9.94 else trial_s
            spikes.append(np.concatenate([times[start + j] + shift * j for j in range(block)]))
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
# Extensions (second version of the report): a second stimulus, the Victor-Purpura distance,
# the random time cutoff control, k-medoids, and scores across two stimuli.
# ---------------------------------------------------------------------------

def load_stimuli(stimuli, n_neurons=40, min_rate=1.0, seed=0, trial_s=9.94):
    """The same neurons under several stimuli.

    `stimuli` is a list of dicts, one per stimulus, with the keys
        path      table file (None = the retina frozen-noise table)
        session   value of the `session` column to keep (None = all)
        query     optional pandas query on the table, for example "piece == 0"
        n         number of samples to keep per neuron (the first n, in trial order)
    Eligible neurons fire at least `min_rate` Hz in the FIRST stimulus (as in `load_trials`)
    and have at least `n` samples in every stimulus. Returns one dict per stimulus, like
    `load_trials`, all with the same `neurons` and the same label numbering.
    """
    tables, ok = [], None
    for k, st in enumerate(stimuli):
        df = load_dataset(st["path"]) if st.get("path") is not None else load_dataset()
        if st.get("session") is not None:
            df = df[df.session == st["session"]]
        if st.get("query"):
            df = df.query(st["query"])
        tables.append(df)
        count = df.groupby("neuron_id").size()
        good = set(count.index[count >= st["n"]])
        if k == 0:
            summ = neuron_summary(df, trial_s)
            good &= set(summ[summ.rate_hz >= min_rate].index)
        ok = good if ok is None else ok & good
    ok = np.array(sorted(ok))
    if len(ok) < n_neurons:
        raise ValueError(f"only {len(ok)} neurons have enough samples in every stimulus (need {n_neurons})")
    rng = np.random.default_rng(seed)
    neurons = sorted(rng.choice(ok, n_neurons, replace=False))
    out = []
    for df, st in zip(tables, stimuli):
        isis, spikes, labels = [], [], []
        by = {nid: g.sort_values("trial") for nid, g in df[df.neuron_id.isin(neurons)].groupby("neuron_id")}
        for label, nid in enumerate(neurons):
            for t in list(by[nid]["spike_times"])[:st["n"]]:
                t = np.asarray(t, dtype=float)
                spikes.append(t)
                isis.append(np.diff(t))
                labels.append(label)
        out.append(dict(isis=isis, spikes=spikes, labels=np.array(labels), neurons=neurons))
    return out


def random_time_cut(spikes, duration, seed=0, max_cut=0.25):
    """Control: cut a random amount of time off the start AND off the end of every sample.

    The two cuts are independent, each uniform between 0 and `max_cut` * duration, so between
    50 % and 100 % of the sample is kept (75 % on average) and the kept part differs from
    sample to sample. The ISI distribution of the neuron is (statistically) unchanged, but two
    samples of the same neuron no longer cover the same part of the stimulus and no longer
    have comparable spike counts. What a method loses under this cut is what it owed to the
    spike count and to seeing the same piece of the stimulus.
    Returns (spike times from the new start, ISIs).
    """
    rng = np.random.default_rng(seed)
    out_t, out_i = [], []
    for t in spikes:
        a = rng.uniform(0, max_cut) * duration
        b = duration - rng.uniform(0, max_cut) * duration
        t = np.asarray(t, dtype=float)
        t = t[(t >= a) & (t < b)] - a
        out_t.append(t)
        out_i.append(np.diff(t))
    return out_t, out_i


def cloud_line_log(isis, floor=1e-4):
    """LINE cloud whose H0 bars have length log10(ISI / 0.1 ms) instead of ISI.

    H0 of points on a line gives one bar [0, gap) per gap between neighbours. So we place the
    points with gaps log10(ISI) - log10(floor): the barcode is the LINE barcode with the bar
    lengths read on a logarithmic scale. `floor` = 0.1 ms is below every real ISI, so all gaps
    are positive.
    """
    gaps = np.maximum(np.log10(np.maximum(isis, floor)) - np.log10(floor), 1e-9)
    return np.concatenate([[0.0], np.cumsum(gaps)])[:, None]


def normalise_pcfs(pcfs, n):
    """Divide every stable rank by its value at 0 (its number of bars).

    The result starts at 1 and gives the SHARE of bars longer than t, so the number of bars
    (the spike count) no longer enters the distance. Empty barcodes stay the zero function.
    """
    out = []
    for i in range(n):
        a = np.asarray(pcfs[i].to_numpy(), dtype=float).reshape(-1, 2)
        top = a[0, 1] if len(a) and a[0, 1] > 0 else 1.0
        out.append(sb.Pcf(np.c_[a[:, 0], a[:, 1] / top]))
    return sb.PcfTensor(out)


def pcf_matrix(pcfs):
    """L1 distance matrix between stable ranks (no neuron means, so no labels needed)."""
    return np.asarray(sb.pdist(pcfs, p=1).to_dense(), dtype=float)


def vector_matrix(X):
    """L1 distance matrix between plain feature vectors."""
    X = np.asarray(X, dtype=float).reshape(len(X), -1)
    return np.abs(X[:, None, :] - X[None, :, :]).sum(-1)


def combine_matrices(mats):
    """Add distance matrices after dividing each by the median of its off-diagonal entries."""
    tot = 0
    for D in mats:
        v = D[np.triu_indices(len(D), 1)]
        v = v[v > 0]
        tot = tot + D / (np.median(v) if len(v) else 1.0)
    return tot


def _vp_kernel():
    """Victor-Purpura for all pairs, compiled with numba (threads: NUMBA_NUM_THREADS)."""
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


_VP = None


def victor_purpura(spikes, q):
    """Victor-Purpura spike-train distance between all pairs of samples (n x n matrix).

    The distance between two spike trains is the cheapest way to turn one into the other with
    three moves: delete a spike (cost 1), add a spike (cost 1), shift a spike by dt seconds
    (cost q * dt). q (per second) sets the time precision: for q = 0 the distance is the
    difference of the spike counts, for large q every spike that is not at exactly the same
    time is deleted and re-added. It is computed by dynamic programming, like an edit distance.
    Uses numba if it is installed; the fallback gives the same numbers, slowly.
    """
    global _VP
    trains = [np.asarray(t, dtype=np.float64) for t in spikes]
    try:
        if _VP is None:
            _VP = _vp_kernel()
        flat = np.concatenate(trains) if trains else np.zeros(0)
        start = np.r_[0, np.cumsum([len(t) for t in trains])].astype(np.int64)
        return _VP(flat, start, float(q))
    except ImportError:
        n = len(trains)
        out = np.zeros((n, n))
        for i in range(n):
            a = trains[i]
            for j in range(i + 1, n):
                b = trains[j]
                D = np.arange(len(b) + 1, dtype=float)
                idx = np.arange(len(b) + 1)
                for k, x in enumerate(a, 1):
                    c = np.empty(len(b) + 1)
                    c[0] = k
                    c[1:] = np.minimum(D[1:] + 1.0, D[:-1] + q * np.abs(x - b))
                    D = idx + np.minimum.accumulate(c - idx)      # also allows D[j-1] + 1
                out[i, j] = out[j, i] = D[-1]
        return out


def kmedoids(D, k, seed=0, restarts=5, iters=100):
    """k-medoids clustering of a distance matrix: like k-means, but every centre is one of
    the samples (a "medoid"), so it works for any distance, also without a notion of mean.
    No labels are used. Returns (cluster index of every sample, indices of the k medoids)."""
    rng = np.random.default_rng(seed)
    n = len(D)
    best, best_med, best_cost = None, None, np.inf
    for _ in range(restarts):
        medoids = [int(rng.integers(n))]                       # k-means++ style start
        near = D[medoids[0]].copy()
        for _ in range(k - 1):
            w = near ** 2
            medoids.append(int(rng.choice(n, p=w / w.sum())) if w.sum() > 0 else int(rng.integers(n)))
            near = np.minimum(near, D[medoids[-1]])
        medoids = np.array(medoids)
        for _ in range(iters):
            assign = D[:, medoids].argmin(1)
            new = medoids.copy()
            for c in range(k):
                members = np.flatnonzero(assign == c)
                if len(members):
                    new[c] = members[D[np.ix_(members, members)].sum(0).argmin()]
            if np.array_equal(new, medoids):
                break
            medoids = new
        assign = D[:, medoids].argmin(1)
        cost = D[np.arange(n), medoids[assign]].sum()
        if cost < best_cost:
            best, best_med, best_cost = assign, medoids, cost
    return best, best_med


def _match(labels, clusters, k):
    """Pair every cluster with one neuron so that as many samples as possible agree
    (Hungarian method). Returns (accuracy, neuron given to each cluster)."""
    from scipy.optimize import linear_sum_assignment
    table = np.zeros((labels.max() + 1, k))
    np.add.at(table, (labels, clusters), 1)
    rows, cols = linear_sum_assignment(-table)
    name = np.full(k, -1)
    name[cols] = rows
    return float(table[rows, cols].sum() / len(labels)), name


def score_kmedoids(D, labels, seeds=3):
    """Unsupervised score: cluster all samples into (number of neurons) groups with k-medoids,
    pair clusters with neurons, and count the samples that sit in their neuron's cluster.
    Mean over `seeds` runs of the clustering (each with 5 restarts)."""
    k = labels.max() + 1
    return float(np.mean([_match(labels, kmedoids(D, k, seed=s)[0], k)[0] for s in range(seeds)]))


def scores_across(D, labels, groups, seeds=3):
    """Scores for samples recorded under several stimuli. D is the distance matrix of all
    samples, `labels` the neuron of every sample, and `groups` a list with the indices of the
    samples of each stimulus (the first one is the reference stimulus).

        nn_within, kmed_within   inside the first stimulus only, as in the within part
        nn_pooled, kmed_pooled   ALL STIMULI MIXED: every neuron gives the same number of
                      samples from every stimulus. The samples go into one pool; which
                      stimulus a sample came from is not used. 1-NN: is the nearest other
                      sample of the pool from the same neuron? k-medoids: sort the pool into
                      (number of neurons) buckets without labels, pair buckets with neurons,
                      and count the samples that sit in their neuron's bucket.
        nn_other      a stricter 1-NN: the nearest sample is searched only among the samples
                      of the OTHER stimuli. In the mixed pool a sample can find its neighbour
                      in its own stimulus; here it cannot, so this shows what carries over.
    """
    k = labels.max() + 1
    first = groups[0]
    D0 = D[np.ix_(first, first)]
    r = dict(nn_within=score_1nn(D0, labels[first]), kmed_within=score_kmedoids(D0, labels[first], seeds))
    m = min(np.bincount(labels[g], minlength=k).min() for g in groups)      # samples per neuron and stimulus
    pool = np.array([i for g in groups for lab in range(k) for i in g[labels[g] == lab][:m]])
    Dp, yp = D[np.ix_(pool, pool)], labels[pool]
    r["nn_pooled"] = score_1nn(Dp, yp)
    r["kmed_pooled"] = score_kmedoids(Dp, yp, seeds)
    stim = np.concatenate([np.full(len(g), j) for j, g in enumerate(groups)])
    idx = np.concatenate(groups)
    Dx = D[np.ix_(idx, idx)].copy()
    Dx[stim[:, None] == stim[None, :]] = np.inf                             # own stimulus not allowed
    r["nn_other"] = float(np.mean(labels[idx][Dx.argmin(1)] == labels[idx]))
    return r


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
