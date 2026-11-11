"""Spike trains -> distance matrix -> neuron re-identification.

Run from the repo root:
    python scripts/run_pipeline.py --list
    python scripts/run_pipeline.py m3_fixed m3_fixed_shuffled
    python scripts/run_pipeline.py "time_bars_*" "isi_bars_*" isi_histogram
    python scripts/run_pipeline.py --neurons 60 --jobs 4     # every config, on 60 neurons

Every config gives one or more distance matrices. Each one adds a row to
results/results.csv and is saved in results/distances/<session>_<neurons>/, where
compare.py reads it. Saved work is reused, so rerunning is free; use --force after
changing code or settings. See scripts/README.md for what each method does.
"""
import argparse
import fnmatch
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from barcodes import burst_bars, isi_bars, peak_events, shuffle_isis, smoothed_rate, time_bars
from embeddings import EMBEDDINGS
from load_dataset import DEFAULT_PATH, load_dataset
from spike_distance import matching_matrix, vp_distance_matrix

RESULTS = Path(__file__).resolve().parent.parent / "results"


@dataclass(frozen=True)
class Config:
    method: str = "rips"      # "rips", or a name from METHODS
    params: dict = field(default_factory=dict)  # for the embedding or the method
    # data
    sessions: tuple = ("20180126_sr_le_sp",)  # most spikes per trial, 156 neurons
    min_isis: int = 30        # drop trials with fewer ISIs
    pool_trials: int = 1      # clouds per neuron = n_trials // pool_trials
    # embedding (rips only)
    embedding: str = "isi_delay"
    # cloud size: None keeps every point, an int subsamples to exactly that many.
    # A stable rank starts at the bar count, so unequal sizes leak the spike count.
    n_points: int = None
    n_subsamples: int = 1     # stable ranks are averaged over this many subsamples
    regions: int = 0          # > 0 gives each subsample one barcode per Voronoi region
    # persistent homology
    max_dim: int = 1
    # summary and distance
    summary: str = "stable_rank"   # stable_rank | betti_curve
    p: int = 1
    seed: int = 0


def delay(m, **params):
    return dict(params=dict(m=m, **params))


def method(name, **params):
    return Config(method=name, params=params)


FIXED = dict(n_points=25, n_subsamples=20)
WINDOWS = {"s667": 2 / 3, "s200": 0.2, "s67": 2 / 30, "s20": 0.02}  # s = 2/q for q = 3, 10, 30, 100


CONFIGS = {
    # --- Rips on the ISI delay embedding
    # Every point kept: the cloud size carries the spike count.
    "m1_all": Config(max_dim=0, **delay(1)),
    "m2_all": Config(**delay(2)),
    "m3_all": Config(**delay(3)),
    "m5_all": Config(**delay(5)),
    # Fixed size and log ISIs: neither the rate nor the spike count reaches the barcode.
    "m1_fixed": Config(max_dim=0, **FIXED, **delay(1)),
    "m2_fixed": Config(**FIXED, **delay(2)),
    "m3_fixed": Config(**FIXED, **delay(3)),
    "m5_fixed": Config(**FIXED, **delay(5)),
    # Controls for the above: same ISIs, sequence destroyed.
    "m3_fixed_shuffled": Config(**FIXED, **delay(3, shuffle=True)),
    "m5_fixed_shuffled": Config(**FIXED, **delay(5, shuffle=True)),
    # Pooling 5 trials gives denser clouds, which H1 needs.
    "m3_fixed_pool5": Config(pool_trials=5, n_points=125, n_subsamples=20, **delay(3, center=True)),
    # Plain ISIs in seconds scale with the firing rate, so here the rate CAN reach the barcode.
    "m3_fixed_nolog": Config(**FIXED, **delay(3, log=False)),
    "m3_fixed_nolog_shuffled": Config(**FIXED, **delay(3, log=False, shuffle=True)),
    # Delay regions: split each cloud around fixed anchors, so the barcode keeps where it was.
    "regions_m3_k4": Config(regions=4, **FIXED, **delay(3)),
    "regions_m3_k8": Config(regions=8, **FIXED, **delay(3)),
    "regions_m3_k8_shuffled": Config(regions=8, **FIXED, **delay(3, shuffle=True)),
    # --- Baselines without topology
    "spike_count": method("spike_count"),
    "isi_histogram": method("isi_histogram"),
    "isi_histogram_centered": method("isi_histogram", center=True),
    **{f"window_count_{k}": method("window_count", s=s) for k, s in WINDOWS.items()},
    **{f"van_rossum_{int(sg * 1000)}": method("van_rossum", sigma=sg)
       for sg in (0.005, 0.01, 0.02, 0.05, 0.1)},
    "van_rossum_20_shuffled": method("van_rossum", sigma=0.02, shuffle=True),
    **{f"spike_distance_q{q}": method("spike_distance", q=float(q)) for q in (1, 3, 10, 30, 100)},
    # --- Barcodes written by hand, summarised by stablebear
    **{f"time_bars_{k}": method("time_bars", s=s) for k, s in WINDOWS.items()},
    "time_bars_s67_shuffled": method("time_bars", s=2 / 30, shuffle=True),
    # One bar per burst: real H0 of the thickened train, and it keeps time.
    **{f"burst_bars_{k}": method("burst_bars", s=s) for k, s in WINDOWS.items()},
    "burst_bars_s67_shuffled": method("burst_bars", s=2 / 30, shuffle=True),
    "isi_bars_raw": method("isi_bars", log=False),
    "isi_bars_log": method("isi_bars", log=True),
    # --- H0 of the smoothed rate, compared in time
    **{f"peak_s{int(sg * 1000)}_q{q}": method("peak_matching", sigma=sg, q=float(q))
       for sg in (0.005, 0.01, 0.02, 0.05, 0.1) for q in (3, 10, 30)},
    "peak_s20_q10_shuffled": method("peak_matching", sigma=0.02, q=10.0, shuffle=True),
}


# ------------------------------------------------------------------ data


def make_blocks(df, cfg):
    """Group each neuron's usable trials into blocks of cfg.pool_trials."""
    df = df[df["isis"].map(len) >= cfg.min_isis]
    blocks = []
    for neuron, g in df.groupby("neuron_id", sort=True):
        trials = list(g.itertuples(index=False))
        k = cfg.pool_trials
        for b in range(len(trials) // k):
            blocks.append((neuron, trials[b * k:(b + 1) * k]))
    counts = pd.Series([n for n, _ in blocks]).value_counts()
    return [b for b in blocks if counts[b[0]] >= 2]  # 1-NN needs a second cloud


def pick_neurons(blocks, n, seed):
    """Keep all trials of n random neurons."""
    names = sorted({neuron for neuron, _ in blocks})
    keep = set(np.random.default_rng(seed).choice(names, n, replace=False).tolist())
    return [b for b in blocks if b[0] in keep]


# ------------------------------------------------------------------ clouds


def make_clouds(block_trials, cfg, rng):
    """The clouds of one block. Windows never cross a trial boundary."""
    embed = EMBEDDINGS[cfg.embedding]
    parts = [embed(t.isis, t.spike_times, **cfg.params) for t in block_trials]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    cloud = np.concatenate(parts)
    if cfg.n_points is None:
        return [cloud] * cfg.n_subsamples
    if len(cloud) < cfg.n_points:
        return None
    return [cloud[rng.choice(len(cloud), cfg.n_points, replace=False)]
            for _ in range(cfg.n_subsamples)]


def build_clouds(blocks, cfg):
    """clouds[i][s] is subsample s of block i; labels[i] is its neuron."""
    rng = np.random.default_rng(cfg.seed)
    made = [(n, make_clouds(trials, cfg, rng)) for n, trials in blocks]
    made = [(n, c) for n, c in made if c is not None]
    counts = pd.Series([n for n, _ in made]).value_counts()
    made = [(n, c) for n, c in made if counts[n] >= 2]
    return [c for _, c in made], np.array([n for n, _ in made])


def fit_anchors(df, blocks, cfg):
    """k-means centres of clouds from trials outside this comparison, when there are any."""
    rng = np.random.default_rng(cfg.seed)
    used = {neuron for neuron, _ in blocks}
    pool = df[~df["neuron_id"].isin(used) & (df["isis"].map(len) >= cfg.min_isis)]
    if pool.empty:
        pool = df[df["isis"].map(len) >= cfg.min_isis]
    rows = pool.iloc[rng.choice(len(pool), min(300, len(pool)), replace=False)]
    embed = EMBEDDINGS[cfg.embedding]
    parts = [embed(t.isis, t.spike_times, **cfg.params) for t in rows.itertuples(index=False)]
    points = np.concatenate([p for p in parts if p is not None])
    points = points[rng.choice(len(points), min(8000, len(points)), replace=False)]
    return KMeans(cfg.regions, n_init=3, random_state=cfg.seed).fit(points).cluster_centers_


def split_regions(cloud, anchors):
    """One cloud per anchor. Regions with under 2 points become two equal points, whose
    only bar has length 0, because empty clouds are undocumented in stablebear."""
    nearest = np.linalg.norm(cloud[:, None] - anchors[None], axis=2).argmin(1)
    parts = [cloud[nearest == r] for r in range(len(anchors))]
    return [p if len(p) >= 2 else np.zeros((2, cloud.shape[1])) for p in parts]


# ------------------------------------------------------------------ topology


def barcode_cache_key(cfg, tag):
    d = asdict(cfg)
    for k in ("summary", "p"):  # these do not affect the barcodes
        d.pop(k)
    d["tag"] = tag
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def compute_barcodes(clouds, anchors, cfg, tag, force):
    import stablebear as sb
    from stablebear import persistence
    from stablebear.io import load, save

    path = RESULTS / "cache" / f"{barcode_cache_key(cfg, tag)}.sb"
    if path.exists() and not force:
        return load(str(path))

    shape = (len(clouds), cfg.n_subsamples) + ((cfg.regions,) if cfg.regions else ())
    tensor = sb.zeros(shape, dtype=sb.pcloud64)
    for i, subsamples in enumerate(clouds):
        for s, cloud in enumerate(subsamples):
            if cfg.regions:
                for r, part in enumerate(split_regions(cloud, anchors)):
                    tensor[i, s, r] = part
            else:
                tensor[i, s] = cloud
    # reduced=True drops the infinite H0 bar, so every stable rank decays to 0.
    bcs = persistence.compute_persistent_homology(
        tensor, max_dim=cfg.max_dim, reduced=True, verbose=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    save(bcs, str(path))
    return bcs


def distance_matrices(bcs, cfg):
    """{'H0': D, 'H1': D, 'H0+H1': D}, each a dense (n, n) numpy array."""
    import stablebear as sb
    from stablebear import persistence

    to_pcf = {"stable_rank": persistence.barcode_to_stable_rank,
              "betti_curve": persistence.barcode_to_betti_curve}[cfg.summary]
    pcfs = to_pcf(bcs)  # shape (blocks, subsamples, [regions,] [max_dim + 1])

    def l1(x):
        return np.asarray(sb.pdist(sb.mean(x, dim=1), p=cfg.p).to_dense(), dtype=np.float64)

    out = {}
    for d in range(cfg.max_dim + 1):
        if cfg.regions:
            out[f"H{d}"] = sum(l1(pcfs[:, :, r, d]) for r in range(cfg.regions))
        else:
            # With max_dim=0 stablebear returns no homology axis.
            out[f"H{d}"] = l1(pcfs[:, :, d] if cfg.max_dim else pcfs)
    if cfg.max_dim >= 1:
        # Median-scale each dimension, otherwise H0 swamps H1.
        out["H0+H1"] = sum(scaled(D) for D in out.values())
    return out


def run_rips(df, blocks, cfg, tag, force):
    clouds, labels = build_clouds(blocks, cfg)
    print(f"  {len(clouds)} clouds, median {int(np.median([len(c[0]) for c in clouds]))} points")
    anchors = fit_anchors(df, blocks, cfg) if cfg.regions else None
    return distance_matrices(compute_barcodes(clouds, anchors, cfg, tag, force), cfg), labels


# ------------------------------------------------------------------ methods without Rips
# Each one maps a list of spike time arrays to a distance matrix.


def spike_count(trains):
    n = np.array([len(t) for t in trains], dtype=float)
    return np.abs(n[:, None] - n[None, :])


def isi_histogram(trains, bins=30, center=False):
    """L1 distance between normalised log-ISI histograms.

    center=True removes each trial's mean log ISI first, the fair comparison for the
    fixed-size clouds, which cannot see the rate either.
    """
    half = np.log(1e3) if center else None
    edges = (np.linspace(-half, half, bins + 1) if center
             else np.linspace(np.log(1e-3), np.log(10), bins + 1))
    H = np.empty((len(trains), bins))
    for i, t in enumerate(trains):
        x = np.log(np.maximum(np.diff(t), 1e-4))
        H[i] = np.histogram(x - x.mean() if center else x, edges)[0] / len(x)
    return cdist(H, H, "cityblock")


def window_count(trains, s):
    """L1 distance between counts in a sliding window of s seconds, divided by s.

    With s = 2/q, shifting one spike by d costs min(q*d, 2) and adding one costs 1, as in
    the spike distance. Edits close together can cancel, so this is a lower bound on it.
    Same function as the time_bars method, but needs only numpy. The grid step is s/20, so
    shifts below that are rounded; time_bars integrates exactly.
    """
    dt = s / 20
    grid = np.arange(dt / 2, max(t[-1] for t in trains) + s, dt)
    F = np.stack([np.searchsorted(t, grid, "right") - np.searchsorted(t, grid - s, "right")
                  for t in trains]).astype(np.float32)
    return cdist(F, F, "cityblock") * dt / s


def van_rossum(trains, sigma):
    """L2 distance between Gaussian-smoothed trains."""
    F = np.stack([smoothed_rate(t, sigma)[::max(1, int(sigma / 0.002))] for t in trains])
    return cdist(F, F, "euclidean")


def spike_distance(trains, q, jobs):
    return vp_distance_matrix(trains, q, jobs)


def peak_matching(trains, sigma, q, jobs):
    return matching_matrix([peak_events(t, sigma) for t in trains], q, jobs)


def l1_between(bars, summary):
    """Hand-written barcodes -> stablebear summary -> pairwise L1 distances."""
    import stablebear as sb
    from stablebear import persistence
    from stablebear.persistence import Barcode, BarcodeTensor

    bcs = BarcodeTensor([Barcode(b) for b in bars])
    pcfs = getattr(persistence, f"barcode_to_{summary}")(bcs)
    return np.asarray(sb.pdist(pcfs, p=1).to_dense(), dtype=np.float64)


def time_bars_distance(trains, s):
    return l1_between([time_bars(t, s) for t in trains], "betti_curve") / s


def burst_bars_distance(trains, s):
    return l1_between([burst_bars(t, s) for t in trains], "betti_curve") / s


def isi_bars_distance(trains, log):
    return l1_between([isi_bars(t, log) for t in trains], "stable_rank")


METHODS = {
    "spike_count": spike_count,
    "isi_histogram": isi_histogram,
    "window_count": window_count,
    "van_rossum": van_rossum,
    "spike_distance": spike_distance,
    "peak_matching": peak_matching,
    "time_bars": time_bars_distance,
    "burst_bars": burst_bars_distance,
    "isi_bars": isi_bars_distance,
}
NEEDS_JOBS = {"spike_distance", "peak_matching"}


def run_method(blocks, cfg, jobs):
    params = dict(cfg.params)
    trains = [np.asarray(trials[0].spike_times, dtype=np.float64) for _, trials in blocks]
    if params.pop("shuffle", False):
        rng = np.random.default_rng(cfg.seed)
        trains = [shuffle_isis(t, rng) for t in trains]
    if cfg.method in NEEDS_JOBS:
        params["jobs"] = jobs
    return {"D": METHODS[cfg.method](trains, **params)}, np.array([n for n, _ in blocks])


# ------------------------------------------------------------------ evaluation


def scaled(D):
    """Divide by the median so that distance matrices can be added."""
    return D / np.median(D[D > 0])


def k_medoids(D, k, seed=0, restarts=5, iters=100):
    """k-means for a distance matrix: centres are trials, not averages. No labels used."""
    rng = np.random.default_rng(seed)
    n = len(D)
    best, best_cost = None, np.inf
    for _ in range(restarts):
        medoids = [int(rng.integers(n))]  # k-means++ style start
        near = D[medoids[0]].copy()
        for _ in range(k - 1):
            medoids.append(int(rng.choice(n, p=near ** 2 / (near ** 2).sum())))
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
            best, best_cost = assign, cost
    return best


def matched_accuracy(y, clusters):
    """Share of trials in the right cluster, after pairing each cluster with one neuron
    so that as many trials as possible agree (Hungarian method)."""
    table = np.zeros((y.max() + 1, clusters.max() + 1))
    np.add.at(table, (y, clusters), 1)
    rows, cols = linear_sum_assignment(-table)
    return float(table[rows, cols].sum() / len(y))


def evaluate(D, labels, seed=0):
    """Can we tell which neuron a cloud came from, given a distance matrix?

    acc_1nn      each trial takes the label of its nearest other trial: needs labelled trials.
    cluster_acc  k-medoids with k = number of neurons, no labels used to form the groups;
                 then the share of trials whose cluster is matched to their neuron.
    ami, cluster_ami   agreement of average-linkage and k-medoids clusters with neurons.
    """
    D = np.array(D, dtype=np.float64)
    names, y = np.unique(labels, return_inverse=True)
    n = len(y)
    # Ties are common for integer baselines, so break them at random.
    scale = np.median(D[D > 0])
    jitter = np.random.default_rng(seed).random((n, n)) * 1e-9 * scale
    N = D + jitter + jitter.T
    np.fill_diagonal(N, np.inf)
    acc = float(np.mean(y[N.argmin(1)] == y))
    # Leave-one-out neighbours, so chance is the expected share of same-neuron clouds.
    counts = np.bincount(y)
    chance = float(np.mean((counts[y] - 1) / (n - 1)))
    pred = AgglomerativeClustering(n_clusters=len(names), metric="precomputed",
                                   linkage="average").fit_predict(D)
    medoid = k_medoids(D, len(names), seed)
    return dict(n_clouds=n, n_neurons=len(names), acc_1nn=acc, chance=chance,
                ami=float(adjusted_mutual_info_score(y, pred)),
                cluster_acc=matched_accuracy(y, medoid),
                cluster_ami=float(adjusted_mutual_info_score(y, medoid)))


# ------------------------------------------------------------------ main


def data_tag(session, neurons, seed=0):
    return f"{session}_{neurons or 'all'}" + (f"_seed{seed}" if seed else "")


def run(name, cfg, df, neurons, jobs, force):
    tag = data_tag("+".join(cfg.sessions), neurons, cfg.seed)
    folder = RESULTS / "distances" / tag
    folder.mkdir(parents=True, exist_ok=True)
    if not force and list(folder.glob(f"{name}__*.npy")):
        print(f"{name}: already computed, skipping")
        return None
    print(f"{name}:")
    start = time.time()
    blocks = make_blocks(df, cfg)
    if neurons:
        blocks = pick_neurons(blocks, neurons, cfg.seed)
    if cfg.method == "rips":
        matrices, labels = run_rips(df, blocks, cfg, tag, force)
    else:
        matrices, labels = run_method(blocks, cfg, jobs)
    rows = []
    for view, D in matrices.items():
        np.save(folder / f"{name}__{view}.npy", D.astype(np.float32))
        if cfg.pool_trials == 1:
            np.save(folder / "labels.npy", labels)
        rows.append(dict(config=name, distance=view, **evaluate(D, labels, cfg.seed),
                         seconds=round(time.time() - start, 1)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("configs", nargs="*", help="names or patterns from CONFIGS (default: all)")
    ap.add_argument("--list", action="store_true", help="print config names and exit")
    ap.add_argument("--data", default=DEFAULT_PATH, help=".parquet or .csv dataset")
    ap.add_argument("--session", help="override the session of every config")
    ap.add_argument("--summary", choices=["stable_rank", "betti_curve"])
    ap.add_argument("--neurons", type=int, help="use this many random neurons (faster)")
    ap.add_argument("--seed", type=int, help="another random choice of neurons and subsamples")
    ap.add_argument("--jobs", type=int, default=1, help="processes for the matching distances")
    ap.add_argument("--force", action="store_true", help="recompute what is already saved")
    args = ap.parse_args()

    if args.list:
        print("\n".join(CONFIGS))
        return
    names = [n for n in CONFIGS if not args.configs
             or any(fnmatch.fnmatch(n, p) for p in args.configs)]
    if not names:
        sys.exit(f"no config matches {args.configs}; see --list")
    chosen = {n: CONFIGS[n] for n in names}
    if args.session:
        chosen = {n: replace(c, sessions=(args.session,)) for n, c in chosen.items()}
    if args.seed is not None:
        chosen = {n: replace(c, seed=args.seed) for n, c in chosen.items()}
    if args.summary:
        chosen = {n: replace(c, summary=args.summary) for n, c in chosen.items()}

    sessions = sorted({s for c in chosen.values() for s in c.sessions})
    df = load_dataset(args.data, sessions=sessions)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "results.csv"
    for name, cfg in chosen.items():
        rows = run(name, cfg, df[df["session"].isin(cfg.sessions)],
                   args.neurons, args.jobs, args.force)
        if rows is None:
            continue
        rows = pd.DataFrame(rows)
        print(rows.round(3).to_string(index=False), "\n", flush=True)
        rows["settings"] = json.dumps(asdict(cfg), sort_keys=True)
        rows.to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()
