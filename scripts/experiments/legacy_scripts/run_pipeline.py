"""Embedding -> barcodes -> stable ranks -> distance matrix -> neuron re-identification.

Run from the repo root:
    python scripts/run_pipeline.py --list
    python scripts/run_pipeline.py m1_all m3_fixed m3_fixed_shuffled
    python scripts/run_pipeline.py            # every config in CONFIGS

Each run appends one row per distance to results/results.csv. Barcodes are cached
in results/cache/, keyed on everything that affects them, so changing only the
summary or the evaluation does not recompute persistent homology.
"""
import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import EMBEDDINGS
from load_dataset import DEFAULT_PATH, load_dataset

RESULTS = Path(__file__).resolve().parent.parent / "results"


@dataclass(frozen=True)
class Config:
    # data
    sessions: tuple = ("20180126_sr_le_sp",)  # most spikes per trial, 156 neurons
    min_isis: int = 30        # drop trials with fewer ISIs
    pool_trials: int = 1      # clouds per neuron = n_trials // pool_trials
    # embedding
    embedding: str = "isi_delay"
    params: dict = field(default_factory=dict)
    # cloud size: None keeps every point, an int subsamples to exactly that many.
    # A stable rank starts at the bar count, so unequal sizes leak the spike count.
    n_points: int = None
    n_subsamples: int = 1     # stable ranks are averaged over this many subsamples
    # persistent homology
    max_dim: int = 1
    # summary and distance
    summary: str = "stable_rank"   # stable_rank | betti_curve
    p: int = 1
    seed: int = 0


def delay(m, **params):
    return dict(params=dict(m=m, **params))


def pdf(L, **params):
    return dict(embedding="predictive_state", params=dict(L=L, **params))


FIXED = dict(n_points=25, n_subsamples=20)


CONFIGS = {
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
    "m3_fixed_pool5": Config(pool_trials=5, n_points=125, n_subsamples=20,
                             **delay(3, center=True)),
    # Plain ISIs in seconds. Unlike log ISIs these scale with the firing rate, so the
    # rate CAN reach the barcode here, even at a fixed cloud size.
    "m1_all_nolog": Config(max_dim=0, **delay(1, log=False)),
    "m3_all_nolog": Config(**delay(3, log=False)),
    "m5_all_nolog": Config(**delay(5, log=False)),
    "m1_fixed_nolog": Config(max_dim=0, **FIXED, **delay(1, log=False)),
    "m3_fixed_nolog": Config(**FIXED, **delay(3, log=False)),
    "m5_fixed_nolog": Config(**FIXED, **delay(5, log=False)),
    "m3_fixed_nolog_shuffled": Config(**FIXED, **delay(3, log=False, shuffle=True)),
    "m5_fixed_nolog_shuffled": Config(**FIXED, **delay(5, log=False, shuffle=True)),
    # One pdf of the next ISI per spike, Wasserstein-2 between pdfs.
    "pdf_L1_w2": Config(**FIXED, **pdf(1)),
    "pdf_L2_w2": Config(**FIXED, **pdf(2)),
    "pdf_L3_w2": Config(**FIXED, **pdf(3)),
    "pdf_L1_w2_shuffled": Config(**FIXED, **pdf(1, shuffle=True)),
    "pdf_L2_w2_shuffled": Config(**FIXED, **pdf(2, shuffle=True)),
    "pdf_L3_w2_shuffled": Config(**FIXED, **pdf(3, shuffle=True)),
    # The same with Hellinger distance.
    "pdf_L2_hellinger": Config(**FIXED, **pdf(2, distance="hellinger")),
    "pdf_L2_hellinger_shuffled": Config(**FIXED, **pdf(2, distance="hellinger", shuffle=True)),
}


# ------------------------------------------------------------------ clouds


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


def build_clouds(df, cfg):
    """clouds[i][s] is subsample s of block i; labels[i] is its neuron."""
    rng = np.random.default_rng(cfg.seed)
    clouds, labels, blocks = [], [], []
    for neuron, trials in make_blocks(df, cfg):
        c = make_clouds(trials, cfg, rng)
        if c is not None:
            clouds.append(c)
            labels.append(neuron)
            blocks.append(trials)
    counts = pd.Series(labels).value_counts()
    keep = [i for i, n in enumerate(labels) if counts[n] >= 2]
    return ([clouds[i] for i in keep], np.array([labels[i] for i in keep]),
            [blocks[i] for i in keep])


# ------------------------------------------------------------------ topology


def barcode_cache_key(cfg):
    d = asdict(cfg)
    for k in ("summary", "p"):  # these do not affect the barcodes
        d.pop(k)
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def compute_barcodes(clouds, cfg):
    import stablebear as sb
    from stablebear import persistence
    from stablebear.io import load, save

    path = RESULTS / "cache" / f"{barcode_cache_key(cfg)}.sb"
    if path.exists():
        return load(str(path))

    tensor = sb.zeros((len(clouds), cfg.n_subsamples), dtype=sb.pcloud64)
    for i, subsamples in enumerate(clouds):
        for s, cloud in enumerate(subsamples):
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
    pcfs = to_pcf(bcs)  # shape (blocks, subsamples, max_dim + 1)
    out = {}
    for d in range(cfg.max_dim + 1):
        mean_pcfs = sb.mean(pcfs[:, :, d], dim=1)
        out[f"H{d}"] = np.asarray(sb.pdist(mean_pcfs, p=cfg.p).to_dense(), dtype=np.float64)
    if cfg.max_dim >= 1:
        # Median-scale each dimension, otherwise H0 swamps H1.
        out["H0+H1"] = sum(D / np.median(D[D > 0]) for D in out.values())
    return out


# ------------------------------------------------------------------ baselines


def baseline_spike_count(blocks):
    n = np.array([sum(len(t.isis) for t in trials) for trials in blocks], dtype=float)
    return np.abs(n[:, None] - n[None, :])


def baseline_isi_histogram(blocks, bins=30, center=False):
    """L1 distance between normalised log-ISI histograms. No topology at all.

    center=True removes each trial's mean log ISI first. This is the fair comparison
    for the fixed-size clouds, which cannot see the rate either.
    """
    half = np.log(1e3) if center else None
    edges = (np.linspace(-half, half, bins + 1) if center
             else np.linspace(np.log(1e-3), np.log(10), bins + 1))
    H = np.empty((len(blocks), bins))
    for i, trials in enumerate(blocks):
        logs = [np.log(np.maximum(t.isis, 1e-4)) for t in trials]
        x = np.concatenate([v - v.mean() for v in logs] if center else logs)
        H[i] = np.histogram(x, edges)[0] / len(x)
    return cdist(H, H, "cityblock")


# ------------------------------------------------------------------ evaluation


def evaluate(D, labels, seed=0):
    """Can we tell which neuron a cloud came from, given a distance matrix?"""
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
    return dict(n_clouds=n, n_neurons=len(names), acc_1nn=acc, chance=chance,
                ami=float(adjusted_mutual_info_score(y, pred)))


# ------------------------------------------------------------------ main


def run(name, cfg, df, baselines=True):
    clouds, labels, blocks = build_clouds(df, cfg)
    print(f"{name}: {len(clouds)} clouds, {len(set(labels))} neurons, "
          f"median {int(np.median([len(c[0]) for c in clouds]))} points")
    rows = []
    if baselines:
        for b, D in [("baseline_spike_count", baseline_spike_count(blocks)),
                     ("baseline_isi_histogram", baseline_isi_histogram(blocks)),
                     ("baseline_centered_histogram",
                      baseline_isi_histogram(blocks, center=True))]:
            rows.append(dict(config=name, distance=b, **evaluate(D, labels, cfg.seed)))
    for key, D in distance_matrices(compute_barcodes(clouds, cfg), cfg).items():
        rows.append(dict(config=name, distance=key, **evaluate(D, labels, cfg.seed)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("configs", nargs="*", help="names from CONFIGS (default: all)")
    ap.add_argument("--list", action="store_true", help="print config names and exit")
    ap.add_argument("--data", default=DEFAULT_PATH, help=".parquet or .csv dataset")
    ap.add_argument("--session", help="override the session of every config")
    ap.add_argument("--summary", choices=["stable_rank", "betti_curve"])
    args = ap.parse_args()

    if args.list:
        print("\n".join(CONFIGS))
        return
    chosen = {n: CONFIGS[n] for n in (args.configs or CONFIGS)}
    if args.session:
        chosen = {n: replace(c, sessions=(args.session,)) for n, c in chosen.items()}
    if args.summary:
        chosen = {n: replace(c, summary=args.summary) for n, c in chosen.items()}

    sessions = sorted({s for c in chosen.values() for s in c.sessions})
    df = load_dataset(args.data, sessions=sessions)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "results.csv"
    for name, cfg in chosen.items():
        rows = pd.DataFrame(run(name, cfg, df[df["session"].isin(cfg.sessions)]))
        rows["settings"] = json.dumps(asdict(cfg), sort_keys=True)
        rows.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(rows.drop(columns="settings").round(3).to_string(index=False), "\n")


if __name__ == "__main__":
    main()
