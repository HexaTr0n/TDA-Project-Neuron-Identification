"""Task L10: Rips persistence of the ISI delay embedding on long samples, in stablebear.

Per sample: log ISIs, delay embedding with m = 2, 3 or 5 (coordinates / sqrt(m), as
embeddings.isi_delay), 20 random subsamples of exactly 100 points, Vietoris-Rips H0 and H1 in
stablebear (reduced), stable ranks averaged over the subsamples (sb.mean), L1 distances with
sb.pdist. H0+H1 adds the two median-scaled matrices. Next to the homology, two summaries of the
same cloud show what homology keeps:
  - position: the mean point of the full cloud, Euclidean distance;
  - whole cloud: sliced Wasserstein-1 between the first 100-point subsamples (64 random
    directions, sorted projections, L1);
and ISI fractions (stablebear stable rank / ISI count) as the reference without order.
Control: the ISIs permuted within each sample before embedding (every summary).

Samples with fewer than 100 points are dropped. Neurons: units with >= 10 usable samples,
60 at random (seed 0). Scores: 1-NN, and k-medoids on the same distance matrix (3 restarts),
clusters paired with neurons by the Hungarian method.

Task L11 uses the same script with --embedding predictive: one point per spike, the pdf of
the next log ISI given the last L = 2 ISIs (bandwidth 1, Wasserstein-2 coordinates,
embeddings.predictive_state); writes results/rips_long_predictive_<session>.csv.

Run from the repo root, one session at a time (writes results/rips_long_<session>.csv):
    python scripts/sb_rips_long.py --data data/openscope/glo637908.parquet --session glo637908_grating
    python scripts/sb_rips_long.py --data data/allen/allen829720705.parquet --session allen829720705_movie
    python scripts/sb_rips_long.py --data data/allen/allen829720705.parquet --session allen829720705_spont
    python scripts/sb_rips_long.py --data data/retina_running_20180126_sr_le_sp.parquet \
        --session 20180126_sr_le_sp_running --join-pieces
"""
import argparse
import functools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curve_eval import kmedoids, l1_pdist, matched, nn_score, read
from sb_cross import isi_fraction_pcfs
from sb_length import stretches

MIN_ISI = 1e-4
RESULTS = Path(__file__).resolve().parent.parent / "results"


def samples(df, join_pieces):
    """(neuron_id, spike_times) per sample."""
    if join_pieces:
        return [(n, t) for n, t, _ in stretches(df, True)]
    return [(r.neuron_id, np.asarray(r.spike_times, float)) for r in df.itertuples(index=False)]


@functools.lru_cache(None)
def _legacy():
    """experiments/legacy_scripts/embeddings.py (scripts/embeddings.py has the same name)."""
    import importlib.util
    path = Path(__file__).resolve().parent / "experiments" / "legacy_scripts" / "embeddings.py"
    spec = importlib.util.spec_from_file_location("legacy_embeddings", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def isis(t, shuffle, rng):
    x = np.maximum(np.diff(np.sort(t)), MIN_ISI)
    return rng.permutation(x) if shuffle else x


def cloud(x, m, embedding, bandwidth):
    """delay: m log ISIs per point. predictive: the pdf of the next log ISI given the last
    m ISIs (embeddings.predictive_state, Wasserstein-2 coordinates)."""
    if embedding == "predictive":
        return _legacy().predictive_state(x, L=m, bandwidth=bandwidth, distance="w2")
    return sliding_window_view(np.log(x), m).astype(np.float64) / np.sqrt(m)


def sliced_features(c, directions):
    """Sorted projections, so that L1 between rows / n = sliced Wasserstein-1."""
    return np.sort(c @ directions.T, axis=0).T.ravel() / (len(c) * len(directions))


def scaled(D):
    return D / np.median(D[D > 0])


def rips_distances(clouds, max_dim=1):
    import stablebear as sb
    from stablebear import persistence

    n, s = len(clouds), len(clouds[0])
    tensor = sb.zeros((n, s), dtype=sb.pcloud64)
    for i, subs in enumerate(clouds):
        for j, c in enumerate(subs):
            tensor[i, j] = c
    bcs = persistence.compute_persistent_homology(tensor, max_dim=max_dim, reduced=True, verbose=False)
    pcfs = persistence.barcode_to_stable_rank(bcs)
    out = {}
    for d in range(max_dim + 1):
        out[f"H{d}"] = np.asarray(sb.pdist(sb.mean(pcfs[:, :, d], dim=1), p=1).to_dense(), dtype=float)
    out["H0+H1"] = scaled(out["H0"]) + scaled(out["H1"])
    return out


def score(D, y, k, rng):
    nn = nn_score(D.copy(), y, rng)
    Dm = D.copy()
    np.fill_diagonal(Dm, 0)
    km = matched(y, kmedoids(Dm, k, rng), k)[0]
    return nn, km


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--join-pieces", action="store_true")
    ap.add_argument("--embedding", choices=["delay", "predictive"], default="delay")
    ap.add_argument("--ms", type=int, nargs="+", default=None,
                    help="delay dimensions (default 2 3 5), or history lengths L (default 2)")
    ap.add_argument("--bandwidth", type=float, default=1.0, help="predictive only")
    ap.add_argument("--points", type=int, default=100)
    ap.add_argument("--subsamples", type=int, default=20)
    ap.add_argument("--neurons", type=int, default=60)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--max-per-neuron", type=int, default=0, help="0 keeps all samples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.ms is None:
        args.ms = [2] if args.embedding == "predictive" else [2, 3, 5]
    rng = np.random.default_rng(args.seed)
    df = read(args.data, [args.session])
    S = samples(df, args.join_pieces)
    need = args.points + max(args.ms) + (1 if args.embedding == "predictive" else 0)  # ISIs for 100 points at the largest m
    S = [(n, t) for n, t in S if len(t) - 1 >= need]
    counts = pd.Series([n for n, _ in S]).value_counts()
    names = sorted(counts.index[counts >= args.min_samples])
    keep = set(rng.choice(names, min(args.neurons, len(names)), replace=False).tolist())
    S = [(n, t) for n, t in S if n in keep]
    if args.max_per_neuron:
        by = {}
        for n, t in S:
            by.setdefault(n, []).append(t)
        S = []
        for n in sorted(by):
            idx = rng.permutation(len(by[n]))[:args.max_per_neuron]
            S += [(n, by[n][i]) for i in sorted(idx)]
    y = np.unique([n for n, _ in S], return_inverse=True)[1]
    k = y.max() + 1
    trains = [t for _, t in S]
    print(f"{args.session}: {len(names)} units with >= {args.min_samples} samples of >= {need} ISIs; "
          f"{k} neurons, {len(S)} samples, median {np.median([len(t) for t in trains]):.0f} spikes; "
          f"chance {100 / k:.1f}%", flush=True)

    rows = []

    def add(m, rep, shuffled, D):
        nn, km = score(D, y, k, np.random.default_rng(args.seed))
        rows.append(dict(session=args.session, m=m, summary=rep, shuffled=shuffled,
                         nn=nn, kmedoids=km, neurons=k, samples=len(S)))
        print(f"  m={m} {rep:<12} {'shuffled' if shuffled else 'real':<8} 1-NN {100 * nn:5.1f}  "
              f"k-medoids {100 * km:5.1f}", flush=True)

    import stablebear as sb
    Dref = np.asarray(sb.pdist(sb.PcfTensor(isi_fraction_pcfs(trains)), p=1).to_dense(), dtype=float)
    add(0, "ISI fractions", False, Dref)

    for m in args.ms:
        dim = 32 if args.embedding == "predictive" else m
        directions = rng.normal(size=(64, dim))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        for shuffled in (False, True):
            t0 = time.time()
            r = np.random.default_rng(args.seed + 1)
            clouds, pos, sw = [], [], []
            for t in trains:
                c = cloud(isis(t, shuffled, r), m, args.embedding, args.bandwidth)
                subs = [c[r.choice(len(c), args.points, replace=False)] for _ in range(args.subsamples)]
                clouds.append(subs)
                pos.append(c.mean(0))
                sw.append(sliced_features(subs[0], directions))
            pos = np.array(pos)
            Dpos = np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1))
            add(m, "position", shuffled, Dpos)
            add(m, "whole cloud", shuffled, l1_pdist(np.array(sw)))
            for rep, D in rips_distances(clouds).items():
                add(m, rep, shuffled, D)
            print(f"  m={m} {'shuffled' if shuffled else 'real'} done in {time.time() - t0:.0f} s", flush=True)
            tag = "" if args.embedding == "delay" else f"_{args.embedding}"
            out = RESULTS / f"rips_long{tag}_{args.session}.csv"
            out.parent.mkdir(exist_ok=True)
            pd.DataFrame(rows).to_csv(out, index=False)

    print("\n" + pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
