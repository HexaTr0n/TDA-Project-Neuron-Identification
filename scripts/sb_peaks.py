"""Task L13: sublevel peak matching on Allen natural-movie repeats.

Each movie repeat (30 s) is one sample. Per sample, the peaks of the Gaussian-smoothed rate
(barcodes.peak_events: H0 of the superlevel sets, one bar per peak, mass = prominence * sigma *
sqrt(2 pi) spikes) for sigma = 10, 20, 50 ms, then
  - peaks matched in time: spike_distance.matching_matrix, q = 10 per s;
  - peaks with time removed: bars [0, mass] -> stable rank in stablebear, L1 (sb.pdist);
  - reference: square-root time bars in stablebear (as L9), ISI fractions (stablebear).
Control: ISIs permuted within each repeat (sigma 20 ms, and the references).

Units: >= 10 repeats with >= 30 ISIs. Reliability = mean pairwise PSTH correlation (50 ms bins,
as L9). Two sets: 60 random units (seed 0) and the 60 most reliable. 1-NN, and k-medoids on the
same distances (3 restarts).

Run from the repo root:
    python scripts/sb_peaks.py --data data/allen/allen829720705.parquet --session allen829720705_movie
Writes results/peaks_<session>.csv.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import barcodes
import curves as C
from barcodes import shuffle_isis
from curve_eval import kmedoids, matched, nn_score, read
from sb_check import sqrt_betti_pcfs
from sb_cross import isi_fraction_pcfs
from sb_reliability import reliability
from spike_distance import matching_matrix

RESULTS = Path(__file__).resolve().parent.parent / "results"


def sb_l1(pcfs):
    import stablebear as sb
    return np.asarray(sb.pdist(pcfs if not isinstance(pcfs, list) else sb.PcfTensor(pcfs), p=1).to_dense(), float)


def mass_ranks(events):
    from stablebear import persistence
    from stablebear.persistence import Barcode, BarcodeTensor

    bars = [np.c_[np.zeros(len(m)), m] if len(m) else np.zeros((1, 2)) for _, m in events]
    return persistence.barcode_to_stable_rank(BarcodeTensor([Barcode(b) for b in bars]))


def score(D, y, k, seed):
    rng = np.random.default_rng(seed)
    nn = nn_score(D.copy(), y, rng)
    Dm = D.copy()
    np.fill_diagonal(Dm, 0)
    return nn, matched(y, kmedoids(Dm, k, rng), k)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    ap.add_argument("--q", type=float, default=10.0)
    ap.add_argument("--neurons", type=int, default=60)
    ap.add_argument("--min-isis", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = read(args.data, [args.session])
    dur = float(df["duration"].median())
    barcodes.DURATION = dur  # smoothed_rate reads this global
    df = df[df["isis"].map(len) >= args.min_isis]
    counts = df["neuron_id"].value_counts()
    df = df[df["neuron_id"].isin(counts.index[counts >= 10])]
    rel = df.groupby("neuron_id")["spike_times"].apply(lambda s: reliability([np.asarray(x, float) for x in s], dur))
    print(f"{args.session}: {len(rel)} units with >= 10 repeats of >= {args.min_isis} ISIs, {dur:.1f} s; "
          f"reliability median {rel.median():.3f}, max {rel.max():.3f}", flush=True)

    rng = np.random.default_rng(args.seed)
    names = sorted(rel.index)
    sets = {"60 random": rng.choice(names, min(args.neurons, len(names)), replace=False),
            "60 most reliable": rel.sort_values(ascending=False).index[:args.neurons]}
    rows = []
    for label, sel in sets.items():
        sub = df[df["neuron_id"].isin(sel)].sort_values(["neuron_id", "trial"]).reset_index(drop=True)
        y = np.unique(sub["neuron_id"], return_inverse=True)[1]
        k = y.max() + 1
        real = [np.sort(np.asarray(t, float)) for t in sub["spike_times"]]
        r = np.random.default_rng(args.seed + 1)
        shuf = [shuffle_isis(t, r) for t in real]
        print(f"\n{label}: {k} units, {len(real)} repeats, median {np.median([len(t) for t in real]):.0f} "
              f"spikes, reliability median {rel.loc[list(sel)].median():.3f}", flush=True)

        def add(name, shuffled, D, extra=None):
            nn, km = score(D, y, k, args.seed)
            rows.append(dict(session=args.session, units=label, signature=name, shuffled=shuffled,
                             nn=nn, kmedoids=km, reliability_median=float(rel.loc[list(sel)].median()),
                             samples=len(y), **(extra or {})))
            print(f"  {name:<34} {'shuffled' if shuffled else 'real':<8} 1-NN {100 * nn:5.1f}  "
                  f"k-medoids {100 * km:5.1f}", flush=True)
            pd.DataFrame(rows).to_csv(RESULTS / f"peaks_{args.session}.csv", index=False)

        for shuffled, trains in ((False, real), (True, shuf)):
            add("ISI fractions", shuffled, sb_l1(isi_fraction_pcfs(trains)))
            add("sqrt time bars", shuffled, sb_l1(sqrt_betti_pcfs(trains, C.S)))
            for sg in args.sigmas:
                if shuffled and sg != 0.02:
                    continue
                t0 = time.time()
                ev = [barcodes.peak_events(t, sg) for t in trains]
                npk = float(np.median([len(e[0]) for e in ev]))
                add(f"peaks matched in time, sigma {sg * 1000:.0f} ms", shuffled,
                    matching_matrix(ev, args.q, jobs=args.jobs), dict(peaks_median=npk))
                add(f"peaks, time removed, sigma {sg * 1000:.0f} ms", shuffled, sb_l1(mass_ranks(ev)),
                    dict(peaks_median=npk))
                print(f"    ({npk:.0f} peaks per repeat, {time.time() - t0:.0f} s)", flush=True)

    print("\n" + pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
