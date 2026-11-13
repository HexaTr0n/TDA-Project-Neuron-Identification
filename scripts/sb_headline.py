"""Task L22: headline homology numbers, every row built from barcodes only (stablebear).

Rows (each with ISIs shuffled, and its twin without bars):
  delay regions            whole delay cloud (m = 3, every point, one barcode), 32 k-means anchors
                           fitted on outside neurons, Rips H0 per region, stable ranks summed
                           over regions; point count left in (L20)
     twin: region counts   points per region
  ISI bars                 stable rank of the bars [0, log(ISI / 1e-4)], NOT divided by the count
     twin: spike count     number of spikes
  delay regions + ISI bars the two distances, each divided by its median over pairs of samples
                           of outside neurons, added
     twin                  region counts + spike count, scaled the same way
  + burst bars             (within a repeated stimulus only: column "within A") Betti curve of
                           barcodes.burst_bars (s = 67 ms), H0 that keeps time; twin: burst count
References on the same samples: ISI fractions, burst table, multiscale sqrt time bars.

Datasets: as sb_feature_cloud.DS (retina10 --drop-piece 2, retina40clean, openscope10, allen30,
openscope124), plus
  allen30adj         Allen movie block 1 <-> the spontaneous windows before it (0-38 min), the
                     pair adjacent in time (L24); neurons of allen30
  retina:<session>   frozen trials of one retina session (data/isi_frozennoise.parquet), within;
                     outside neurons (anchors, medians) from the other eight sessions
Neurons: L17 selection with --seed (60 neurons); one-sided sets as L19 (choose_one).
Scores as L17: 1-NN, k-medoids, k-means with sb.mean (k-means skipped with --no-kmeans).

Run from the repo root:
    python scripts/sb_headline.py --dataset retina10 --drop-piece 2 --seed 0
Writes results/headline_<dataset>[_without_piece2]_seed<seed>.csv.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
import barcodes
import curves as C
from barcodes import shuffle_isis
from curve_eval import read
from sb_cross import isi_fraction_pcfs
from sb_feature_cloud import DS, choose_one, evaluate_one, samples_keyed
from sb_full_table import (betti_of, choose, delay_cloud, evaluate, part_pdist, rips_parts, samples,
                           split_regions, sqrt_betti, to_list)

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
K = 32
MIN_ISI = 1e-4


class Opt:
    def __init__(self, seed, one):
        self.neurons, self.seed = 60, seed
        self.min_isis, self.min_samples, self.max_per_side = (103, 10, 1000) if one else (30, 5, 30)


def isi_bar_pcfs(trains):
    from stablebear import persistence
    from stablebear.persistence import Barcode, BarcodeTensor
    bars = []
    for t in trains:
        L = np.log(np.maximum(np.diff(t), MIN_ISI) / MIN_ISI)
        bars.append(np.c_[np.zeros(len(L)), L] if len(L) else np.zeros((1, 2)))
    return to_list(persistence.barcode_to_stable_rank(BarcodeTensor([Barcode(b) for b in bars])), len(bars))


def representations(trains, anchors):
    """name -> list of parts (sb_full_table format)."""
    clouds = [delay_cloud(t) if len(t) >= 5 else np.zeros((2, 3)) for t in trains]
    reg = rips_parts([[split_regions(c, anchors)] for c in clouds], 0)[0]
    lab = [np.linalg.norm(c[:, None] - anchors[None], axis=2).argmin(1) for c in clouds]
    counts = np.array([np.bincount(l, minlength=len(anchors)) for l in lab], dtype=float)
    bursts = [barcodes.burst_bars(t, C.S) for t in trains]
    return {
        "regions": [("pcf", p) for p in reg],
        "region counts": [("vec", counts, "l1")],
        "ISI bars": [("pcf", isi_bar_pcfs(trains))],
        "spike count": [("vec", np.array([[len(t)] for t in trains], float), "l1")],
        "burst bars": [("pcf", betti_of(bursts))],
        "burst count": [("vec", np.array([[len(b)] for b in bursts], float), "l1")],
    }


def med(parts):
    D = sum(part_pdist(p) for p in parts)
    v = D[np.triu_indices(len(D), 1)]
    v = v[v > 0]
    return float(np.median(v)) if len(v) else 1.0


ROWS = {  # row name -> list of representation names, each its own group
    "delay regions (k=32, H0)": ["regions"],
    "  twin: region counts": ["region counts"],
    "ISI bars": ["ISI bars"],
    "  twin: spike count": ["spike count"],
    "delay regions + ISI bars": ["regions", "ISI bars"],
    "  twin: region counts + spike count": ["region counts", "spike count"],
    "delay regions + ISI bars + burst bars": ["regions", "ISI bars", "burst bars"],
    "  twin: + burst count": ["region counts", "spike count", "burst count"],
}


def load(name, args):
    """(A, B or None, pool rows, candidate count)."""
    if name.startswith("retina:"):
        sess = name.split(":", 1)[1]
        allr = pd.read_parquet(REPO / "data" / "isi_frozennoise.parquet", columns=["neuron_id", "session", "spike_times"])
        allr["spike_times"] = allr["spike_times"].map(lambda a: np.asarray(a, np.float64))
        df = allr[allr["session"] == sess]
        A, _, n = choose_one(df[["neuron_id", "spike_times"]], Opt(args.seed, False))
        # most sessions have <= 60 usable cells, so outside neurons come from the other sessions
        pool = allr[allr["session"] != sess][["neuron_id", "spike_times"]]
        return A, None, pool, n
    if name == "allen30adj":
        A_all, B_all = samples("allen30")
        A, B, pool, _ = choose(A_all, B_all, Opt(args.seed, False))
        kept = sorted(set(A["neuron_id"]))
        raw = read(REPO / "data/allen/allen829720705.parquet", ["allen829720705_movie", "allen829720705_spont"])
        raw = raw[raw["neuron_id"].isin(kept) & (raw["spike_times"].map(len) - 1 >= 30)]
        m = raw["session"].str.endswith("movie")
        X, Y = raw[m & (raw["start"] < 3600)], raw[~m & (raw["start"] < 38 * 60)]
        ok = sorted(set(X["neuron_id"].value_counts().loc[lambda s: s >= 3].index)
                    & set(Y["neuron_id"].value_counts().loc[lambda s: s >= 3].index))
        X, Y = X[X["neuron_id"].isin(ok)], Y[Y["neuron_id"].isin(ok)]
        return X[["neuron_id", "spike_times"]].reset_index(drop=True), Y[["neuron_id", "spike_times"]].reset_index(drop=True), pool, len(ok)
    if DS[name]["b"] is None:
        A_all, _, A0, _ = samples_keyed(name, None)
        del A0
        A, pool, n = choose_one(A_all, Opt(args.seed, True))
        return A, None, pool, n
    A_all, B_all = samples(name, args.drop_piece)
    A, B, pool, n = choose(A_all, B_all, Opt(args.seed, False))
    return A, B, pool, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--drop-piece", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-kmeans", action="store_true")
    ap.add_argument("--no-references", action="store_true")
    args = ap.parse_args()

    A, B, pool, n_cand = load(args.dataset, args)
    one = B is None
    both = A if one else pd.concat([A, B]).reset_index(drop=True)
    names = sorted(set(both["neuron_id"]))
    y = both["neuron_id"].map({n: i for i, n in enumerate(names)}).to_numpy()
    k = len(names)
    ia, ib = np.arange(len(A)), np.arange(len(A), len(both))
    trains = [np.sort(np.asarray(t, float)) for t in both["spike_times"]]
    print(f"{args.dataset} seed {args.seed}: {n_cand} candidates, {k} neurons; {len(ia)} + {len(ib)} samples; "
          f"median {np.median([len(t) for t in trains]):.0f} spikes; chance {100 / k:.1f}%", flush=True)

    rng = np.random.default_rng(args.seed)
    pool = pool[pool["spike_times"].map(len) > 5]
    pool = pool.iloc[rng.choice(len(pool), min(300, len(pool)), replace=False)]
    pool_trains = [np.sort(np.asarray(t, float)) for t in pool["spike_times"]]
    pts = np.concatenate([delay_cloud(t) for t in pool_trains])
    pts = pts[rng.choice(len(pts), min(8000, len(pts)), replace=False)]
    anchors = KMeans(K, n_init=3, random_state=args.seed).fit(pts).cluster_centers_
    t0 = time.time()
    P = representations(pool_trains, anchors)
    pool_med = {r: med(parts) for r, parts in P.items()}
    print(f"outside-neuron medians ({len(pool_trains)} samples, {time.time() - t0:.0f} s): "
          + ", ".join(f"{r} {v:.3g}" for r, v in pool_med.items()), flush=True)

    tag = "" if args.drop_piece is None else f"_without_piece{args.drop_piece}"
    out = RESULTS / f"headline_{args.dataset.replace(':', '_')}{tag}_seed{args.seed}.csv"
    rows = []
    rs = np.random.default_rng(args.seed + 1)
    shuffled = [shuffle_isis(t, rs) for t in trains]
    for cond, tr in (("real", trains), ("ISIs shuffled", shuffled)):
        t0 = time.time()
        R = representations(tr, anchors)
        comp_med = {r: med(parts) for r, parts in R.items()}
        print(f"  representations ({cond}) in {time.time() - t0:.0f} s", flush=True)
        todo = {name: [(R[r], comp_med[r] / pool_med[r]) for r in reps] for name, reps in ROWS.items()}
        if not args.no_references:
            todo["ref: ISI fractions"] = [([("pcf", isi_fraction_pcfs(tr))], 1.0)]
            todo["ref: burst table"] = [([("vec", np.array([C.burst_table(t) for t in tr]), "l1")], 1.0)]
            todo["ref: multiscale sqrt time bars"] = [([("pcf", sqrt_betti(tr, s))], 1.0) for s in (0.02, C.S, 0.2)]
        for name, groups in todo.items():
            t1 = time.time()
            r = (evaluate_one(groups, True, y, k, args.seed, not args.no_kmeans) if one
                 else evaluate(groups, True, y, ia, ib, k, args.seed, not args.no_kmeans))
            rows.append(dict(dataset=args.dataset, seed=args.seed, row=name, control=cond, neurons=k,
                             samples_A=len(ia), samples_B=len(ib), **r))
            f = lambda key: f"{100 * r[key]:5.1f}" if key in r else "    -"
            print(f"  {name:<40} {cond:<13} | A {f('nn_A')} {f('kmedoids_A')} {f('kmeans_A')}"
                  f" | B {f('nn_B')} {f('kmedoids_B')} {f('kmeans_B')} | A->B {f('nn_Ato')} {f('kmedoids_Ato')}"
                  f" {f('kmeans_Ato')} | B->A {f('nn_Bto')} {f('kmedoids_Bto')} {f('kmeans_Bto')}"
                  f"  ({time.time() - t1:.0f} s)", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
