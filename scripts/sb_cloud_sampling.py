"""Task L20: three ways to cut a delay cloud to size for Rips.

  random        P points drawn at random, 20 draws, stable ranks averaged (as L17)
  consecutive   P consecutive points: a random number of points cut off the start and the end,
                20 draws averaged; samples with fewer than P points use all of them (counted)
  whole         every point, one barcode; reported raw and with the stable rank divided by the
                number of points (otherwise the spike count leaks in); regions k = 8, 16, 32, 64
  crop70/50     a random amount of time cut off the start and the end so that 70% / 50% of the
                sample remains, every point of it, one barcode, no averaging (user, 2026-11-13)
P = 25 on short samples, 100 on long ones (retina40, openscope124).

Signatures (stablebear): delay Rips m = 3 (H0, H0+H1) and delay regions k = 8 (H0, H0+H1; anchors fitted
on outside neurons, as L17). Control: ISIs permuted within every sample.

Neurons and samples as L17 (sb_full_table.samples / choose, seed 0); openscope124 as L19
(grating blocks, units with >= 10 blocks of >= 103 ISIs, within only). Scores as L17: 1-NN,
k-medoids, k-means with sb.mean, within and across.

Run from the repo root:
    python scripts/sb_cloud_sampling.py --dataset retina10 --drop-piece 2
Writes results/cloud_sampling_<dataset>[_without_piece2].csv.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from barcodes import shuffle_isis
from sb_feature_cloud import DS, POINTS, choose_one, evaluate_one, samples_keyed
from sb_full_table import (choose, delay_cloud, evaluate, rips_parts, samples, split_regions)

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"


def draws(clouds, mode, P, S, rng):
    """Per sample a list of point clouds (S of them, or 1 for whole) and how many were short."""
    out, short = [], 0
    for c in clouds:
        n = len(c)
        if mode == "whole":
            out.append([c])
        elif mode == "random":
            short += n < P
            out.append([c[rng.choice(n, P, replace=n < P)] for _ in range(S)])
        else:  # consecutive
            if n <= P:
                short += 1
                out.append([c] * S)
            else:
                starts = rng.integers(0, n - P + 1, S)
                out.append([c[s:s + P] for s in starts])
    return out, short


def scale_pcfs(pcfs, factors):
    import stablebear as sb
    out = []
    for p, f in zip(pcfs, factors):
        a = np.asarray(p, dtype=np.float64).copy()
        a[:, 1] *= f
        out.append(sb.Pcf(a))
    return out


def crop_clouds(trains, length, keep, rng):
    """Cut a random amount of time off the start and the end, keep the fraction `keep` of the
    sample, and return the delay cloud of every spike left (one cloud per sample)."""
    out, short = [], 0
    for t in trains:
        a = rng.uniform(0, (1 - keep) * length)
        u = t[(t >= a) & (t < a + keep * length)]
        c = delay_cloud(u) if len(u) >= 5 else np.zeros((0, 3))
        if len(c) < 2:
            short += 1
            c = np.zeros((2, 3))
        out.append(c)
    return out, short


def group_of(name, parts, H1=None):
    """parts: list of PCF lists (regions) for H0; H1 the same for H1 or None."""
    g = [([("pcf", p) for p in parts], 1.0)]
    if H1 is not None:
        g.append(([("pcf", p) for p in H1], 1.0))
    return g


def signatures(clouds, anchors, Ks, mode, P, S, rng, divide):
    """{name: groups}. anchors: {K: centres}; Ks: region counts used in this mode."""
    if mode in ("random", "consecutive"):
        subs, short = draws(clouds, mode, P, S, rng)
    else:  # whole or crop: one barcode of every point
        subs, short = [[c] for c in clouds], 0
    sigs = {}
    plain = rips_parts([[[x] for x in sub] for sub in subs], 1)
    sigs["delay Rips m=3, H0"] = ([plain[0][0]], None)
    sigs["delay Rips m=3, H0+H1"] = ([plain[0][0]], [plain[1][0]])
    for K in Ks:
        reg = rips_parts([[split_regions(x, anchors[K]) for x in sub] for sub in subs], 1)
        sigs[f"delay regions k={K}, H0"] = (reg[0], None)
        sigs[f"delay regions k={K}, H0+H1"] = (reg[0], reg[1])
    groups = {n: group_of(n, h0, h1) for n, (h0, h1) in sigs.items()}
    if divide:
        f = [1.0 / max(len(c), 1) for c in clouds]
        for n, (h0, h1) in sigs.items():
            groups[n + ", / points"] = group_of(n, [scale_pcfs(p, f) for p in h0],
                                                None if h1 is None else [scale_pcfs(p, f) for p in h1])
    return groups, short


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=list(DS), required=True)
    ap.add_argument("--neurons", type=int, default=60)
    ap.add_argument("--subsamples", type=int, default=20)
    ap.add_argument("--drop-piece", type=int)
    ap.add_argument("--modes", nargs="+", default=["whole", "crop70", "crop50", "random", "consecutive"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    one = DS[args.dataset]["b"] is None
    args.min_isis, args.min_samples, args.max_per_side = (103, 10, 1000) if one else (30, 5, 30)
    P, S = POINTS[args.dataset], args.subsamples

    if one:
        A_all, _, A0, _ = samples_keyed(args.dataset, None)
        del A0
        A, pool, n_cand = choose_one(A_all, args)
        both = A
    else:
        A_all, B_all = samples(args.dataset, args.drop_piece)
        A, B, pool, n_cand = choose(A_all, B_all, args)
        both = pd.concat([A, B]).reset_index(drop=True)
    names = sorted(set(both["neuron_id"]))
    y = both["neuron_id"].map({n: i for i, n in enumerate(names)}).to_numpy()
    k = len(names)
    ia, ib = np.arange(len(A)), np.arange(len(A), len(both))

    pool_trains = [np.sort(np.asarray(t, float)) for t in pool["spike_times"] if len(t) > 4][:300]
    pts = np.concatenate([delay_cloud(t) for t in pool_trains])
    rng = np.random.default_rng(args.seed)
    pts = pts[rng.choice(len(pts), min(8000, len(pts)), replace=False)]
    anchors = {K: KMeans(K, n_init=3, random_state=args.seed).fit(pts).cluster_centers_ for K in (8, 16, 32, 64)}
    length = DS[args.dataset]["length"]

    trains = [np.sort(np.asarray(t, float)) for t in both["spike_times"]]
    rs = np.random.default_rng(args.seed + 1)
    conds = {"real": trains, "ISIs shuffled": [shuffle_isis(t, rs) for t in trains]}
    npts = np.array([len(t) - 3 for t in trains])
    print(f"{args.dataset}: {n_cand} candidates, {k} kept; {len(ia)} + {len(ib)} samples; delay points median "
          f"{np.median(npts):.0f}, min {npts.min()}, max {npts.max()}; P = {P}, {S} draws; chance {100 / k:.1f}%",
          flush=True)

    tag = "" if args.drop_piece is None else f"_without_piece{args.drop_piece}"
    out_path = RESULTS / f"cloud_sampling_{args.dataset}{tag}.csv"
    rows = []
    for mode in args.modes:
        for cond, tr in conds.items():
            t0 = time.time()
            r0 = np.random.default_rng(args.seed + 2)
            if mode.startswith("crop"):
                clouds, short0 = crop_clouds(tr, length, int(mode[4:]) / 100, r0)
            else:
                clouds, short0 = [delay_cloud(t) for t in tr], 0
            Ks = (8, 16, 32, 64) if mode == "whole" else (8,)
            G, short = signatures(clouds, anchors, Ks, mode, P, S, r0, divide=mode == "whole")
            short += short0
            print(f"  {mode} {cond}: barcodes in {time.time() - t0:.0f} s, short samples {short}", flush=True)
            for name, groups in G.items():
                t1 = time.time()
                r = (evaluate_one(groups, True, y, k, args.seed, True) if one
                     else evaluate(groups, True, y, ia, ib, k, args.seed, True))
                rows.append(dict(dataset=args.dataset, sampling=mode, method=name, control=cond, points=P,
                                 short=short, neurons=k, samples_A=len(ia), samples_B=len(ib), **r))
                f = lambda key: f"{100 * r[key]:5.1f}" if key in r else "    -"
                print(f"  {mode:<11} {name:<36} {cond:<13} | A {f('nn_A')} {f('kmedoids_A')} {f('kmeans_A')}"
                      f" | B {f('nn_B')} {f('kmedoids_B')} {f('kmeans_B')} | A->B {f('nn_Ato')} {f('kmedoids_Ato')}"
                      f" {f('kmeans_Ato')} | B->A {f('nn_Bto')} {f('kmedoids_Bto')} {f('kmeans_Bto')}"
                      f"  ({time.time() - t1:.0f} s)", flush=True)
                pd.DataFrame(rows).to_csv(out_path, index=False)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
