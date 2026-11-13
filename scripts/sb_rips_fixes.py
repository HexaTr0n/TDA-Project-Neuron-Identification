"""Task L21: three fixes for what Rips cannot do, on whole clouds (every point, one barcode).

  1. H1 on the whole cloud: plain delay Rips (m = 3) and delay regions (8 and 16 anchors).
  2. H1 as a Betti curve instead of a stable rank (keeps the scale at which each loop lives).
     Readouts: H0 SR, H1 SR, H1 Betti, H0 SR + H1 SR, H0 SR + H1 Betti.
     Rips, barcodes, stable ranks and Betti curves all in stablebear.
  3. Time-filtered complex. Delay points (m = 3, raw log ISIs, as exp32) enter in spike order: a
     point enters at the time of its last spike; an edge joins two points closer than r from the
     moment both are in (r = 0.5, 1, 2). H0 bars live on the time axis. The bars are computed
     here by union-find (stablebear builds Rips only by distance, not by entry time) and handed to
     stablebear as Barcodes:
       Betti curve over time      barcode_to_betti_curve of the bars (infinite bars end at the
                                  sample length)
       births so far              Betti curve of bars [birth, length)
       lifetimes                  barcode_to_stable_rank of the bars
       Betti + births             both, median-scaled
Control: ISIs permuted within every sample. Neurons and samples as L20 (L17 selection; L19 for
openscope124). Scores as L17: 1-NN, k-medoids, k-means with sb.mean, within and across.

Run from the repo root:
    python scripts/sb_rips_fixes.py --dataset retina10 --drop-piece 2
Writes results/rips_fixes_<dataset>[_without_piece2].csv.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from barcodes import shuffle_isis
from sb_feature_cloud import DS, choose_one, evaluate_one, samples_keyed
from sb_full_table import choose, delay_cloud, evaluate, samples, split_regions, to_list

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
M = 3


def whole_barcodes(clouds, R):
    """Stable ranks and Betti curves per (region, dim) for one barcode per sample.
    clouds[i] is a list of R point clouds. Returns {(kind, r, d): list of n PCFs}."""
    import stablebear as sb
    from stablebear import persistence

    n = len(clouds)
    out = {}
    chunk = 400
    for s0 in range(0, n, chunk):
        ids = range(s0, min(n, s0 + chunk))
        T = sb.zeros((len(ids), R), dtype=sb.pcloud64)
        for a, i in enumerate(ids):
            for r in range(R):
                T[a, r] = clouds[i][r]
        bcs = persistence.compute_persistent_homology(T, max_dim=1, reduced=True, verbose=False)
        SR = persistence.barcode_to_stable_rank(bcs)
        BC = persistence.barcode_to_betti_curve(bcs)
        for r in range(R):
            for d in (0, 1):
                for kind, X in (("sr", SR), ("bc", BC)):
                    out.setdefault((kind, r, d), []).extend(to_list(X[:, r, d], len(ids)))
        del T, bcs, SR, BC
    return out


def rips_groups(clouds, anchors):
    """Readouts for the plain cloud (anchors None) or regions."""
    if anchors is None:
        cl, R = [[c] for c in clouds], 1
    else:
        cl, R = [split_regions(c, anchors) for c in clouds], len(anchors)
    B = whole_barcodes(cl, R)
    part = lambda kind, d: ([("pcf", B[(kind, r, d)]) for r in range(R)], 1.0)
    return {"H0": [part("sr", 0)],
            "H1 stable rank": [part("sr", 1)],
            "H1 Betti curve": [part("bc", 1)],
            "H0 + H1 stable rank": [part("sr", 0), part("sr", 1)],
            "H0 + H1 Betti curve": [part("sr", 0), part("bc", 1)]}


def time_filtered_bars(t, r, length):
    """H0 bars (birth, death) in seconds when delay points enter in spike order at scale r."""
    x = np.log(np.maximum(np.diff(t), 1e-4))
    if len(x) < M + 1:
        return np.zeros((0, 2))
    c = np.stack([x[i:len(x) - M + 1 + i] for i in range(M)], axis=1)
    when = t[M:M + len(c)]
    near = cdist(c, c) <= r
    parent = np.arange(len(c))
    born = when.copy()
    bars = []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for k in range(len(c)):
        roots = {find(j) for j in np.flatnonzero(near[k, :k])}
        if not roots:
            continue
        old = min(roots, key=lambda q: born[q])
        parent[k] = old
        for q in roots - {old}:
            bars.append((born[q], when[k]))
            parent[q] = old
    alive = {find(i) for i in range(len(c))}
    end = max(length, float(t[-1]))  # a sample can run a few ms past its nominal length
    bars += [(born[q], end) for q in alive]
    b = np.array(bars, dtype=np.float64).reshape(-1, 2)
    b[:, 1] = np.maximum(b[:, 1], b[:, 0])
    return b


def tf_groups(trains, r, length):
    from stablebear import persistence
    from stablebear.persistence import Barcode, BarcodeTensor

    bars = [time_filtered_bars(t, r, length) for t in trains]
    bars = [b if len(b) else np.zeros((1, 2)) for b in bars]
    n = len(bars)
    betti = to_list(persistence.barcode_to_betti_curve(BarcodeTensor([Barcode(b) for b in bars])), n)
    births = to_list(persistence.barcode_to_betti_curve(
        BarcodeTensor([Barcode(np.c_[b[:, 0], np.maximum(b[:, 0], length)]) for b in bars])), n)
    life = to_list(persistence.barcode_to_stable_rank(BarcodeTensor([Barcode(b) for b in bars])), n)
    nb = np.median([len(b) for b in bars])
    return {"Betti curve over time": [([("pcf", betti)], 1.0)],
            "births so far": [([("pcf", births)], 1.0)],
            "lifetimes (stable rank)": [([("pcf", life)], 1.0)],
            "Betti + births": [([("pcf", betti)], 1.0), ([("pcf", births)], 1.0)]}, nb


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=list(DS), required=True)
    ap.add_argument("--neurons", type=int, default=60)
    ap.add_argument("--drop-piece", type=int)
    ap.add_argument("--parts", nargs="+", default=["rips", "time"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    one = DS[args.dataset]["b"] is None
    args.min_isis, args.min_samples, args.max_per_side = (103, 10, 1000) if one else (30, 5, 30)
    length = DS[args.dataset]["length"]

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
    pts = pts[np.random.default_rng(args.seed).choice(len(pts), min(8000, len(pts)), replace=False)]
    anchors = {K: KMeans(K, n_init=3, random_state=args.seed).fit(pts).cluster_centers_ for K in (8, 16)}

    trains = [np.sort(np.asarray(t, float)) for t in both["spike_times"]]
    rs = np.random.default_rng(args.seed + 1)
    conds = {"real": trains, "ISIs shuffled": [shuffle_isis(t, rs) for t in trains]}
    print(f"{args.dataset}: {n_cand} candidates, {k} kept; {len(ia)} + {len(ib)} samples; median "
          f"{np.median([len(t) for t in trains]):.0f} spikes; chance {100 / k:.1f}%", flush=True)

    tag = "" if args.drop_piece is None else f"_without_piece{args.drop_piece}"
    out_path = RESULTS / f"rips_fixes_{args.dataset}{tag}.csv"
    rows = []

    def score(family, name, groups, cond, extra=None):
        t1 = time.time()
        r = (evaluate_one(groups, True, y, k, args.seed, True) if one
             else evaluate(groups, True, y, ia, ib, k, args.seed, True))
        rows.append(dict(dataset=args.dataset, family=family, readout=name, control=cond, neurons=k,
                         samples_A=len(ia), samples_B=len(ib), **(extra or {}), **r))
        f = lambda key: f"{100 * r[key]:5.1f}" if key in r else "    -"
        print(f"  {family:<26} {name:<24} {cond:<13} | A {f('nn_A')} {f('kmedoids_A')} {f('kmeans_A')}"
              f" | B {f('nn_B')} {f('kmedoids_B')} {f('kmeans_B')} | A->B {f('nn_Ato')} {f('kmedoids_Ato')}"
              f" {f('kmeans_Ato')} | B->A {f('nn_Bto')} {f('kmedoids_Bto')} {f('kmeans_Bto')}"
              f"  ({time.time() - t1:.0f} s)", flush=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    for cond, tr in conds.items():
        if "rips" in args.parts:
            clouds = [delay_cloud(t) for t in tr]
            for fam, anc in (("whole cloud, plain", None), ("whole cloud, regions k=8", anchors[8]),
                             ("whole cloud, regions k=16", anchors[16])):
                t0 = time.time()
                G = rips_groups(clouds, anc)
                print(f"    {fam} {cond}: barcodes in {time.time() - t0:.0f} s", flush=True)
                for name, groups in G.items():
                    score(fam, name, groups, cond)
        if "time" in args.parts:
            for r in (0.5, 1.0, 2.0):
                t0 = time.time()
                G, nb = tf_groups(tr, r, length)
                print(f"    time-filtered r={r} {cond}: bars in {time.time() - t0:.0f} s, median {nb:.0f} bars",
                      flush=True)
                for name, groups in G.items():
                    score(f"time-filtered r={r:g}", name, groups, cond, dict(median_bars=nb))
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
