"""L19 extra: was Rips held back by cloud size?

Same neurons and samples as sb_feature_cloud.py (retina40 via sb_full_table.choose, openscope124
via choose_one). Cloud: delay points (log ISI before, log ISI after) of every spike with a
neighbour on both sides (= coordinates 1-2 of L19, delay embedding m = 2), scaled by the mean and
sd of outside neurons; 16 k-means anchors on 20,000 outside points.

For cloud sizes P = 25, 50, 100, 200, 400 (20 subsamples each, with replacement only where a
sample has fewer than P points):
  regional H0       Rips H0 inside each region (stablebear), stable ranks averaged, summed over regions
  region counts     the twin without homology
  ISI fractions, P ISIs   stable rank of the P ISIs before the same subsampled points, averaged
                          over the 20 subsamples (the ISI distribution at equal data)
Plus ISI fractions on every ISI (the usual reference), and ISIs shuffled as the control.

Run from the repo root:
    python scripts/sb_cloud_size.py --dataset retina40
Writes results/cloud_size_<dataset>.csv.
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
from sb_cross import isi_fraction_pcfs
from sb_feature_cloud import DS, MIN_ISI, choose_one, evaluate_one, samples_keyed
from sb_full_table import choose, evaluate, to_list

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
SIZES = (25, 50, 100, 200, 400)


def delay2(t):
    x = np.log(np.maximum(np.diff(t), MIN_ISI))
    return np.c_[x[:-1], x[1:]]


def run_size(clouds, isis, anchors, P, S, seed):
    """{readout: groups} for one cloud size."""
    import stablebear as sb
    from stablebear import persistence

    rng = np.random.default_rng(seed)
    n, K = len(clouds), len(anchors)
    idx = [[rng.choice(len(c), P, replace=len(c) < P) for _ in range(S)] for c in clouds]
    short = sum(len(c) < P for c in clouds)
    labs = [np.linalg.norm(c[:, None] - anchors[None], axis=2).argmin(1) for c in clouds]
    counts = np.array([np.mean([np.bincount(lab[i], minlength=K) for i in ix], 0) for lab, ix in zip(labs, idx)])
    # ISI fractions of the P ISIs before the chosen points, averaged over subsamples
    fake = [np.r_[0, np.cumsum(isis[j][i])] for j, ix in enumerate(idx) for i in ix]
    pc = isi_fraction_pcfs(fake)
    isi_p = [to_list(sb.mean(sb.PcfTensor(pc[j * S:(j + 1) * S])), 1)[0] for j in range(n)]
    H0 = [[None] * n for _ in range(K)]
    chunk = max(1, 40000 // (S * K))
    for s0 in range(0, n, chunk):
        ids = range(s0, min(n, s0 + chunk))
        T = sb.zeros((len(ids), S, K), dtype=sb.pcloud64)
        for a, j in enumerate(ids):
            for s, i in enumerate(idx[j]):
                pts, lab = clouds[j][i], labs[j][i]
                for r in range(K):
                    p = pts[lab == r]
                    T[a, s, r] = p if len(p) >= 2 else np.zeros((2, 2))
        bcs = persistence.compute_persistent_homology(T, max_dim=0, reduced=True, verbose=False)
        pcfs = persistence.barcode_to_stable_rank(bcs)
        for r in range(K):
            m = to_list(sb.mean(pcfs[:, :, r, 0], dim=1), len(ids))
            for a, j in enumerate(ids):
                H0[r][j] = m[a]
        del T, bcs, pcfs
    return {"regional H0": [([("pcf", H0[r]) for r in range(K)], 1.0)],
            "region counts": [([("vec", counts, "l1")], 1.0)],
            "ISI fractions, same ISIs": [([("pcf", isi_p)], 1.0)]}, short


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=["retina40", "openscope124"], required=True)
    ap.add_argument("--neurons", type=int, default=60)
    ap.add_argument("--subsamples", type=int, default=20)
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    one = DS[args.dataset]["b"] is None
    args.min_isis, args.min_samples, args.max_per_side = (103, 10, 1000) if one else (30, 5, 30)

    A_all, B_all, A0, B0 = samples_keyed(args.dataset, None)
    del A0, B0
    if one:
        A, poolA, _ = choose_one(A_all, args)
        both = A
        pool = poolA
    else:
        A, B, _, _ = choose(A_all, B_all, args)
        both = pd.concat([A, B]).reset_index(drop=True)
        kept = set(A["neuron_id"])
        pool = pd.concat([A_all[~A_all["neuron_id"].isin(kept)], B_all[~B_all["neuron_id"].isin(kept)]])
    del A_all, B_all
    pool = pool[pool["spike_times"].map(len) > 3]
    pool = pool.iloc[np.random.default_rng(args.seed + 3).choice(len(pool), min(600, len(pool)), replace=False)]
    pts = np.concatenate([delay2(np.sort(np.asarray(t, float))) for t in pool["spike_times"]])
    mu, sd = pts.mean(0), pts.std(0)
    pts = (pts - mu) / sd
    pts = pts[np.random.default_rng(args.seed).choice(len(pts), min(20000, len(pts)), replace=False)]
    anchors = KMeans(16, n_init=3, random_state=args.seed).fit(pts).cluster_centers_

    names = sorted(set(both["neuron_id"]))
    y = both["neuron_id"].map({n: i for i, n in enumerate(names)}).to_numpy()
    k = len(names)
    ia, ib = np.arange(len(A)), np.arange(len(A), len(both))
    trains = [np.sort(np.asarray(t, float)) for t in both["spike_times"]]
    rs = np.random.default_rng(args.seed + 1)
    conds = {"real": trains, "ISIs shuffled": [shuffle_isis(t, rs) for t in trains]}
    pts_n = np.array([len(t) - 2 for t in trains])
    print(f"{args.dataset}: {k} neurons, {len(ia)} + {len(ib)} samples; points per sample median "
          f"{np.median(pts_n):.0f}, min {pts_n.min()}", flush=True)

    rows = []
    out_path = RESULTS / f"cloud_size_{args.dataset}.csv"

    def score(groups, name, P, cond, short):
        t0 = time.time()
        r = (evaluate_one(groups, False, y, k, args.seed, False) if one
             else evaluate(groups, False, y, ia, ib, k, args.seed, False))
        rows.append(dict(dataset=args.dataset, readout=name, points=P, control=cond, short=short, **r))
        f = lambda key: f"{100 * r[key]:5.1f}" if key in r else "    -"
        print(f"  P={P:<4} {name:<26} {cond:<14} | A {f('nn_A')} {f('kmedoids_A')} | B {f('nn_B')} {f('kmedoids_B')}"
              f" | A->B {f('nn_Ato')} | B->A {f('nn_Bto')}  short {short}  ({time.time() - t0:.0f} s)", flush=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    for cond, tr in conds.items():
        score([([("pcf", isi_fraction_pcfs(tr))], 1.0)], "ISI fractions, every ISI", 0, cond, 0)
        clouds = [(delay2(t) - mu) / sd for t in tr]
        isis = [np.maximum(np.diff(t), MIN_ISI)[:-1] for t in tr]  # ISI before each point
        for P in args.sizes:
            t0 = time.time()
            R, short = run_size(clouds, isis, anchors, P, args.subsamples, args.seed + 2)
            print(f"    P={P} {cond}: readouts in {time.time() - t0:.0f} s", flush=True)
            for name, groups in R.items():
                if cond == "ISIs shuffled" and name == "ISI fractions, same ISIs":
                    continue  # the same ISIs in another order: identical up to subsampling
                score(groups, name, P, cond, short)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
