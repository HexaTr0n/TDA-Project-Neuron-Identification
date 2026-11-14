"""L22 extra (web request): regional H0 against its counting twin at k = 8, 16, 32 (whole clouds).

Same samples and loader as sb_headline.py (seed 0). For each k: anchors on outside neurons, whole
delay cloud (m = 3, every point), Rips H0 per region (stablebear) against the number of points per
region; ISIs shuffled as the control. 1-NN and k-medoids only.

Run from the repo root:
    python scripts/sb_twins.py --dataset retina10 --drop-piece 2
Writes results/twins_<dataset>[_without_piece2].csv.
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
from sb_feature_cloud import evaluate_one
from sb_full_table import delay_cloud, evaluate, rips_parts, split_regions
from sb_headline import load

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--drop-piece", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32])
    args = ap.parse_args()
    A, B, pool, _ = load(args.dataset, args)
    one = B is None
    both = A if one else pd.concat([A, B]).reset_index(drop=True)
    names = sorted(set(both["neuron_id"]))
    y = both["neuron_id"].map({n: i for i, n in enumerate(names)}).to_numpy()
    k = len(names)
    ia, ib = np.arange(len(A)), np.arange(len(A), len(both))
    trains = [np.sort(np.asarray(t, float)) for t in both["spike_times"]]
    rng = np.random.default_rng(args.seed)
    pool = pool[pool["spike_times"].map(len) > 5]
    pool = pool.iloc[rng.choice(len(pool), min(300, len(pool)), replace=False)]
    pts = np.concatenate([delay_cloud(np.sort(np.asarray(t, float))) for t in pool["spike_times"]])
    pts = pts[rng.choice(len(pts), min(8000, len(pts)), replace=False)]
    rs = np.random.default_rng(args.seed + 1)
    conds = {"real": trains, "ISIs shuffled": [shuffle_isis(t, rs) for t in trains]}
    tag = "" if args.drop_piece is None else f"_without_piece{args.drop_piece}"
    out = REPO / "results" / f"twins_{args.dataset.replace(':', '_')}{tag}.csv"
    rows = []
    for K in args.ks:
        anchors = KMeans(K, n_init=3, random_state=args.seed).fit(pts).cluster_centers_
        for cond, tr in conds.items():
            clouds = [delay_cloud(t) if len(t) >= 5 else np.zeros((2, 3)) for t in tr]
            t0 = time.time()
            reg = rips_parts([[split_regions(c, anchors)] for c in clouds], 0)[0]
            lab = [np.linalg.norm(c[:, None] - anchors[None], axis=2).argmin(1) for c in clouds]
            counts = np.array([np.bincount(l, minlength=K) for l in lab], dtype=float)
            for name, groups in (("regional H0", [([("pcf", p) for p in reg], 1.0)]),
                                 ("region counts", [([("vec", counts, "l1")], 1.0)])):
                r = (evaluate_one(groups, False, y, k, args.seed, False) if one
                     else evaluate(groups, False, y, ia, ib, k, args.seed, False))
                rows.append(dict(dataset=args.dataset, K=K, readout=name, control=cond, **r))
                f = lambda key: f"{100 * r[key]:5.1f}" if key in r else "    -"
                print(f"  K={K:<3} {name:<14} {cond:<13} | A {f('nn_A')} {f('kmedoids_A')} | B {f('nn_B')}"
                      f" {f('kmedoids_B')} | A->B {f('nn_Ato')} | B->A {f('nn_Bto')}  ({time.time() - t0:.0f} s)",
                      flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
