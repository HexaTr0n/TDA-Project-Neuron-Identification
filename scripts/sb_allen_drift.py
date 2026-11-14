"""Task L24 (part 2): Allen, stimulus change against drift, for single-neuron signatures.

Allen 829720705: natural movie in two blocks (block 1 at 38-53 min, block 2 at 123-138 min);
30 s spontaneous windows at 0-38 min ("early", 17 windows) and 73-103 min ("late", 60).
Pairs:
  movie 1 <-> spont early    stimulus change, close in time (spont 1-38 min before the movie)
  movie 2 <-> spont late     stimulus change, 20-50 min apart
  movie 1 <-> movie 2        same stimulus, ~70 min apart
  spont early <-> spont late same stimulus (grey), ~35-100 min apart
Neurons: the 60 of L17 allen30 (sb_full_table.choose, seed 0); per pair, those with >= 3 samples
of >= 30 ISIs on both sides. All samples of each group are used.
Signatures: ISI fractions, burst table, delay regions k = 8 H0 (L17 version: 25 random points, 20
draws) and delay regions k = 8 H0 on the whole cloud (every point); ISIs shuffled as control.
Scores as L17 (sb_full_table.evaluate): within each side, and across (1-NN, k-medoids, k-means).

Run from the repo root:
    python scripts/sb_allen_drift.py
Writes results/allen_drift.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curves as C
from barcodes import shuffle_isis
from curve_eval import read
from sb_cross import isi_fraction_pcfs
from sb_full_table import choose, delay_cloud, evaluate, rips_parts, samples, split_regions

REPO = Path(__file__).resolve().parent.parent
PATH = REPO / "data/allen/allen829720705.parquet"


class Args:
    neurons, min_isis, min_samples, max_per_side, seed = 60, 30, 5, 30, 0


def signatures(trains, anchors, rng):
    out = {"ISI fractions": [([("pcf", isi_fraction_pcfs(trains))], 1.0)],
           "burst table": [([("vec", np.array([C.burst_table(t) for t in trains]), "l1")], 1.0)]}
    clouds = [delay_cloud(t) for t in trains]
    subs = [[c[rng.choice(len(c), 25, replace=len(c) < 25)] for _ in range(20)] for c in clouds]
    reg = rips_parts([[split_regions(x, anchors) for x in sub] for sub in subs], 0)
    out["delay regions k=8, H0 (25 points)"] = [([("pcf", p) for p in reg[0]], 1.0)]
    reg = rips_parts([[split_regions(c, anchors)] for c in clouds], 0)
    out["delay regions k=8, H0 (whole cloud)"] = [([("pcf", p) for p in reg[0]], 1.0)]
    return out


def main():
    A_all, B_all = samples("allen30")
    A, B, pool, _ = choose(A_all, B_all, Args)
    kept = sorted(set(A["neuron_id"]))
    raw = read(PATH, ["allen829720705_movie", "allen829720705_spont"])
    raw = raw[raw["neuron_id"].isin(kept) & (raw["spike_times"].map(len) - 1 >= 30)]
    m = raw["session"].str.endswith("movie")
    groups = {
        "movie 1": raw[m & (raw["start"] < 60 * 60)],
        "movie 2": raw[m & (raw["start"] >= 60 * 60)],
        "spont early": raw[~m & (raw["start"] < 38 * 60)],
        "spont late": raw[~m & (raw["start"] > 70 * 60)],
    }
    for g, df in groups.items():
        print(f"{g}: {df['trial'].nunique()} windows, {df['neuron_id'].nunique()} neurons, {len(df)} samples, "
              f"{df['start'].min() / 60:.1f}-{(df['start'].max() + 30) / 60:.1f} min", flush=True)
    pool_trains = [np.sort(np.asarray(t, float)) for t in pool["spike_times"] if len(t) > 4][:300]
    pts = np.concatenate([delay_cloud(t) for t in pool_trains])
    pts = pts[np.random.default_rng(0).choice(len(pts), min(8000, len(pts)), replace=False)]
    anchors = KMeans(8, n_init=3, random_state=0).fit(pts).cluster_centers_

    pairs = [("movie 1", "spont early", "stimulus change, close in time"),
             ("movie 2", "spont late", "stimulus change, 20-50 min apart"),
             ("movie 1", "movie 2", "same stimulus, 70 min apart"),
             ("spont early", "spont late", "same stimulus (grey), 35-100 min apart")]
    rows = []
    for a, b, what in pairs:
        X, Y = groups[a], groups[b]
        ok = lambda df: df["neuron_id"].value_counts()
        ca, cb = ok(X), ok(Y)
        names = sorted(set(ca.index[ca >= 3]) & set(cb.index[cb >= 3]))
        X, Y = X[X["neuron_id"].isin(names)], Y[Y["neuron_id"].isin(names)]
        both = pd.concat([X, Y]).reset_index(drop=True)
        y = both["neuron_id"].map({n: i for i, n in enumerate(names)}).to_numpy()
        ia, ib = np.arange(len(X)), np.arange(len(X), len(both))
        trains = [np.sort(np.asarray(t, float)) for t in both["spike_times"]]
        rs = np.random.default_rng(1)
        for cond, tr in (("real", trains), ("ISIs shuffled", [shuffle_isis(t, rs) for t in trains])):
            S = signatures(tr, anchors, np.random.default_rng(2))
            for name, g in S.items():
                t0 = time.time()
                r = evaluate(g, True, y, ia, ib, len(names), 0, True)
                rows.append(dict(pair=f"{a} <-> {b}", what=what, method=name, control=cond, neurons=len(names),
                                 samples_A=len(X), samples_B=len(Y), **r))
                f = lambda k: f"{100 * r[k]:5.1f}"
                print(f"  {a:>11} <-> {b:<11} {name:<36} {cond:<13} | A {f('nn_A')} {f('kmedoids_A')} {f('kmeans_A')}"
                      f" | B {f('nn_B')} {f('kmedoids_B')} {f('kmeans_B')} | A->B {f('nn_Ato')} {f('kmedoids_Ato')}"
                      f" {f('kmeans_Ato')} | B->A {f('nn_Bto')} {f('kmedoids_Bto')} {f('kmeans_Bto')}"
                      f"  [{len(names)} neurons, {len(X)}+{len(Y)}] ({time.time() - t0:.0f} s)", flush=True)
                pd.DataFrame(rows).to_csv(REPO / "results" / "allen_drift.csv", index=False)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
