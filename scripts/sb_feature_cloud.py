"""Task L19: the spike-feature cloud, one point per spike with several qualities as coordinates.

Coordinates of spike k at time t (spikes with a neighbour on both sides):
  1 log ISI before (floor 1e-4 s)          2 log ISI after
  3, 4, 5 log(1 + own spikes within +-10, +-33, +-100 ms, the spike itself not counted)
  6 sqrt(other units' spikes within +-10 ms)
  7 (A - B) / (A + B + 1), A = other units' spikes in (t, t + 10 ms], B in [t - 10 ms, t)
  8 t / sample length (one-stimulus version only; also with weight 4)
"Other units" = every unit of the session on the same trial / time window, minus the unit itself.

Scaling: mean and sd of each coordinate from spikes of neurons outside the comparison.
Regions: k-means anchors (K = 16, 64) on 20,000 pooled scaled spikes of those outside neurons.
Readouts (same subsamples: --points per subsample, 20 subsamples):
  H0, H0+H1        Rips inside each region (stablebear), stable ranks averaged over subsamples,
                   summed over regions (a region with < 2 points gets two coincident points)
  region counts    points per region, averaged over subsamples, L1
  sliced W1        the pooled subsamples projected on 64 fixed directions, 50 quantiles each, L1
  mean point       L2
Controls: (a) ISIs shuffled, every coordinate recomputed; (b) other units from another trial.
Versions: stimulus-free (1-7, within each stimulus and across, as L17) and one-stimulus (1-8, within
stimulus A only). Ablations at K = 16.

Neurons and samples: scripts/sb_full_table.py's selection (same seed); openscope124 is the
grating blocks whole, units with >= 10 blocks of >= 103 ISIs, 60 at random, within only.

Run from the repo root:
    python scripts/sb_feature_cloud.py --dataset retina10 --drop-piece 2
Writes results/feature_cloud_<dataset>[_without_piece2].csv.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curves as C
from barcodes import shuffle_isis
from curve_eval import kmedoids, matched, read
from sb_cross import isi_fraction_pcfs
from sb_full_table import (DATASETS, choose, combine, evaluate, kmeans_mean, nn_within, sqrt_betti,
                           to_list)

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
MIN_ISI = 1e-4
W = 0.01
OWN_WINDOWS = (0.01, 0.033, 0.1)

DS = dict(DATASETS)
DS["openscope124"] = dict(path="data/openscope/glo637908.parquet", a="glo637908_grating", b=None,
                          length=124.0, names=("gratings 124 s", None))
POINTS = {"retina10": 25, "openscope10": 25, "allen30": 25, "allen30b": 25, "retina40": 100, "retina40clean": 100, "openscope124": 100}

COORDS = ["isi before", "isi after", "own 10", "own 33", "own 100", "others 10", "lead/lag", "time"]
SETS = {  # name: (coordinate indices, K values)
    "all 1-7": ([0, 1, 2, 3, 4, 5, 6], (16, 64)),
    "without 1-2": ([2, 3, 4, 5, 6], (16,)),
    "without 3-5": ([0, 1, 5, 6], (16,)),
    "without 6-7": ([0, 1, 2, 3, 4], (16,)),
    "only 1-2": ([0, 1], (16,)),
    "only 3-5": ([2, 3, 4], (16,)),
    "only 6-7": ([5, 6], (16,)),
}
ONE_STIM = {"all 1-8": ([0, 1, 2, 3, 4, 5, 6, 7], 1.0), "all 1-8, time x4": ([0, 1, 2, 3, 4, 5, 6, 7], 4.0)}


# ------------------------------------------------------------------ samples with their time segments
# A segment (trial, source offset, destination offset, length) says where in the session a
# piece of a sample came from, so the other units' spikes can be cut out the same way.


def samples_keyed(name, drop_piece):
    cfg = DS[name]
    path = REPO / cfg["path"]
    A0 = read(path, [cfg["a"]])
    seg1 = lambda df: [[(r.trial, 0.0, 0.0, float(r.duration))] for r in df.itertuples(index=False)]
    if cfg["b"] is None:
        A = A0.assign(segs=seg1(A0))
        return A[["neuron_id", "spike_times", "segs"]], None, A0, None
    B0 = read(path, [cfg["b"]])
    B = B0 if drop_piece is None else B0[B0["piece"] != drop_piece]
    if name.startswith("retina40"):  # as sb_full_table.samples
        rows = []
        for nid, g in A0.sort_values("trial").groupby("neuron_id"):
            g = list(g.itertuples(index=False))
            for j in range(len(g) // 4):
                part = g[4 * j:4 * j + 4]
                t = np.concatenate([r.spike_times + i * r.duration for i, r in enumerate(part)])
                segs = [(r.trial, 0.0, i * r.duration, r.duration) for i, r in enumerate(part)]
                rows.append(dict(neuron_id=nid, spike_times=t, segs=segs))
        A = pd.DataFrame(rows)
        rows = []
        for (nid, _), g in B.sort_values("piece").groupby(["neuron_id", "block"]):  # as sb_length.stretches
            if list(g["piece"]) != list(range(len(g))):
                continue
            g = list(g.itertuples(index=False))
            t = np.concatenate([r.spike_times + j * r.duration for j, r in enumerate(g)])
            segs = [(r.trial, 0.0, j * r.duration, r.duration) for j, r in enumerate(g)]
            rows.append(dict(neuron_id=nid, spike_times=t, segs=segs))
        B = pd.DataFrame(rows)
    elif name == "openscope10":  # as sb_cross.cut_samples
        rows = []
        for r in A0.itertuples(index=False):
            for j in range(int(r.duration // 10.0 + 1e-9)):
                t = r.spike_times[(r.spike_times >= j * 10.0) & (r.spike_times < (j + 1) * 10.0)] - j * 10.0
                rows.append(dict(neuron_id=r.neuron_id, spike_times=t, segs=[(r.trial, j * 10.0, 0.0, 10.0)]))
        A = pd.DataFrame(rows)
        B = B.assign(segs=seg1(B))
    else:
        A, B = A0.assign(segs=seg1(A0)), B.assign(segs=seg1(B))
    return A[["neuron_id", "spike_times", "segs"]], B[["neuron_id", "spike_times", "segs"]], A0, B0


def choose_one(A, args):
    """sb_full_table.choose for one side only."""
    rng = np.random.default_rng(args.seed)
    A = A[A["spike_times"].map(len) - 1 >= args.min_isis]
    ca = A["neuron_id"].value_counts()
    names = sorted(ca.index[ca >= args.min_samples])
    keep = sorted(rng.choice(names, min(args.neurons, len(names)), replace=False).tolist())
    out = []
    for n in keep:
        g = A[A["neuron_id"] == n]
        if len(g) > args.max_per_side:
            g = g.iloc[np.sort(rng.choice(len(g), args.max_per_side, replace=False))]
        out.append(g)
    return pd.concat(out).reset_index(drop=True), A[~A["neuron_id"].isin(keep)], len(names)


def population(raw):
    """trial -> sorted float32 spike times of every unit on that trial."""
    pop = {}
    for tr, g in raw.groupby("trial"):
        pop[tr] = np.sort(np.concatenate(g["spike_times"].tolist()).astype(np.float32))
    return pop


def own_lookup(raw, neurons):
    sub = raw[raw["neuron_id"].isin(neurons)]
    return {(r.neuron_id, r.trial): np.sort(np.asarray(r.spike_times, np.float32)) for r in sub.itertuples(index=False)}


def cut(arr, segs, shift=None):
    parts = []
    for tr, so, do, L in segs:
        a = arr(shift[tr] if shift else tr)
        lo, hi = np.searchsorted(a, [so, so + L])
        parts.append(a[lo:hi] - np.float32(so) + np.float32(do))
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


# ------------------------------------------------------------------ the cloud


def count(a, lo, hi, left, right):
    return np.searchsorted(a, hi, right) - np.searchsorted(a, lo, left)


def features(t, pop, own, length):
    """(n - 2) x 8 coordinates. pop includes the unit's own original spikes, own is that original
    train (both float32), so others = pop - own in every window."""
    isi = np.maximum(np.diff(t), MIN_ISI)
    tk = t[1:-1]
    cols = [np.log(isi[:-1]), np.log(isi[1:])]
    for w in OWN_WINDOWS:
        cols.append(np.log1p(np.searchsorted(t, tk + w, "right") - np.searchsorted(t, tk - w, "left") - 1))
    t32 = tk.astype(np.float32)
    w = np.float32(W)
    oth = lambda lo, hi, left, right: np.maximum(count(pop, lo, hi, left, right) - count(own, lo, hi, left, right), 0)
    around = oth(t32 - w, t32 + w, "left", "right")
    after = oth(t32, t32 + w, "right", "right")
    before = oth(t32 - w, t32, "left", "left")
    cols.append(np.sqrt(around))
    cols.append((after - before) / (after + before + 1.0))
    cols.append(tk / length)
    return np.stack(cols, 1).astype(np.float64)


def clouds_of(df, trains, popd, ownd, length, shift=None):
    out = []
    for r, t in zip(df.itertuples(index=False), trains):
        pop = cut(lambda tr: popd[tr], r.segs, shift)
        own = cut(lambda tr: ownd[(r.neuron_id, tr)], r.segs, shift)
        out.append(features(t, pop, own, length))
    return out


# ------------------------------------------------------------------ readouts


def readouts(clouds, anchors, P, S, rng, dirs, rips=True):
    """{readout: (groups, has_mean)} in sb_full_table's format."""
    import stablebear as sb
    from stablebear import persistence

    n, K, d = len(clouds), len(anchors), anchors.shape[1]
    subs, labs = [], []
    short = 0
    for c in clouds:
        lab = np.linalg.norm(c[:, None] - anchors[None], axis=2).argmin(1)
        rep = len(c) < P
        short += rep
        idx = [rng.choice(len(c), P, replace=rep) for _ in range(S)]
        subs.append([c[i] for i in idx])
        labs.append([lab[i] for i in idx])
    counts = np.array([np.mean([np.bincount(l, minlength=K) for l in ls], 0) for ls in labs])
    pooled = [np.concatenate(s) for s in subs]
    mean = np.array([p.mean(0) for p in pooled])
    q = (np.arange(50) + 0.5) / 50
    sw = np.array([np.quantile(p @ dirs[:, :d].T, q, axis=0).T.ravel() for p in pooled]) / (50 * len(dirs))
    out = {"region counts": ([([("vec", counts, "l1")], 1.0)], True),
           "sliced W1": ([([("vec", sw, "l1")], 1.0)], True),
           "mean point": ([([("vec", mean, "l2")], 1.0)], True)}
    if not rips:
        return out, short
    H = {0: [[None] * n for _ in range(K)], 1: [[None] * n for _ in range(K)]}
    chunk = max(1, 60000 // (S * K))
    for s0 in range(0, n, chunk):
        ids = range(s0, min(n, s0 + chunk))
        T = sb.zeros((len(ids), S, K), dtype=sb.pcloud64)
        for a, i in enumerate(ids):
            for s in range(S):
                for r in range(K):
                    pts = subs[i][s][labs[i][s] == r]
                    T[a, s, r] = pts if len(pts) >= 2 else np.zeros((2, d))
        bcs = persistence.compute_persistent_homology(T, max_dim=1, reduced=True, verbose=False)
        pcfs = persistence.barcode_to_stable_rank(bcs)
        for dim in (0, 1):
            for r in range(K):
                m = to_list(sb.mean(pcfs[:, :, r, dim], dim=1), len(ids))
                for a, i in enumerate(ids):
                    H[dim][r][i] = m[a]
        del T, bcs, pcfs
    out["H0"] = ([([("pcf", H[0][r]) for r in range(K)], 1.0)], True)
    out["H0+H1"] = ([([("pcf", H[0][r]) for r in range(K)], 1.0), ([("pcf", H[1][r]) for r in range(K)], 1.0)], True)
    return out, short


def evaluate_one(groups, has_mean, y, k, seed, kmeans_on):
    D, meds = combine(groups)
    r = {"nn_A": nn_within(D, y)}
    Dm = D.copy()
    np.fill_diagonal(Dm, 0)
    r["kmedoids_A"] = matched(y, kmedoids(Dm, k, np.random.default_rng(seed)), k)[0]
    if has_mean and kmeans_on:
        lab, _ = kmeans_mean(groups, meds, np.arange(len(y)), k, np.random.default_rng(seed))
        r["kmeans_A"] = matched(y, lab, k)[0]
    return r


def references(trains):
    """ISI fractions, burst table, multiscale sqrt time bars on the same samples."""
    isi = isi_fraction_pcfs(trains)
    return {"ISI fractions": ([([("pcf", isi)], 1.0)], True),
            "burst table": ([([("vec", np.array([C.burst_table(t) for t in trains]), "l1")], 1.0)], True),
            "multiscale sqrt time bars": ([([("pcf", sqrt_betti(trains, s))], 1.0) for s in (0.02, C.S, 0.2)], True)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=list(DS), required=True)
    ap.add_argument("--neurons", type=int, default=60)
    ap.add_argument("--min-isis", type=int)
    ap.add_argument("--min-samples", type=int)
    ap.add_argument("--max-per-side", type=int)
    ap.add_argument("--subsamples", type=int, default=20)
    ap.add_argument("--drop-piece", type=int)
    ap.add_argument("--no-kmeans-k64", action="store_true", help="skip k-means for K = 64 Rips readouts")
    ap.add_argument("--skip", nargs="*", default=[], help="parts to skip: free, one, ablations, controls")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    one_sided = DS[args.dataset]["b"] is None
    args.min_isis = args.min_isis or (103 if one_sided else 30)
    args.min_samples = args.min_samples or (10 if one_sided else 5)
    args.max_per_side = args.max_per_side or (1000 if one_sided else 30)
    cfg = DS[args.dataset]
    L, P, S = cfg["length"], POINTS[args.dataset], args.subsamples
    tag = "" if args.drop_piece is None else f"_without_piece{args.drop_piece}"
    out_path = RESULTS / f"feature_cloud_{args.dataset}{tag}.csv"

    A_all, B_all, A0, B0 = samples_keyed(args.dataset, args.drop_piece)
    rng0 = np.random.default_rng(args.seed + 3)
    if one_sided:
        A, poolA, n_cand = choose_one(A_all, args)
        both = A
        kept = sorted(set(A["neuron_id"]))
        poolA = poolA[poolA["spike_times"].map(len) > 3]
        pool = [("A", poolA.iloc[rng0.choice(len(poolA), min(300, len(poolA)), replace=False)])]
    else:
        A, B, _, n_cand = choose(A_all, B_all, args)
        both = pd.concat([A, B]).reset_index(drop=True)
        kept = sorted(set(A["neuron_id"]))
        # outside neurons: up to 300 samples per side, for scaling and anchors
        out = lambda df: df[~df["neuron_id"].isin(kept) & (df["spike_times"].map(len) > 3)]
        pa, pb = out(A_all), out(B_all)
        pool = [("A", pa.iloc[rng0.choice(len(pa), min(300, len(pa)), replace=False)]),
                ("B", pb.iloc[rng0.choice(len(pb), min(300, len(pb)), replace=False)])]
    del A_all, B_all
    pool_neurons = set().union(*[set(p["neuron_id"]) for _, p in pool])

    t0 = time.time()
    popd = {"A": population(A0)}
    ownd = {"A": own_lookup(A0, set(kept) | pool_neurons)}
    if not one_sided:
        popd["B"] = population(B0)
        ownd["B"] = own_lookup(B0, set(kept) | pool_neurons)
    del A0, B0
    shift = {s: dict(zip(sorted(p), np.roll(sorted(p), -1))) for s, p in popd.items()}
    print(f"population tables in {time.time() - t0:.0f} s", flush=True)

    names = sorted(set(both["neuron_id"]))
    y = both["neuron_id"].map({n: i for i, n in enumerate(names)}).to_numpy()
    k = len(names)
    ia = np.arange(len(A))
    ib = np.arange(len(A), len(both))
    side = np.array(["A"] * len(A) + ["B"] * (len(both) - len(A)))
    trains = [np.sort(np.asarray(t, float)) for t in both["spike_times"]]
    rsh = np.random.default_rng(args.seed + 1)
    shuffled = [shuffle_isis(t, rsh) for t in trains]
    sp = lambda ix: np.median([len(trains[i]) for i in ix]) if len(ix) else 0
    print(f"{args.dataset}: {cfg['names']}, {L:.0f} s; {n_cand} candidates, {k} kept; {len(ia)} + {len(ib)} samples, "
          f"median {sp(ia):.0f} / {sp(ib):.0f} spikes; {P} points x {S} subsamples; chance {100 / k:.1f}%", flush=True)

    def all_clouds(tr, other_trial=False):
        out = [None] * len(both)
        for s in popd:
            ix = np.flatnonzero(side == s)
            cl = clouds_of(both.iloc[ix], [tr[i] for i in ix], popd[s], ownd[s], L,
                           shift[s] if other_trial else None)
            for i, c in zip(ix, cl):
                out[i] = c
        return out

    t0 = time.time()
    conds = {"real": all_clouds(trains)}
    if "controls" not in args.skip:
        conds["ISIs shuffled"] = all_clouds(shuffled)
        conds["others from another trial"] = all_clouds(trains, other_trial=True)
    pool_cl = {s: clouds_of(p, [np.sort(np.asarray(t, float)) for t in p["spike_times"]], popd[s], ownd[s], L)
               for s, p in pool}
    print(f"clouds in {time.time() - t0:.0f} s; median {np.median([len(c) for c in conds['real']]):.0f} points",
          flush=True)
    free_pool = np.concatenate([c for cl in pool_cl.values() for c in cl])
    a_pool = np.concatenate(pool_cl["A"])
    stats = lambda X: (X.mean(0), np.maximum(X.std(0), 1e-9))
    mu_f, sd_f = stats(free_pool)
    mu_a, sd_a = stats(a_pool)
    print("coordinate mean / sd (outside neurons): " + ", ".join(
        f"{c} {m:.2f}/{s:.2f}" for c, m, s in zip(COORDS, mu_f, sd_f)), flush=True)
    dirs = np.random.default_rng(7).normal(size=(64, 8))

    rows = []

    def record(version, cset, K, readout, cond, res, short, secs):
        rows.append(dict(dataset=args.dataset, version=version, coords=cset, K=K, readout=readout, control=cond,
                         neurons=k, samples_A=len(ia), samples_B=len(ib), short_clouds=short, **res))
        f = lambda key: f"{100 * res[key]:5.1f}" if key in res else "    -"
        print(f"  {version:<14} {cset:<18} K={K:<3} {readout:<14} {cond:<26} | A {f('nn_A')} {f('kmedoids_A')} "
              f"{f('kmeans_A')} | B {f('nn_B')} {f('kmedoids_B')} {f('kmeans_B')} | A->B {f('nn_Ato')} "
              f"{f('kmedoids_Ato')} {f('kmeans_Ato')} | B->A {f('nn_Bto')} {f('kmedoids_Bto')} {f('kmeans_Bto')}"
              f"  ({secs:.0f} s)", flush=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    def run(version, cset, cols, weight, K, mu, sd, pool_pts, idx):
        scale = lambda X: (X[:, cols] - mu[cols]) / sd[cols] * w_vec
        w_vec = np.ones(len(cols))
        if 7 in cols:
            w_vec[cols.index(7)] = weight
        pts = scale(pool_pts)
        pts = pts[np.random.default_rng(args.seed).choice(len(pts), min(20000, len(pts)), replace=False)]
        anchors = KMeans(K, n_init=3, random_state=args.seed).fit(pts).cluster_centers_
        for cond, cl in conds.items():
            if cond == "others from another trial" and not ({5, 6} & set(cols)):
                continue  # identical to real
            t0 = time.time()
            R, short = readouts([scale(cl[i]) for i in idx], anchors, P, S, np.random.default_rng(args.seed + 2), dirs)
            for readout, (groups, has_mean) in R.items():
                km = not (args.no_kmeans_k64 and K == 64 and readout.startswith("H"))
                t1 = time.time()
                if version == "one-stimulus" or one_sided:
                    res = evaluate_one(groups, has_mean, y[idx], k, args.seed, km)
                else:
                    res = evaluate(groups, has_mean, y, ia, ib, k, args.seed, km)
                record(version, cset, K, readout, cond, res, short, time.time() - t1)
            print(f"    ({time.time() - t0:.0f} s for {cset} K={K} {cond})", flush=True)

    every = np.arange(len(both))
    # references on the same samples
    for cond, tr in (("real", trains), ("ISIs shuffled", shuffled)):
        for name, (groups, has_mean) in references(tr).items():
            t1 = time.time()
            res = (evaluate_one(groups, has_mean, y, k, args.seed, True) if one_sided
                   else evaluate(groups, has_mean, y, ia, ib, k, args.seed, True))
            record("reference", name, 0, "-", cond, res, 0, time.time() - t1)

    if "free" not in args.skip:
        for cset, (cols, Ks) in SETS.items():
            if cset != "all 1-7" and "ablations" in args.skip:
                continue
            for K in Ks:
                run("stimulus-free", cset, cols, 1.0, K, mu_f, sd_f, free_pool, every)
    if "one" not in args.skip:
        for cset, (cols, weight) in ONE_STIM.items():
            for K in (16, 64):
                if one_sided:
                    run("one-stimulus", cset, cols, weight, K, mu_a, sd_a, a_pool, every)
                else:
                    run("one-stimulus", cset, cols, weight, K, mu_a, sd_a, a_pool, ia)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
