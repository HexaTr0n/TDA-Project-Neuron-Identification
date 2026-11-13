"""Task L17: every method, within and across a real stimulus change, equal sample lengths.

Datasets (--dataset), stimulus A <-> stimulus B:
  retina10    retina 20180126_sr_le_sp frozen noise trials (9.94 s) <-> unrepeated noise pieces (10 s)
  retina40    the same, 4 consecutive frozen trials joined <-> 4 consecutive unrepeated pieces joined
  openscope10 OpenScope glo637908 grating blocks cut into 10 s pieces <-> 10 s grey-screen windows
  allen30     Allen 829720705 natural movie repeats (30 s) <-> spontaneous grey-screen windows (30 s)

Neurons: >= --min-samples samples with >= --min-isis ISIs on both sides, 60 at random (seed 0);
at most --max-per-side random samples per neuron and side.

Methods (stablebear for every barcode, PCF, mean and PCF distance), each with ISIs permuted
within every sample as the control:
  multiscale sqrt time bars   square-root Betti curves of bars [t, t+s), s = 20, 67, 200 ms
  sqrt time bars + ISI fractions   (weights 1, 0.5, as curve_eval)
  ISI fractions               stable rank of log ISI bars / ISI count
  burst table                 curves.burst_table (bursts at 20 scales with >= m spikes / count)
  burst bars                  Betti curve of barcodes.burst_bars, s = 67 ms
  peak matching               barcodes.peak_events sigma 20 ms, matching_matrix q = 10
  delay Rips m=3              log-ISI delay cloud, --points points x 20 subsamples, H0 / H0+H1
  delay regions k=8           same clouds split by 8 k-means anchors (fitted on unselected
                              neurons), one barcode per region, H0 / H0+H1
  pdf per spike               predictive_state (L=2, bandwidth 1, W2): position (mean point),
                              H0 of --points-point subsamples
Several parts are median-scaled and added; parts of one barcode family (regions) are summed raw.

Scores, all from one distance matrix over A and B together:
  within A, within B          1-NN; k-medoids; k-means with mean curves (sb.mean) where a mean exists
  A -> B, B -> A              1-NN into the other stimulus; clustering: cluster one stimulus
                              (k-medoids / k-means), pair clusters with neurons there, assign the
                              other stimulus's samples to the nearest medoid / centre

Run from the repo root:
    python scripts/sb_full_table.py --dataset retina10
Writes results/full_table_<dataset>.csv.
"""
import argparse
import functools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
import barcodes
import curves as C
from barcodes import shuffle_isis
from curve_eval import kmedoids, matched, read
from sb_cross import cut_samples, isi_fraction_pcfs
from sb_length import stretches
from spike_distance import matching_matrix

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
MIN_ISI = 1e-4

DATASETS = {
    "retina10": dict(path="data/retina_running_20180126_sr_le_sp.parquet",
                     a="20180126_sr_le_sp", b="20180126_sr_le_sp_running", length=10.0,
                     names=("frozen", "unrepeated")),
    "retina40": dict(path="data/retina_running_20180126_sr_le_sp.parquet",
                     a="20180126_sr_le_sp", b="20180126_sr_le_sp_running", length=40.0,
                     names=("frozen x4", "unrepeated x4")),
    "openscope10": dict(path="data/openscope/glo637908.parquet", a="glo637908_grating",
                        b="glo637908_spont", length=10.0, names=("gratings", "grey")),
    "allen30": dict(path="data/allen/allen829720705.parquet", a="allen829720705_movie",
                    b="allen829720705_spont", length=30.0, names=("movie", "spontaneous")),
    # retina 40 s without the repeating 5 s chunk (L24; load_retina_running.py --cut 24.8 30.0)
    "retina40clean": dict(path="data/retina_running_20180126_sr_le_sp_clean.parquet",
                          a="20180126_sr_le_sp", b="20180126_sr_le_sp_running", length=40.0,
                          names=("frozen x4", "unrepeated x4, clean")),
    # second cortical session (L24): Allen FC 789848216, same protocol
    "allen30b": dict(path="data/allen/allen789848216.parquet", a="allen789848216_movie",
                     b="allen789848216_spont", length=30.0, names=("movie", "spontaneous")),
}


# ------------------------------------------------------------------ samples


def samples(name, drop_piece=None):
    """(table A, table B) with neuron_id, spike_times."""
    cfg = DATASETS[name]
    path = REPO / cfg["path"]
    A, B = read(path, [cfg["a"]]), read(path, [cfg["b"]])
    if drop_piece is not None:
        # Retina: unrepeated piece 2 turned out to repeat across blocks (reliability 0.24).
        B = B[B["piece"] != drop_piece]
    if name.startswith("retina40"):
        rows = []
        for nid, g in A.sort_values("trial").groupby("neuron_id"):
            g = list(g.itertuples(index=False))
            for j in range(len(g) // 4):
                part = g[4 * j:4 * j + 4]
                t = np.concatenate([r.spike_times + i * r.duration for i, r in enumerate(part)])
                rows.append(dict(neuron_id=nid, spike_times=t))
        A = pd.DataFrame(rows)
        B = pd.DataFrame([dict(neuron_id=n, spike_times=t) for n, t, _ in stretches(B, True)])
    elif name == "openscope10":
        A = cut_samples(A, 10.0)
    return A[["neuron_id", "spike_times"]], B[["neuron_id", "spike_times"]]


def choose(A, B, args):
    rng = np.random.default_rng(args.seed)
    ok = lambda df: df[df["spike_times"].map(len) - 1 >= args.min_isis]
    A, B = ok(A), ok(B)
    ca, cb = A["neuron_id"].value_counts(), B["neuron_id"].value_counts()
    names = sorted(set(ca.index[ca >= args.min_samples]) & set(cb.index[cb >= args.min_samples]))
    keep = sorted(rng.choice(names, min(args.neurons, len(names)), replace=False).tolist())
    rest = sorted(set(A["neuron_id"]) - set(keep))

    def cap(df):
        out = []
        for n in keep:
            g = df[df["neuron_id"] == n]
            if len(g) > args.max_per_side:
                g = g.iloc[np.sort(rng.choice(len(g), args.max_per_side, replace=False))]
            out.append(g)
        return pd.concat(out).reset_index(drop=True)

    pool = A[A["neuron_id"].isin(rest)]
    return cap(A), cap(B), pool, len(names)


# ------------------------------------------------------------------ representations
# A part is ("pcf", list of sb.Pcf) | ("vec", array, metric) | ("dist", matrix).
# A method is a list of groups (parts, weight); a group's parts are summed raw, groups are
# divided by their median distance and added with their weights.


def sb_pdist(pcfs):
    import stablebear as sb
    return np.asarray(sb.pdist(sb.PcfTensor(pcfs), p=1).to_dense(), float)


def sb_cdist(pcfs, centres):
    import stablebear as sb
    return np.asarray(sb.cdist(sb.PcfTensor(pcfs), sb.PcfTensor(centres), p=1), float)


def vec_dist(X, Y, metric):
    if metric == "l2":
        return np.sqrt(np.maximum(((X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None] - 2 * X @ Y.T), 0))
    from curve_eval import l1_cross
    return l1_cross(np.ascontiguousarray(X, dtype=np.float64), np.ascontiguousarray(Y, dtype=np.float64))


def part_pdist(p):
    if p[0] == "pcf":
        return sb_pdist(p[1])
    if p[0] == "vec":
        return vec_dist(p[1], p[1], p[2])
    return p[1]


def to_list(T, n):
    import stablebear as sb
    return [sb.Pcf(np.asarray(T[i], dtype=np.float64)) for i in range(n)]


def sqrt_betti(trains, s):
    from sb_check import sqrt_betti_pcfs
    return sqrt_betti_pcfs(trains, s)


def betti_of(bars):
    from stablebear import persistence
    from stablebear.persistence import Barcode, BarcodeTensor
    T = persistence.barcode_to_betti_curve(BarcodeTensor([Barcode(b) for b in bars]))
    return to_list(T, len(bars))


def delay_cloud(t, m=3):
    x = np.log(np.maximum(np.diff(t), MIN_ISI))
    return sliding_window_view(x, m).astype(np.float64) / np.sqrt(m)


@functools.lru_cache(None)
def _legacy():
    import importlib.util
    path = REPO / "scripts" / "experiments" / "legacy_scripts" / "embeddings.py"
    spec = importlib.util.spec_from_file_location("legacy_embeddings", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def predictive_cloud(t):
    return _legacy().predictive_state(np.maximum(np.diff(t), MIN_ISI), L=2, bandwidth=1.0, distance="w2")


def rips_parts(clouds_per_sample, max_dim):
    """clouds_per_sample[i][s][r]: region r of subsample s of sample i (r = 0 without regions).
    Returns per homology dimension a list of parts (one per region), each a list of PCFs
    averaged over subsamples."""
    import stablebear as sb
    from stablebear import persistence

    n, S, R = len(clouds_per_sample), len(clouds_per_sample[0]), len(clouds_per_sample[0][0])
    tensor = sb.zeros((n, S, R), dtype=sb.pcloud64)
    for i, subs in enumerate(clouds_per_sample):
        for s, regs in enumerate(subs):
            for r, c in enumerate(regs):
                tensor[i, s, r] = c
    bcs = persistence.compute_persistent_homology(tensor, max_dim=max_dim, reduced=True, verbose=False)
    pcfs = persistence.barcode_to_stable_rank(bcs)  # (n, S, R, dims)
    out = {}
    for d in range(max_dim + 1):
        out[d] = [to_list(sb.mean(pcfs[:, :, r, d], dim=1), n) for r in range(R)]
    return out


def split_regions(cloud, anchors):
    nearest = np.linalg.norm(cloud[:, None] - anchors[None], axis=2).argmin(1)
    parts = [cloud[nearest == r] for r in range(len(anchors))]
    return [p if len(p) >= 2 else np.zeros((2, cloud.shape[1])) for p in parts]


def methods(trains, pool_trains, args, rng):
    """{name: (groups, has_mean)} for one list of trains."""
    out = {}
    s3 = [sqrt_betti(trains, s) for s in (0.02, C.S, 0.2)]
    isi = isi_fraction_pcfs(trains)
    out["multiscale sqrt time bars"] = ([([("pcf", p)], 1.0) for p in s3], True)
    out["sqrt time bars + ISI fractions"] = ([([("pcf", s3[1])], 1.0), ([("pcf", isi)], 0.5)], True)
    out["ISI fractions"] = ([([("pcf", isi)], 1.0)], True)
    out["burst table"] = ([([("vec", np.array([C.burst_table(t) for t in trains]), "l1")], 1.0)], True)
    out["burst bars"] = ([([("pcf", betti_of([barcodes.burst_bars(t, C.S) for t in trains]))], 1.0)], True)
    if not args.skip_peaks:
        ev = [barcodes.peak_events(t, 0.02) for t in trains]
        out["peak matching"] = ([([("dist", matching_matrix(ev, 10.0, jobs=args.jobs))], 1.0)], False)

    P, S = args.points, args.subsamples
    clouds = [delay_cloud(t) for t in trains]
    subs = [[c[rng.choice(len(c), P, replace=False)] for _ in range(S)] for c in clouds]
    plain = rips_parts([[[x] for x in sub] for sub in subs], 1)
    out["delay Rips m=3, H0"] = ([([("pcf", plain[0][0])], 1.0)], True)
    out["delay Rips m=3, H0+H1"] = ([([("pcf", plain[0][0])], 1.0), ([("pcf", plain[1][0])], 1.0)], True)

    pts = np.concatenate([delay_cloud(t) for t in pool_trains])
    pts = pts[rng.choice(len(pts), min(8000, len(pts)), replace=False)]
    anchors = KMeans(8, n_init=3, random_state=args.seed).fit(pts).cluster_centers_
    reg = rips_parts([[split_regions(x, anchors) for x in sub] for sub in subs], 1)
    out["delay regions k=8, H0"] = ([([("pcf", p) for p in reg[0]], 1.0)], True)
    out["delay regions k=8, H0+H1"] = ([([("pcf", p) for p in reg[0]], 1.0),
                                        ([("pcf", p) for p in reg[1]], 1.0)], True)

    pc = [predictive_cloud(t) for t in trains]
    out["pdf per spike, position"] = ([([("vec", np.array([c.mean(0) for c in pc]), "l2")], 1.0)], True)
    psubs = [[[c[rng.choice(len(c), P, replace=False)]] for _ in range(S)] for c in pc]
    out["pdf per spike, H0"] = ([([("pcf", rips_parts(psubs, 0)[0][0])], 1.0)], True)
    return out


# ------------------------------------------------------------------ evaluation


def combine(groups):
    """Total distance matrix and the median of each group (for centre distances)."""
    total, meds = 0, []
    for parts, w in groups:
        D = sum(part_pdist(p) for p in parts)
        med = np.median(D[D > 0]) if (D > 0).any() else 1.0
        meds.append(med)
        total = total + w * D / med
    return total, meds


def centre_dist(groups, meds, idx, cent):
    """Distance of samples idx to centres cent (cent[g][j] = centre of part j of group g)."""
    total = 0
    for (parts, w), med, cg in zip(groups, meds, cent):
        D = 0
        for p, c in zip(parts, cg):
            if p[0] == "pcf":
                D = D + sb_cdist([p[1][i] for i in idx], c)
            else:
                D = D + vec_dist(p[1][idx], c, p[2])
        total = total + w * D / med
    return total


def centres(groups, idx_by_cluster):
    import stablebear as sb
    out = []
    for parts, _ in groups:
        cg = []
        for p in parts:
            if p[0] == "pcf":
                cg.append([to_list(sb.mean(sb.PcfTensor([p[1][i] for i in m])), 1)[0] for m in idx_by_cluster])
            else:
                cg.append(np.stack([p[1][m].mean(0) for m in idx_by_cluster]))
        out.append(cg)
    return out


def kmeans_mean(groups, meds, idx, k, rng, restarts=3, iters=25):
    """k-means with mean curves on samples idx; returns labels and the centres."""
    best = (np.inf, None, None)
    n = len(idx)
    for _ in range(restarts):
        lab = None
        cl = [[idx[i]] for i in rng.choice(n, k, replace=False)]
        cent = centres(groups, cl)
        for _ in range(iters):
            Dc = centre_dist(groups, meds, idx, cent)
            new = Dc.argmin(1)
            if lab is not None and np.array_equal(new, lab):
                break
            lab = new
            far = np.argsort(-Dc[np.arange(n), lab])
            cl, f = [], 0
            for c in range(k):
                m = np.flatnonzero(lab == c)
                if len(m):
                    cl.append(idx[m])
                else:
                    cl.append(idx[[far[f]]])
                    f += 1
            cent = centres(groups, cl)
        Dc = centre_dist(groups, meds, idx, cent)
        lab = Dc.argmin(1)
        cost = Dc[np.arange(n), lab].sum()
        if cost < best[0]:
            best = (cost, lab, cent)
    return best[1], best[2]


def nn_within(D, y):
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    return float(np.mean(y[D.argmin(1)] == y))


def evaluate(groups, has_mean, y, ia, ib, k, seed, kmeans_on):
    D, meds = combine(groups)
    r = {}
    for side, ix, other in (("A", ia, ib), ("B", ib, ia)):
        Dw = D[np.ix_(ix, ix)]
        r[f"nn_{side}"] = nn_within(Dw, y[ix])
        Dm = Dw.copy()
        np.fill_diagonal(Dm, 0)
        lab = kmedoids(Dm, k, np.random.default_rng(seed))
        acc, name = matched(y[ix], lab, k)
        r[f"kmedoids_{side}"] = acc
        # medoid of each cluster, then assign the other side
        med = np.array([ix[np.flatnonzero(lab == c)][Dm[np.ix_(lab == c, lab == c)].sum(1).argmin()]
                        if (lab == c).any() else ix[0] for c in range(k)])
        to = D[np.ix_(other, med)].argmin(1)
        r[f"kmedoids_{side}to"] = float(np.mean(name[to] == y[other]))
        r[f"nn_{side}to"] = float(np.mean(y[other][D[np.ix_(ix, other)].argmin(1)] == y[ix]))
        if has_mean and kmeans_on:
            labm, cent = kmeans_mean(groups, meds, ix, k, np.random.default_rng(seed))
            accm, namem = matched(y[ix], labm, k)
            r[f"kmeans_{side}"] = accm
            to = centre_dist(groups, meds, other, cent).argmin(1)
            r[f"kmeans_{side}to"] = float(np.mean(namem[to] == y[other]))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--neurons", type=int, default=60)
    ap.add_argument("--min-isis", type=int, default=30)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--max-per-side", type=int, default=30)
    ap.add_argument("--points", type=int, default=25)
    ap.add_argument("--subsamples", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--skip-peaks", action="store_true")
    ap.add_argument("--no-kmeans", action="store_true")
    ap.add_argument("--only", nargs="*", help="method names to keep (default all)")
    ap.add_argument("--drop-piece", type=int, help="retina10: leave out this unrepeated piece")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]
    barcodes.DURATION = cfg["length"]  # peak_events reads this global

    A, B, pool, n_cand = choose(*samples(args.dataset, args.drop_piece), args)
    both = pd.concat([A, B]).reset_index(drop=True)
    names = sorted(set(both["neuron_id"]))
    y = both["neuron_id"].map({n: i for i, n in enumerate(names)}).to_numpy()
    k = len(names)
    ia, ib = np.arange(len(A)), np.arange(len(A), len(both))
    sp = lambda df: np.median(df["spike_times"].map(len))
    print(f"{args.dataset}: {cfg['names'][0]} <-> {cfg['names'][1]}, {cfg['length']:.0f} s; {n_cand} candidate "
          f"neurons, {k} kept; {len(A)} + {len(B)} samples, median {sp(A):.0f} / {sp(B):.0f} spikes; "
          f"Rips clouds of {args.points} points; chance {100 / k:.1f}%", flush=True)

    pool_trains = [np.sort(t) for t in pool["spike_times"]][:300] or [np.sort(t) for t in A["spike_times"]]
    rows = []
    tag = "" if args.drop_piece is None else f"_without_piece{args.drop_piece}"
    out_path = RESULTS / f"full_table_{args.dataset}{tag}.csv"
    for shuffled in (False, True):
        r = np.random.default_rng(args.seed + 1)
        trains = [np.sort(np.asarray(t, float)) for t in both["spike_times"]]
        if shuffled:
            trains = [shuffle_isis(t, r) for t in trains]
        t0 = time.time()
        M = methods(trains, pool_trains, args, np.random.default_rng(args.seed + 2))
        print(f"  representations ({'shuffled' if shuffled else 'real'}) in {time.time() - t0:.0f} s", flush=True)
        for name, (groups, has_mean) in M.items():
            if args.only and name not in args.only:
                continue
            t0 = time.time()
            res = evaluate(groups, has_mean, y, ia, ib, k, args.seed, not args.no_kmeans)
            rows.append(dict(dataset=args.dataset, method=name, shuffled=shuffled, neurons=k,
                             samples_A=len(A), samples_B=len(B), **res))
            f = lambda key: f"{100 * res[key]:5.1f}" if key in res else "    -"
            print(f"  {name:<32} {'shuf' if shuffled else 'real'} | within A {f('nn_A')} {f('kmedoids_A')} {f('kmeans_A')}"
                  f" | within B {f('nn_B')} {f('kmedoids_B')} {f('kmeans_B')} | A->B {f('nn_Ato')} {f('kmedoids_Ato')}"
                  f" {f('kmeans_Ato')} | B->A {f('nn_Bto')} {f('kmedoids_Bto')} {f('kmeans_Bto')}  ({time.time() - t0:.0f} s)",
                  flush=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
