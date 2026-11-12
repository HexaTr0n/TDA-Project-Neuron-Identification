"""Cross-stimulus test with ISI fractions, all in stablebear.

ISI fractions of a train: bars [0, log(max(ISI, 1e-4) / 1e-4)], one per ISI ->
barcode_to_stable_rank (the number of ISIs longer than x) -> divided by the number of ISIs.
(The shift by log(1e-4) keeps bars positive; L1 distances are unchanged.) Distances are
sb.cdist / sb.pdist with p = 1, k-means centres are sb.mean of PCFs.

Two sessions of the same units, A and B (for example frozen trials and unrepeated noise):
    within_A, within_B   1-NN inside one session
    A_to_B, B_to_A       1-NN of a sample of one session among all samples of the other
    *_far                the same, but never matching a sample of the same block (on the
                         retina: the unrepeated noise just before that frozen trial), so
                         closeness in time cannot do the work
    clust_A_to_B ...     k-means on one session without labels, clusters named by neuron
                         (Hungarian), the other session assigned to the nearest centre;
                         mean over --kmeans-seeds seeds
    rate_*               the same 1-NN scores with |log(1 + spike count)| only: what the
                         firing rate alone gives

Run from the repo root:
    python scripts/sb_cross.py --data data/retina_running_20180126_sr_le_sp.parquet \\
        --session 20180126_sr_le_sp --cross-session 20180126_sr_le_sp_running
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curve_eval import matched, read, select
from sb_check import sb_kmeans

MIN_ISI = 1e-4


def isi_fraction_pcfs(trains):
    import stablebear as sb
    from stablebear import persistence
    from stablebear.persistence import Barcode, BarcodeTensor

    bars = []
    for t in trains:
        L = np.log(np.maximum(np.diff(t), MIN_ISI) / MIN_ISI)
        bars.append(np.c_[np.zeros(len(L)), L] if len(L) else np.zeros((1, 2)))
    ranks = persistence.barcode_to_stable_rank(BarcodeTensor([Barcode(b) for b in bars]))
    out = []
    for i, t in enumerate(trains):
        a = np.asarray(ranks[i], dtype=np.float64)
        out.append(sb.Pcf(np.c_[a[:, 0], a[:, 1] / max(len(t) - 1, 1)]))
    return out


def nn(D, yq, yr, forbid=None):
    D = D.astype(float).copy()
    if forbid is not None:
        D[forbid] = np.inf
    return float(np.mean(yr[D.argmin(1)] == yq))


def main():
    import stablebear as sb

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--session", required=True, help="session A")
    ap.add_argument("--cross-session", required=True, help="session B")
    ap.add_argument("--neurons", type=int, default=60)
    ap.add_argument("--min-isis", type=int, default=30, help="for session A, as run_pipeline.py")
    ap.add_argument("--cross-min-isis", type=int, default=10, help="for session B")
    ap.add_argument("--kmeans-seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A = select(read(args.data, [args.session]), args.neurons, args.min_isis, args.seed)
    B = read(args.data, [args.cross_session])
    B = B[(B["isis"].map(len) >= args.cross_min_isis) & B["neuron_id"].isin(A["neuron_id"].unique())]
    B = B.reset_index(drop=True)
    names = np.unique(A["neuron_id"])
    ya, yb = np.searchsorted(names, A["neuron_id"]), np.searchsorted(names, B["neuron_id"])
    k = len(names)
    na, nb = A["spike_times"].map(len), B["spike_times"].map(len)
    print(f"A = {args.session}: {k} neurons, {len(A)} samples of {A['duration'].median():.2f} s, "
          f"median {na.median():.0f} spikes")
    print(f"B = {args.cross_session}: {B['neuron_id'].nunique()} neurons, {len(B)} samples of "
          f"{B['duration'].median():.2f} s, median {nb.median():.0f} spikes", flush=True)

    t0 = time.time()
    pa = isi_fraction_pcfs(list(A["spike_times"]))
    pb = isi_fraction_pcfs(list(B["spike_times"]))
    TA, TB = sb.PcfTensor(pa), sb.PcfTensor(pb)
    DAA = np.asarray(sb.pdist(TA, p=1).to_dense(), dtype=float)
    DBB = np.asarray(sb.pdist(TB, p=1).to_dense(), dtype=float)
    DAB = np.asarray(sb.cdist(TA, TB, p=1), dtype=float)
    np.fill_diagonal(DAA, np.inf)
    np.fill_diagonal(DBB, np.inf)

    has_block = "block" in A and "block" in B
    same_block = (A["block"].to_numpy()[:, None] == B["block"].to_numpy()[None, :]) if has_block else None

    ra, rb = np.log1p(na.to_numpy(float)), np.log1p(nb.to_numpy(float))
    RAA, RBB, RAB = (np.abs(ra[:, None] - ra[None, :]), np.abs(rb[:, None] - rb[None, :]),
                     np.abs(ra[:, None] - rb[None, :]))
    rng = np.random.default_rng(args.seed)
    for M in (RAA, RBB, RAB):                         # integer counts tie: break at random
        M += rng.random(M.shape) * 1e-9
    np.fill_diagonal(RAA, np.inf)
    np.fill_diagonal(RBB, np.inf)

    rows = []
    for name, (aa, bb, ab) in (("ISI fractions (stablebear)", (DAA, DBB, DAB)),
                               ("rate only", (RAA, RBB, RAB))):
        r = dict(feature=name, within_A=nn(aa, ya, ya), within_B=nn(bb, yb, yb),
                 A_to_B=nn(ab, ya, yb), B_to_A=nn(ab.T, yb, ya))
        if has_block:
            r["A_to_B_far"] = nn(ab, ya, yb, same_block)
            r["B_to_A_far"] = nn(ab.T, yb, ya, same_block.T)
        rows.append(r)

    clust = {"clust_A_to_B": [], "clust_B_to_A": [], "clust_A": [], "clust_B": []}
    for s in range(args.kmeans_seeds):
        for src, dst, ys, yd, key in ((pa, pb, ya, yb, "A"), (pb, pa, yb, ya, "B")):
            lab, cent = sb_kmeans_with_centres(src, k, np.random.default_rng(s))
            share, name = matched(ys, lab, k)
            pred = name[np.asarray(sb.cdist(sb.PcfTensor(dst), cent, p=1)).argmin(1)]
            clust[f"clust_{key}"].append(share)
            clust[f"clust_{key}_to_{'B' if key == 'A' else 'A'}"].append(float(np.mean(pred == yd)))
    rows[0].update({key: float(np.mean(v)) for key, v in clust.items()})

    print(f"\nchance about {1 / k:.3f}; {time.time() - t0:.0f} s")
    print(pd.DataFrame(rows).set_index("feature").T.round(3).to_string())
    print("\nclustering per seed:", {key: np.round(v, 3).tolist() for key, v in clust.items()})


def sb_kmeans_with_centres(pcfs, k, rng):
    """sb_check.sb_kmeans, also returning the centres as a PcfTensor."""
    import stablebear as sb

    lab = sb_kmeans(pcfs, k, rng)
    cent = sb.concatenate([sb.mean(sb.PcfTensor([pcfs[i] for i in np.flatnonzero(lab == c)]))
                           for c in range(k) if np.any(lab == c)])
    used = [c for c in range(k) if np.any(lab == c)]
    remap = np.full(k, -1)
    remap[used] = np.arange(len(used))
    return remap[lab], cent


if __name__ == "__main__":
    main()
