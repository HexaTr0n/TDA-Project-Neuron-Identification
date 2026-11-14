"""Task L2: square-root time bars and k-means with mean curves on real stablebear.

The curves in curves.py are numpy grids. Here the same signature goes through stablebear:
time bars -> Barcode -> barcode_to_betti_curve (an exact PCF) -> square root of its values ->
Pcf. Distances are sb.pdist / sb.cdist with p = 1, and k-means centres are sb.mean of the
members' PCFs. The script prints the stablebear scores next to the numpy ones on the same
trials, and how closely the two distance matrices agree.

Run from the repo root:
    python scripts/sb_check.py --session 20180126_sr_le_sp --neurons 60
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curves as C
from curve_eval import RETINA, kmeans, l1_pdist, matched, nn_score, read, select


def sqrt_betti_pcfs(trains, s):
    import stablebear as sb
    from stablebear import persistence
    from stablebear.persistence import Barcode, BarcodeTensor

    curves = persistence.barcode_to_betti_curve(
        BarcodeTensor([Barcode(np.c_[t, t + s]) for t in trains]))
    out = []
    for i in range(len(trains)):
        a = np.asarray(curves[i], dtype=np.float64)
        out.append(sb.Pcf(np.c_[a[:, 0], np.sqrt(a[:, 1])]))
    return out


def sb_kmeans(pcfs, k, rng, restarts=3, iters=25):
    """k-means with mean curves, every step in stablebear."""
    import stablebear as sb

    T = sb.PcfTensor(pcfs)
    n = len(pcfs)
    best = (np.inf, None)
    for _ in range(restarts):
        cent = sb.concatenate([sb.PcfTensor([pcfs[i]]) for i in rng.choice(n, k, replace=False)])
        lab = None
        for _ in range(iters):
            Dc = np.asarray(sb.cdist(T, cent, p=1))
            new = Dc.argmin(1)
            if lab is not None and np.array_equal(new, lab):
                break
            lab = new
            far = np.argsort(-Dc[np.arange(n), lab])
            parts, f = [], 0
            for c in range(k):
                m = np.flatnonzero(lab == c)
                if len(m):
                    parts.append(sb.mean(sb.PcfTensor([pcfs[i] for i in m])))
                else:
                    parts.append(sb.PcfTensor([pcfs[far[f]]]))
                    f += 1
            cent = sb.concatenate(parts)
        Dc = np.asarray(sb.cdist(T, cent, p=1))
        lab = Dc.argmin(1)
        cost = Dc[np.arange(n), lab].sum()
        if cost < best[0]:
            best = (cost, lab)
    return best[1]


def main():
    import stablebear as sb

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(RETINA))
    ap.add_argument("--session", default="20180126_sr_le_sp")
    ap.add_argument("--neurons", type=int, default=60)
    ap.add_argument("--min-isis", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kmeans-seeds", type=int, default=5, help="k-means runs with seeds 0..n-1")
    args = ap.parse_args()

    rows = select(read(args.data, [args.session]), args.neurons, args.min_isis, args.seed)
    y = np.unique(rows["neuron_id"], return_inverse=True)[1]
    k = y.max() + 1
    trains = list(rows["spike_times"])
    durs = rows["duration"].to_numpy(float)
    print(f"{args.session}: {k} neurons, {len(rows)} trials", flush=True)

    t0 = time.time()
    pcfs = sqrt_betti_pcfs(trains, C.S)
    Dsb = np.asarray(sb.pdist(sb.PcfTensor(pcfs), p=1).to_dense(), dtype=np.float64)
    sb_nn = nn_score(Dsb.copy(), y, np.random.default_rng(args.seed))
    sb_km = np.array([matched(y, sb_kmeans(pcfs, k, np.random.default_rng(s)), k)[0]
                      for s in range(args.kmeans_seeds)])
    t_sb = time.time() - t0

    t0 = time.time()
    F = np.stack([C.sqrt_time_bars(t, d) for t, d in zip(trains, durs)]).astype(np.float32)
    Dnp = l1_pdist(F)
    np_nn = nn_score(Dnp.copy(), y, np.random.default_rng(args.seed))
    np_km = np.array([matched(y, kmeans(F, k, np.random.default_rng(s))[0], k)[0]
                      for s in range(args.kmeans_seeds)])
    t_np = time.time() - t0

    iu = np.triu_indices(len(y), 1)
    rel = np.abs(Dsb[iu] - Dnp[iu]) / np.maximum(Dsb[iu], 1e-12)
    same_nn = np.mean((Dsb + np.diag(np.full(len(y), np.inf))).argmin(1)
                      == (Dnp + np.diag(np.full(len(y), np.inf))).argmin(1))
    print(f"\nk-means over seeds 0..{args.kmeans_seeds - 1} (3 restarts each): mean, sd, min, max")
    print(f"{'':10s} {'1-NN':>6s} {'k-means':>8s} {'sd':>6s} {'min':>6s} {'max':>6s} {'seconds':>8s}")
    for name, nn, km, sec in (("stablebear", sb_nn, sb_km, t_sb), ("numpy grid", np_nn, np_km, t_np)):
        print(f"{name:10s} {nn:6.3f} {km.mean():8.3f} {km.std():6.3f} {km.min():6.3f} {km.max():6.3f} {sec:8.1f}")
    print("per seed, stablebear:", np.round(sb_km, 3), " numpy:", np.round(np_km, 3))
    print(f"\ndistance matrices: correlation {np.corrcoef(Dsb[iu], Dnp[iu])[0, 1]:.6f}, "
          f"median relative difference {np.median(rel):.2e}, max {rel.max():.2e}")
    print(f"same nearest neighbour for {same_nn:.3f} of trials")


if __name__ == "__main__":
    main()
