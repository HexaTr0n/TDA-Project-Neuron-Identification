"""Task L18: the Victor-Purpura spike distance as the reference, on the cells of L17.

Same neurons and samples as scripts/sb_full_table.py (its samples() and choose(), same seed),
so the numbers sit next to L17's.

Distances:
  spike distance q = 0, 1, 3, 10, 30, 100 per s   spike_distance.vp_distance_matrix_fast, the same
                                                   recursion as vp_distance_matrix compiled with numba
                                                   (q = 0 is the spike-count difference); the numpy
                                                   version is timed on a 500-sample subset
  van Rossum sigma = 10, 20, 50 ms                L2 between Gaussian-smoothed trains (1 ms grid)
  spike distance + 0.5 x ISI fractions            at the best q of each cell, median-scaled
  shuffled control (ISIs permuted)                spike distance at the best q of each cell
  timing references                               multiscale sqrt time bars and ISI fractions
                                                   (stablebear), timed on the same samples

Cells: within A, within B (1-NN, k-medoids), A -> B, B -> A (1-NN; k-medoids on one side,
assign the other side to the nearest medoid), as in sb_full_table.evaluate.

Honest q: for every cell, q is chosen by 1-NN on half the neurons and scored on the other half,
50 random splits (as compare.py). Chance on a half is 1 / 30. Next to it: the best q
picked on the scored half itself (optimistic), so the gap is the cost of choosing.

Distance matrices are cached in results/spike_distance/<dataset>/ as float32 .npy.

Run from the repo root:
    python scripts/sb_spike_distance.py --dataset allen30
Writes results/spike_distance_<dataset>[_without_piece2].csv and ..._timing.csv.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import barcodes
import curves as C
from barcodes import shuffle_isis
from curve_eval import kmedoids, matched
from sb_cross import isi_fraction_pcfs
from sb_full_table import DATASETS, choose, nn_within, samples, sb_pdist, sqrt_betti
from spike_distance import vp_distance_matrix, vp_distance_matrix_fast

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
QS = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0)
SIGMAS = (0.01, 0.02, 0.05)


def van_rossum(trains, sigma, length, dt=0.001):
    """L2 between trains smoothed with a Gaussian of width sigma, units of sqrt(spikes)."""
    n = int(round(length / dt)) + 1
    step = max(1, int(sigma / (2 * dt)))
    F = np.zeros((len(trains), len(range(0, n, step))), np.float32)
    for i, t in enumerate(trains):
        b = np.zeros(n)
        np.add.at(b, np.minimum((t / dt).astype(int), n - 1), 1.0)
        F[i] = gaussian_filter1d(b, sigma / dt, mode="constant")[::step]
    sq = (F.astype(np.float64) ** 2).sum(1)
    G = F @ F.T
    return np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * G, 0) * step)


def scaled(D):
    m = np.median(D[D > 0]) if (D > 0).any() else 1.0
    return D / m


def score(D, y, ia, ib, k, seed):
    r = {}
    for side, ix, other in (("A", ia, ib), ("B", ib, ia)):
        Dw = D[np.ix_(ix, ix)].astype(np.float64)
        r[f"nn_{side}"] = nn_within(Dw, y[ix])
        np.fill_diagonal(Dw, 0)
        lab = kmedoids(Dw, k, np.random.default_rng(seed))
        acc, name = matched(y[ix], lab, k)
        r[f"kmedoids_{side}"] = acc
        med = np.array([ix[np.flatnonzero(lab == c)][Dw[np.ix_(lab == c, lab == c)].sum(1).argmin()]
                        if (lab == c).any() else ix[0] for c in range(k)])
        r[f"kmedoids_{side}to"] = float(np.mean(name[D[np.ix_(other, med)].argmin(1)] == y[other]))
        r[f"nn_{side}to"] = float(np.mean(y[other][D[np.ix_(ix, other)].argmin(1)] == y[ix]))
    return r


def nn_cell(D, y, src, dst):
    """1-NN accuracy of samples src among samples dst (dst may equal src: within)."""
    S = D[np.ix_(src, dst)].astype(np.float64)
    if src is dst:
        np.fill_diagonal(S, np.inf)
    return float(np.mean(y[dst][S.argmin(1)] == y[src]))


CELLS = {"nn_A": ("A", "A"), "nn_B": ("B", "B"), "nn_Ato": ("A", "B"), "nn_Bto": ("B", "A")}


def held_out(mats, y, ia, ib, splits):
    """{cell: (mean held-out 1-NN, how often each q was picked)} over the splits."""
    side = {"A": ia, "B": ib}
    out = {}
    for cell, (s, d) in CELLS.items():
        scores, picks, oracle = [], [], []
        for half_a, half_b in splits:
            def sc(D, half):
                src = side[s][np.isin(y[side[s]], half)]
                dst = src if s == d else side[d][np.isin(y[side[d]], half)]
                return nn_cell(D, y, src, dst)
            best = max(mats, key=lambda q: sc(mats[q], half_a))
            picks.append(best)
            scores.append(sc(mats[best], half_b))
            oracle.append(max(sc(D, half_b) for D in mats.values()))
        out[cell] = (float(np.mean(scores)), pd.Series(picks).value_counts().to_dict(), float(np.mean(oracle)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--neurons", type=int, default=60)
    ap.add_argument("--min-isis", type=int, default=30)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--max-per-side", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--time-subset", type=int, default=500)
    ap.add_argument("--drop-piece", type=int)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]
    barcodes.DURATION = cfg["length"]

    A, B, _, n_cand = choose(*samples(args.dataset, args.drop_piece), args)
    both = pd.concat([A, B]).reset_index(drop=True)
    names = sorted(set(both["neuron_id"]))
    y = both["neuron_id"].map({n: i for i, n in enumerate(names)}).to_numpy()
    k = len(names)
    ia, ib = np.arange(len(A)), np.arange(len(A), len(both))
    trains = [np.sort(np.asarray(t, float)) for t in both["spike_times"]]
    rs = np.random.default_rng(args.seed + 1)
    shuffled = [shuffle_isis(t, rs) for t in trains]
    nspk = np.array([len(t) for t in trains])
    print(f"{args.dataset}: {cfg['names'][0]} <-> {cfg['names'][1]}; {n_cand} candidates, {k} kept; "
          f"{len(A)} + {len(B)} samples; median {np.median(nspk[ia]):.0f} / {np.median(nspk[ib]):.0f} spikes",
          flush=True)

    tag = "" if args.drop_piece is None else f"_without_piece{args.drop_piece}"
    cache = RESULTS / "spike_distance" / f"{args.dataset}{tag}"
    cache.mkdir(parents=True, exist_ok=True)
    timing = []

    def get(name, fn):
        p = cache / f"{name}.npy"
        if p.exists():
            return np.load(p)
        t0 = time.time()
        D = fn().astype(np.float32)
        timing.append(dict(distance=name, seconds=time.time() - t0, samples=len(trains)))
        print(f"  {name}: {timing[-1]['seconds']:.0f} s", flush=True)
        np.save(p, D)
        return D

    # references, timed
    def isi_D():
        return sb_pdist(isi_fraction_pcfs(trains))

    def ms_D():
        return sum(scaled(sb_pdist(sqrt_betti(trains, s))) for s in (0.02, C.S, 0.2))

    Disi = get("isi_fractions", isi_D)
    get("multiscale_sqrt_time_bars", ms_D)

    # the repo's numpy version, timed on --time-subset samples only (cost grows with n^2)
    sub = np.random.default_rng(args.seed).choice(len(trains), min(args.time_subset, len(trains)), replace=False)
    t0 = time.time()
    Dn = vp_distance_matrix([trains[i] for i in sub], 10.0, args.jobs)
    tn = time.time() - t0
    t0 = time.time()
    Df = vp_distance_matrix_fast([trains[i] for i in sub], 10.0)
    tf = time.time() - t0
    timing.append(dict(distance=f"vp_q10 numpy ({args.jobs} processes), subset", seconds=tn, samples=len(sub)))
    timing.append(dict(distance="vp_q10 numba (first call includes compiling), subset", seconds=tf, samples=len(sub)))
    print(f"  subset of {len(sub)}: numpy {tn:.1f} s, numba {tf:.1f} s, max difference {np.abs(Dn - Df).max():.2e}",
          flush=True)

    vp = {q: get(f"vp_q{q:g}", lambda q=q: vp_distance_matrix_fast(trains, q)) for q in QS}
    vr = {s: get(f"vr_s{int(s * 1000)}", lambda s=s: van_rossum(trains, s, cfg["length"])) for s in SIGMAS}

    rng = np.random.default_rng(0)
    splits = []
    for _ in range(args.splits):
        p = rng.permutation(k)
        splits.append((p[:k // 2], p[k // 2:]))

    rows = []

    def add(method, D, setting, extra=None):
        r = score(D, y, ia, ib, k, args.seed)
        rows.append(dict(dataset=args.dataset, method=method, setting=setting, **r, **(extra or {})))
        f = lambda key: f"{100 * r[key]:5.1f}"
        print(f"  {method:<28} {setting:<10} | A {f('nn_A')} {f('kmedoids_A')} | B {f('nn_B')} {f('kmedoids_B')}"
              f" | A->B {f('nn_Ato')} {f('kmedoids_Ato')} | B->A {f('nn_Bto')} {f('kmedoids_Bto')}", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS / f"spike_distance_{args.dataset}{tag}.csv", index=False)

    for q, D in vp.items():
        add("spike distance", D, f"q={q:g}")
    for s, D in vr.items():
        add("van Rossum", D, f"s={int(s * 1000)}ms")

    # honest q choice
    ho = held_out(vp, y, ia, ib, splits)
    ho_vr = held_out(vr, y, ia, ib, splits)
    hrow = dict(dataset=args.dataset, method="spike distance", setting="held-out q")
    hrow_vr = dict(dataset=args.dataset, method="van Rossum", setting="held-out sigma")
    for cell, (v, picks, orc) in ho.items():
        hrow[cell] = v
        hrow[cell + "_oracle"] = orc
        hrow[cell + "_picks"] = str({f"{q:g}": c for q, c in picks.items()})
        print(f"  held-out q  {cell}: {100 * v:.1f} (best on the same half {100 * orc:.1f})  picks {hrow[cell + '_picks']}", flush=True)
    for cell, (v, picks, orc) in ho_vr.items():
        hrow_vr[cell] = v
        hrow_vr[cell + "_oracle"] = orc
        hrow_vr[cell + "_picks"] = str({f"{s * 1000:g}": c for s, c in picks.items()})
        print(f"  held-out sigma {cell}: {100 * v:.1f} (best on the same half {100 * orc:.1f})  picks {hrow_vr[cell + '_picks']}", flush=True)
    rows += [hrow, hrow_vr]

    # best q per cell (1-NN on all neurons), combined and shuffled at that q
    best = {cell: max(QS, key=lambda q: [r for r in rows if r.get("setting") == f"q={q:g}"][0][cell])
            for cell in CELLS}
    print(f"  best q per cell: {best}", flush=True)
    for q in sorted(set(best.values())):
        cells = ",".join(c for c, v in best.items() if v == q)
        add("spike distance + 0.5 ISI fractions", scaled(vp[q]) + 0.5 * scaled(Disi), f"q={q:g}",
            dict(best_for=cells))
        Ds = get(f"vp_q{q:g}_shuffled", lambda q=q: vp_distance_matrix_fast(shuffled, q))
        add("spike distance, shuffled", Ds, f"q={q:g}", dict(best_for=cells))
    pd.DataFrame(rows).to_csv(RESULTS / f"spike_distance_{args.dataset}{tag}.csv", index=False)
    if timing:
        pd.DataFrame(timing).to_csv(RESULTS / f"spike_distance_{args.dataset}{tag}_timing.csv", index=False)
        print(pd.DataFrame(timing).to_string(index=False), flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
