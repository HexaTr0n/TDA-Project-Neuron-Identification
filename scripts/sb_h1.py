"""Task L23: H0+H1, make it work or close it. OpenScope 124 s only (the one dataset where H1 on the
whole cloud beat its shuffled control in L21).

Whole delay cloud (m = 3, every point), plain and in 16 regions (anchors on outside neurons). Rips
H0 and H1 per region in stablebear; bars kept as arrays so H1 can be filtered:
  (a) each degree's distance divided by its median over pairs of outside-neuron samples, H1
      weighted 0.25, 0.5, 1 and added to H0;
  (b) only H1 bars longer than the 95th percentile of H1 bar lengths in the same sample's
      ISI-shuffled cloud (same size by construction), as stable rank and as Betti curve;
  (c) (a) and (b) on samples with >= 200 points only.
Control: ISIs permuted within every sample (the filter threshold for the shuffled run comes from
a second, independent shuffle). Scores: 1-NN and k-medoids within gratings (no k-means).

Run from the repo root:
    python scripts/sb_h1.py            # (an optional dataset argument is only for testing)
Writes results/h1_openscope124.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from barcodes import shuffle_isis
from sb_feature_cloud import evaluate_one
from sb_full_table import delay_cloud, part_pdist, split_regions, to_list
from sb_headline import load

REPO = Path(__file__).resolve().parent.parent
DATASET = sys.argv[1] if len(sys.argv) > 1 else "openscope124"   # another dataset only for testing
OUT = REPO / "results" / f"h1_{DATASET.replace(':', '_')}.csv"


class A_:
    seed, drop_piece = 0, None


def bars_of(clouds, R):
    """[(H0 list over regions, H1 list over regions)] per sample; each a (n, 2) array."""
    import stablebear as sb
    from stablebear import persistence
    out = []
    for s0 in range(0, len(clouds), 300):
        part = clouds[s0:s0 + 300]
        T = sb.zeros((len(part), R), dtype=sb.pcloud64)
        for a, regs in enumerate(part):
            for r, c in enumerate(regs):
                T[a, r] = c
        B = persistence.compute_persistent_homology(T, max_dim=1, reduced=True, verbose=False)
        for a in range(len(part)):
            out.append(([np.asarray(B[a, r, 0].to_numpy()) for r in range(R)],
                        [np.asarray(B[a, r, 1].to_numpy()) for r in range(R)]))
        del T, B
    return out


def pcfs(barlists, kind):
    """barlists[i][r] -> per region a list of PCFs (stable rank or Betti curve)."""
    from stablebear import persistence
    from stablebear.persistence import Barcode, BarcodeTensor
    n, R = len(barlists), len(barlists[0])
    f = persistence.barcode_to_stable_rank if kind == "sr" else persistence.barcode_to_betti_curve
    out = []
    for r in range(R):
        bs = [b[r] if len(b[r]) else np.zeros((1, 2)) for b in barlists]
        out.append(to_list(f(BarcodeTensor([Barcode(np.asarray(x, np.float64)) for x in bs])), n))
    return out


def filt(h1, thr):
    return [[b[(b[:, 1] - b[:, 0]) > t] if len(b) else b for b in regs] for regs, t in zip(h1, thr)]


def threshold(h1_shuf):
    out = []
    for regs in h1_shuf:
        L = np.concatenate([b[:, 1] - b[:, 0] for b in regs if len(b)]) if any(len(b) for b in regs) else np.zeros(1)
        out.append(np.percentile(L, 95) if len(L) else np.inf)
    return out


def med(parts):
    D = sum(part_pdist(("pcf", p)) for p in parts)
    v = D[np.triu_indices(len(D), 1)]
    v = v[v > 0]
    return float(np.median(v)) if len(v) else 1.0


def main():
    A, _, pool, _ = load(DATASET, A_)
    names = sorted(set(A["neuron_id"]))
    y = A["neuron_id"].map({n: i for i, n in enumerate(names)}).to_numpy()
    trains = [np.sort(np.asarray(t, float)) for t in A["spike_times"]]
    rng = np.random.default_rng(0)
    pool = pool[pool["spike_times"].map(len) > 5]
    pool = pool.iloc[rng.choice(len(pool), min(300, len(pool)), replace=False)]
    ptrains = [np.sort(np.asarray(t, float)) for t in pool["spike_times"]]
    pts = np.concatenate([delay_cloud(t) for t in ptrains])
    pts = pts[rng.choice(len(pts), min(8000, len(pts)), replace=False)]
    anchors = KMeans(16, n_init=3, random_state=0).fit(pts).cluster_centers_
    s1, s2 = np.random.default_rng(1), np.random.default_rng(2)
    sets = {"real": trains, "ISIs shuffled": [shuffle_isis(t, s1) for t in trains],
            "shuffle for threshold": [shuffle_isis(t, s2) for t in trains], "outside": ptrains}
    npts = np.array([len(t) - 3 for t in trains])
    big = np.flatnonzero(npts >= 200)
    print(f"{DATASET}: {len(names)} neurons, {len(trains)} samples, points median {np.median(npts):.0f}; "
          f">= 200 points: {len(big)} samples", flush=True)
    rows = []

    for fam, anc in (("plain", None), ("regions k=16", anchors)):
        R = 1 if anc is None else 16
        B = {}
        for name, tr in sets.items():
            t0 = time.time()
            clouds = [[delay_cloud(t)] if anc is None else split_regions(delay_cloud(t), anc) for t in tr]
            B[name] = bars_of(clouds, R)
            print(f"  {fam} {name}: bars in {time.time() - t0:.0f} s", flush=True)
        thr_real = threshold([b[1] for b in B["ISIs shuffled"]])          # real samples: their own shuffle
        thr_shuf = threshold([b[1] for b in B["shuffle for threshold"]])  # shuffled samples: an independent one
        thr_out = threshold([b[1] for b in bars_of(
            [[delay_cloud(shuffle_isis(t, s2))] if anc is None else split_regions(delay_cloud(shuffle_isis(t, s2)), anc)
             for t in ptrains], R)])
        meds = {}
        P_out = {"H0": pcfs([b[0] for b in B["outside"]], "sr"), "H1 sr": pcfs([b[1] for b in B["outside"]], "sr"),
                 "H1 bc": pcfs([b[1] for b in B["outside"]], "bc"),
                 "H1f sr": pcfs(filt([b[1] for b in B["outside"]], thr_out), "sr"),
                 "H1f bc": pcfs(filt([b[1] for b in B["outside"]], thr_out), "bc")}
        for k, v in P_out.items():
            meds[k] = med(v)
        print(f"  {fam} outside medians: " + ", ".join(f"{k} {v:.3g}" for k, v in meds.items()), flush=True)
        for cond, thr in (("real", thr_real), ("ISIs shuffled", thr_shuf)):
            bb = B[cond]
            P = {"H0": pcfs([b[0] for b in bb], "sr"), "H1 sr": pcfs([b[1] for b in bb], "sr"),
                 "H1 bc": pcfs([b[1] for b in bb], "bc"),
                 "H1f sr": pcfs(filt([b[1] for b in bb], thr), "sr"),
                 "H1f bc": pcfs(filt([b[1] for b in bb], thr), "bc")}
            kept = np.mean([sum(len(x) for x in f) / max(1, sum(len(x) for x in o))
                            for f, o in zip(filt([b[1] for b in bb], thr), [b[1] for b in bb])])
            readouts = {"H0": [("H0", 1.0)]}
            for h in ("H1 sr", "H1 bc", "H1f sr", "H1f bc"):
                readouts[h] = [(h, 1.0)]
                for w in (0.25, 0.5, 1.0):
                    readouts[f"H0 + {w:g} x {h}"] = [("H0", 1.0), (h, w)]
            for subset, ix in (("all samples", np.arange(len(trains))), (">= 200 points", big)):
                if len(ix) < 20:
                    continue
                cmed = {h: med([[P[h][r][i] for i in ix] for r in range(R)]) for h in P}
                yy = y[ix]
                keep_n = sorted(set(yy))
                ymap = {v: i for i, v in enumerate(keep_n)}
                y2 = np.array([ymap[v] for v in yy])
                for name, terms in readouts.items():
                    groups = []
                    for h, w in terms:
                        parts = [("pcf", [P[h][r][i] for i in ix]) for r in range(R)]
                        # combine() divides by the comparison median; rescale so the net divisor is the outside median
                        groups.append((parts, w * cmed[h] / meds[h]))
                    t1 = time.time()
                    r = evaluate_one(groups, False, y2, len(keep_n), 0, False)
                    rows.append(dict(family=fam, readout=name, control=cond, subset=subset, samples=len(ix),
                                     neurons=len(keep_n), h1_kept=kept if "H1f" in name else np.nan, **r))
                    print(f"  {fam:<13} {subset:<13} {name:<22} {cond:<13} | 1-NN {100 * r['nn_A']:5.1f} "
                          f"k-medoids {100 * r['kmedoids_A']:5.1f}  ({time.time() - t1:.0f} s)", flush=True)
                    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
