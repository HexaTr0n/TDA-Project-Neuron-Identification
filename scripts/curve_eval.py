"""Score the curve signatures of curves.py: 1-NN, k-means with mean curves, k-medoids, and a
cross-stimulus test.

Run from the repo root:
    python scripts/curve_eval.py --session 20180126_sr_le_sp --neurons 60
    python scripts/curve_eval.py --session all --neurons 60       # every retina session, averaged
    python scripts/curve_eval.py --data data/openscope/glo637908.parquet \\
        --session glo637908_grating --neurons 60
    python scripts/curve_eval.py --data data/openscope/glo637908.parquet \\
        --session glo637908_grating --cross-session glo637908_spont --neurons 60

Scores (all are shares of trials, so chance is about 1 / neurons):
    nn        1-NN on whole trials, leave one out
    kmeans    k-means with mean curves: each trial goes to the nearest centre by L1, a centre
              is the mean curve of its members; k = neurons, 3 restarts, 25 iterations. Share
              of trials in the right cluster after pairing clusters with neurons (Hungarian)
    kmedoids  the same with medoids of the L1 matrix, for comparison
    x_nn      cross-stimulus 1-NN. Each trial is cut in two; a half-trial looks for its
              nearest half-trial among the OTHER half of OTHER trials. Both directions are
              averaged. With --cross-session the two halves are instead the two sessions
              (for example grating blocks and spontaneous windows of the same units).
    x_kmeans  cluster one half without labels, name the clusters by neuron (Hungarian),
              assign the other half's trials to the nearest centre. Both directions averaged.

Neurons are chosen as in run_pipeline.py (trials with >= --min-isis ISIs, then --neurons
random neurons with seed 0), so on the retina they are the same 60 neurons. With several
sessions the mean over sessions is printed as well, not the best one. Results are appended
to results/curve_results.csv.

A "_shuffled" method shuffles the ISIs within each trial before the curve is computed. The
control for coincidence is coincidence_next_trial: the other neurons come from the next
trial, which keeps only what the stimulus explains.
"""
import argparse
import gc
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curves as C
from barcodes import DURATION, shuffle_isis

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "curve_results.csv"
RETINA = ROOT / "data" / "isi_frozennoise.parquet"

# name: ([(signature, weight), ...], control). With more than one part, each part is
# divided by its median pairwise L1 distance (300 random trials), then weighted.
METHODS = {
    "time_bars": ([(C.time_bars, 1.0)], "shuffle"),
    "sqrt_time_bars": ([(C.sqrt_time_bars, 1.0)], "shuffle"),
    "multiscale_sqrt_time_bars": ([(partial(C.sqrt_time_bars, s=0.02), 1.0),
                                   (C.sqrt_time_bars, 1.0),
                                   (partial(C.sqrt_time_bars, s=0.2), 1.0)], "shuffle"),
    "isi_fractions": ([(C.isi_fractions, 1.0)], None),
    "sqrt_time_bars+isi_fractions": ([(C.sqrt_time_bars, 1.0), (C.isi_fractions, 0.5)], "shuffle"),
    "burst_bars": ([(C.burst_bars, 1.0)], "shuffle"),
    "current_isi": ([(C.current_isi, 1.0)], "shuffle"),
    "burst_size": ([(C.burst_size, 1.0)], "shuffle"),
    "coincidence": ([("coincidence", 1.0)], "next_trial"),
    "cond_isi_fractions": ([(C.cond_isi_fractions, 1.0)], "shuffle"),
    "burst_table": ([(C.burst_table, 1.0)], "shuffle"),
    "isi_joint": ([(C.isi_joint, 1.0)], "shuffle"),
    "isi_order": ([(C.isi_order, 1.0)], "shuffle"),
}


# Only these ignore when spikes happen, so only they can compare trials of different
# lengths or stimuli; --cross-session skips the rest.
STIMULUS_FREE = ("isi_fractions", "cond_isi_fractions", "burst_table", "isi_joint", "isi_order")


def all_names():
    out = []
    for name, (_, control) in METHODS.items():
        out.append(name)
        if control == "shuffle":
            out.append(name + "_shuffled")
        elif control == "next_trial":
            out.append(name + "_next_trial")
    return out


# ------------------------------------------------------------------ L1 kernels


@njit(parallel=True, fastmath=True, cache=True)
def l1_pdist(X):
    n, p = X.shape
    D = np.zeros((n, n))
    for i in prange(n):
        for j in range(i + 1, n):
            s = 0.0
            for k in range(p):
                s += abs(X[i, k] - X[j, k])
            D[i, j] = s
    for i in range(n):
        for j in range(i):
            D[i, j] = D[j, i]
    return D


@njit(parallel=True, fastmath=True, cache=True)
def l1_cross(X, Y):
    n, p = X.shape
    D = np.zeros((n, Y.shape[0]))
    for i in prange(n):
        for j in range(Y.shape[0]):
            s = 0.0
            for k in range(p):
                s += abs(X[i, k] - Y[j, k])
            D[i, j] = s
    return D


# ------------------------------------------------------------------ data


def read(path, sessions):
    df = pd.read_parquet(path, filters=[("session", "in", list(sessions))])
    for c in ("isis", "spike_times"):
        df[c] = df[c].map(lambda a: np.asarray(a, dtype=np.float64))
    if "duration" not in df:
        df["duration"] = DURATION
    return df.reset_index(drop=True)


def select(df, n, min_isis, seed):
    """Same neurons as run_pipeline.pick_neurons: usable trials, >= 2 of them, n at random."""
    ok = df[df["isis"].map(len) >= min_isis]
    counts = ok["neuron_id"].value_counts()
    names = sorted(counts.index[counts >= 2])
    if n and n < len(names):
        names = np.random.default_rng(seed).choice(names, n, replace=False).tolist()
    return ok[ok["neuron_id"].isin(names)].sort_values(["neuron_id", "trial"]).reset_index(drop=True)


def coincidence_weights(full, rows, next_trial=False):
    """For every spike of every row: other neurons' spikes within +-COINCIDENCE seconds,
    on the same trial, or on the next trial of the session (the control)."""
    trials = np.sort(full["trial"].unique())
    nxt = dict(zip(trials, np.roll(trials, -1)))
    own = {(r.neuron_id, r.trial): r.spike_times for r in full.itertuples(index=False)}
    pop = {k: np.sort(np.concatenate(g.tolist())).astype(np.float64)
           for k, g in full.groupby("trial")["spike_times"]}
    c = C.COINCIDENCE

    def count(ref, t):
        return np.searchsorted(ref, t + c, "right") - np.searchsorted(ref, t - c, "left")

    out = []
    for r in rows.itertuples(index=False):
        tr = nxt[r.trial] if next_trial else r.trial
        mine = own.get((r.neuron_id, tr), np.zeros(0))
        out.append((count(pop[tr], r.spike_times) - count(mine, r.spike_times)).astype(np.float64))
    return out


# ------------------------------------------------------------------ features


def median_l1(F, rng, n=300):
    idx = rng.choice(len(F), min(n, len(F)), replace=False)
    D = l1_pdist(np.ascontiguousarray(F[idx]))
    return float(np.median(D[np.triu_indices(len(idx), 1)])) or 1.0


def features(parts, trains, durs, weights, rng):
    """Curves of every part, side by side, as one float32 matrix (rows = trains)."""
    blocks = []
    for fn, wt in parts:
        if fn == "coincidence":
            F = np.stack([C.coincidence(t, w, d) for t, w, d in zip(trains, weights, durs)])
        else:
            F = np.stack([fn(t, d) for t, d in zip(trains, durs)])
        F = F.astype(np.float32)
        if len(parts) > 1:
            F *= np.float32(wt / median_l1(F, rng))
        blocks.append(F)
    return blocks[0] if len(blocks) == 1 else np.hstack(blocks)


def halves(trains, durs, weights):
    """Cut each trial at dur / 2: (first halves, second halves), shifted to start at 0."""
    A, B = ([], [], []), ([], [], [])
    for t, d, w in zip(trains, durs, weights):
        h = d / 2
        first = t < h
        A[0].append(t[first]); A[1].append(h); A[2].append(w[first] if w is not None else None)
        B[0].append(t[~first] - h); B[1].append(h); B[2].append(w[~first] if w is not None else None)
    return A, B


# ------------------------------------------------------------------ scores


def nn_score(D, y, rng):
    D = D + rng.random(D.shape) * 1e-9 * (np.median(D[D > 0]) if (D > 0).any() else 1)
    np.fill_diagonal(D, np.inf)
    return float(np.mean(y[D.argmin(1)] == y))


def matched(y, lab, k):
    """Share of trials in the right cluster after pairing clusters with neurons; the pairing."""
    M = np.zeros((k, y.max() + 1))
    np.add.at(M, (lab, y), 1)
    r, c = linear_sum_assignment(-M)
    name = np.full(k, -1)
    name[r] = c
    return M[r, c].sum() / len(y), name


def kmeans(X, k, rng, restarts=3, iters=25):
    """k-means with mean curves and L1 assignment. Returns (labels, centres)."""
    best = (np.inf, None, None)
    n = len(X)
    for _ in range(restarts):
        cent = X[rng.choice(n, k, replace=False)].copy()
        lab = None
        for _ in range(iters):
            Dc = l1_cross(X, cent)
            new = Dc.argmin(1)
            if lab is not None and np.array_equal(new, lab):
                break
            lab = new
            far = np.argsort(-Dc[np.arange(n), lab])
            f = 0
            for c in range(k):
                m = lab == c
                if m.any():
                    cent[c] = X[m].mean(0)
                else:                               # empty cluster: restart it far away
                    cent[c] = X[far[f]]
                    f += 1
        Dc = l1_cross(X, cent)
        lab = Dc.argmin(1)
        cost = Dc[np.arange(n), lab].sum()
        if cost < best[0]:
            best = (cost, lab, cent)
    return best[1], best[2]


def kmedoids(D, k, rng, restarts=3, iters=25):
    best = (np.inf, None)
    n = len(D)
    for _ in range(restarts):
        med = rng.choice(n, k, replace=False)
        for _ in range(iters):
            lab = D[:, med].argmin(1)
            new = med.copy()
            for c in range(k):
                m = np.flatnonzero(lab == c)
                if len(m):
                    new[c] = m[D[np.ix_(m, m)].sum(1).argmin()]
            if np.array_equal(new, med):
                break
            med = new
        lab = D[:, med].argmin(1)
        cost = D[np.arange(n), med[lab]].sum()
        if cost < best[0]:
            best = (cost, lab)
    return best[1]


def cross_nn(FA, FB, ya, yb, exclude_same_row):
    D = l1_cross(FA, FB)
    if exclude_same_row:
        np.fill_diagonal(D, np.inf)                 # never the other half of the same trial
    return float(np.mean(yb[D.argmin(1)] == ya))


def cross_kmeans(FA, FB, ya, yb, k, rng):
    lab, cent = kmeans(FA, k, rng)
    _, name = matched(ya, lab, k)
    pred = name[l1_cross(FB, cent).argmin(1)]
    return float(np.mean(pred == yb))


# ------------------------------------------------------------------ one method on one session


def run(name, rows, full, other, rng, args):
    base = name.removesuffix("_shuffled").removesuffix("_next_trial")
    parts, _ = METHODS[base]
    y = np.unique(rows["neuron_id"], return_inverse=True)[1]
    k = y.max() + 1
    trains = list(rows["spike_times"])
    if name.endswith("_shuffled"):
        srng = np.random.default_rng(args.seed)
        trains = [shuffle_isis(t, srng) if len(t) > 1 else t for t in trains]
    durs = rows["duration"].to_numpy(float)
    weights = [None] * len(trains)
    if base == "coincidence":
        weights = coincidence_weights(full, rows, next_trial=name.endswith("_next_trial"))

    out = dict(method=name, n_trials=len(rows), n_neurons=int(k))
    F = features(parts, trains, durs, weights, rng)
    out["points"] = F.shape[1]
    D = l1_pdist(F)
    out["nn"] = nn_score(D, y, rng)
    out["kmeans"] = matched(y, kmeans(F, k, rng)[0], k)[0]
    out["kmedoids"] = matched(y, kmedoids(D, k, rng), k)[0]
    del F, D
    gc.collect()

    if other is None:                               # halves of the same trials
        A, B = halves(trains, durs, weights)
        FA = features(parts, A[0], A[1], A[2], rng)
        FB = features(parts, B[0], B[1], B[2], rng)
        ya = yb = y
        same_row = True
    else:                                           # the same units in another session
        if base not in STIMULUS_FREE:
            return out                              # needs a shared timeline: skip
        o = other[other["neuron_id"].isin(rows["neuron_id"].unique())]
        names = np.unique(rows["neuron_id"])
        yb = np.searchsorted(names, o["neuron_id"].to_numpy())
        other_trains = list(o["spike_times"])
        if name.endswith("_shuffled"):
            srng = np.random.default_rng(args.seed + 1)
            other_trains = [shuffle_isis(t, srng) if len(t) > 1 else t for t in other_trains]
        FA = features(parts, trains, durs, weights, rng)
        FB = features(parts, other_trains, o["duration"].to_numpy(float), [None] * len(o), rng)
        ya = y
        same_row = False
    out["x_nn"] = (cross_nn(FA, FB, ya, yb, same_row) + cross_nn(FB, FA, yb, ya, same_row)) / 2
    out["x_kmeans"] = (cross_kmeans(FA, FB, ya, yb, k, rng) + cross_kmeans(FB, FA, yb, ya, k, rng)) / 2
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("methods", nargs="*", help=f"default: all of {all_names()}")
    ap.add_argument("--data", default=str(RETINA))
    ap.add_argument("--session", nargs="+", default=["20180126_sr_le_sp"], help="names, or 'all'")
    ap.add_argument("--cross-session", help="score the cross test against this session instead of halves")
    ap.add_argument("--neurons", type=int, default=60)
    ap.add_argument("--min-isis", type=int, default=30)
    ap.add_argument("--cross-min-isis", type=int, default=10, help="for --cross-session trials")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--area-prefix", nargs="+", help="keep units whose area starts with one of these")
    args = ap.parse_args()

    methods = args.methods or all_names()
    unknown = set(methods) - set(all_names())
    if unknown:
        sys.exit(f"unknown methods: {sorted(unknown)}")
    sessions = args.session
    if sessions == ["all"]:
        sessions = sorted(pd.read_parquet(args.data, columns=["session"])["session"].unique())
        sessions = [s for s in sessions if not s.endswith("_spont")]

    results = []
    for session in sessions:
        full = read(args.data, [session])
        if args.area_prefix:
            full = full[full["area"].astype(str).str.startswith(tuple(args.area_prefix))].reset_index(drop=True)
        rows = select(full, args.neurons, args.min_isis, args.seed)
        if not any(m.startswith("coincidence") for m in methods):
            full = None
        other = None
        if args.cross_session:
            other = read(args.data, [args.cross_session])
            other = other[other["isis"].map(len) >= args.cross_min_isis]
            other = other[other["neuron_id"].isin(rows["neuron_id"].unique())].reset_index(drop=True)
        gc.collect()
        print(f"\n{session}: {rows['neuron_id'].nunique()} neurons, {len(rows)} trials of "
              f"{rows['duration'].iloc[0]:.2f} s"
              + (f"; cross test against {args.cross_session} ({len(other)} trials)" if other is not None else
                 "; cross test on half-trials"), flush=True)
        for m in methods:
            t0 = time.time()
            r = run(m, rows, full, other, np.random.default_rng(args.seed), args)
            r.update(session=session, data=Path(args.data).name, seconds=round(time.time() - t0, 1),
                     cross=args.cross_session or "halves")
            results.append(r)
            print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in r.items() if k not in ("session", "data", "cross")),
                  flush=True)
            gc.collect()

    df = pd.DataFrame(results)
    cols = [c for c in ("nn", "kmeans", "kmedoids", "x_nn", "x_kmeans") if c in df]
    print("\nPer session:" if len(sessions) > 1 else "")
    print(df.pivot_table(index="method", columns="session", values="nn", sort=False).round(3).to_string()
          if len(sessions) > 1 else "")
    print(f"\nMean over {len(sessions)} session(s); chance about {1 / args.neurons:.3f}:")
    print(df.groupby("method", sort=False)[cols].mean().round(3).to_string())
    RESULTS.parent.mkdir(exist_ok=True)
    df.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
    print(f"\nappended to {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
