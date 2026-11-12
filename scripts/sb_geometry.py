"""Task L16: match neuron geometries across stimuli.

Under each condition, the neuron-by-neuron distance matrix D = 1 - correlation of the
Gaussian-smoothed trains (--bin 10 ms bins, sigma 20 ms), over all samples of the condition laid
end to end (1 s gaps). Then hide which neuron is which in condition B and pair the neurons of A
with those of B:
  - sorted profile: each neuron's sorted row of distances (who dropped), L1 between profiles,
    Hungarian matching;
  - refined: graph-matching iterations from that start. With the current pairing pi, the cost of
    pairing i with j is sum_k |D_A[i, k] - D_B[j, pi(k)]| / n; re-solve, --iterations times;
  - rate only: pair by |log rate_A - log rate_B| (Hungarian), the baseline for "the profile is just
    the rate";
  - sorted profile + rate: the two costs median-scaled and added.
Score: share of neurons paired with themselves; a random pairing gives 1/n (and the 95% of 200
random pairings is printed). Reference pairs with the same stimulus: first half against second
half of A, and of B (in the Allen session the movie halves are two blocks 70 min apart),
and even against odd samples. Time-matched: A's first half against the B samples recorded within
--near s of it.

Units: quality-good units with >= --min-rate Hz in both conditions.

Run from the repo root:
    python scripts/sb_geometry.py --data data/allen/allen829720705.parquet \
        --a allen829720705_movie --b allen829720705_spont
Writes results/geometry_<a>_<b>.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curve_eval import read
from sb_partners import corr_rows

RESULTS = Path(__file__).resolve().parent.parent / "results"


def trains_on_one_clock(df, units, trials, gap=1.0):
    dur = float(df["duration"].median())
    off = {tr: i * (dur + gap) for i, tr in enumerate(trials)}
    g = df[df["neuron_id"].isin(units) & df["trial"].isin(trials)]
    out = {u: [] for u in units}
    for r in g.itertuples(index=False):
        out[r.neuron_id].append(np.asarray(r.spike_times, float) + off[r.trial])
    return [np.sort(np.concatenate(out[u])) if out[u] else np.zeros(0) for u in units], len(trials) * (dur + gap)


def distance_matrix(trains, total, bin_s, sigma):
    edges = np.arange(0, total + bin_s / 2, bin_s)
    X = np.empty((len(trains), len(edges) - 1), np.float32)
    for i, t in enumerate(trains):
        X[i] = gaussian_filter1d(np.histogram(t, edges)[0].astype(np.float32), sigma / bin_s)
    D = 1 - corr_rows(X, X).astype(np.float64)
    np.fill_diagonal(D, 0)
    return (D + D.T) / 2, np.array([len(t) for t in trains]) / total


def hungarian(cost):
    r, c = linear_sum_assignment(cost)
    pi = np.empty(len(cost), int)
    pi[r] = c
    return pi


def l1_cost(P, Q):
    return np.stack([np.abs(p[None, :] - Q).sum(1) for p in P]) / P.shape[1]  # row by row: RAM


def refine(DA, DB, pi, iterations):
    n = len(DA)
    for _ in range(iterations):
        DBp = DB[:, pi]  # column k of B reordered as A's neuron k
        cost = np.empty((n, n))
        for i in range(n):
            cost[i] = np.abs(DA[i][None, :] - DBp).sum(1) / n
        new = hungarian(cost)
        if np.array_equal(new, pi):
            break
        pi = new
    return pi


def pairings(DA, DB, rA, rB, iterations):
    n = len(DA)
    PA = np.sort(DA, axis=1)[:, 1:]  # drop the zero self-distance
    PB = np.sort(DB, axis=1)[:, 1:]
    c_prof = l1_cost(PA, PB)
    lA, lB = np.log(np.maximum(rA, 1e-3)), np.log(np.maximum(rB, 1e-3))  # silent in one half
    c_rate = np.abs(lA[:, None] - lB[None, :])
    scaled = lambda C: C / np.median(C)
    out = {}
    pi = hungarian(c_prof)
    out["sorted profile"] = pi
    out["sorted profile, refined"] = refine(DA, DB, pi, iterations)
    out["rate only"] = hungarian(c_rate)
    both = hungarian(scaled(c_prof) + scaled(c_rate))
    out["sorted profile + rate"] = both
    out["sorted profile + rate, refined"] = refine(DA, DB, both, iterations)
    return {k: float(np.mean(v == np.arange(n))) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--min-rate", type=float, default=1.0)
    ap.add_argument("--max-units", type=int, default=0, help="0 keeps all")
    ap.add_argument("--bin", type=float, default=0.01)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--near", type=float, default=1500.0, help="s around A's first half, time-matched test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    dfa, dfb = read(args.data, [args.a]), read(args.data, [args.b])

    def rates(df):
        tot = float(df["duration"].median()) * df["trial"].nunique()
        return df.groupby("neuron_id")["spike_times"].apply(lambda s: sum(len(x) for x in s)) / tot

    ra, rb = rates(dfa), rates(dfb)
    units = sorted(set(ra.index[ra >= args.min_rate]) & set(rb.index[rb >= args.min_rate]))
    if args.max_units and len(units) > args.max_units:
        units = sorted(rng.choice(units, args.max_units, replace=False).tolist())
    n = len(units)
    null = np.array([np.mean(rng.permutation(n) == np.arange(n)) for _ in range(200)])
    print(f"{n} units >= {args.min_rate} Hz in both; random pairing {100 / n:.2f}% "
          f"(95% of 200: {100 * np.quantile(null, 0.95):.2f}%)", flush=True)

    ta, tb = sorted(dfa["trial"].unique()), sorted(dfb["trial"].unique())
    # Time-matched: A's first half against the B samples recorded within --near s of it, so
    # that slow drift of the recording does not count as a stimulus effect.
    sa = dfa.drop_duplicates("trial").set_index("trial")["start"]
    sbs = dfb.drop_duplicates("trial").set_index("trial")["start"]
    a1 = ta[:len(ta) // 2]
    lo, hi = sa.loc[a1].min() - args.near, sa.loc[a1].max() + float(dfa["duration"].median()) + args.near
    b_near = [t for t in tb if lo <= sbs.loc[t] <= hi]
    mats = {}
    for name, df, trials in ((args.a, dfa, ta), (args.b, dfb, tb),
                             (f"{args.a} 1st half", dfa, ta[:len(ta) // 2]),
                             (f"{args.a} 2nd half", dfa, ta[len(ta) // 2:]),
                             (f"{args.b} 1st half", dfb, tb[:len(tb) // 2]),
                             (f"{args.b} 2nd half", dfb, tb[len(tb) // 2:]),
                             (f"{args.a} even", dfa, ta[0::2]), (f"{args.a} odd", dfa, ta[1::2]),
                             (f"{args.b} even", dfb, tb[0::2]), (f"{args.b} odd", dfb, tb[1::2]),
                             (f"{args.b} near {args.a} 1st half", dfb, b_near)):
        tr, total = trains_on_one_clock(df, units, trials)
        mats[name] = distance_matrix(tr, total, args.bin, args.sigma)
        print(f"  {name}: {len(trials)} samples, {total:.0f} s, median distance "
              f"{np.median(mats[name][0][np.triu_indices(n, 1)]):.3f}", flush=True)

    rows = []
    for test, (x, y) in {f"{args.a} -> {args.b} (stimulus change)": (args.a, args.b),
                         f"{args.a} halves (same stimulus)": (f"{args.a} 1st half", f"{args.a} 2nd half"),
                         f"{args.b} halves (same stimulus)": (f"{args.b} 1st half", f"{args.b} 2nd half"),
                         f"{args.a} even vs odd samples": (f"{args.a} even", f"{args.a} odd"),
                         f"{args.b} even vs odd samples": (f"{args.b} even", f"{args.b} odd"),
                         f"{args.a} 1st half -> {args.b} within {args.near:.0f} s (stimulus change, time-matched)":
                             (f"{args.a} 1st half", f"{args.b} near {args.a} 1st half")}.items():
        (DA, rA), (DB, rB) = mats[x], mats[y]
        corr = np.corrcoef(DA[np.triu_indices(n, 1)], DB[np.triu_indices(n, 1)])[0, 1]
        res = pairings(DA, DB, rA, rB, args.iterations)
        for k, v in res.items():
            rows.append(dict(test=test, method=k, self_paired=v, units=n, random=1 / n,
                             random_p95=float(np.quantile(null, 0.95)), matrix_corr=corr))
        rate_corr = float(np.corrcoef(np.log(np.maximum(rA, 1e-3)), np.log(np.maximum(rB, 1e-3)))[0, 1])
        for r in rows[-len(res):]:
            r["log_rate_corr"] = rate_corr
        print(f"{test}: correlation of the two distance matrices {corr:.3f}, of log rates {rate_corr:.3f}", flush=True)
        for k, v in res.items():
            print(f"    {k:<32} {100 * v:5.1f}% paired with themselves", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / f"geometry_{args.a}_{args.b}.csv", index=False)


if __name__ == "__main__":
    main()
