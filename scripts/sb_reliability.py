"""Task L9: response reliability and time-locked identity on OpenScope, using single grating
presentations joined by condition. Distances in stablebear.

The oddball blocks repeat groups of five: A A A B (or A A A A) and a blank, 25 times. A
condition is (orientation, position in the group). Each block becomes one trial built only from
identical conditions in a fixed order: the 1 s windows (0.5 s grating + 0.5 s grey) of the first
--n-a presentations of A at positions 0, 1 and 2 and the first --n-b presentations of B at
position 3, laid end to end. Window m of every trial therefore shows the same stimulus, as the
frozen movie does on the retina. Blocks without enough presentations of each condition are
dropped (the blocks where 45 degrees is A).

Per unit: reliability = mean pairwise correlation of the trials' PSTHs (50 ms bins), as in
build_isi_dataset.py. Then, for 60 random units and for the 60 most reliable units: 1-NN and
k-means (sb.mean) with square-root time bars (stablebear Betti curves, sqrt values), the
ISI-shuffled control, and ISI fractions; and across all units with >= 30 ISIs per trial, how
each unit's own 1-NN hit rate follows its reliability (Spearman).

Run from the repo root:
    python scripts/sb_reliability.py data/openscope/sub-637908_ses-1213341633_ogen.nwb
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curves as C
from barcodes import shuffle_isis
from curve_eval import matched
from load_openscope import GRATING_TABLE, blocked_spans, overlaps, read_nwb
from sb_check import sb_kmeans, sqrt_betti_pcfs
from sb_cross import isi_fraction_pcfs

A_ORI, B_ORI = 135.0, 45.0


def condition_windows(table, avoid, n_a, n_b):
    """{block: onsets in canonical order}, for blocks with enough presentations of each condition."""
    g = table.sort_values("start_time").copy()
    g["pos"] = g.groupby("stimulus_block").cumcount() % 5
    g["ori"] = pd.to_numeric(g["orientation"], errors="coerce")
    out = {}
    for b, gb in g.groupby("stimulus_block"):
        if overlaps(gb["start_time"].min(), gb["stop_time"].max() + 0.5, avoid):
            continue
        parts = [gb.loc[(gb["ori"] == A_ORI) & (gb["pos"] == j), "start_time"].to_numpy()[:n_a] for j in (0, 1, 2)]
        parts.append(gb.loc[(gb["ori"] == B_ORI) & (gb["pos"] == 3), "start_time"].to_numpy()[:n_b])
        if all(len(p) == n for p, n in zip(parts, (n_a, n_a, n_a, n_b))):
            out[int(b)] = np.concatenate(parts)
    return out


def join(spikes, onsets, window=1.0):
    pieces = []
    for m, o in enumerate(onsets):
        s = spikes[np.searchsorted(spikes, o):np.searchsorted(spikes, o + window)]
        pieces.append(s - o + m * window)
    return np.concatenate(pieces) if pieces else np.zeros(0)


def reliability(trains, dur, bin_s=0.05):
    bins = np.arange(0, dur + bin_s, bin_s)
    c = np.array([np.histogram(t, bins)[0] for t in trains], float)
    c = c[c.std(axis=1) > 0]
    if len(c) < 2:
        return np.nan
    r = np.corrcoef(c)
    return float(r[np.triu_indices(len(c), 1)].mean())


def evaluate(df, label, seeds, with_isi=True):
    import stablebear as sb

    y = np.unique(df["neuron_id"], return_inverse=True)[1]
    k = y.max() + 1
    trains = list(df["spike_times"])
    rows = []
    rng = np.random.default_rng(0)
    variants = {"sqrt time bars": trains,
                "sqrt time bars, ISIs shuffled": [shuffle_isis(t, rng) if len(t) > 1 else t for t in trains]}
    per_unit = None
    for name, tr in list(variants.items()) + ([("ISI fractions", trains)] if with_isi else []):
        pcfs = isi_fraction_pcfs(tr) if name == "ISI fractions" else sqrt_betti_pcfs(tr, C.S)
        D = np.asarray(sb.pdist(sb.PcfTensor(pcfs), p=1).to_dense(), dtype=float)
        np.fill_diagonal(D, np.inf)
        hit = y[D.argmin(1)] == y
        km = [matched(y, sb_kmeans(pcfs, k, np.random.default_rng(s)), k)[0] for s in range(seeds)]
        rows.append(dict(units=label, signature=name, nn=hit.mean(), kmeans=np.mean(km) if km else np.nan))
        if name == "sqrt time bars":
            per_unit = pd.Series(hit).groupby(df["neuron_id"].to_numpy()).mean()
    return rows, per_unit


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("nwb")
    ap.add_argument("--n-a", type=int, default=20)
    ap.add_argument("--n-b", type=int, default=15)
    ap.add_argument("--neurons", type=int, default=60)
    ap.add_argument("--min-isis", type=int, default=30)
    ap.add_argument("--kmeans-seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-unit-units", type=int, default=400,
                    help="random units for the per-unit reliability analysis (RAM: dense distances)")
    args = ap.parse_args()

    d = read_nwb(args.nwb)
    avoid = blocked_spans(d["intervals"], d["opto"])
    onsets = condition_windows(d["intervals"][GRATING_TABLE], avoid, args.n_a, args.n_b)
    dur = float(3 * args.n_a + args.n_b)
    units = d["units"]
    units = units[units["quality"].astype(str) == "good"]
    print(f"{len(onsets)} blocks with {args.n_a} x 3 A windows and {args.n_b} B windows: "
          f"trials of {dur:.0f} s; {len(units)} good units", flush=True)

    rows = []
    for u, spikes in units["spike_times"].items():
        spikes = np.sort(np.asarray(spikes, float))
        for trial, (b, ons) in enumerate(sorted(onsets.items())):
            t = join(spikes, ons)
            rows.append(dict(neuron_id=f"u{u}", trial=trial, block=b, spike_times=t, n=len(t)))
    df = pd.DataFrame(rows)
    ok = df[df["n"] >= args.min_isis + 1]
    counts = ok["neuron_id"].value_counts()
    ok = ok[ok["neuron_id"].isin(counts.index[counts >= 10])]
    rel = ok.groupby("neuron_id")["spike_times"].apply(lambda s: reliability(list(s), dur))
    print(f"units with >= 10 trials of >= {args.min_isis} ISIs: {len(rel)}; reliability median "
          f"{rel.median():.3f}, quartiles {rel.quantile(.25):.3f} {rel.quantile(.75):.3f}, "
          f"max {rel.max():.3f}; spikes per trial median {ok['n'].median():.0f}", flush=True)

    rng = np.random.default_rng(args.seed)
    names = sorted(rel.index)
    random_units = rng.choice(names, min(args.neurons, len(names)), replace=False)
    top_units = rel.sort_values(ascending=False).index[:args.neurons]
    results = []
    for label, sel in (("60 random", random_units), ("60 most reliable", top_units)):
        sub = ok[ok["neuron_id"].isin(sel)].reset_index(drop=True)
        r, _ = evaluate(sub, label, args.kmeans_seeds)
        for x in r:
            x["reliability_median"] = float(rel.loc[list(sel)].median())
        results += r
        print(pd.DataFrame(r).round(3).to_string(index=False), flush=True)

    # Per unit: does its own 1-NN hit rate follow its reliability? A random subset of units,
    # because the dense distance matrix over all of them would not fit in RAM. No k-means.
    many = rng.choice(names, min(args.per_unit_units, len(names)), replace=False)
    _, per_unit = evaluate(ok[ok["neuron_id"].isin(many)].reset_index(drop=True), "per unit", 0, with_isi=False)
    rho, p = spearmanr(rel.loc[per_unit.index], per_unit)
    q = pd.qcut(rel.loc[per_unit.index], 5, labels=False)
    print(f"\n{len(per_unit)} random units: per-unit 1-NN (sqrt time bars) against reliability, "
          f"Spearman {rho:.3f} (p = {p:.1e})")
    print("by reliability quintile (mean reliability, mean 1-NN):")
    print(pd.DataFrame({"reliability": rel.loc[per_unit.index], "nn": per_unit}).groupby(q.values).mean().round(3).to_string())
    print("\n" + pd.DataFrame(results).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
