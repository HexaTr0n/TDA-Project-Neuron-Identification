"""Task L14 (part 1): ISI fractions in rescaled time, the population clock of exp21.

On each trial (grating block, movie repeat or spontaneous window, shared by all units), a unit's
spike at time t moves to (number of other units' spikes before t) / (all other units' spikes)
* duration. Equal steps then mean equal amounts of population activity, so speed-ups that the
whole population shares cancel. Then ISI fractions in stablebear (stable rank of log ISIs /
ISI count, as sb_cross.isi_fraction_pcfs) on the rescaled times. The population is every unit
in the file (all quality-good units), minus the unit itself.

Neurons: the 60 of curve_eval.select (>= 30 ISIs, seed 0). 1-NN and k-medoids (3 restarts).
Control: the unit's ISIs permuted (in real time) before rescaling. With --cross-session, also
1-NN from every sample of one session to the samples of the other (stablebear cdist).

Run from the repo root:
    python scripts/sb_rescaled.py --data data/openscope/glo637908.parquet --session glo637908_grating
    python scripts/sb_rescaled.py --data data/allen/allen829720705.parquet --session allen829720705_movie \
        --cross-session allen829720705_spont
Writes results/rescaled_<session>.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from barcodes import shuffle_isis
from curve_eval import kmedoids, matched, nn_score, read, select
from sb_cross import isi_fraction_pcfs

RESULTS = Path(__file__).resolve().parent.parent / "results"


def population(df):
    """{trial: sorted spike times of every unit}."""
    return {tr: np.sort(np.concatenate([np.asarray(t, float) for t in g["spike_times"]]))
            for tr, g in df.groupby("trial")}


def rescale(t, own, pop, dur):
    """Others' spikes before each time t, as a fraction of all others' spikes, times dur."""
    n_other = len(pop) - len(own)
    before = np.searchsorted(pop, t, side="left") - np.searchsorted(own, t, side="left")
    return np.sort(before / max(n_other, 1) * dur)


def signatures(rows, pops, seed):
    """{name: list of PCFs} for real and rescaled time, real and shuffled ISIs."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("ISI fractions", "ISI fractions, rescaled time")}
    sh = {k: [] for k in out}
    for r in rows.itertuples(index=False):
        own = np.sort(np.asarray(r.spike_times, float))
        s = shuffle_isis(own, rng) if len(own) > 1 else own
        for name, store, t in (("ISI fractions", out, own), ("ISI fractions", sh, s)):
            store[name].append(t)
        out["ISI fractions, rescaled time"].append(rescale(own, own, pops[r.trial], r.duration))
        sh["ISI fractions, rescaled time"].append(rescale(s, own, pops[r.trial], r.duration))
    return ({k: isi_fraction_pcfs(v) for k, v in out.items()},
            {k: isi_fraction_pcfs(v) for k, v in sh.items()})


def score(D, y, k, seed):
    rng = np.random.default_rng(seed)
    nn = nn_score(D.copy(), y, rng)
    Dm = D.copy()
    np.fill_diagonal(Dm, 0)
    return nn, matched(y, kmedoids(Dm, k, rng), k)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--cross-session")
    ap.add_argument("--neurons", type=int, default=60)
    ap.add_argument("--min-isis", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import stablebear as sb

    full = read(args.data, [args.session])
    rows = select(full, args.neurons, args.min_isis, args.seed)
    y = np.unique(rows["neuron_id"], return_inverse=True)[1]
    k = y.max() + 1
    pops = population(full)
    print(f"{args.session}: {k} neurons, {len(rows)} samples, median {rows['isis'].map(len).median() + 1:.0f} "
          f"spikes, {rows['duration'].median():.0f} s; population median "
          f"{np.median([len(p) for p in pops.values()]):.0f} spikes per trial", flush=True)
    real, shuf = signatures(rows, pops, args.seed)
    out = []
    for shuffled, sigs in ((False, real), (True, shuf)):
        for name, pcfs in sigs.items():
            D = np.asarray(sb.pdist(sb.PcfTensor(pcfs), p=1).to_dense(), float)
            nn, km = score(D, y, k, args.seed)
            out.append(dict(session=args.session, test="within", signature=name, shuffled=shuffled, nn=nn, kmedoids=km))
            print(f"  {name:<30} {'shuffled' if shuffled else 'real':<8} 1-NN {100 * nn:5.1f}  k-medoids {100 * km:5.1f}", flush=True)

    if args.cross_session:
        full_b = read(args.data, [args.cross_session])
        names = sorted(set(rows["neuron_id"]))
        rows_b = full_b[(full_b["isis"].map(len) >= args.min_isis) & full_b["neuron_id"].isin(names)]
        rows_b = rows_b.reset_index(drop=True)
        real_b, shuf_b = signatures(rows_b, population(full_b), args.seed)
        lab = {n: i for i, n in enumerate(np.unique(rows["neuron_id"]))}
        yb = rows_b["neuron_id"].map(lab).to_numpy()
        print(f"{args.cross_session}: {len(set(rows_b['neuron_id']))} of the neurons, {len(rows_b)} samples", flush=True)
        for shuffled, A, B in ((False, real, real_b), (True, shuf, shuf_b)):
            for name in A:
                X = np.asarray(sb.cdist(sb.PcfTensor(A[name]), sb.PcfTensor(B[name]), p=1), float)
                present = np.isin(y, yb)  # A samples whose neuron also has B samples
                a2b = float(np.mean((yb[X.argmin(1)] == y)[present])) if len(yb) else np.nan
                b2a = float(np.mean(y[X.argmin(0)] == yb)) if len(yb) else np.nan
                for test, v in ((f"{args.session} -> {args.cross_session}", a2b),
                                (f"{args.cross_session} -> {args.session}", b2a)):
                    out.append(dict(session=args.session, test=test, signature=name, shuffled=shuffled, nn=v))
                print(f"  cross {name:<30} {'shuffled' if shuffled else 'real':<8} A->B {100 * a2b:5.1f}  "
                      f"B->A {100 * b2a:5.1f}", flush=True)
    pd.DataFrame(out).to_csv(RESULTS / f"rescaled_{args.session}.csv", index=False)
    print("\n" + pd.DataFrame(out).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
