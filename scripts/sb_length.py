"""Task L6: how neuron identity from ISI fractions grows with sample length. In stablebear.

Long stretches of one condition are cut into samples of L seconds, for each L in --lengths:
    retina     the unrepeated noise before each frozen trial: the 4 pieces of 10 s of a block are
               joined into one 40 s stretch (load_retina_running.py)
    OpenScope  each 124 s grating block is one stretch
Every neuron then gets the same number of samples at every length (--per-neuron, drawn at
random from all its pieces), so the scores compare lengths fairly: 1-NN and k-means with mean
curves (sb.mean / sb.cdist), ISI fractions as in sb_cross.py. Pieces with fewer than 2 ISIs give
an empty barcode and stay in, as they would in practice.

Run from the repo root:
    python scripts/sb_length.py --data data/retina_running_20180126_sr_le_sp.parquet \\
        --select-session 20180126_sr_le_sp --session 20180126_sr_le_sp_running --join-pieces
    python scripts/sb_length.py --data data/openscope/glo637908.parquet \\
        --select-session glo637908_grating --session glo637908_grating
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curve_eval import matched, read, select
from sb_check import sb_kmeans
from sb_cross import isi_fraction_pcfs


def stretches(df, join_pieces):
    """(neuron_id, spike_times, length) of contiguous stretches."""
    if not join_pieces:
        return [(r.neuron_id, r.spike_times, r.duration) for r in df.itertuples(index=False)]
    out = []
    for (nid, _), g in df.sort_values("piece").groupby(["neuron_id", "block"]):
        if list(g["piece"]) != list(range(len(g))):
            continue
        t = np.concatenate([r.spike_times + j * r.duration for j, r in enumerate(g.itertuples(index=False))])
        out.append((nid, t, float(g["duration"].sum())))
    return out


def cut(stretch_list, L):
    out = []
    for nid, t, length in stretch_list:
        for j in range(int(length // L + 1e-9)):
            a = j * L
            out.append((nid, t[(t >= a) & (t < a + L)] - a))
    return out


def main():
    import stablebear as sb

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--select-session", required=True, help="neurons chosen here, as run_pipeline.py")
    ap.add_argument("--session", required=True, help="the stretches come from here")
    ap.add_argument("--join-pieces", action="store_true", help="join consecutive pieces of a block")
    ap.add_argument("--lengths", type=float, nargs="+", default=[2.5, 5, 10, 20, 40])
    ap.add_argument("--per-neuron", type=int, default=20)
    ap.add_argument("--neurons", type=int, default=60)
    ap.add_argument("--min-isis", type=int, default=30)
    ap.add_argument("--kmeans-seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = np.unique(select(read(args.data, [args.select_session]), args.neurons, args.min_isis,
                             args.seed)["neuron_id"])
    df = read(args.data, [args.session])
    df = df[df["neuron_id"].isin(names)].reset_index(drop=True)
    st = stretches(df, args.join_pieces)
    lens = pd.Series([s[2] for s in st])
    print(f"{args.session}: {len(names)} neurons, {len(st)} stretches of {lens.median():.1f} s "
          f"(min {lens.min():.1f}); {args.per_neuron} samples per neuron and length", flush=True)

    rows = []
    for L in args.lengths:
        rng = np.random.default_rng(args.seed)
        pieces = pd.DataFrame(cut(st, L), columns=["neuron_id", "spike_times"])
        counts = pieces["neuron_id"].value_counts()
        keep = counts.index[counts >= args.per_neuron]
        if len(keep) < len(names):
            print(f"  {L:g} s: {len(names) - len(keep)} neurons have fewer than {args.per_neuron} pieces, dropped")
        pieces = pd.concat([g.iloc[rng.choice(len(g), args.per_neuron, replace=False)]
                            for _, g in pieces[pieces["neuron_id"].isin(keep)].groupby("neuron_id")],
                           ignore_index=True)
        y = np.unique(pieces["neuron_id"], return_inverse=True)[1]
        k = y.max() + 1
        trains = list(pieces["spike_times"])
        n = np.array([len(t) for t in trains])
        pcfs = isi_fraction_pcfs(trains)
        D = np.asarray(sb.pdist(sb.PcfTensor(pcfs), p=1).to_dense(), dtype=float)
        D += rng.random(D.shape) * 1e-12
        np.fill_diagonal(D, np.inf)
        R = np.abs(np.log1p(n)[:, None] - np.log1p(n)[None, :]) + rng.random((len(n), len(n))) * 1e-9
        np.fill_diagonal(R, np.inf)
        km = [matched(y, sb_kmeans(pcfs, k, np.random.default_rng(s)), k)[0] for s in range(args.kmeans_seeds)]
        rows.append(dict(length_s=L, neurons=k, samples=len(y), median_spikes=float(np.median(n)),
                         share_lt_2_isis=float(np.mean(n < 3)),
                         nn=float(np.mean(y[D.argmin(1)] == y)), kmeans=float(np.mean(km)),
                         kmeans_sd=float(np.std(km)), rate_nn=float(np.mean(y[R.argmin(1)] == y)),
                         chance=1 / k))
        print("  " + "  ".join(f"{a}={b:.3f}" if isinstance(b, float) else f"{a}={b}"
                               for a, b in rows[-1].items()), flush=True)
    print()
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
