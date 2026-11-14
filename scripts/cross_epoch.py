"""Does a neuron's spontaneous activity find its grating activity? A cross-stimulus test.

Both epochs are cut into windows of the same length, so spike counts and bar counts are
comparable. Each spontaneous window looks for its nearest grating window, and it scores
when that window comes from the same neuron (1-NN across epochs). The reverse direction and
the within-epoch scores are printed too. Only features that ignore when spikes happen are
used, because the two epochs share no stimulus timeline.

Run from the repo root, after load_openscope.py:
    python scripts/cross_epoch.py data/openscope/glo637908.parquet --neurons 60
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_dataset import load_dataset
from run_pipeline import isi_bars_distance, isi_histogram, scaled


def windows(df, window):
    """Cut every sample into consecutive windows of `window` seconds."""
    rows = []
    for r in df.itertuples(index=False):
        for k in range(int(r.duration // window + 1e-9)):
            t = r.spike_times[(r.spike_times >= k * window) & (r.spike_times < (k + 1) * window)]
            rows.append((r.neuron_id, r.session, t - k * window))
    return pd.DataFrame(rows, columns=["neuron_id", "session", "spike_times"])


def rate_distance(trains):
    n = np.array([len(t) for t in trains], dtype=float)
    return np.abs(np.log1p(n)[:, None] - np.log1p(n)[None, :])


def score(D, y, query, ref):
    """Share of query rows whose nearest reference row has the same neuron."""
    S = D[np.ix_(query, ref)].astype(float).copy()
    same = query[:, None] == ref[None, :]
    S[same] = np.inf                                 # a row never matches itself
    return float(np.mean(y[ref][S.argmin(1)] == y[query]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data")
    ap.add_argument("--window", type=float, default=10.0, help="seconds per window in both epochs")
    ap.add_argument("--per-neuron", type=int, default=10, help="windows per neuron and epoch")
    ap.add_argument("--min-isis", type=int, default=10)
    ap.add_argument("--neurons", type=int, help="random subset of neurons")
    ap.add_argument("--no-stablebear", action="store_true", help="skip the ISI bars (needs stablebear)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_dataset(args.data)
    grat = sorted(s for s in df["session"].unique() if s.endswith("_grating"))
    spont = sorted(s for s in df["session"].unique() if s.endswith("_spont"))
    if not grat or not spont:
        sys.exit("need a *_grating and a *_spont session in the file")
    w = windows(df[df["session"].isin(grat + spont)], args.window)
    w["epoch"] = np.where(w["session"].isin(grat), "grating", "spont")
    w = w[w["spike_times"].map(len) > args.min_isis]

    rng = np.random.default_rng(args.seed)
    ok = w.groupby(["neuron_id", "epoch"]).size().unstack(fill_value=0)
    names = ok.index[(ok >= max(2, args.per_neuron)).all(axis=1)].to_numpy()
    if args.neurons:
        names = rng.choice(names, min(args.neurons, len(names)), replace=False)
    parts = [g.iloc[rng.choice(len(g), args.per_neuron, replace=False)]
             for (n, _), g in w[w["neuron_id"].isin(names)].groupby(["neuron_id", "epoch"])]
    w = pd.concat(parts, ignore_index=True)
    y = np.unique(w["neuron_id"], return_inverse=True)[1]
    trains = [np.asarray(t, dtype=np.float64) for t in w["spike_times"]]
    g_idx = np.flatnonzero(w["epoch"] == "grating")
    s_idx = np.flatnonzero(w["epoch"] == "spont")

    D = {"rate": rate_distance(trains), "ISI histogram": isi_histogram(trains)}
    if not args.no_stablebear:
        D["log ISI bars"] = isi_bars_distance(trains, log=True)
        D["log ISI bars + histogram"] = scaled(D["log ISI bars"]) + 0.5 * scaled(D["ISI histogram"])
    rows = [dict(feature=k,
                 spont_to_grating=score(M, y, s_idx, g_idx),
                 grating_to_spont=score(M, y, g_idx, s_idx),
                 within_grating=score(M, y, g_idx, g_idx),
                 within_spont=score(M, y, s_idx, s_idx)) for k, M in D.items()]
    print(f"{len(names)} neurons, {args.per_neuron} windows of {args.window:g} s per neuron and epoch. "
          f"Chance about {1 / len(names):.3f}")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
