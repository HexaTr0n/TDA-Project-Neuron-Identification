"""Victor-Purpura spike train distance as a baseline.

Cost of an edit script: inserting or deleting a spike costs 1, shifting one by dt
seconds costs q*|dt|. Shifting is only worth it when |dt| < 2/q, so 1/q is the
timescale. q=0 gives |n_a - n_b|, the spike count baseline. A very large q counts
every spike that does not coincide exactly.

This compares whole trials and needs no embedding or homology, so stablebear is not
involved. It uses absolute spike times, so unlike the ISI clouds it can see responses
locked to the repeated stimulus.

Run from the repo root:
    python scripts/spike_distance.py --q 1 10 100 1000 --jobs 4

Each q appends a row to results/results.csv and saves its matrix to results/vp_q*.npy.
Uses the same trials as the default Config in run_pipeline.py, so the numbers are
directly comparable. Cost grows with (total spikes)^2.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_dataset import DEFAULT_PATH, load_dataset
from run_pipeline import RESULTS, Config, evaluate, make_blocks

CHUNK = 256  # trains compared at once. Similar lengths, so little padding is wasted.


def vp_one_to_many(x, Y, lengths, q):
    """Distance from train x to each row of the zero-padded matrix Y.

    This is the standard dynamic program, D[i, j] = min(D[i-1, j] + 1, D[i, j-1] + 1,
    D[i-1, j-1] + q*|x[i] - y[j]|), with the D[i, j-1] term solved in one step per row:
    D[i, j] = j + cummin(A - j), where A is the minimum of the other two terms.
    Padding sits to the right of each train and never affects entries before it.
    """
    m, l = Y.shape
    j = np.arange(l + 1, dtype=np.float64)
    D = np.tile(j, (m, 1))
    A = np.empty_like(D)
    for i, t in enumerate(x, start=1):
        A[:, 0] = i
        A[:, 1:] = np.minimum(D[:, 1:] + 1, D[:, :-1] + q * np.abs(t - Y))
        D = np.minimum.accumulate(A - j, axis=1) + j
    return D[np.arange(m), lengths]


def _pad(trains):
    Y = np.zeros((len(trains), max(len(t) for t in trains)))
    for i, t in enumerate(trains):
        Y[i, :len(t)] = t
    return Y


_trains, _lengths, _q = None, None, None


def _init(trains, q):
    global _trains, _lengths, _q
    _trains, _lengths, _q = trains, np.array([len(t) for t in trains]), q


def _row(i):
    """Distances from train i to every longer train (trains are sorted by length)."""
    out = np.empty(len(_trains) - i - 1)
    if _lengths[i] == 0:  # only deletions or only insertions
        return i, _lengths[i + 1:].astype(float)
    for s in range(i + 1, len(_trains), CHUNK):
        e = min(s + CHUNK, len(_trains))
        out[s - i - 1:e - i - 1] = vp_one_to_many(
            _trains[i], _pad(_trains[s:e]), _lengths[s:e], _q)
    return i, out


def vp_distance_matrix(trains, q, jobs=1):
    """Pairwise Victor-Purpura distances. trains: list of sorted spike time arrays."""
    n = len(trains)
    order = np.argsort([len(t) for t in trains], kind="stable")
    ordered = [np.asarray(trains[i], dtype=np.float64) for i in order]
    D = np.zeros((n, n))
    with ProcessPoolExecutor(jobs, initializer=_init, initargs=(ordered, q)) as pool:
        for i, row in pool.map(_row, range(n - 1), chunksize=8):
            D[order[i], order[i + 1:]] = row
            D[order[i + 1:], order[i]] = row
    return D


def pick_neurons(blocks, n, seed):
    """Keep all trials of n random neurons."""
    names = sorted({neuron for neuron, _ in blocks})
    keep = set(np.random.default_rng(seed).choice(names, n, replace=False).tolist())
    return [b for b in blocks if b[0] in keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--q", type=float, nargs="+", default=[1, 10, 100, 1000],
                    help="shift costs in 1/s")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--data", default=DEFAULT_PATH, help=".parquet or .csv dataset")
    ap.add_argument("--session", help="override the session of the default Config")
    ap.add_argument("--neurons", type=int, help="use this many random neurons (faster)")
    args = ap.parse_args()

    cfg = Config(sessions=(args.session,)) if args.session else Config()
    df = load_dataset(args.data, sessions=list(cfg.sessions))
    blocks = make_blocks(df, cfg)
    if args.neurons:
        blocks = pick_neurons(blocks, args.neurons, cfg.seed)
    labels = np.array([neuron for neuron, _ in blocks])
    trains = [trials[0].spike_times for _, trials in blocks]
    print(f"{len(trains)} trials, {len(set(labels))} neurons, "
          f"{sum(len(t) for t in trains)} spikes")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "results.csv"
    for q in args.q:
        start = time.time()
        D = vp_distance_matrix(trains, q, args.jobs)
        np.save(RESULTS / f"vp_q{q:g}.npy", D)
        row = pd.DataFrame([dict(config="victor_purpura", distance=f"q={q:g}",
                                 **evaluate(D, labels, cfg.seed),
                                 settings=json.dumps(asdict(cfg), sort_keys=True))])
        row.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(row.drop(columns="settings").round(3).to_string(index=False),
              f"({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
