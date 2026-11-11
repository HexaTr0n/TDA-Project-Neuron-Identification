"""Does a signature survive a change of stimulus?

Run from the repo root:
    python scripts/cross_segment.py --neurons 60

Each trial is cut in half. The two halves show different parts of the movie, so they are
different stimuli with the same statistics. A half-trial is then matched to its nearest
half-trial among

    same  the same half of the other trials (same stimulus, as in run_pipeline.py)
    cross the other half of the other trials (different stimulus)

and scored by whether that one comes from the same neuron. The other half of the same
trial is never a candidate. Methods locked to movie time must fall to chance on "cross";
they are included as a check.
"""
import argparse
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pipeline as rp
from barcodes import DURATION, shuffle_isis
from load_dataset import DEFAULT_PATH, load_dataset

Trial = namedtuple("Trial", "isis spike_times")
MIN_ISIS = 20   # per half
RIPS = dict(n_points=15, n_subsamples=20)


def halves(t):
    t = np.asarray(t, dtype=np.float64)
    return t[t < DURATION / 2], t[t >= DURATION / 2] - DURATION / 2


def nearest_is_same_neuron(D, y, exclude_diagonal=True):
    D = np.array(D, dtype=np.float64)
    if exclude_diagonal:
        np.fill_diagonal(D, np.inf)
    return np.mean(y[D.argmin(1)] == y)


def score(D, y):
    """D covers first halves then second halves, 2n x 2n."""
    n = len(y)
    aa, bb, ab = D[:n, :n], D[n:, n:], D[:n, n:]
    same = (nearest_is_same_neuron(aa, y) + nearest_is_same_neuron(bb, y)) / 2
    cross = (nearest_is_same_neuron(ab, y) + nearest_is_same_neuron(ab.T, y)) / 2
    return dict(same=same, cross=cross)


def rips_distances(trains, shuffle, tag, seed):
    cfg = rp.Config(params=dict(m=3, shuffle=shuffle), seed=seed, **RIPS)
    blocks = [(i, [Trial(np.diff(t), t)]) for i, t in enumerate(trains)]
    rng = np.random.default_rng(seed)
    clouds = [rp.make_clouds(trials, cfg, rng) for _, trials in blocks]
    return rp.distance_matrices(rp.compute_barcodes(clouds, None, cfg, tag, True), cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=DEFAULT_PATH)
    ap.add_argument("--session", default=rp.Config().sessions[0])
    ap.add_argument("--neurons", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-rips", action="store_true", help="skip the delay embedding")
    args = ap.parse_args()

    df = load_dataset(args.data, sessions=[args.session])
    blocks = rp.make_blocks(df, rp.Config(min_isis=2 * MIN_ISIS))
    if args.neurons:
        blocks = rp.pick_neurons(blocks, args.neurons, args.seed)
    pairs = [(n, halves(trials[0].spike_times)) for n, trials in blocks]
    pairs = [(n, h) for n, h in pairs if min(len(h[0]), len(h[1])) > MIN_ISIS]
    counts = pd.Series([n for n, _ in pairs]).value_counts()
    pairs = [(n, h) for n, h in pairs if counts[n] >= 2]
    y = np.unique([n for n, _ in pairs], return_inverse=True)[1]
    trains = [h[0] for _, h in pairs] + [h[1] for _, h in pairs]
    print(f"{len(y)} trials of {y.max() + 1} neurons, each cut in two. Chance about {1 / (y.max() + 1):.3f}")

    rng = np.random.default_rng(args.seed)
    shuffled = [shuffle_isis(t, rng) for t in trains]
    D = {
        "spike count": rp.spike_count(trains),
        "ISI histogram": rp.isi_histogram(trains),
        "ISI bars, raw": rp.isi_bars_distance(trains, False),
        "ISI bars, log": rp.isi_bars_distance(trains, True),
        "time bars s=67 ms (check)": rp.time_bars_distance(trains, 2 / 30),
        "burst bars s=67 ms (check)": rp.burst_bars_distance(trains, 2 / 30),
    }
    if not args.no_rips:
        tag = f"cross_{args.session}_{args.neurons}_{args.seed}"
        real = rips_distances(trains, False, tag, args.seed)
        fake = rips_distances(trains, True, tag, args.seed)
        for k in real:
            D[f"delay m=3 {k}"] = real[k]
            D[f"delay m=3 {k}, shuffled"] = fake[k]
        base = rp.scaled(D["ISI bars, log"])
        D["ISI bars, log + delay H0+H1"] = base + 0.5 * rp.scaled(real["H0+H1"])
        D["ISI bars, log + delay H0+H1, shuffled"] = base + 0.5 * rp.scaled(fake["H0+H1"])
    D["ISI bars, log + spike count"] = rp.scaled(D["ISI bars, log"]) + 0.5 * rp.scaled(D["spike count"])

    table = pd.DataFrame({k: score(v, y) for k, v in D.items()}).T.round(3)
    print(table.to_string())
    out = rp.RESULTS / f"cross_segment_{rp.data_tag(args.session, args.neurons, args.seed)}.csv"
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out)


if __name__ == "__main__":
    main()
