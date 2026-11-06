"""Compare the saved distance matrices from run_pipeline.py, with settings chosen fairly.

Run from the repo root, after run_pipeline.py, with the same --session and --neurons:
    python scripts/compare.py --neurons 60

Each family is one method with several settings. "1-NN" is the best setting on all neurons,
so it flatters families with many settings. "Held-out" fixes that: settings are chosen on
half the neurons and scored on the other half, over many random splits. The same splits
are used for every row. Chance on a half is one over the number of neurons in it.
"""
import argparse
import fnmatch
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline import RESULTS, Config, data_tag, scaled

FAMILIES = {
    "Delay embedding, H0": ["m?_fixed__H0"],
    "Delay embedding, H0+H1": ["m?_fixed__H0+H1"],
    "Delay regions": ["regions_m3_k?__H*"],
    "Time bars": ["time_bars_s???__D", "time_bars_s??__D"],
    "Window count": ["window_count_s*__D"],
    "ISI bars": ["isi_bars_*__D"],
    "Peak matching": ["peak_s*_q*__D"],
    "van Rossum": ["van_rossum_*__D"],
    "Spike distance": ["spike_distance_q*__D"],
    "ISI histogram": ["isi_histogram__D"],
    "Spike count": ["spike_count__D"],
}
PARTNERS = {"+ histogram": "isi_histogram__D", "+ log ISI bars": "isi_bars_log__D"}


def hits(D, y, idx):
    S = np.array(D[np.ix_(idx, idx)], dtype=np.float64)
    np.fill_diagonal(S, np.inf)
    return np.mean(y[idx][S.argmin(1)] == y[idx])


def held_out(settings, y, splits):
    """settings: {name: matrix}. Mean score on B of the setting that does best on A."""
    scores = []
    for a, b in splits:
        best = max(settings, key=lambda k: hits(settings[k], y, a))
        scores.append(hits(settings[best], y, b))
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", default=Config().sessions[0])
    ap.add_argument("--neurons", type=int)
    ap.add_argument("--weight", type=float, default=0.5, help="weight of the added partner")
    ap.add_argument("--splits", type=int, default=50)
    args = ap.parse_args()

    folder = RESULTS / "distances" / data_tag(args.session, args.neurons)
    if not (folder / "labels.npy").exists():
        sys.exit(f"nothing saved in {folder}; run run_pipeline.py first")
    y = np.unique(np.load(folder / "labels.npy"), return_inverse=True)[1]
    files = {p.stem: p for p in folder.glob("*__*.npy")}
    matrices = {k: np.load(p) for k, p in files.items()
                if np.load(p, mmap_mode="r").shape[0] == len(y)}

    rng = np.random.default_rng(0)
    k = y.max() + 1
    splits = []
    for _ in range(args.splits):
        p = rng.permutation(k)
        splits.append((np.flatnonzero(np.isin(y, p[:k // 2])),
                       np.flatnonzero(np.isin(y, p[k // 2:]))))

    everyone = np.arange(len(y))
    rows = []
    for family, patterns in FAMILIES.items():
        settings = {n: D for n, D in matrices.items()
                    if any(fnmatch.fnmatch(n, p) for p in patterns) and "shuffled" not in n}
        if not settings:
            continue
        one_nn = {n: hits(D, y, everyone) for n, D in settings.items()}
        best = max(one_nn, key=one_nn.get)
        row = dict(family=family, settings=len(settings), best=best.replace("__D", ""),
                   one_nn=one_nn[best], held_out=held_out(settings, y, splits))
        for label, partner in PARTNERS.items():
            if partner in matrices and partner not in settings:
                extra = args.weight * scaled(matrices[partner])
                row[label] = held_out({n: scaled(D) + extra for n, D in settings.items()}, y, splits)
        rows.append(row)
    table = pd.DataFrame(rows).round(3)
    print(f"{len(y)} trials, {k} neurons. Chance: 1-NN about {1 / k:.3f}, held-out about {2 / k:.3f}")
    print(table.to_string(index=False, na_rep="-"))

    controls = {n: hits(D, y, everyone) for n, D in matrices.items() if "shuffled" in n}
    if controls:
        print("\nShuffled controls, 1-NN:")
        print(pd.Series(controls).round(3).to_string())
    table.to_csv(folder / "comparison.csv", index=False)


if __name__ == "__main__":
    main()
