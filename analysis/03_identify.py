"""Step 3 - the main experiment: can one trial tell which neuron it came from?

For each method we compute a distance between every pair of trials, then two scores:

    1-NN          for each trial, is the closest OTHER trial from the same neuron?
    nearest mean  for each trial, is the closest neuron mean the right neuron?

Both are leave-one-out cross-validation: each trial in turn is the test trial and all the
other trials are the training set (the test trial is not in any neuron mean).

Chance level is 1 / n_neurons (2.5 % with 40 neurons).

Methods:
    B1  firing rate            |rate_i - rate_j|                               (no topology)
    B2  ISI histogram          L1 between log10-ISI histograms                 (no topology)
    T1  LINE H0                L1 between H0 stable ranks of the spike times on a line
    T2  DELAY H0 / H1 / H0+H1  L1 between stable ranks of the delay cloud
    C   DELAY H0+H1, shuffled  T2 after shuffling the ISIs inside every trial (control)

Everything is repeated for `--seeds` different random draws of neurons; the table gives
mean +/- std over the draws. Then the same is done with longer samples (2 and 3 trials
glued together) to see how the scores grow with the amount of data.

Outputs:
    results/identification.csv / .md      scores per method (10 s samples)
    results/sample_length.csv              scores against sample length
    figures/analysis/id_scores.png         bar chart of the scores
    figures/analysis/id_sample_length.png  scores against sample length
    figures/analysis/id_distance_matrix.png  distance matrices sorted by neuron (seed 0)
    figures/analysis/id_mds.png            2-D picture (MDS) of the trials of 8 neurons

Usage:  python analysis/03_identify.py [--n-neurons 40] [--seeds 5] [--min-rate 1]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tda_utils import (RESULTS_DIR, classical_mds, cloud_delay, cloud_line, combine,
                       load_trials, pcf_distances, savefig, score_1nn, score_nearest_mean,
                       shuffle_isis, stable_ranks, vector_distances)

TRIAL_S = 9.94  # duration of one frozen-noise trial
METHOD_ORDER = ["B1 firing rate", "B2 ISI histogram", "T1 LINE H0", "T2 DELAY H0",
                "T2 DELAY H1", "T2 DELAY H0+H1", "C DELAY H0+H1 shuffled"]


def all_distances(data, block, seed):
    """Return {method name: (D, M)} for every method, for one set of samples."""
    isis, labels = data["isis"], data["labels"]
    out = {}

    # B1: firing rate = number of spikes / duration. One number per sample.
    rate = np.array([len(x) + 1 for x in isis]) / (block * TRIAL_S)
    out["B1 firing rate"] = vector_distances(rate[:, None], labels)

    # B2: histogram of log10 ISI, 40 bins from 1 ms to 10 s. Raw counts (not normalised),
    # like the stable rank, which also counts bars.
    bins = np.linspace(-3, 1, 41)
    hist = np.array([np.histogram(np.log10(x), bins)[0] for x in isis])
    out["B2 ISI histogram"] = vector_distances(hist, labels)

    # T1: H0 of the spike times on a line (bars = ISIs).
    (line,) = stable_ranks([cloud_line(x) for x in isis], 0)
    out["T1 LINE H0"] = pcf_distances(line, labels)

    # T2: H0 and H1 of the delay cloud, alone and combined.
    h0, h1 = stable_ranks([cloud_delay(x) for x in isis], 1)
    out["T2 DELAY H0"] = pcf_distances(h0, labels)
    out["T2 DELAY H1"] = pcf_distances(h1, labels)
    out["T2 DELAY H0+H1"] = combine([out["T2 DELAY H0"], out["T2 DELAY H1"]])

    # C: same as T2 H0+H1, after shuffling the order of the ISIs inside each sample.
    s0, s1 = stable_ranks([cloud_delay(x) for x in shuffle_isis(isis, seed)], 1)
    out["C DELAY H0+H1 shuffled"] = combine([pcf_distances(s0, labels), pcf_distances(s1, labels)])

    for name, (D, M) in out.items():
        assert np.allclose(D, D.T) and np.allclose(np.diag(D), 0), f"{name}: bad distance matrix"
    return out


def run(args, block, seed):
    data = load_trials(args.n_neurons, args.min_rate, args.n_trials, seed=seed, block=block)
    dists = all_distances(data, block, seed)
    rows = [dict(method=name, seed=seed, block=block, seconds=block * 10,
                 nn1=score_1nn(D, data["labels"]), nearest_mean=score_nearest_mean(M, data["labels"]))
            for name, (D, M) in dists.items()]
    return rows, data, dists


def figure_matrices(data, dists, k=10):
    """Distance matrices of the trials of k neurons, sorted by neuron (white lines separate
    the neurons). Light squares on the diagonal = trials of the same neuron are close."""
    labels = data["labels"]
    show = np.linspace(0, labels.max(), k).astype(int)
    keep = np.flatnonzero(np.isin(labels, show))
    edges = np.cumsum([np.sum(labels == c) for c in show])[:-1] - 0.5
    names = ["B2 ISI histogram", "T1 LINE H0", "T2 DELAY H0+H1", "C DELAY H0+H1 shuffled"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for ax, name in zip(axes.flat, names):
        D = dists[name][0][np.ix_(keep, keep)]
        im = ax.imshow(D, cmap="viridis_r", vmin=0, vmax=np.percentile(D, 95), interpolation="nearest")
        for e in edges:
            ax.axhline(e, color="w", lw=0.6)
            ax.axvline(e, color="w", lw=0.6)
        ax.set_title(name)
        ax.set_xlabel(f"trial ({k} neurons x {len(keep) // k} trials)")
        ax.set_ylabel("trial (sorted by neuron)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    savefig(fig, "id_distance_matrix.png")


def figure_mds(data, dists, k=8):
    """MDS: place trials in 2-D so that their distances are approximately kept."""
    labels = data["labels"]
    show = np.linspace(0, labels.max(), k).astype(int)
    keep = np.isin(labels, show)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name in zip(axes, ["T1 LINE H0", "T2 DELAY H0+H1"]):
        D = dists[name][0][np.ix_(keep, keep)]
        xy = classical_mds(D)
        for j, c in enumerate(show):
            m = labels[keep] == c
            ax.scatter(xy[m, 0], xy[m, 1], s=14, color=plt.cm.tab10(j), label=data["neurons"][c])
        ax.set_title(f"MDS of {name} distances\n({k} neurons, 1 dot = 1 trial)")
        ax.set_xticks([]), ax.set_yticks([])
    axes[0].legend(fontsize=9)
    savefig(fig, "id_mds.png")


def figure_scores(table, chance):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(table))
    w = 0.38
    for off, col, lab in [(-w / 2, "nn1", "1-NN"), (w / 2, "nearest_mean", "nearest mean")]:
        ax.bar(x + off, 100 * table[f"{col}_mean"], w, yerr=100 * table[f"{col}_std"],
               capsize=3, label=lab)
    ax.axhline(100 * chance, color="k", ls="--", lw=1, label=f"chance ({100 * chance:.1f} %)")
    ax.set_xticks(x)
    ax.set_xticklabels(table["method"], rotation=20, ha="right")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Which neuron is this 10 s trial from?  (mean ± std over neuron draws)")
    ax.legend()
    savefig(fig, "id_scores.png")


def figure_length(length):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in ["B1 firing rate", "B2 ISI histogram", "T1 LINE H0", "T2 DELAY H0+H1",
                 "C DELAY H0+H1 shuffled"]:
        g = length[length.method == name].groupby("seconds")["nearest_mean"]
        ax.errorbar(g.mean().index, 100 * g.mean(), yerr=100 * g.std(), marker="o", capsize=3, label=name)
    ax.set_xlabel("sample length (s)")
    ax.set_ylabel("nearest-mean accuracy (%)")
    ax.set_title("More data per sample -> easier to identify the neuron")
    ax.legend(fontsize=10)
    savefig(fig, "id_sample_length.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-neurons", type=int, default=40)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--min-rate", type=float, default=1.0)
    p.add_argument("--seeds", type=int, default=5, help="number of random neuron draws")
    a = p.parse_args()
    chance = 1 / a.n_neurons
    RESULTS_DIR.mkdir(exist_ok=True)

    all_rows = []
    for seed in range(a.seeds):
        for block in (1, 2, 3):
            print(f"seed {seed}, samples of {block * 10} s ...", flush=True)
            rows, data, dists = run(a, block, seed)
            all_rows += rows
            if seed == 0 and block == 1:
                # Sanity check: with random labels the 1-NN score must drop to chance.
                rng = np.random.default_rng(0)
                fake = rng.permutation(data["labels"])
                print(f"  check: 1-NN with shuffled neuron labels = "
                      f"{100 * score_1nn(dists['T2 DELAY H0+H1'][0], fake):.1f} % (chance {100 * chance:.1f} %)")
                figure_matrices(data, dists)
                figure_mds(data, dists)
    df = pd.DataFrame(all_rows)

    # --- Main table: 10 s samples --------------------------------------------------------
    one = df[df.block == 1]
    table = (one.groupby("method")[["nn1", "nearest_mean"]].agg(["mean", "std"]))
    table.columns = [f"{a_}_{b}" for a_, b in table.columns]
    table = table.reindex(METHOD_ORDER).reset_index()
    table.to_csv(RESULTS_DIR / "identification.csv", index=False)
    with open(RESULTS_DIR / "identification.md", "w") as f:
        f.write(f"Accuracy (%) for 10 s samples, {a.n_neurons} neurons x {a.n_trials} trials, "
                f"mean ± std over {a.seeds} neuron draws. Chance = {100 * chance:.1f} %.\n\n")
        f.write("| method | 1-NN | nearest mean |\n|---|---|---|\n")
        for _, r in table.iterrows():
            f.write(f"| {r.method} | {100 * r.nn1_mean:.1f} ± {100 * r.nn1_std:.1f} "
                    f"| {100 * r.nearest_mean_mean:.1f} ± {100 * r.nearest_mean_std:.1f} |\n")
    print("\n" + open(RESULTS_DIR / "identification.md").read())
    figure_scores(table, chance)

    # --- Sample length ----------------------------------------------------------------------
    df.to_csv(RESULTS_DIR / "sample_length.csv", index=False)
    summary = df.groupby(["method", "seconds"])["nearest_mean"].mean().unstack().reindex(METHOD_ORDER)
    print("Nearest-mean accuracy (%) against sample length (s):")
    print((100 * summary).round(1).to_string())
    figure_length(df)


if __name__ == "__main__":
    main()
