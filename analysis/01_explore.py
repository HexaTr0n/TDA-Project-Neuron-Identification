"""Step 1 - look at the data before doing any topology.

Figures (in figures/analysis/):
    explore_rasters.png       spikes of 3 neurons over all their trials
    explore_isi_hist.png      distribution of log10 ISI for the same 3 neurons
    explore_delay_clouds.png  the DELAY point cloud (log ISI_i, log ISI_i+1) of one trial
                              of each neuron, next to the cloud of all their trials

The 3 neurons are picked among the selected neurons so that they have clearly different
firing rates (slow, medium, fast), which makes the differences easy to see.

Usage:  python analysis/01_explore.py [--n-neurons 40] [--seed 0] [--min-rate 1]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tda_utils import cloud_delay, load_trials, savefig

COLORS = ["tab:blue", "tab:orange", "tab:green"]


def pick_examples(data, k=3):
    """Labels of k neurons with low, medium and high mean firing rate."""
    n_spikes = np.array([len(x) + 1 for x in data["isis"]])
    rate = np.array([n_spikes[data["labels"] == c].mean() for c in range(len(data["neurons"]))])
    order = np.argsort(rate)
    return [order[int(q * (len(order) - 1))] for q in np.linspace(0.1, 0.9, k)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-neurons", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-rate", type=float, default=1.0)
    a = p.parse_args()

    data = load_trials(a.n_neurons, a.min_rate, seed=a.seed)
    labels, isis, neurons = data["labels"], data["isis"], data["neurons"]
    ex = pick_examples(data)
    print("Example neurons:", [neurons[c] for c in ex])

    # --- Rasters: one row per trial, one tick per spike ---------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for ax, c, col in zip(axes, ex, COLORS):
        trials = [data["spikes"][i] for i in np.flatnonzero(labels == c)]
        ax.eventplot(trials, colors=col, linelengths=0.8, linewidths=0.7)
        ax.set_ylabel("trial")
        ax.set_title(neurons[c])
    axes[-1].set_xlabel("time in the trial (s)")
    fig.suptitle("Spike rasters of three neurons (same movie in every trial)")
    savefig(fig, "explore_rasters.png")

    # --- ISI histograms --------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-3, 1, 50)
    for c, col in zip(ex, COLORS):
        all_isis = np.concatenate([isis[i] for i in np.flatnonzero(labels == c)])
        ax.hist(np.log10(all_isis), bins=bins, histtype="step", lw=2, color=col,
                density=True, label=neurons[c])
    ax.set_xlabel("log10 ISI (s)    [-3 = 1 ms, -2 = 10 ms, -1 = 100 ms, 0 = 1 s]")
    ax.set_ylabel("density")
    ax.legend(fontsize=10)
    ax.set_title("ISI distributions (all trials pooled)")
    savefig(fig, "explore_isi_hist.png")

    # --- Delay clouds ----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True, sharey=True)
    for j, (c, col) in enumerate(zip(ex, COLORS)):
        idx = np.flatnonzero(labels == c)
        # Show the trial with the median number of spikes, a typical trial for this neuron.
        typical = idx[np.argsort([len(isis[i]) for i in idx])[len(idx) // 2]]
        one = cloud_delay(isis[typical])
        # Trials with < 2 ISIs give a 1-point placeholder cloud at (0, 0): leave them out here.
        everything = np.vstack([cloud_delay(isis[i]) for i in idx if len(isis[i]) >= 2])
        axes[0, j].scatter(one[:, 0], one[:, 1], s=12, color=col)
        axes[0, j].set_title(f"{neurons[c]}\na typical trial ({len(one)} points)")
        axes[1, j].scatter(everything[:, 0], everything[:, 1], s=3, alpha=0.3, color=col)
        axes[1, j].set_title(f"all {len(idx)} trials")
    for ax in axes[1]:
        ax.set_xlabel("log10 ISI$_i$ (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("log10 ISI$_{i+1}$ (s)")
    fig.suptitle("Delay point clouds: each point is a pair of consecutive ISIs")
    savefig(fig, "explore_delay_clouds.png")


if __name__ == "__main__":
    main()
