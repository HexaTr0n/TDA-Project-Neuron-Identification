"""Step 2 - from one trial to barcodes and stable ranks, then average stable ranks per neuron.

Figures (in figures/analysis/):
    sr_line_example.png   LINE cloud of one trial: the spike times, the H0 barcode
                          (one bar per ISI) and the H0 stable rank
    sr_delay_example.png  DELAY cloud of the same trial: the cloud, the H0 and H1 barcodes
                          and the two stable ranks
    sr_mean_per_neuron.png  mean stable rank +/- one standard deviation over the 30 trials,
                          for 5 neurons, for LINE H0, DELAY H0 and DELAY H1

If the bands of different neurons are well separated, the stable ranks can tell the neurons
apart. This is the classic figure of the earlier student projects.

Usage:  python analysis/02_stable_ranks.py [--n-neurons 40] [--seed 0] [--min-rate 1] [--show 5]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tda_utils import (bars_of, cloud_delay, cloud_line, load_trials, pcf_on_grid,
                       pcf_to_xy, savefig, stable_ranks)


def plot_barcode(ax, bars, color, title):
    """Draw each bar [birth, death) as a horizontal segment, longest at the top."""
    bars = bars[np.argsort(bars[:, 1] - bars[:, 0])]
    for k, (b, d) in enumerate(bars):
        ax.plot([b, d], [k, k], color=color, lw=1.2)
    ax.set_yticks([])
    ax.set_xlabel("scale t")
    ax.set_title(f"{title} ({len(bars)} bars)")


def plot_pcf(ax, pcf, color, title):
    t, v = pcf_to_xy(pcf)
    ax.step(np.append(t, t[-1] * 1.05), np.append(v, 0), where="post", color=color)
    ax.set_xlabel("t")
    ax.set_ylabel("bars longer than t")
    ax.set_title(title)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-neurons", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-rate", type=float, default=1.0)
    p.add_argument("--show", type=int, default=5, help="neurons in the mean stable rank figure")
    a = p.parse_args()

    data = load_trials(a.n_neurons, a.min_rate, seed=a.seed)
    labels, isis, neurons = data["labels"], data["isis"], data["neurons"]

    # A typical trial: the one with the median number of spikes among all samples.
    i = int(np.argsort([len(x) for x in isis])[len(isis) // 2])
    print(f"Example trial: {neurons[labels[i]]}, {len(isis[i]) + 1} spikes")

    # --- LINE cloud of one trial ------------------------------------------------------------
    line = cloud_line(isis[i])
    (h0,) = bars_of(line, 0)
    # Sanity check of the theory: the H0 bars of points on a line are exactly the gaps (ISIs).
    assert np.allclose(np.sort(h0[:, 1] - h0[:, 0]), np.sort(isis[i])), "H0 bars should be the ISIs"
    (sr0,) = stable_ranks([line, line], 0)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    ax[0].eventplot([line[:, 0]], colors="k", linelengths=0.6)
    ax[0].set_yticks([])
    ax[0].set_xlabel("spike time (s)")
    ax[0].set_title("LINE cloud: spike times as points on R")
    plot_barcode(ax[1], h0, "tab:blue", "H0 barcode: bar i = ISI i")
    plot_pcf(ax[2], sr0[0], "tab:blue", "H0 stable rank")
    savefig(fig, "sr_line_example.png")

    # --- DELAY cloud of the same trial -----------------------------------------------------
    cloud = cloud_delay(isis[i])
    h0, h1 = bars_of(cloud, 1)
    sr0, sr1 = stable_ranks([cloud, cloud], 1)
    # Layout: the cloud on the left; H0 (barcode, stable rank) on the top row, H1 below.
    fig = plt.figure(figsize=(13, 7.5))
    grid = fig.add_gridspec(2, 3)
    ax_cloud = fig.add_subplot(grid[:, 0])
    ax_cloud.scatter(cloud[:, 0], cloud[:, 1], s=14, color="k")
    ax_cloud.set_xlabel("log10 ISI$_i$")
    ax_cloud.set_ylabel("log10 ISI$_{i+1}$")
    ax_cloud.set_title("DELAY cloud")
    plot_barcode(fig.add_subplot(grid[0, 1]), h0, "tab:blue", "H0 barcode")
    plot_pcf(fig.add_subplot(grid[0, 2]), sr0[0], "tab:blue", "H0 stable rank")
    plot_barcode(fig.add_subplot(grid[1, 1]), h1, "tab:red", "H1 barcode")
    plot_pcf(fig.add_subplot(grid[1, 2]), sr1[0], "tab:red", "H1 stable rank")
    savefig(fig, "sr_delay_example.png")

    # --- Mean stable rank +/- std per neuron ------------------------------------------------
    print("Computing stable ranks of all trials...")
    line_sr = stable_ranks([cloud_line(x) for x in isis], 0)[0]
    delay_sr = stable_ranks([cloud_delay(x) for x in isis], 1)
    panels = [("LINE H0 (ISI lengths, s)", line_sr, np.linspace(0, 1.0, 400)),
              ("DELAY H0", delay_sr[0], np.linspace(0, 0.5, 400)),
              ("DELAY H1", delay_sr[1], np.linspace(0, 0.2, 400))]
    show = np.linspace(0, len(neurons) - 1, a.show).astype(int)   # spread over the neurons
    colors = plt.cm.tab10(np.arange(a.show))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (title, pcfs, grid) in zip(axes, panels):
        for c, col in zip(show, colors):
            vals = np.array([pcf_on_grid(pcfs[k], grid) for k in np.flatnonzero(labels == c)])
            m, s = vals.mean(0), vals.std(0)
            ax.plot(grid, m, color=col, label=neurons[c])
            ax.fill_between(grid, m - s, m + s, color=col, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("t")
        ax.set_ylabel("stable rank (mean ± std over trials)")
    axes[0].legend(fontsize=9)
    savefig(fig, "sr_mean_per_neuron.png")


if __name__ == "__main__":
    main()
