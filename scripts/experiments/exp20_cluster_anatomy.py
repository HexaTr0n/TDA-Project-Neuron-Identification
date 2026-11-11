"""Who gets misclustered? Uses the distance matrices saved by run_pipeline.py --neurons 60.

Run from the repo root: python scripts/experiments/exp20_cluster_anatomy.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import run_pipeline as rp
from load_dataset import load_dataset

folder = rp.RESULTS / "distances" / "20180126_sr_le_sp_60"
names = np.load(folder / "labels.npy"); uniq, y = np.unique(names, return_inverse=True)
csv = REPO / "data" / "isi_frozennoise.csv"
df = load_dataset(csv if csv.exists() else rp.DEFAULT_PATH, sessions=["20180126_sr_le_sp"])
blocks = rp.pick_neurons(rp.make_blocks(df, rp.Config()), 60, 0)
assert [b[0] for b in blocks] == list(names)
trial = np.array([b[1][0].trial for b in blocks]); count = np.array([len(b[1][0].spike_times) for b in blocks])
L = lambda n: np.load(folder / f"{n}__D.npy").astype(float)
VIEWS = {"time bars + log ISI bars": rp.scaled(L("time_bars_s67")) + .5 * rp.scaled(L("isi_bars_log")),
         "burst bars + log ISI bars": rp.scaled(L("burst_bars_s67")) + .5 * rp.scaled(L("isi_bars_log")),
         "log ISI bars": L("isi_bars_log")}
n_neur = len(uniq); rate = np.array([count[y == k].mean() for k in range(n_neur)]) / 9.94

for name, D in VIEWS.items():
    print(f"\n===== {name}")
    print("  clusters | neuron kept together | cluster is one neuron | neurons per cluster | matched accuracy")
    for k in (5, 10, 20, 30, 45, 60, 90, 120):
        c = rp.k_medoids(D, k)
        tab = np.zeros((n_neur, k)); np.add.at(tab, (y, c), 1)
        together = (tab.max(1) / tab.sum(1)).mean()          # share of a neuron's trials in its main cluster
        pure = (tab.max(0).sum() / tab.sum())                 # share of trials that are their cluster's majority neuron
        home = tab.argmax(1); per_cluster = np.bincount(home, minlength=k); per_cluster = per_cluster[per_cluster > 0].mean()
        print(f"  {k:8d} | {together:20.2f} | {pure:21.2f} | {per_cluster:19.1f} | {rp.matched_accuracy(y, c):.2f}")
        if k == 60:
            keep60, tab60, c60 = tab.max(1) / tab.sum(1), tab, c
        if k == 20:
            # are neurons that share a cluster similar in rate?
            same = home[:, None] == home[None, :]; iu = np.triu_indices(n_neur, 1)
            dr = np.abs(np.log(rate)[:, None] - np.log(rate)[None, :])
            print(f"           at 20 clusters: log-rate gap between neurons sharing a cluster {dr[iu][same[iu]].mean():.2f}, others {dr[iu][~same[iu]].mean():.2f}")
    # ---- k = 60: who is wrong?
    s = np.sort(keep60)
    print(f"  at 60 clusters, share of a neuron's trials in its main cluster: {(keep60 >= .9).sum()} neurons above 0.9, "
          f"{((keep60 >= .5) & (keep60 < .9)).sum()} between 0.5 and 0.9, {(keep60 < .5).sum()} below 0.5")
    print(f"  kept-together against firing rate: Spearman {spearmanr(keep60, rate)[0]:+.2f}   (rates {rate.min():.1f} to {rate.max():.1f} Hz)")
    home = tab60.argmax(1); owner = np.full(60, -1)
    for cl in range(60):
        if (home == cl).any():
            owner[cl] = np.flatnonzero(home == cl)[tab60[home == cl, cl].argmax()]
    shared = np.bincount(home, minlength=60)
    print(f"  clusters that are the main cluster of: no neuron {(shared == 0).sum()}, one neuron {(shared == 1).sum()}, several {(shared > 1).sum()}")
    ok = home[y] == c60                                           # trial sits in its neuron's main cluster
    by_trial = np.array([ok[trial == t].mean() for t in range(30)])
    print(f"  share of trials in their neuron's main cluster, by position in the recording: first five {by_trial[:5].mean():.2f}, "
          f"middle {by_trial[10:20].mean():.2f}, last five {by_trial[-5:].mean():.2f}; spread across trials {by_trial.std():.3f}")
    # same trial numbers wrong across neurons? compare with chance for independent errors
    p = np.array([ok[trial == t].mean() for t in range(30)]); print(f"  worst three trial numbers: {np.argsort(p)[:3].tolist()} with {np.sort(p)[:3].round(2).tolist()}")
    # is a split neuron split by time?  second cluster's trials early or late
    z = []
    for k in range(n_neur):
        m = y == k; cl = c60[m]; tr = trial[m]
        if (cl == home[k]).mean() < .9 and (cl != home[k]).sum() >= 3:
            r = spearmanr(tr, cl != home[k])[0]; z.append(abs(r))
    rng = np.random.default_rng(0); null = []
    for _ in range(200):
        for k in range(n_neur):
            m = y == k; cl = c60[m]
            if (cl == home[k]).mean() < .9 and (cl != home[k]).sum() >= 3:
                null.append(abs(spearmanr(rng.permutation(trial[m]), cl != home[k])[0]))
    print(f"  split neurons ({len(z)}): strays tied to trial order, |Spearman| {np.mean(z):.2f} against {np.mean(null):.2f} if order did not matter")
