"""Step 4 - the extended experiment: the same question, with more controls and a second stimulus.

Step 3 asked: can one trial tell which neuron it came from, when all trials show the SAME movie?
This step keeps that protocol (40 random neurons firing at least 1 Hz, 30 samples each, several
random draws, mean +/- std) and adds:

  * a second dataset: Neuropixels recordings from mouse visual cortex (Allen Visual Coding),
    30 s repeats of a natural movie, next to the retina (10 s repeats of frozen noise);
  * a second question, ACROSS stimuli: we know the neurons from one stimulus (A) and must
    recognise them in samples recorded under another stimulus (B);
  * k-medoids clustering as the second score (it needs no labels and works for any distance);
  * the Victor-Purpura spike-train distance as a reference without topology;
  * two controls for the delay embedding: shuffled ISIs and a random time cutoff.

Two parts are run for every dataset:

    within   every sample shows the same stimulus (as in step 3)
                1-NN          is the closest OTHER sample from the same neuron?
                k-medoids     cluster all samples without labels; share in the right cluster
    across   the same neurons under several stimuli
             first stimulus only   1-NN and k-medoids inside the first stimulus (as "within",
                                   but on the neurons of this part)
             all stimuli mixed     every neuron gives the same number of samples from every
                                   stimulus; the samples go into one pool and the stimulus of
                                   a sample is not used. 1-NN and k-medoids on that pool:
                                   k-medoids sorts the unlabelled pool into (number of neurons)
                                   buckets, and we count the samples in their neuron's bucket.
             other stimuli only    a stricter 1-NN: the nearest sample is searched only among
                                   samples of the OTHER stimuli, so a sample cannot be matched
                                   to its own stimulus

Stimuli (the lists are in DATASETS below; a stimulus is added by adding one line there):
    retina       frozen noise (the same 10 s movie every time)
                 unrepeated noise shown just before each frozen trial (new noise every time)
    neuropixels  natural movie, first block of 30 repeats (30 s each), and every other stimulus
                 in the table recorded within --max-gap minutes of that block: the grey screen,
                 and after `scripts/load_allen.py --all-stimuli` also gratings, flashes, ...

Methods (all distances are between two samples):
    B1  firing rate                |rate_i - rate_j|                              (no topology)
    B2  ISI histogram              L1 between log10-ISI histograms                (no topology)
    B3  Victor-Purpura, q = ...    edit distance between spike trains             (no topology)
    T1  LINE H0                    L1 between H0 stable ranks of the spike times on a line
    T1  LINE H0, log scale         the same barcode with bar lengths log10(ISI / 0.1 ms)
    T1  LINE H0, log scale, share  ... and each stable rank divided by its number of bars
    T2  DELAY H0 / H1 / H0+H1      L1 between stable ranks of the delay cloud (log10 ISI_i, log10 ISI_i+1)
    T2  DELAY 3D H0 / H1 / H0+H1   the same with three consecutive ISIs per point
    C1  ..., ISIs shuffled         control: same ISIs in random order (no order information)
    C2  ..., random time cutoff    control: a random part of the start and of the end of every
                                   sample is removed (no common spike count, no common piece
                                   of the stimulus)
    C3  DELAY H0, share            control: stable rank divided by its number of bars

Outputs (dataset = retina or neuropixels):
    results/extended_<dataset>_within.csv / .md    scores, mean +/- std over the neuron draws
    results/extended_<dataset>_across.csv / .md
    figures/analysis/ext_<dataset>_within.png      bar charts
    figures/analysis/ext_<dataset>_across.png      top: 1-NN, bottom: k-medoids
    figures/analysis/ext_<dataset>_vp.png          Victor-Purpura score against q
    figures/analysis/ext_<dataset>_distance_matrix.png   distances between the samples of 10
                                                   neurons, sorted by neuron, then stimulus

Usage:
    python analysis/04_extended.py --dataset retina
    python analysis/04_extended.py --dataset neuropixels
    python analysis/04_extended.py --dataset retina --part within --seeds 1 --n-neurons 10   # quick test
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tda_utils import (RESULTS_DIR, ROOT, load_dataset, cloud_delay, cloud_line, cloud_line_log, combine,
                       combine_matrices, load_stimuli, load_trials, normalise_pcfs, pcf_distances,
                       pcf_matrix, random_time_cut, savefig, score_1nn, score_kmedoids,
                       scores_across, shuffle_isis, stable_ranks,
                       vector_distances, vector_matrix, victor_purpura)
import tda_utils

tda_utils.SAVE_SVG = True                 # every figure also as .svg

Q_VALUES = [0.3, 1, 3, 10, 30, 100]       # Victor-Purpura cost per second of shift

DATASETS = {
    # The retina "within" part uses the same table and the same random draws as step 3, so its
    # B1, B2, T1, T2 and C rows must reproduce results/identification.md.
    "retina": dict(
        label="retina (frozen noise, 10 s)", trial_s=9.94,
        within=dict(path=None, session=None, query=None, n_trials=30),
        # The unrepeated noise only exists for one recording, so the across part uses that one.
        across=[dict(name="frozen noise", path=ROOT / "data" / "retina_running_20180126_sr_le_sp_clean.parquet",
                     session="20180126_sr_le_sp", query=None, n=30),
                dict(name="unrepeated noise", path=ROOT / "data" / "retina_running_20180126_sr_le_sp_clean.parquet",
                     session="20180126_sr_le_sp_running", query="piece == 0", n=30)]),
    "neuropixels": dict(
        label="Neuropixels (natural movie, 30 s)", trial_s=30.0,
        within=dict(path=ROOT / "data" / "allen" / "allen829720705.parquet",
                    session="allen829720705_movie", query="start < 3600", n_trials=30),
        # The stimulus list is built from the table when the script runs (see
        # neuropixels_stimuli): the first movie block, plus every other stimulus with enough
        # 30 s windows recorded within --max-gap minutes of that block.
        across="neuropixels"),
}

NPX_PATH = ROOT / "data" / "allen" / "allen829720705.parquet"
NPX_MOVIE = "allen829720705_movie"


def neuropixels_stimuli(max_gap_min=30.0, n_max=15, n_min=10, path=None):
    """Stimuli for the Neuropixels "across" part, chosen from what the table contains.

    Stimulus 1 is the first block of natural-movie repeats (30 samples, as in the within part).
    Every other `session` of the table (grey screen, and the stimuli added by
    `load_allen.py --all-stimuli`) is used with the windows that start within `max_gap_min`
    minutes of that block, if it has at least `n_min` of them. Neurons drift slowly, so stimuli
    recorded far apart in time would be told apart by the drift and not by the stimulus.
    All other stimuli use the same number of samples per neuron: the smallest count, at most
    `n_max`.
    """
    path = path or NPX_PATH
    df = load_dataset(path)
    movie = df[(df.session == NPX_MOVIE) & (df.start < 3600)]
    lo = float(movie.start.min()) - 60 * max_gap_min
    hi = float((movie.start + movie.duration).max()) + 60 * max_gap_min
    found = {}
    for name, g in df[df.session != NPX_MOVIE].groupby("session"):
        g = g[(g.start >= lo) & (g.start + g.duration <= hi)]
        n = g.groupby("neuron_id").size().min() if len(g) else 0
        if n >= n_min:
            found[name] = int(n)
    if not found:
        raise ValueError(f"no stimulus besides the movie has {n_min} windows within {max_gap_min} min of it")
    n = min(n_max, min(found.values()))
    out = [dict(name="natural movie", path=path, session=NPX_MOVIE, query="start < 3600", n=30)]
    for name in sorted(found):
        label = name.replace("allen829720705_", "").replace("spont", "grey screen").replace("_", " ")
        out.append(dict(name=label, path=path, session=name, query=f"start >= {lo} and start + duration <= {hi}", n=n))
    print("Neuropixels stimuli: " + ", ".join(f"{st['name']} ({st['n']})" for st in out), flush=True)
    return out


STEP3_ROWS = ["B1 firing rate", "B2 ISI histogram", "T1 LINE H0", "T2 DELAY H0", "T2 DELAY H1",
              "T2 DELAY H0+H1", "C1 DELAY H0+H1, ISIs shuffled"]
ORDER = (["B1 firing rate", "B2 ISI histogram"] + [f"B3 Victor-Purpura, q = {q:g}/s" for q in Q_VALUES]
         + ["T1 LINE H0", "T1 LINE H0, log scale", "T1 LINE H0, log scale, share",
            "T2 DELAY H0", "T2 DELAY H1", "T2 DELAY H0+H1",
            "C1 DELAY H0, ISIs shuffled", "C1 DELAY H0+H1, ISIs shuffled",
            "C2 LINE H0, random time cutoff", "C2 DELAY H0, random time cutoff",
            "C3 DELAY H0, share",
            "T2 DELAY 3D H0", "T2 DELAY 3D H1", "T2 DELAY 3D H0+H1",
            "C1 DELAY 3D H0, ISIs shuffled", "C1 DELAY 3D H0+H1, ISIs shuffled",
            "C2 DELAY 3D H0, random time cutoff"])


class _Scored(dict):
    """Stands in for the result dict of `all_distances` when a `sink` is given: every matrix is
    handed to `sink(name, (D, M))` as soon as it exists and only kept if a later method needs it
    (the H0 and H1 matrices for the H0+H1 sum). This keeps memory at a few matrices instead of
    all of them, which matters for large pools of samples (many stimuli or neurons)."""
    KEEP = ("T2 DELAY H0", "T2 DELAY H1")

    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def __setitem__(self, name, value):
        D = value[0]
        n = len(D)
        assert D.shape == (n, n) and np.allclose(D, D.T) and np.allclose(np.diag(D), 0), f"{name}: bad distance matrix"
        self.sink(name, value)
        if name in self.KEEP:
            super().__setitem__(name, value)
        if name == "T2 DELAY H0+H1":
            for k in self.KEEP:
                self.pop(k, None)


def all_distances(isis, spikes, duration, seed, labels=None, sink=None):
    """{method name: (D, M)} for one list of samples.

    With `sink`, every (D, M) is passed to `sink(name, (D, M))` right away and not kept (to
    save memory); the return value is then empty.

    D is the n x n distance matrix. M (distances to the leave-one-out neuron means, for the
    nearest-mean score of step 3) is only computed when `labels` is given, and only for the
    methods that have a mean; otherwise it is None.
    `duration` is one number, or one number per sample (seconds).
    """
    n = len(isis)
    duration = np.broadcast_to(np.asarray(duration, dtype=float), (n,))
    out = {} if sink is None else _Scored(sink)

    def pcf(pcfs):
        return pcf_distances(pcfs, labels) if labels is not None else (pcf_matrix(pcfs), None)

    def vec(X):
        return vector_distances(X, labels) if labels is not None else (vector_matrix(X), None)

    # B1, B2: exactly as in step 3.
    rate = np.array([len(x) + 1 for x in isis]) / duration
    out["B1 firing rate"] = vec(rate[:, None])
    bins = np.linspace(-3, 1, 41)
    out["B2 ISI histogram"] = vec(np.array([np.histogram(np.log10(x), bins)[0] for x in isis]))

    # B3: Victor-Purpura, one matrix per time precision q.
    for q in Q_VALUES:
        out[f"B3 Victor-Purpura, q = {q:g}/s"] = (victor_purpura(spikes, q), None)

    # T1: H0 of the spike times on a line (bars = ISIs), then the two variants.
    (line,) = stable_ranks([cloud_line(x) for x in isis], 0)
    out["T1 LINE H0"] = pcf(line)
    (line_log,) = stable_ranks([cloud_line_log(x) for x in isis], 0)
    out["T1 LINE H0, log scale"] = (pcf_matrix(line_log), None)
    out["T1 LINE H0, log scale, share"] = (pcf_matrix(normalise_pcfs(line_log, n)), None)

    # T2: H0 and H1 of the delay cloud, alone and combined (as in step 3).
    h0, h1 = stable_ranks([cloud_delay(x) for x in isis], 1)
    out["T2 DELAY H0"] = pcf(h0)
    out["T2 DELAY H1"] = pcf(h1)
    if labels is not None:
        out["T2 DELAY H0+H1"] = combine([out["T2 DELAY H0"], out["T2 DELAY H1"]])
    else:
        out["T2 DELAY H0+H1"] = (combine_matrices([out["T2 DELAY H0"][0], out["T2 DELAY H1"][0]]), None)

    # C1: the delay cloud after shuffling the order of the ISIs inside each sample.
    s0, s1 = stable_ranks([cloud_delay(x) for x in shuffle_isis(isis, seed)], 1)
    out["C1 DELAY H0, ISIs shuffled"] = (pcf_matrix(s0), None)
    if labels is not None:
        out["C1 DELAY H0+H1, ISIs shuffled"] = combine([pcf_distances(s0, labels), pcf_distances(s1, labels)])
    else:
        out["C1 DELAY H0+H1, ISIs shuffled"] = (combine_matrices([pcf_matrix(s0), pcf_matrix(s1)]), None)

    # C2: a random part of the start and of the end of every sample is removed.
    cut_isis = []
    for i in range(n):
        cut_isis.append(random_time_cut([spikes[i]], duration[i], seed=1000 * seed + i)[1][0])
    (cut_line,) = stable_ranks([cloud_line(x) for x in cut_isis], 0)
    out["C2 LINE H0, random time cutoff"] = (pcf_matrix(cut_line), None)
    (cut_h0,) = stable_ranks([cloud_delay(x) for x in cut_isis], 0)
    out["C2 DELAY H0, random time cutoff"] = (pcf_matrix(cut_h0), None)

    # C3: the delay cloud's H0 stable rank divided by its number of bars.
    out["C3 DELAY H0, share"] = (pcf_matrix(normalise_pcfs(h0, n)), None)

    # The delay cloud with three consecutive ISIs per point, with the same controls.
    a0, a1 = stable_ranks([cloud_delay(x, 3) for x in isis], 1)
    D0, D1 = pcf_matrix(a0), pcf_matrix(a1)
    out["T2 DELAY 3D H0"] = (D0, None)
    out["T2 DELAY 3D H1"] = (D1, None)
    out["T2 DELAY 3D H0+H1"] = (combine_matrices([D0, D1]), None)
    s0, s1 = stable_ranks([cloud_delay(x, 3) for x in shuffle_isis(isis, seed)], 1)
    D0 = pcf_matrix(s0)
    out["C1 DELAY 3D H0, ISIs shuffled"] = (D0, None)
    out["C1 DELAY 3D H0+H1, ISIs shuffled"] = (combine_matrices([D0, pcf_matrix(s1)]), None)
    (cut3,) = stable_ranks([cloud_delay(x, 3) for x in cut_isis], 0)
    out["C2 DELAY 3D H0, random time cutoff"] = (pcf_matrix(cut3), None)

    if sink is not None:
        return {}
    for name, (D, _) in out.items():
        assert D.shape == (n, n) and np.allclose(D, D.T) and np.allclose(np.diag(D), 0), f"{name}: bad distance matrix"
    return out


def sample_seconds(cfg, block):
    return block * (10.0 if cfg["trial_s"] == 9.94 else cfg["trial_s"]) if block > 1 else cfg["trial_s"]


def run_within(cfg, args, seed, block=1):
    w = cfg["within"]
    data = load_trials(args.n_neurons, args.min_rate, w["n_trials"], seed=seed, block=block, path=w["path"],
                       session=w["session"], query=w["query"], trial_s=cfg["trial_s"])
    labels = data["labels"]
    rows = []

    def score(name, value):
        km, km_chance = score_kmedoids(value[0], labels, args.kmedoids_seeds, chance=True)
        rows.append(dict(method=name, seed=seed, block=block, nn1=score_1nn(value[0], labels),
                         kmedoids=km, kmedoids_chance=km_chance))

    all_distances(data["isis"], data["spikes"], sample_seconds(cfg, block), seed, sink=score)
    return rows


def stimulus_list(cfg, args):
    if cfg["across"] == "neuropixels":
        return neuropixels_stimuli(args.max_gap)
    return cfg["across"]


FIG_METHODS = ["B2 ISI histogram", "T1 LINE H0", "T1 LINE H0, log scale, share",
               "T2 DELAY H0+H1", "C1 DELAY H0+H1, ISIs shuffled"]   # + the best Victor-Purpura q

# Methods in the bar figures, fixed in advance by their role (not by their score), with the
# short label used on the axis. Groups: references, LINE, DELAY, DELAY 3D; each control sits
# next to the method it controls and is hatched. "B3" is replaced by the Victor-Purpura q with
# the best 1-NN score (the whole q sweep is in ext_<dataset>_vp). The tables have every row.
# Set BAR_GROUPS = None to plot every row instead.
BAR_GROUPS = [
    [("B1 firing rate", "firing\nrate"), ("B2 ISI histogram", "ISI\nhistogram"), ("B3", "Victor-\nPurpura")],
    [("T1 LINE H0", "LINE\nH0"), ("T1 LINE H0, log scale, share", "LINE H0\nlog share"),
     ("C2 LINE H0, random time cutoff", "LINE H0\ncutoff")],
    [("T2 DELAY H0", "DELAY\nH0"), ("C1 DELAY H0, ISIs shuffled", "DELAY H0\nshuffled"),
     ("C2 DELAY H0, random time cutoff", "DELAY H0\ncutoff"), ("T2 DELAY H1", "DELAY\nH1")],
    [("T2 DELAY 3D H0", "DELAY 3D\nH0"), ("C1 DELAY 3D H0, ISIs shuffled", "DELAY 3D H0\nshuffled")],
]


def run_across(cfg, args, seed, stimuli, figure=None, k_fig=10, block=1):
    """Scores of one neuron draw. With `figure` (a dict), also keeps the distance matrices of
    `k_fig` neurons for the figure: for each neuron its samples of stimulus 1, then of
    stimulus 2, and so on, with the same number of samples per stimulus as the mixed pool."""
    data = load_stimuli(stimuli, args.n_neurons, args.min_rate, seed=seed, trial_s=cfg["trial_s"], block=block)
    isis = [x for d in data for x in d["isis"]]
    spikes = [x for d in data for x in d["spikes"]]
    labels = np.concatenate([d["labels"] for d in data])
    edges = np.r_[0, np.cumsum([len(d["labels"]) for d in data])]
    groups = [np.arange(edges[j], edges[j + 1]) for j in range(len(data))]
    rows = []
    if figure is not None:
        k = labels.max() + 1
        m = min(np.bincount(labels[g], minlength=k).min() for g in groups)
        show = np.linspace(0, k - 1, k_fig).astype(int)
        keep = np.array([i for c in show for g in groups for i in g[labels[g] == c][:m]])
        figure.update(keep=keep, m=m, n_stim=len(groups), k=k_fig, mats={})

    def score(name, value):                                        # no labels: no neuron means
        rows.append(dict(method=name, seed=seed, block=block,
                         **scores_across(value[0], labels, groups, args.kmedoids_seeds)))
        if figure is not None and (name in FIG_METHODS or name.startswith("B3")):
            figure["mats"][name] = value[0][np.ix_(figure["keep"], figure["keep"])]

    all_distances(isis, spikes, sample_seconds(cfg, block), seed, sink=score)        # scores each matrix, then drops it
    if figure is not None:                                         # keep the Victor-Purpura q that does best here
        vp = [r for r in rows if r["method"].startswith("B3")]
        best = max(vp, key=lambda r: r["nn_within"])["method"]
        figure["mats"] = {n: D for n, D in figure["mats"].items() if not n.startswith("B3") or n == best}
    return rows


def figure_matrices_across(fig_data, stimuli, dataset):
    """Like figure 7 of step 3, with the samples of several stimuli. Rows and columns: k neurons;
    inside each neuron, its samples of stimulus 1, then stimulus 2, ... (thick white lines
    separate neurons, thin dotted lines separate stimuli). Light = small distance. A light
    square that covers the whole neuron block means the neuron is recognised across stimuli;
    light only in the small squares on the diagonal means only within one stimulus."""
    mats, m, S, k = fig_data["mats"], fig_data["m"], fig_data["n_stim"], fig_data["k"]
    names = [n for n in FIG_METHODS if n in mats] + [n for n in mats if n.startswith("B3")]
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.6 * rows), squeeze=False)
    per = m * S
    for ax, name in zip(axes.flat, names):
        D = mats[name]
        im = ax.imshow(D, cmap="viridis_r", vmin=0, vmax=np.percentile(D, 95), interpolation="nearest")
        for j in range(1, k * S):
            e = j * m - 0.5
            thick = j % S == 0
            ax.axhline(e, color="w", lw=0.8 if thick else 0.3, ls="-" if thick else ":")
            ax.axvline(e, color="w", lw=0.8 if thick else 0.3, ls="-" if thick else ":")
        ax.set_title(name)
        ax.set_xlabel(f"sample ({k} neurons x {S} stimuli x {m} samples)")
        ax.set_ylabel("sample (sorted by neuron, then stimulus)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    order = ", ".join(f"{j + 1}. {st['name']}" for j, st in enumerate(stimuli))
    fig.suptitle(f"{dataset}: distances between samples of {k} neurons. Inside each neuron: {order}", fontsize=12)
    savefig(fig, f"ext_{dataset}_distance_matrix.png")


def summarise(df, cols):
    t = df.groupby("method")[cols].agg(["mean", "std"])
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    return t.reindex([m for m in ORDER if m in t.index]).reset_index()


def cell(r, col):
    m, s = r[f"{col}_mean"], r[f"{col}_std"]
    if np.isnan(m):
        return "-"
    return f"{100 * m:.1f} ± {100 * (0 if np.isnan(s) else s):.1f}"


def write_table(table, cols, heads, path, title):
    with open(path, "w") as f:
        f.write(title + "\n\n| method | " + " | ".join(heads) + " |\n|---|" + "---|" * len(heads) + "\n")
        for _, r in table.iterrows():
            f.write(f"| {r.method} | " + " | ".join(cell(r, c) for c in cols) + " |\n")
    print("\n" + open(path).read(), flush=True)


def bar_rows(table, col):
    """The rows shown in the bar figures, with axis label, group number and control flag."""
    if BAR_GROUPS is None:
        return table.assign(short=table.method, group=0, control=table.method.str.startswith("C"))
    vp = table[table.method.str.startswith("B3")]
    rows = []
    for g, group in enumerate(BAR_GROUPS):
        for method, short in group:
            if method == "B3":
                if vp.empty:
                    continue
                method = vp.loc[vp[f"{col}_mean"].idxmax(), "method"]
                short = f"Victor-Purpura\nq = {method.split('= ')[1]}"
            if method in set(table.method):
                rows.append(dict(table[table.method == method].iloc[0], short=short, group=g,
                                 control=method.startswith("C")))
    return pd.DataFrame(rows)


def figure_bars(table, cols, labs, chance, title, name):
    """One panel: for every method one bar per score column (1-NN, k-medoids), grouped as in
    BAR_GROUPS, controls hatched. Dashed line: 1-NN chance (1 / number of neurons)."""
    t = bar_rows(table, cols[0])
    x = np.arange(len(t)) + 0.6 * t["group"].to_numpy()          # a gap between the groups
    fig, ax = plt.subplots(figsize=(11, 4.2) if BAR_GROUPS is not None else (13, 5))
    w = 0.8 / len(cols)
    for j, (col, lab) in enumerate(zip(cols, labs)):
        ax.bar(x + (j - (len(cols) - 1) / 2) * w, 100 * t[f"{col}_mean"], w, color=f"C{j}",
               yerr=100 * t[f"{col}_std"].fillna(0), capsize=2, error_kw=dict(lw=0.8), label=lab,
               hatch=["///" if c else "" for c in t["control"]], edgecolor="white", linewidth=0)
    ax.axhline(100 * chance, color="k", ls="--", lw=1, label=f"chance, 1-NN ({100 * chance:.1f} %)")
    ax.bar([np.nan], [np.nan], color="white", hatch="///", edgecolor="grey", linewidth=0.5, label="control")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    if BAR_GROUPS is None:
        ax.set_xticklabels(t["short"], rotation=35, ha="right", fontsize=9)
    else:
        ax.set_xticklabels(t["short"], fontsize=8)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.5, ncol=4, frameon=False)
    ax.set_title(title, pad=24)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, name)


def figure_vp(tables, chance, title, name):
    """Victor-Purpura score against q, for every (table, column, label) given."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for table, col, lab in tables:
        t = table[table.method.str.startswith("B3")]
        ax.errorbar(Q_VALUES, 100 * t[f"{col}_mean"], yerr=100 * t[f"{col}_std"].fillna(0),
                    marker="o", capsize=3, label=lab)
    ax.axhline(100 * chance, color="k", ls="--", lw=1, label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("q: cost of shifting a spike by one second")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    savefig(fig, name)


LENGTH_METHODS = ["B1 firing rate", "B2 ISI histogram", "T1 LINE H0", "T1 LINE H0, log scale, share",
                  "T2 DELAY H0", "C1 DELAY H0, ISIs shuffled"]       # + the best Victor-Purpura q


def run_length(cfg, a, chance):
    """Scores for samples made of 1, 2 and 3 glued samples: within one stimulus (as step 3)
    and with all stimuli mixed."""
    stimuli = stimulus_list(cfg, a)
    within_n = cfg["within"]["n_trials"]
    rows = []
    for block in (1, 2, 3):
        sec = sample_seconds(cfg, block) if block > 1 else (10.0 if cfg["trial_s"] == 9.94 else cfg["trial_s"])
        for seed in range(a.seeds):
            t0 = time.time()
            if within_n // block >= 2:
                rows += [dict(r, seconds=sec, part="within", nn=r["nn1"], kmed=r["kmedoids"],
                              kmed_chance=r["kmedoids_chance"]) for r in run_within(cfg, a, seed, block=block)]
            if min(st["n"] for st in stimuli) // block >= 2:
                rows += [dict(r, seconds=sec, part="mixed", nn=r["nn_pooled"], kmed=r["kmed_pooled"],
                              kmed_chance=r["kmed_pooled_chance"])
                         for r in run_across(cfg, a, seed, stimuli, block=block)]
            print(f"{a.dataset} length {sec:g} s, draw {seed}: {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"extended_{a.dataset}_length.csv", index=False)
    one = df[(df.block == 1) & (df.part == "within")]
    vp = one[one.method.str.startswith("B3")].groupby("method")["nn"].mean().idxmax()
    methods = LENGTH_METHODS[:2] + [vp] + LENGTH_METHODS[2:]
    # The table has every method (the figures only the ones in `methods`), mean ± std over draws.
    t = df.groupby(["method", "seconds", "part"])[["nn", "kmed", "kmed_chance"]].agg(["mean", "std"]).unstack("part")
    order = [m for m in ORDER if m in set(df.method)]

    def fmt(r, c, p):
        m, sd = r.get((c, "mean", p), np.nan), r.get((c, "std", p), np.nan)
        return "-" if np.isnan(m) else f"{100 * m:.1f} ± {100 * (0 if np.isnan(sd) else sd):.1f}"

    with open(RESULTS_DIR / f"extended_{a.dataset}_length.md", "w") as f:
        f.write(f"SAMPLE LENGTH. {a.dataset}: accuracy (%), mean ± std over {a.seeds} neuron draws. "
                f"1-NN chance = {100 * chance:.1f} %.\n\n| method | seconds | 1-NN, one stimulus | "
                f"k-medoids, one stimulus | k-medoids chance | 1-NN, all stimuli mixed | "
                f"k-medoids, all stimuli mixed | k-medoids chance |\n|---|---|---|---|---|---|---|---|\n")
        for (m, sec), r in t.reindex(order, level=0).iterrows():
            vals = [fmt(r, c, p) for p in ("within", "mixed") for c in ("nn", "kmed", "kmed_chance")]
            f.write(f"| {m} | {sec:g} | " + " | ".join(vals) + " |\n")
    print("\n" + open(RESULTS_DIR / f"extended_{a.dataset}_length.md").read(), flush=True)

    for part, suffix, what in (("within", "within", "one stimulus"), ("mixed", "across", "all stimuli mixed")):
        d = df[df.part == part]
        if d.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
        for ax, col, ylab in zip(axes, ["nn", "kmed"], ["1-NN", "k-medoids"]):
            for name in methods:
                g = d[d.method == name].groupby("seconds")[col]
                ax.errorbar(g.mean().index, 100 * g.mean(), yerr=100 * g.std(), marker="o", capsize=3, label=name)
            ax.axhline(100 * chance, color="k", ls="--", lw=1, label="1-NN chance")
            ax.set_title(ylab)
            ax.set_xlabel("sample length (s)")
            ax.set_xticks(sorted(d.seconds.unique()))
            ax.set_ylim(0, 100)
        axes[0].set_ylabel("accuracy (%)")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{a.dataset}, {what}: longer samples")
        savefig(fig, f"ext_{a.dataset}_length_{suffix}.png")


def replot(cfg, a, chance):
    """Bar and q-sweep figures from results/extended_<dataset>_within.csv and _across.csv."""
    vp = []
    f = RESULTS_DIR / f"extended_{a.dataset}_within.csv"
    if f.exists():
        table = pd.read_csv(f)
        figure_bars(table, ["nn1", "kmedoids"], ["1-NN", "k-medoids"], chance,
                    f"{cfg['label']}: within one stimulus", f"ext_{a.dataset}_within.png")
        vp += [(table, "nn1", "within, 1-NN"), (table, "kmedoids", "within, k-medoids")]
    f = RESULTS_DIR / f"extended_{a.dataset}_across.csv"
    if f.exists():
        table = pd.read_csv(f)
        names = ", ".join(st["name"] for st in cfg["across"]) if isinstance(cfg["across"], list) else "all stimuli"
        figure_bars(table, ["nn_pooled", "kmed_pooled"], ["1-NN", "k-medoids"], chance,
                    f"{a.dataset}: across stimuli ({names} in one pool)", f"ext_{a.dataset}_across.png")
        vp += [(table, "nn_pooled", "all stimuli mixed, 1-NN"), (table, "kmed_pooled", "all stimuli mixed, k-medoids")]
    if vp:
        figure_vp(vp, chance, f"{a.dataset}: Victor-Purpura distance against its time precision",
                  f"ext_{a.dataset}_vp.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=list(DATASETS), required=True)
    p.add_argument("--part", choices=["within", "across", "both", "length"], default="both",
                   help="length: the across scores for samples of 1, 2 and 3 glued samples")
    p.add_argument("--n-neurons", type=int, default=40)
    p.add_argument("--min-rate", type=float, default=1.0)
    p.add_argument("--seeds", type=int, default=5, help="number of random neuron draws")
    p.add_argument("--kmedoids-seeds", type=int, default=3, help="k-medoids runs averaged per score")
    p.add_argument("--figure-only", action="store_true",
                   help="across part: only draw 0 and the distance-matrix figure; the tables are not rewritten")
    p.add_argument("--replot", action="store_true",
                   help="only redraw the bar and q-sweep figures from the tables in results/ (no computing)")
    p.add_argument("--max-gap", type=float, default=30.0,
                   help="Neuropixels: use stimuli recorded within this many minutes of the first movie block")
    a = p.parse_args()
    cfg = DATASETS[a.dataset]
    chance = 1 / a.n_neurons
    RESULTS_DIR.mkdir(exist_ok=True)
    head = f"{a.n_neurons} neurons, mean ± std over {a.seeds} neuron draws. Chance = {100 * chance:.1f} %."
    vp = []

    if a.replot:
        replot(cfg, a, chance)
        return

    if a.part in ("within", "both"):
        rows = []
        for seed in range(a.seeds):
            t0 = time.time()
            rows += run_within(cfg, a, seed)
            print(f"{a.dataset} within, draw {seed}: {time.time() - t0:.0f} s", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_DIR / f"extended_{a.dataset}_within_runs.csv", index=False)
        cols = ["nn1", "kmedoids", "kmedoids_chance"]
        table = summarise(df, cols)
        table.to_csv(RESULTS_DIR / f"extended_{a.dataset}_within.csv", index=False)
        write_table(table, cols, ["1-NN", "k-medoids", "k-medoids chance"],
                    RESULTS_DIR / f"extended_{a.dataset}_within.md",
                    f"WITHIN one stimulus. {cfg['label']}: accuracy (%), {cfg['within']['n_trials']} samples per "
                    f"neuron, {head}")
        figure_bars(table, ["nn1", "kmedoids"], ["1-NN", "k-medoids"], chance,
                    f"{cfg['label']}: within one stimulus", f"ext_{a.dataset}_within.png")
        vp += [(table, "nn1", "within, 1-NN"), (table, "kmedoids", "within, k-medoids")]

    if a.part in ("across", "both"):
        stimuli = stimulus_list(cfg, a)
        rows, fig_data = [], {}
        for seed in range(1 if a.figure_only else a.seeds):
            t0 = time.time()
            rows += run_across(cfg, a, seed, stimuli, figure=fig_data if seed == 0 else None)
            print(f"{a.dataset} across, draw {seed}: {time.time() - t0:.0f} s", flush=True)
        figure_matrices_across(fig_data, stimuli, a.dataset)
        if a.figure_only:
            return
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_DIR / f"extended_{a.dataset}_across_runs.csv", index=False)
        cols = ["nn_within", "kmed_within", "nn_pooled", "kmed_pooled", "nn_other", "kmed_within_chance",
                "kmed_pooled_chance"]
        table = summarise(df, cols)
        table.to_csv(RESULTS_DIR / f"extended_{a.dataset}_across.csv", index=False)
        m = min(st["n"] for st in stimuli)
        first = stimuli[0]["name"]
        listing = ", ".join(f"{st['name']} ({st['n']} samples per neuron)" for st in stimuli)
        write_table(table, cols, [f"1-NN, {first} only", f"k-medoids, {first} only", "1-NN, all stimuli mixed",
                                  "k-medoids, all stimuli mixed", "1-NN, other stimuli only",
                                  f"k-medoids chance, {first} only", "k-medoids chance, all stimuli mixed"],
                    RESULTS_DIR / f"extended_{a.dataset}_across.md",
                    f"ACROSS stimuli. {a.dataset}: {listing}. All stimuli mixed = {m} samples per neuron from "
                    f"each stimulus in one unlabelled pool. Accuracy (%), {head}")
        figure_bars(table, ["nn_pooled", "kmed_pooled"], ["1-NN", "k-medoids"], chance,
                    f"{a.dataset}: across stimuli (" + ", ".join(st["name"] for st in stimuli) + " in one pool)",
                    f"ext_{a.dataset}_across.png")
        vp += [(table, "nn_pooled", "all stimuli mixed, 1-NN"), (table, "kmed_pooled", "all stimuli mixed, k-medoids")]

    if a.part == "length":
        run_length(cfg, a, chance)
        return
    figure_vp(vp, chance, f"{a.dataset}: Victor-Purpura distance against its time precision", f"ext_{a.dataset}_vp.png")


if __name__ == "__main__":
    main()
