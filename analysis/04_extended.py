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

    within   every sample shows stimulus A (as in step 3)
                1-NN          is the closest OTHER sample from the same neuron?
                k-medoids     cluster all samples without labels; share in the right cluster
                nearest mean  as in step 3 (only for the methods of step 3)
    across   the same neurons under stimulus A and stimulus B
                1-NN          every B sample takes the neuron of its closest A sample
                k-medoids     cluster the A samples, then give every B sample to the cluster
                              of its closest medoid
             The within scores of stimulus A are printed next to them, on the same neurons.

Stimuli:
    retina       A = frozen noise (the same 10 s movie every time)
                 B = unrepeated noise shown just before each frozen trial (new noise every time)
    neuropixels  A = natural movie, first block of 30 repeats (30 s each)
                 B = spontaneous activity on a grey screen, recorded just before that block

Methods (all distances are between two samples):
    B1  firing rate                |rate_i - rate_j|                              (no topology)
    B2  ISI histogram              L1 between log10-ISI histograms                (no topology)
    B3  Victor-Purpura, q = ...    edit distance between spike trains             (no topology)
    T1  LINE H0                    L1 between H0 stable ranks of the spike times on a line
    T1  LINE H0, log scale         the same barcode with bar lengths log10(ISI / 0.1 ms)
    T1  LINE H0, log scale, share  ... and each stable rank divided by its number of bars
    T2  DELAY H0 / H1 / H0+H1      L1 between stable ranks of the delay cloud
    C1  ..., ISIs shuffled         control: same ISIs in random order (no order information)
    C2  ..., random time cutoff    control: a random part of the start and of the end of every
                                   sample is removed (no common spike count, no common piece
                                   of the stimulus)
    C3  DELAY H0, share            control: stable rank divided by its number of bars

Outputs (dataset = retina or neuropixels):
    results/extended_<dataset>_within.csv / .md    scores, mean +/- std over the neuron draws
    results/extended_<dataset>_across.csv / .md
    figures/analysis/ext_<dataset>_within.png      bar charts
    figures/analysis/ext_<dataset>_across.png
    figures/analysis/ext_<dataset>_vp.png          Victor-Purpura score against q

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

from tda_utils import (RESULTS_DIR, ROOT, cloud_delay, cloud_line, cloud_line_log, combine,
                       combine_matrices, load_pair, load_trials, normalise_pcfs, pcf_distances,
                       pcf_matrix, random_time_cut, savefig, score_1nn, score_kmedoids,
                       score_nearest_mean, scores_across, shuffle_isis, stable_ranks,
                       vector_distances, vector_matrix, victor_purpura)

Q_VALUES = [0.3, 1, 3, 10, 30, 100]       # Victor-Purpura cost per second of shift

DATASETS = {
    # The retina "within" part uses the same table and the same random draws as step 3, so its
    # B1, B2, T1, T2 and C rows must reproduce results/identification.md.
    "retina": dict(
        label="retina (frozen noise, 10 s)", trial_s=9.94,
        within=dict(path=None, session=None, query=None, n_trials=30),
        # The unrepeated noise only exists for one recording, so the across part uses that one.
        across=dict(path=ROOT / "data" / "retina_running_20180126_sr_le_sp_clean.parquet",
                    session_a="20180126_sr_le_sp", query_a=None, n_a=30,
                    session_b="20180126_sr_le_sp_running", query_b="piece == 0", n_b=30,
                    name_a="frozen noise", name_b="unrepeated noise")),
    "neuropixels": dict(
        label="Neuropixels (natural movie, 30 s)", trial_s=30.0,
        within=dict(path=ROOT / "data" / "allen" / "allen829720705.parquet",
                    session="allen829720705_movie", query="start < 3600", n_trials=30),
        across=dict(path=ROOT / "data" / "allen" / "allen829720705.parquet",
                    session_a="allen829720705_movie", query_a="start < 3600", n_a=30,
                    session_b="allen829720705_spont", query_b="start < 2280", n_b=15,
                    name_a="natural movie", name_b="grey screen")),
}

STEP3_ROWS = ["B1 firing rate", "B2 ISI histogram", "T1 LINE H0", "T2 DELAY H0", "T2 DELAY H1",
              "T2 DELAY H0+H1", "C1 DELAY H0+H1, ISIs shuffled"]
ORDER = (["B1 firing rate", "B2 ISI histogram"] + [f"B3 Victor-Purpura, q = {q:g}/s" for q in Q_VALUES]
         + ["T1 LINE H0", "T1 LINE H0, log scale", "T1 LINE H0, log scale, share",
            "T2 DELAY H0", "T2 DELAY H1", "T2 DELAY H0+H1",
            "C1 DELAY H0, ISIs shuffled", "C1 DELAY H0+H1, ISIs shuffled",
            "C2 LINE H0, random time cutoff", "C2 DELAY H0, random time cutoff",
            "C3 DELAY H0, share"])


def all_distances(isis, spikes, duration, seed, labels=None):
    """{method name: (D, M)} for one list of samples.

    D is the n x n distance matrix. M (distances to the leave-one-out neuron means, for the
    nearest-mean score of step 3) is only computed when `labels` is given, and only for the
    methods that have a mean; otherwise it is None.
    `duration` is one number, or one number per sample (seconds).
    """
    n = len(isis)
    duration = np.broadcast_to(np.asarray(duration, dtype=float), (n,))
    out = {}

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

    for name, (D, _) in out.items():
        assert D.shape == (n, n) and np.allclose(D, D.T) and np.allclose(np.diag(D), 0), f"{name}: bad distance matrix"
    return out


def run_within(cfg, args, seed):
    w = cfg["within"]
    data = load_trials(args.n_neurons, args.min_rate, w["n_trials"], seed=seed, path=w["path"],
                       session=w["session"], query=w["query"], trial_s=cfg["trial_s"])
    labels = data["labels"]
    dists = all_distances(data["isis"], data["spikes"], cfg["trial_s"], seed, labels)
    rows = []
    for name, (D, M) in dists.items():
        rows.append(dict(method=name, seed=seed, nn1=score_1nn(D, labels),
                         kmedoids=score_kmedoids(D, labels, args.kmedoids_seeds),
                         nearest_mean=score_nearest_mean(M, labels) if M is not None else np.nan))
    return rows


def run_across(cfg, args, seed):
    c = cfg["across"]
    A, B = load_pair(args.n_neurons, args.min_rate, c["n_a"], c["n_b"], seed=seed, path_a=c["path"],
                     session_a=c["session_a"], path_b=c["path"], session_b=c["session_b"],
                     query_a=c["query_a"], query_b=c["query_b"], trial_s=cfg["trial_s"])
    isis, spikes = A["isis"] + B["isis"], A["spikes"] + B["spikes"]
    labels = np.concatenate([A["labels"], B["labels"]])
    ia, ib = np.arange(len(A["labels"])), len(A["labels"]) + np.arange(len(B["labels"]))
    dists = all_distances(isis, spikes, cfg["trial_s"], seed)       # no labels: no neuron means
    rows = []
    for name, (D, _) in dists.items():
        rows.append(dict(method=name, seed=seed, **scores_across(D, labels, ia, ib, args.kmedoids_seeds)))
    return rows


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


def figure_bars(table, cols, labs, chance, title, name):
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(table))
    w = 0.8 / len(cols)
    for j, (col, lab) in enumerate(zip(cols, labs)):
        ax.bar(x + (j - (len(cols) - 1) / 2) * w, 100 * table[f"{col}_mean"], w,
               yerr=100 * table[f"{col}_std"].fillna(0), capsize=2, label=lab)
    ax.axhline(100 * chance, color="k", ls="--", lw=1, label=f"chance ({100 * chance:.1f} %)")
    ax.set_xticks(x)
    ax.set_xticklabels(table["method"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)          # room for the legend above the bars
    ax.legend(loc="upper center", ncol=len(cols) + 1, fontsize=10)
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=list(DATASETS), required=True)
    p.add_argument("--part", choices=["within", "across", "both"], default="both")
    p.add_argument("--n-neurons", type=int, default=40)
    p.add_argument("--min-rate", type=float, default=1.0)
    p.add_argument("--seeds", type=int, default=5, help="number of random neuron draws")
    p.add_argument("--kmedoids-seeds", type=int, default=3, help="k-medoids runs averaged per score")
    a = p.parse_args()
    cfg = DATASETS[a.dataset]
    chance = 1 / a.n_neurons
    RESULTS_DIR.mkdir(exist_ok=True)
    head = f"{a.n_neurons} neurons, mean ± std over {a.seeds} neuron draws. Chance = {100 * chance:.1f} %."
    vp = []

    if a.part in ("within", "both"):
        rows = []
        for seed in range(a.seeds):
            t0 = time.time()
            rows += run_within(cfg, a, seed)
            print(f"{a.dataset} within, draw {seed}: {time.time() - t0:.0f} s", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_DIR / f"extended_{a.dataset}_within_runs.csv", index=False)
        cols = ["nn1", "kmedoids", "nearest_mean"]
        table = summarise(df, cols)
        table.to_csv(RESULTS_DIR / f"extended_{a.dataset}_within.csv", index=False)
        write_table(table, cols, ["1-NN", "k-medoids", "nearest mean"],
                    RESULTS_DIR / f"extended_{a.dataset}_within.md",
                    f"WITHIN one stimulus. {cfg['label']}: accuracy (%), {cfg['within']['n_trials']} samples per "
                    f"neuron, {head}")
        figure_bars(table, ["nn1", "kmedoids"], ["1-NN", "k-medoids"], chance,
                    f"{cfg['label']}: which neuron is this sample from? (same stimulus)", f"ext_{a.dataset}_within.png")
        vp += [(table, "nn1", "within, 1-NN"), (table, "kmedoids", "within, k-medoids")]

    if a.part in ("across", "both"):
        c = cfg["across"]
        rows = []
        for seed in range(a.seeds):
            t0 = time.time()
            rows += run_across(cfg, a, seed)
            print(f"{a.dataset} across, draw {seed}: {time.time() - t0:.0f} s", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_DIR / f"extended_{a.dataset}_across_runs.csv", index=False)
        cols = ["nn_within", "kmed_within", "nn_across", "kmed_across"]
        table = summarise(df, cols)
        table.to_csv(RESULTS_DIR / f"extended_{a.dataset}_across.csv", index=False)
        write_table(table, cols, [f"1-NN within {c['name_a']}", "k-medoids within",
                                  f"1-NN {c['name_a']} -> {c['name_b']}", "k-medoids across"],
                    RESULTS_DIR / f"extended_{a.dataset}_across.md",
                    f"ACROSS stimuli. {a.dataset}: A = {c['name_a']} ({c['n_a']} samples per neuron), "
                    f"B = {c['name_b']} ({c['n_b']} samples per neuron). Accuracy (%), {head}")
        figure_bars(table, ["nn_within", "nn_across"], [f"1-NN within {c['name_a']}", f"1-NN across to {c['name_b']}"],
                    chance, f"{a.dataset}: same stimulus against a new stimulus", f"ext_{a.dataset}_across.png")
        vp += [(table, "nn_across", "across, 1-NN")]

    figure_vp(vp, chance, f"{a.dataset}: Victor-Purpura distance against its time precision", f"ext_{a.dataset}_vp.png")


if __name__ == "__main__":
    main()
