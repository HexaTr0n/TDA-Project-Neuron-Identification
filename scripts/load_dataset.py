"""Load the frozen-noise ISI dataset (mouse retinal ganglion cells,
Karamanlis & Gollisch 2021) built by build_isi_dataset.py.

One row per (neuron, trial). Each trial is one ~9.94 s presentation of the
same frozen white-noise movie, so the trials of a neuron are repeats.

Columns:
    neuron_id    unique id "<session>_C<channel>_<cluster>"
    session      recording session
    trial        0..n-1, in time order (30-34 trials per neuron)
    isis         np.ndarray of interspike intervals, in seconds
    spike_times  np.ndarray of spike times from trial start, in seconds
Trials with fewer than 2 spikes have an empty `isis` array.

Usage in Python (copy this file next to your code, or add scripts/ to sys.path):
    from load_dataset import load_dataset, isis_by_neuron
    df = load_dataset("data/isi_frozennoise.parquet")    # or the .csv
    isis = isis_by_neuron(df)        # {neuron_id: [array per trial, ...]}

From the command line, to print a summary:
    python scripts/load_dataset.py data/isi_frozennoise.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "isi_frozennoise.parquet"
LIST_COLUMNS = ["isis", "spike_times"]


def _parse(cell):
    """CSV cells are semicolon-separated numbers; an empty cell is an empty list."""
    if not isinstance(cell, str) or cell == "":
        return np.empty(0)
    return np.array(cell.split(";"), dtype=float)


def load_dataset(path=DEFAULT_PATH, sessions=None, min_rate=None):
    """Load the dataset as a DataFrame whose `isis`/`spike_times` are numpy arrays.

    path      .parquet (faster) or .csv
    sessions  optional list of session names to keep
    min_rate  optional: drop neurons whose mean rate (Hz) is below this
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        for c in LIST_COLUMNS:
            df[c] = df[c].map(lambda x: np.asarray(x, dtype=float))
    else:
        df = pd.read_csv(path, keep_default_na=False)
        for c in LIST_COLUMNS:
            df[c] = df[c].map(_parse)

    if sessions is not None:
        df = df[df["session"].isin(sessions)]
    if min_rate is not None:
        rates = neuron_summary(df)["rate_hz"]
        df = df[df["neuron_id"].isin(rates.index[rates >= min_rate])]
    return df.sort_values(["neuron_id", "trial"]).reset_index(drop=True)


def isis_by_neuron(df, pooled=False):
    """{neuron_id: list of per-trial ISI arrays}, or one concatenated array if pooled."""
    out = {nid: list(g["isis"]) for nid, g in df.groupby("neuron_id", sort=True)}
    if pooled:
        out = {nid: np.concatenate(v) for nid, v in out.items()}
    return out


def spike_trains(df, neuron_id):
    """List of spike-time arrays (one per trial) for one neuron, in trial order."""
    return list(df.loc[df["neuron_id"] == neuron_id].sort_values("trial")["spike_times"])


def neuron_summary(df, trial_duration=9.94):
    """Per-neuron table: session, number of trials, spikes, ISIs, mean rate (Hz)."""
    d = df.assign(n_spikes=df["spike_times"].map(len), n_isis=df["isis"].map(len))
    s = d.groupby("neuron_id").agg(session=("session", "first"), n_trials=("trial", "size"),
                                   n_spikes=("n_spikes", "sum"), n_isis=("n_isis", "sum"))
    s["rate_hz"] = s["n_spikes"] / (s["n_trials"] * trial_duration)
    return s


if __name__ == "__main__":
    df = load_dataset(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH)
    s = neuron_summary(df)
    print(f"{len(df)} rows, {len(s)} neurons, {df['session'].nunique()} sessions")
    print(s.groupby("session").agg(neurons=("n_trials", "size"), trials=("n_trials", "max"),
                                   median_rate_hz=("rate_hz", "median")).round(2).to_string())
    nid = s.index[0]
    print(f"\nexample: {nid}, trial 0 -> {len(df.isis[0])} ISIs, first 5: {np.round(df.isis[0][:5], 4)}")
