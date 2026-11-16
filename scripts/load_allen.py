"""Task L5 data: one Allen Visual Coding Neuropixels session (functional connectivity set,
DANDI 000022) as the pipeline's table.

Unlike OpenScope, this set has a frozen repeat: natural movie one (30 s, 900 frames) is shown
60 times in "natural_movie_one_more_repeats". Two sessions in one file:

    allen<id>_movie   one sample per movie repeat, from its frame 0, cut to the shortest
                      repeat (about 30 s); times from the repeat's start
    allen<id>_spont   the spontaneous (grey screen) periods cut into windows of the same length

Columns as load_openscope.py: neuron_id, session, trial, isis, spike_times, duration, block,
start, area, layer, opto_z, waveform_duration, cell_type. Good units only.

With --all-stimuli, every other stimulus of the session is added too, as
    allen<id>_<stimulus>   each block of that stimulus cut into windows of the movie's length
                           (gratings, flashes, dot motion, the shuffled movie, ...; receptive
                           field mapping and invalid times are left out)

Run from the repo root:
    python scripts/load_allen.py data/allen/sub-811322604_ses-829720705.nwb --inspect
    python scripts/load_allen.py data/allen/sub-811322604_ses-829720705.nwb
    python scripts/load_allen.py data/allen/sub-811322604_ses-829720705.nwb --all-stimuli
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_openscope import blocked_spans, build_table, inspect, read_nwb, save, summary, window_samples

MOVIE_TABLE = "natural_movie_one_more_repeats_presentations"
SPONT_TABLE = "spontaneous_presentations"


def movie_samples(table, avoid):
    t = table.sort_values("start_time")
    starts = t.loc[t["frame"].astype(float) == 0, "start_time"].to_numpy(float)
    ends = np.r_[starts[1:], t["stop_time"].max()]
    ends = np.minimum(ends, starts + 31.0)                  # a repeat is 30 s; gaps excluded
    d = float(np.floor(np.min(ends - starts) * 100) / 100)
    keep = [(k, float(s)) for k, (s, e) in enumerate(zip(starts, ends))
            if not any(s < b1 and b0 < s + d for b0, b1 in avoid)]
    return [(k, s, d) for k, s in keep], f"{len(starts)} movie repeats, {len(keep)} kept, each {d:.2f} s"


def stimulus_blocks(table, gap=5.0):
    """[(start, stop)] of the blocks of one stimulus table: by `stimulus_block` if the table
    has it, otherwise presentations closer than `gap` seconds are joined."""
    t = table.sort_values("start_time")
    if "stimulus_block" in t:
        g = t.groupby("stimulus_block")
        return sorted(zip(g["start_time"].min().astype(float), g["stop_time"].max().astype(float)))
    out = []
    for a, b in zip(t["start_time"].astype(float), t["stop_time"].astype(float)):
        if out and a - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def other_stimuli(intervals, avoid, window):
    """{stimulus name: [(-1, start, window), ...]} for every presentation table except the movie
    repeats, the grey screen, receptive field mapping (gabors) and invalid times."""
    out = {}
    for name, t in intervals.items():
        if not name.endswith("_presentations") or name in (MOVIE_TABLE, SPONT_TABLE):
            continue
        if re.search(r"invalid|receptive_field|gabor", name) or not len(t) or "start_time" not in t:
            continue
        spans = pd.DataFrame(stimulus_blocks(t), columns=["start_time", "stop_time"])
        samples, _ = window_samples(spans, avoid, window)
        if samples:
            out[name[: -len("_presentations")]] = samples
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("nwb")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--all-stimuli", action="store_true",
                    help="also add every other stimulus of the session, cut into windows")
    args = ap.parse_args()
    if args.inspect:
        inspect(args.nwb)
        return
    d = read_nwb(args.nwb)
    sid = re.search(r"ses-(\d+)", Path(args.nwb).name).group(1)
    avoid = blocked_spans(d["intervals"], d["opto"])
    movie, note_m = movie_samples(d["intervals"][MOVIE_TABLE], avoid)
    dur = movie[0][2]
    spont, note_s = window_samples(d["intervals"][SPONT_TABLE], avoid, dur)
    sessions = {f"allen{sid}_movie": movie, f"allen{sid}_spont": spont}
    if args.all_stimuli:
        for name, samples in other_stimuli(d["intervals"], avoid, dur).items():
            sessions[f"allen{sid}_{name}"] = samples
            print(f"{name}: {len(samples)} windows of {dur:g} s, first at {samples[0][1]:.0f} s, "
                  f"last at {samples[-1][1]:.0f} s")
    df = build_table(d["units"], d["electrodes"], sessions, d["opto"], sid)
    df["neuron_id"] = df["neuron_id"].str.replace(f"glo{sid}_", f"allen{sid}_", regex=False)
    out = Path(args.out) if args.out else Path(args.nwb).with_name(f"allen{sid}.parquet")
    save(df, out)
    print(note_m)
    print(note_s)
    print(f"genotype: {d['genotype']}")
    print(summary(df).to_string())
    per = df.drop_duplicates("neuron_id")
    print("cell types:", per["cell_type"].value_counts().to_dict())
    print("areas:", per["area"].value_counts().head(20).to_dict())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
