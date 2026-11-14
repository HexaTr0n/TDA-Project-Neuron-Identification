"""Task L5 data: one Allen Visual Coding Neuropixels session (functional connectivity set,
DANDI 000022) as the pipeline's table.

Unlike OpenScope, this set has a frozen repeat: natural movie one (30 s, 900 frames) is shown
60 times in "natural_movie_one_more_repeats". Two sessions in one file:

    allen<id>_movie   one sample per movie repeat, from its frame 0, cut to the shortest
                      repeat (about 30 s); times from the repeat's start
    allen<id>_spont   the spontaneous (grey screen) periods cut into windows of the same length

Columns as load_openscope.py: neuron_id, session, trial, isis, spike_times, duration, block,
start, area, layer, opto_z, waveform_duration, cell_type. Good units only.

Run from the repo root:
    python scripts/load_allen.py data/allen/sub-811322604_ses-829720705.nwb --inspect
    python scripts/load_allen.py data/allen/sub-811322604_ses-829720705.nwb
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("nwb")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--out")
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
