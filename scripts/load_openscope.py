"""Turn one OpenScope Global/Local Oddball session (DANDI 000253) into the pipeline's table.

The retina data had one frozen movie, repeated about 30 times. This session has no such
repeat, so a "trial" here is one of two kinds of sample, kept as two sessions in one file:

    glo<subject>_grating   one sample per grating block (125 presentations, about 125 s),
                           cut to the median block length; shorter blocks are dropped.
                           Times are from block start.
    glo<subject>_spont     the 10 s grey-screen intermissions between grating blocks, cut
                           into windows of --spont-window seconds. The file's own
                           spontaneous_presentations table is mostly 0.5 s gaps (and two
                           rows with stop < start), so it is not used.

One presentation is only 0.5 s and holds a few spikes, so consecutive presentations of the
same block are joined into one sample.

The same unit has the same neuron_id in both, so cross_epoch.py can ask whether a neuron's
spontaneous windows find its grating windows. Columns are those of load_dataset.py
(neuron_id, session, trial, isis, spike_times), plus duration, block, start, area, layer
(cortical layer parsed from the area name, "" outside cortex), opto_z, waveform_duration and
cell_type. The mouse is Pvalb-Cre;Ai32, so light drives PV interneurons: cell_type is
"PV_tagged" when opto_z > OPTO_Z, otherwise "narrow" or "wide" by waveform duration.

Run from the repo root:
    python scripts/load_openscope.py data/openscope/sub-637908_ses-1213341633_ogen.nwb --inspect
    python scripts/load_openscope.py data/openscope/sub-637908_ses-1213341633_ogen.nwb

The second line writes data/openscope/glo637908.parquet. Then, for example:
    python scripts/run_pipeline.py --data data/openscope/glo637908.parquet \\
        --session glo637908_grating --neurons 60 time_bars_s67 isi_bars_log isi_histogram
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

GRATING_TABLE = "init_grating_presentations"
SPONT_TABLE = "init_intermission_presentations"
OPTO_Z = 5.0            # opto_z above this: the unit follows the light
NARROW_MS = 0.4         # waveform_duration below this (ms): narrow spike, putative fast spiking
LAYER = re.compile(r"^(VIS\w*?)(1|2/3|4|5|6a|6b)$")
BIG_COLUMNS = {"waveform_mean", "waveform_sd", "waveforms", "spike_amplitudes"}


# ------------------------------------------------------------------ reading the NWB file
# Everything that touches pynwb is in this section, so a wrong field name fails here.


def _table(t, exclude=()):
    try:
        return t.to_dataframe(exclude=set(exclude) & set(t.colnames))
    except TypeError:                       # older hdmf without `exclude`
        return t.to_dataframe()


def read_nwb(path):
    from pynwb import NWBHDF5IO

    io = NWBHDF5IO(str(path), "r", load_namespaces=True)
    nwb = io.read()
    out = dict(subject=getattr(nwb.subject, "subject_id", None) if nwb.subject else None,
               genotype=getattr(nwb.subject, "genotype", None) if nwb.subject else None)
    out["units"] = _table(nwb.units, BIG_COLUMNS)
    out["electrodes"] = _table(nwb.electrodes) if nwb.electrodes is not None else None
    out["intervals"] = {name: _table(t) for name, t in nwb.intervals.items()}
    out["opto"] = None
    try:
        out["opto"] = _table(nwb.processing["optotagging"]["optogenetic_stimulation"])
    except (KeyError, AttributeError):
        pass
    out["_io"] = io                         # keep the file open while spike_times are read
    return out


def inspect(path):
    d = read_nwb(path)
    u = d["units"]
    print(f"subject: {d['subject']}, genotype: {d['genotype']}")
    print(f"\nunits: {u.shape}\ncolumns: {list(u.columns)}")
    print(u.drop(columns=[c for c in ("spike_times",) if c in u]).iloc[0].to_string())
    if "quality" in u:
        print("quality:", u["quality"].value_counts().to_dict())
    if d["electrodes"] is not None:
        e = d["electrodes"]
        print(f"\nelectrodes: {e.shape}, columns {list(e.columns)}")
        if "location" in e:
            print("locations:", e["location"].value_counts().to_dict())
    for name, t in d["intervals"].items():
        print(f"\ninterval table {name}: {t.shape}\ncolumns: {list(t.columns)}")
        print(t.head(3).to_string())
    if d["opto"] is not None:
        print(f"\noptotagging: {d['opto'].shape}, columns {list(d['opto'].columns)}")
        print(d["opto"].head(3).to_string())


# ------------------------------------------------------------------ samples


def overlaps(a0, a1, spans):
    return any(a0 < b1 and b0 < a1 for b0, b1 in spans)


def blocked_spans(intervals, opto):
    """Periods to avoid: invalid times, receptive field mapping, and the optotagging run."""
    spans = []
    for name, t in intervals.items():
        if re.search(r"invalid|receptive_field", name) and len(t):
            spans += list(zip(t["start_time"], t["stop_time"]))
    if opto is not None and len(opto):
        end = opto["start_time"] + (opto["duration"] if "duration" in opto else 1.0)
        spans.append((float(opto["start_time"].min()) - 1.0, float(end.max()) + 1.0))
    return spans


def grating_samples(table, avoid, duration=None, same_sequence=False):
    """[(block, start, duration)], one per grating block, and a description."""
    rows = []
    for block, g in table.groupby("stimulus_block", sort=True):
        g = g.sort_values("start_time")
        start, stop = float(g["start_time"].min()), float(g["stop_time"].max())
        seq = tuple(np.round(pd.to_numeric(g["orientation"], errors="coerce").fillna(-1), 1)) \
            if "orientation" in g else ()
        if not overlaps(start, stop, avoid):
            rows.append((block, start, stop, seq))
    if not rows:
        sys.exit(f"no usable grating blocks in {GRATING_TABLE}")
    common, n_common = Counter(r[3] for r in rows).most_common(1)[0]
    note = (f"{len(rows)} grating blocks, {len({r[3] for r in rows})} distinct orientation "
            f"sequences; the most common one is shared by {n_common} blocks")
    if same_sequence:
        rows = [r for r in rows if r[3] == common]
        note += "; keeping only those"
    d = duration or float(np.median([stop - start for _, start, stop, _ in rows]))
    rows = [r for r in rows if r[2] - r[1] >= d - 1e-6]
    return [(int(b), s, d) for b, s, _, _ in rows], note + f"; {len(rows)} blocks of at least {d:.2f} s kept, each cut to that"


def window_samples(table, avoid, window):
    """[(-1, start, window)]: spontaneous periods cut into windows, leftovers dropped."""
    out = []
    for s, e in zip(table["start_time"], table["stop_time"]):
        for k in range(int((e - s) // window)):
            a = s + k * window
            if not overlaps(a, a + window, avoid):
                out.append((-1, float(a), float(window)))
    return out, f"{len(out)} spontaneous windows of {window:g} s"


def opto_score(spikes, opto):
    """(spikes during light - spikes just before) / sqrt(sum + 1), over all light pulses."""
    if opto is None or not len(opto):
        return np.nan
    s = opto["start_time"].to_numpy(float)
    d = opto["duration"].to_numpy(float) if "duration" in opto else np.full(len(s), 1.0)
    on = np.searchsorted(spikes, s + d) - np.searchsorted(spikes, s)
    off = np.searchsorted(spikes, s) - np.searchsorted(spikes, s - d)
    return float((on.sum() - off.sum()) / np.sqrt(on.sum() + off.sum() + 1))


def layer_of(area):
    m = LAYER.match(area)
    return m.group(2) if m else ""


def cell_type(opto_z, waveform_duration):
    if opto_z > OPTO_Z:
        return "PV_tagged"
    if not np.isfinite(waveform_duration):
        return "unknown"
    return "narrow" if waveform_duration < NARROW_MS else "wide"


def build_table(units, electrodes, sessions, opto, subject, keep_quality="good"):
    """units: DataFrame with spike_times (and quality, peak_channel_id if present).
    sessions: {session name: [(block, start, duration), ...]}."""
    if keep_quality and "quality" in units:
        units = units[units["quality"].astype(str) == keep_quality]
    area = {}
    if electrodes is not None and "location" in electrodes and "peak_channel_id" in units:
        loc = electrodes["location"].astype(str)
        area = {u: loc.get(int(c), "unknown") for u, c in units["peak_channel_id"].items()}
    rows = []
    for unit, spikes in units["spike_times"].items():
        spikes = np.sort(np.asarray(spikes, dtype=np.float64))
        z = opto_score(spikes, opto)
        a = area.get(unit, "unknown")
        wd = float(units["waveform_duration"].get(unit, np.nan)) if "waveform_duration" in units else np.nan
        extra = dict(area=a, layer=layer_of(a), opto_z=z, waveform_duration=wd,
                     cell_type=cell_type(z, wd))
        for session, samples in sessions.items():
            for trial, (block, start, dur) in enumerate(samples):
                t = spikes[np.searchsorted(spikes, start):np.searchsorted(spikes, start + dur)] - start
                rows.append(dict(neuron_id=f"glo{subject}_u{unit}", session=session, trial=trial,
                                 isis=np.diff(t), spike_times=t, duration=dur, block=block,
                                 start=start, **extra))
    return pd.DataFrame(rows)


def save(df, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:                                   # same format as data/isi_frozennoise.csv
        d = df.copy()
        for c in ("isis", "spike_times"):
            d[c] = d[c].map(lambda a: ";".join(f"{x:.6f}" for x in a))
        d.to_csv(out, index=False)


def summary(df):
    s = df.assign(n=df["spike_times"].map(len)).groupby("session").agg(
        neurons=("neuron_id", "nunique"), samples=("trial", "size"),
        seconds=("duration", "first"), median_spikes=("n", "median"))
    s["samples_per_neuron"] = (s["samples"] / s["neurons"]).round(1)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("nwb")
    ap.add_argument("--inspect", action="store_true", help="print the file's tables and stop")
    ap.add_argument("--out", help="output .parquet or .csv (default: next to the NWB file)")
    ap.add_argument("--spont-window", type=float, default=10.0, help="seconds per spontaneous sample")
    ap.add_argument("--grating-duration", type=float, help="seconds per grating sample (default: shortest block)")
    ap.add_argument("--same-sequence", action="store_true",
                    help="keep only grating blocks with the most common orientation sequence")
    ap.add_argument("--all-units", action="store_true", help="keep units of every quality")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.nwb)
        return
    d = read_nwb(args.nwb)
    subject = d["subject"] or re.search(r"sub-(\w+?)_", Path(args.nwb).name).group(1)
    avoid = blocked_spans(d["intervals"], d["opto"])
    grat, note_g = grating_samples(d["intervals"][GRATING_TABLE], avoid,
                                   args.grating_duration, args.same_sequence)
    sessions = {f"glo{subject}_grating": grat}
    note_s = "no spontaneous table"
    if SPONT_TABLE in d["intervals"]:
        spont, note_s = window_samples(d["intervals"][SPONT_TABLE], avoid, args.spont_window)
        if spont:
            sessions[f"glo{subject}_spont"] = spont
    df = build_table(d["units"], d["electrodes"], sessions, d["opto"], subject,
                     keep_quality=None if args.all_units else "good")
    out = Path(args.out) if args.out else Path(args.nwb).with_name(f"glo{subject}.parquet")
    save(df, out)
    print(note_g)
    print(note_s)
    print(summary(df).to_string())
    per = df.drop_duplicates("neuron_id")
    print("\nunits by area:", per["area"].value_counts().to_dict())
    print(f"genotype: {d['genotype']}")
    print("cell types:", per["cell_type"].value_counts().to_dict())
    print("cortical layers:", per.loc[per["layer"] != "", "layer"].value_counts().to_dict())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
