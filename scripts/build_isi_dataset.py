"""Build the frozen-noise ISI dataset (one row per neuron x frozen trial).

For every extracted session that has a frozen-noise stimulus
(stimpara.stimulus == 'frozencheckerflicker'), every good unit and every
complete frozen segment, spikes in [ftimes[s], ftimes[s+F]) are shifted to
start at 0 and converted to interspike intervals.

Frozen segment k starts at pulse s = R + k*(R+F), with R = RunningFrames and
F = FrozenFrames counted in frame pulses (checked per session, see
check_session). Sessions that fail a consistency check abort the run.

Usage:
    python scripts/build_isi_dataset.py --extracted-dir extracted \
        --out data/isi_frozennoise.csv --fig-dir figures
"""
import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio

PULSE_MS = (32.0, 34.5)  # expected pulse interval: 30 Hz (pulseRate=2 at 60 Hz)


def read_stimpara(path):
    """Read the `stimpara` struct of a v7.3 .mat file; uint16 arrays are strings."""
    out = {}
    with h5py.File(path, "r") as f:
        for k, v in f["stimpara"].items():
            a = np.array(v)
            out[k] = "".join(map(chr, a.ravel())) if a.dtype == np.uint16 else a.ravel()
            if not isinstance(out[k], str) and out[k].size == 1:
                out[k] = out[k].item()
    return out


def read_ftimes(path):
    """Frame-pulse times in ms. Most files are MATLAB v5, some are v7.3."""
    try:
        return sio.loadmat(path)["ftimes"].ravel().astype(float)
    except NotImplementedError:
        with h5py.File(path, "r") as f:
            return np.array(f["ftimes"]).ravel().astype(float)


def read_good_cells(path, min_quality):
    """Columns: channel, cluster[, quality]. A quality filter is applied only
    if a third column exists (none of the published sessions have one: every
    listed unit is a good unit, per the manual)."""
    rows = np.atleast_2d(np.loadtxt(path))
    if rows.shape[1] >= 3 and min_quality is not None:
        rows = rows[rows[:, 2] >= min_quality]
    return [(int(r[0]), int(r[1])) for r in rows]


def read_spikes(path):
    """Spike times (s); files are either one per line or comma-separated."""
    txt = path.read_text().replace(",", " ").split()
    return np.sort(np.array(txt, dtype=float))


def find_frozen_stimulus(session):
    for p in sorted((session / "stimuli").glob("*.mat")):
        para = read_stimpara(p)
        if para.get("stimulus") == "frozencheckerflicker":
            n = p.name.split("_")[0]
            ft = list((session / "frametimes").glob(f"{n}_*frametimings.mat"))
            if len(ft) != 1:
                raise RuntimeError(f"{session.name}: expected 1 frametimes file for stimulus {n}, got {ft}")
            return n, para, ft[0]
    return None


def check_session(name, para, ftimes):
    """Check that R and F are in pulses and return the trial start indices."""
    R, F = int(para["RunningFrames"]), int(para["FrozenFrames"])
    nblinks, pulse_rate = int(para["Nblinks"]), int(para.get("pulseRate", 2))
    dt = np.diff(ftimes)
    med = np.median(dt)
    problems = []
    if not PULSE_MS[0] <= med <= PULSE_MS[1]:
        problems.append(f"median pulse interval {med:.2f} ms is not ~33 ms")
    # The original name gives run/freeze in units of stimulus updates
    # (Nblinks monitor frames each); one pulse every `pulse_rate` monitor frames.
    m = re.search(r"(\d+)run(\d+)freeze", para.get("originalname", ""))
    if m:
        r_name, f_name = (int(x) * nblinks / pulse_rate for x in m.groups())
        if (r_name, f_name) != (R, F):
            problems.append(f"name implies R,F={r_name:g},{f_name:g} pulses but stimpara has {R},{F}")
    if np.any(dt > 1.9 * med):
        problems.append(f"missing pulses at indices {np.where(dt > 1.9 * med)[0].tolist()}")
    if problems:
        raise SystemExit(f"STOP: session {name} looks inconsistent: " + "; ".join(problems))

    starts = []
    k = 0
    while R + k * (R + F) + F < len(ftimes):  # need ftimes[start+F] as trial end
        starts.append(R + k * (R + F))
        k += 1
    late = np.where(dt > 1.4 * med)[0]
    in_trial = [i for i in late if any(s <= i < s + F for s in starts)]
    durs = np.array([ftimes[s + F] - ftimes[s] for s in starts]) / 1000
    print(f"  R={R} F={F} pulses | Nblinks={nblinks} | pulse interval {med:.2f} ms "
          f"| trial duration {durs.mean():.3f} s (range {durs.min():.3f}-{durs.max():.3f}) "
          f"| {len(ftimes)} pulses -> {len(starts)} complete trials")
    if len(late):
        print(f"  note: {len(late)} late pulse(s) (>1.4x interval) at {late.tolist()}, "
              f"{len(in_trial)} inside frozen trials")
    return R, F, starts


def fmt(x):
    return ";".join(f"{v:.6f}" for v in x)


def raster(df_neuron, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    trains = [np.asarray(t) for t in df_neuron.sort_values("trial")["spike_times_list"]]
    ax.eventplot(trains, colors="k", linelengths=0.8, linewidths=0.6)
    ax.set_xlabel("time in frozen trial (s)")
    ax.set_ylabel("trial")
    ax.set_title(df_neuron["neuron_id"].iloc[0])
    ax.set_xlim(0, max((t.max() for t in trains if len(t)), default=10))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def reliability(trains, dur, bin_s=0.05):
    """Mean pairwise correlation between trial PSTHs: >0 if stimulus-locked."""
    bins = np.arange(0, dur + bin_s, bin_s)
    c = np.array([np.histogram(t, bins)[0] for t in trains], float)
    c = c[c.std(axis=1) > 0]
    if len(c) < 2:
        return np.nan
    r = np.corrcoef(c)
    return r[np.triu_indices(len(c), 1)].mean()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extracted-dir", type=Path, default=Path("extracted"))
    p.add_argument("--out", type=Path, default=Path("data/isi_frozennoise.csv"))
    p.add_argument("--fig-dir", type=Path, default=Path("figures"))
    p.add_argument("--min-quality", type=float, default=None,
                   help="keep units with column-3 quality >= this (only if the column exists)")
    p.add_argument("--n-rasters", type=int, default=3)
    p.add_argument("--min-rate", type=float, default=0.5, help="flag neurons below this rate (Hz)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rows, used, skipped = [], [], []
    for session in sorted(d for d in a.extracted_dir.iterdir() if d.is_dir() and not d.name.startswith(".")):
        found = find_frozen_stimulus(session)
        if found is None:
            skipped.append(session.name)
            continue
        n, para, ft_path = found
        ftimes = read_ftimes(ft_path)
        print(f"{session.name}: stimulus {n} ({para.get('originalname', '')})")
        R, F, starts = check_session(session.name, para, ftimes)
        t0, t1 = ftimes[[0, -1]] / 1000

        cells = read_good_cells(session / "list_of_good_cells.txt", a.min_quality)
        nspk = 0
        for ch, cl in cells:
            f = session / "spiketimes" / f"{n}_SP_C{ch}{cl:02d}.txt"
            if not f.exists():
                raise SystemExit(f"STOP: missing spike file {f}")
            spikes = read_spikes(f)
            if len(spikes) and (spikes.min() < t0 - 60 or spikes.max() > t1 + 60):
                raise SystemExit(f"STOP: {f} spikes outside the frame-time range (clock mismatch?)")
            nid = f"{session.name}_C{ch}_{cl}"
            for k, s in enumerate(starts):
                lo, hi = ftimes[s] / 1000, ftimes[s + F] / 1000
                st = spikes[(spikes >= lo) & (spikes < hi)] - lo
                nspk += len(st)
                rows.append(dict(neuron_id=nid, session=session.name, trial=k,
                                 isis_list=np.diff(st), spike_times_list=st, duration=hi - lo))
        used.append((session.name, len(cells), len(starts), nspk))
        print(f"  {len(cells)} good units, {len(starts)} trials, {nspk} spikes in frozen trials\n")

    df = pd.DataFrame(rows)
    out = df[["neuron_id", "session", "trial"]].copy()
    out["isis"] = df["isis_list"].map(fmt)
    out["spike_times"] = df["spike_times_list"].map(fmt)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    pq = df[["neuron_id", "session", "trial"]].assign(
        isis=df["isis_list"].map(list), spike_times=df["spike_times_list"].map(list))
    pq.to_parquet(a.out.with_suffix(".parquet"), index=False)

    # ---- sanity checks ----
    print("=" * 70)
    print(f"{'session':22s} {'units':>6s} {'trials':>7s} {'spikes':>9s}")
    for s, nu, nt, ns in used:
        print(f"{s:22s} {nu:6d} {nt:7d} {ns:9d}")
    print(f"skipped (no frozen noise): {', '.join(skipped) or 'none'}")

    df["nspk"] = df["spike_times_list"].map(len)
    per = df.groupby("neuron_id").agg(session=("session", "first"), trials=("trial", "size"),
                                      spikes=("nspk", "sum"), time=("duration", "sum"))
    per["rate_hz"] = per["spikes"] / per["time"]
    dur = df["duration"].min()
    per["reliability"] = df.groupby("neuron_id")["spike_times_list"].apply(
        lambda ts: reliability(list(ts), dur))
    per.drop(columns="time").to_csv(a.out.with_name("frozennoise_neuron_summary.csv"))
    low = per[per["rate_hz"] < a.min_rate]
    print(f"\nneurons: {len(per)} | rate (Hz) median {per.rate_hz.median():.2f}, "
          f"IQR {per.rate_hz.quantile(.25):.2f}-{per.rate_hz.quantile(.75):.2f}")
    print(f"flagged < {a.min_rate} Hz: {len(low)}")
    for nid, r in low.iterrows():
        print(f"  {nid}: {r.rate_hz:.3f} Hz")
    print(f"trials with < 2 spikes (empty ISI list): {(df.nspk < 2).sum()} / {len(df)}")
    print(f"trial-to-trial PSTH correlation (50 ms bins): median {per.reliability.median():.3f}")

    a.fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for nid in rng.choice(per.index.to_numpy(), a.n_rasters, replace=False):
        raster(df[df.neuron_id == nid], a.fig_dir / f"raster_{nid}.png")
        print(f"raster: {a.fig_dir / f'raster_{nid}.png'} (rate {per.rate_hz[nid]:.2f} Hz, "
              f"reliability {per.reliability[nid]:.3f})")
    print(f"\nwrote {a.out} ({len(out)} rows) and {a.out.with_suffix('.parquet')}")


if __name__ == "__main__":
    main()
