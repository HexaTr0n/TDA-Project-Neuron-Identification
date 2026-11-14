"""Task L3 data: frozen trials and the unrepeated ("running") noise before them, per neuron.

The frozen-noise stimulus alternates R pulses of unrepeated noise and F pulses of the frozen
movie (build_isi_dataset.py). Frozen trial k starts at pulse s = R + k (R + F); the R pulses
before it, [s - R, s), are noise the retina sees only once. This script cuts each such stretch
into pieces of --piece seconds (5 pieces of 10 s for R = 50 s) and writes one table with two
sessions:

    <session>            the frozen trials, exactly as in data/isi_frozennoise.parquet
    <session>_running    the unrepeated pieces; trial = 5 k + j, block = k (the frozen trial
                         that follows), piece = j

Neuron ids are the same in both, so curve_eval.py --cross-session and sb_cross.py can ask
whether a frozen trial finds the same neuron's unrepeated pieces.

--cut A B removes seconds [A, B) of every running stretch and joins the rest before cutting
pieces (L24: seconds 24.8-30.0 hold a fixed 5 s chunk that repeats in every block; with
--cut 24.8 30.0 the 49.65 s stretch becomes 44.45 s of noise seen once, i.e. 4 clean 10 s
pieces). Output then goes to data/retina_running_<session>_clean.parquet.

Run from the repo root, after extracting the session's .rar into extracted/:
    python scripts/load_retina_running.py --extracted-dir extracted --session 20180126_sr_le_sp
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_isi_dataset import check_session, find_frozen_stimulus, read_ftimes, read_good_cells, read_spikes


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extracted-dir", type=Path, default=Path("extracted"))
    ap.add_argument("--session", default="20180126_sr_le_sp")
    ap.add_argument("--piece", type=float, default=10.0, help="seconds per unrepeated piece")
    ap.add_argument("--out", type=Path, help="default: data/retina_running_<session>[_clean].parquet")
    ap.add_argument("--cut", type=float, nargs=2, metavar=("A", "B"),
                    help="drop seconds [A, B) of every running stretch and join the rest")
    args = ap.parse_args()

    session = args.extracted_dir / args.session
    n, para, ft_path = find_frozen_stimulus(session)
    ftimes = read_ftimes(ft_path)
    R, F, starts = check_session(args.session, para, ftimes)
    rows = []
    for ch, cl in read_good_cells(session / "list_of_good_cells.txt", None):
        spikes = read_spikes(session / "spiketimes" / f"{n}_SP_C{ch}{cl:02d}.txt")
        nid = f"{args.session}_C{ch}_{cl}"
        for k, s in enumerate(starts):
            lo, hi = ftimes[s] / 1000, ftimes[s + F] / 1000
            t = spikes[(spikes >= lo) & (spikes < hi)] - lo
            rows.append(dict(neuron_id=nid, session=args.session, trial=k, block=k, piece=-1,
                             isis=np.diff(t), spike_times=t, duration=hi - lo))
            r0, r1 = ftimes[s - R] / 1000, lo
            run = spikes[(spikes >= r0) & (spikes < r1)] - r0
            length = r1 - r0
            if args.cut:
                a0, a1 = args.cut
                run = np.r_[run[run < a0], run[run >= a1] - (a1 - a0)]
                length -= a1 - a0
            for j in range(int(length // args.piece)):
                a = j * args.piece
                t = run[(run >= a) & (run < a + args.piece)] - a
                rows.append(dict(neuron_id=nid, session=f"{args.session}_running", trial=5 * k + j,
                                 block=k, piece=j, isis=np.diff(t), spike_times=t,
                                 duration=args.piece))
    df = pd.DataFrame(rows)
    out = args.out or Path("data") / f"retina_running_{args.session}{'_clean' if args.cut else ''}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    s = df.assign(n=df["spike_times"].map(len)).groupby("session").agg(
        neurons=("neuron_id", "nunique"), samples=("trial", "size"),
        seconds=("duration", "median"), median_spikes=("n", "median"))
    print(s.to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
