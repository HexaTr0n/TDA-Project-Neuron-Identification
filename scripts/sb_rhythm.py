"""Task L12: rhythm signatures, hippocampal and cortical units separately. Stablebear for every
barcode summary and Rips.

Per sample (spikes in [0, duration)):
  - autocorrelogram (ACG): coincidences at lags 1-500 ms, 1 ms bins, Gaussian smoothing 3 ms,
    1 for a Poisson train of the same rate (as exp21_stimulus_free.acg, longer lags so that a
    theta period of about 125 ms fits several times);
      * curve: L1 between ACGs (no topology, reference);
      * bars, lags dropped: peaks and dips of the ACG as bars [0, prominence], stable ranks in
        stablebear, L1 (peaks + dips);
      * peaks matched in lag: peaks with prominence >= 0.3 as (lag, prominence) events, matched
        with spike_distance.matching_matrix (q = 50 per s, as exp21);
  - spectrum windows (extra_embeddings.windowed_spectrum): Gaussian windows sigma 0.5 s every
    0.5 s, log power at 24 frequencies 1-100 Hz relative to Poisson, windows with effective
    spike count >= 4. Cloud position (mean point, Euclidean) and H0 (Rips in stablebear on
    --points random windows, --subsamples times, stable ranks averaged).
  - ISI fractions (stablebear) as the reference.
Control: ISIs permuted within each sample (all signatures).

Samples need >= --min-isis ISIs and, for spectra, >= --points good windows (otherwise the
spectrum rows use only those that qualify, and the counts are printed). Neurons: units of the
group with >= 10 usable samples, 60 at random (seed 0). 1-NN and k-medoids.

Run from the repo root:
    python scripts/sb_rhythm.py --data data/openscope/glo637908.parquet --session glo637908_grating --group hippocampus
    python scripts/sb_rhythm.py --data data/openscope/glo637908.parquet --session glo637908_grating --group cortex
    (same for allen829720705_movie / allen829720705_spont; retina: --group all --join-pieces)
Writes results/rhythm_<session>_<group>.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curve_eval import kmedoids, l1_pdist, matched, nn_score, read
from sb_cross import isi_fraction_pcfs
from sb_length import stretches
from spike_distance import matching_matrix

RESULTS = Path(__file__).resolve().parent.parent / "results"
HIPPOCAMPUS = ("CA1", "CA2", "CA3", "DG", "SUB", "ProS", "POST", "HPF")
MAX_LAG, DT, SMOOTH = 0.5, 0.001, 3.0


def in_group(area, group):
    a = str(area)
    if group == "all":
        return True
    if group == "hippocampus":
        return a.startswith(HIPPOCAMPUS)
    if group == "cortex":
        return a.startswith("VIS")
    raise ValueError(group)


def samples(df, join_pieces, group):
    """(neuron_id, spike_times, duration)."""
    if join_pieces:
        return stretches(df, True)
    return [(r.neuron_id, np.sort(np.asarray(r.spike_times, float)), float(r.duration))
            for r in df.itertuples(index=False) if in_group(getattr(r, "area", ""), group)]


def shuffled(t, rng):
    if len(t) < 3:
        return t
    return t[0] + np.r_[0, np.cumsum(rng.permutation(np.diff(t)))]


def acg(t, dur):
    lags = []
    for k in range(1, len(t)):
        d = t[k:] - t[:-k]
        d = d[d < MAX_LAG]
        if not len(d):
            break
        lags.append(d)
    lags = np.concatenate(lags) if lags else np.zeros(0)
    c = np.histogram(lags, np.arange(0, MAX_LAG + DT / 2, DT))[0].astype(float)
    return gaussian_filter1d(c, SMOOTH, mode="nearest") / max(len(t) ** 2 * DT / dur, 1e-9)


def acg_features(a):
    pk = find_peaks(a)[0]
    dp = find_peaks(-a)[0]
    pp = peak_prominences(a, pk)[0] if len(pk) else np.zeros(0)
    dd = peak_prominences(-a, dp)[0] if len(dp) else np.zeros(0)
    keep = pp >= 0.3
    return pp, dd, ((pk[keep] + 1) * DT, pp[keep])


def stable_rank_pcfs(values):
    """Bars [0, v] -> stablebear stable ranks."""
    from stablebear import persistence
    from stablebear.persistence import Barcode, BarcodeTensor

    bars = [np.c_[np.zeros(len(v)), v] if len(v) else np.zeros((1, 2)) for v in values]
    return persistence.barcode_to_stable_rank(BarcodeTensor([Barcode(b) for b in bars]))


def sb_l1(pcfs):
    import stablebear as sb
    return np.asarray(sb.pdist(pcfs if not isinstance(pcfs, list) else sb.PcfTensor(pcfs), p=1).to_dense(), float)


def spectrum(t, dur, sigma=0.5, fmin=1.0, fmax=100.0, nf=24, min_eff=4.0):
    """Good windows of extra_embeddings.windowed_spectrum, for any duration."""
    centres = np.arange(sigma, dur - sigma / 2, sigma)
    freqs = np.geomspace(fmin, fmax, nf)
    if len(t) < 5:
        return np.zeros((0, nf))
    out = []
    for c0 in range(0, len(centres), 64):  # chunks keep the window x spike matrix small
        c = centres[c0:c0 + 64]
        lo, hi = np.searchsorted(t, c[0] - 5 * sigma), np.searchsorted(t, c[-1] + 5 * sigma)
        ts = t[lo:hi]
        w = np.exp(-(ts[None, :] - c[:, None]) ** 2 / (2 * sigma ** 2))
        S = np.abs(w @ np.exp(-2j * np.pi * freqs[:, None] * ts[None, :]).T) ** 2
        w2 = (w ** 2).sum(1)
        eff = w.sum(1) ** 2 / np.maximum(w2, 1e-12)
        logP = np.log(np.maximum(S / np.maximum(w2, 1e-12)[:, None], 1e-3))
        out.append(logP[eff >= min_eff])
    return np.concatenate(out)


def rips_h0(clouds):
    import stablebear as sb
    from stablebear import persistence

    tensor = sb.zeros((len(clouds), len(clouds[0])), dtype=sb.pcloud64)
    for i, subs in enumerate(clouds):
        for j, c in enumerate(subs):
            tensor[i, j] = c
    bcs = persistence.compute_persistent_homology(tensor, max_dim=0, reduced=True, verbose=False)
    pcfs = persistence.barcode_to_stable_rank(bcs)
    return np.asarray(sb.pdist(sb.mean(pcfs[:, :, 0], dim=1), p=1).to_dense(), float)


def score(D, y, k, seed):
    rng = np.random.default_rng(seed)
    nn = nn_score(D.copy(), y, rng)
    Dm = D.copy()
    np.fill_diagonal(Dm, 0)
    return nn, matched(y, kmedoids(Dm, k, rng), k)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--group", choices=["hippocampus", "cortex", "all"], default="all")
    ap.add_argument("--join-pieces", action="store_true")
    ap.add_argument("--min-isis", type=int, default=100)
    ap.add_argument("--points", type=int, default=40)
    ap.add_argument("--subsamples", type=int, default=10)
    ap.add_argument("--neurons", type=int, default=60)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    S = [s for s in samples(read(args.data, [args.session]), args.join_pieces, args.group)
         if len(s[1]) - 1 >= args.min_isis]
    counts = pd.Series([n for n, _, _ in S]).value_counts()
    names = sorted(counts.index[counts >= args.min_samples])
    keep = set(rng.choice(names, min(args.neurons, len(names)), replace=False).tolist())
    S = [s for s in S if s[0] in keep]
    y = np.unique([n for n, _, _ in S], return_inverse=True)[1]
    k = y.max() + 1
    print(f"{args.session} {args.group}: {len(names)} units with >= {args.min_samples} samples of "
          f">= {args.min_isis} ISIs; {k} neurons, {len(S)} samples, median "
          f"{np.median([len(t) for _, t, _ in S]):.0f} spikes, {np.median([d for _, _, d in S]):.0f} s; "
          f"chance {100 / k:.1f}%", flush=True)

    rows = []

    def add(name, shuf, D, yy=None):
        yy = y if yy is None else yy
        kk = len(np.unique(yy))
        nn, km = score(D, np.unique(yy, return_inverse=True)[1], kk, args.seed)
        rows.append(dict(session=args.session, group=args.group, signature=name, shuffled=shuf,
                         nn=nn, kmedoids=km, neurons=kk, samples=len(yy)))
        print(f"  {name:<32} {'shuffled' if shuf else 'real':<8} 1-NN {100 * nn:5.1f}  "
              f"k-medoids {100 * km:5.1f}  ({len(yy)} samples)", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS / f"rhythm_{args.session}_{args.group}.csv", index=False)

    r = np.random.default_rng(args.seed + 1)
    for shuf in (False, True):
        trains = [shuffled(t, r) if shuf else t for _, t, _ in S]
        durs = [d for _, _, d in S]
        add("ISI fractions", shuf, sb_l1(isi_fraction_pcfs(trains)))
        A = [acg(t, d) for t, d in zip(trains, durs)]
        add("ACG curve", shuf, l1_pdist(np.array(A)))
        F = [acg_features(a) for a in A]
        Dbars = sb_l1(stable_rank_pcfs([f[0] for f in F])) + sb_l1(stable_rank_pcfs([f[1] for f in F]))
        add("ACG bars, lags dropped", shuf, Dbars)
        add("ACG peaks matched in lag", shuf, matching_matrix([f[2] for f in F], 50.0, jobs=args.jobs))
        spec = [spectrum(t, d) for t, d in zip(trains, durs)]
        if not shuf:  # the real spectra choose the samples; the control uses the same ones
            ok = np.array([len(s) >= args.points for s in spec])
            ys = y[ok]
            ys_ok = pd.Series(ys).map(pd.Series(ys).value_counts()) >= 2
            ok[np.flatnonzero(ok)[~ys_ok.to_numpy()]] = False
            idx = np.flatnonzero(ok)
        idx = np.array([i for i in idx if len(spec[i]) >= args.points])
        pos = np.array([spec[i].mean(0) for i in idx])
        add("spectrum position", shuf, np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1)), y[idx])
        clouds = [[spec[i][r.choice(len(spec[i]), args.points, replace=False)] for _ in range(args.subsamples)]
                  for i in idx]
        add("spectrum H0", shuf, rips_h0(clouds), y[idx])
        Dpos = np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1))
        add("ACG bars + spectrum position", shuf,
            Dbars[np.ix_(idx, idx)] / np.median(Dbars[Dbars > 0]) + Dpos / np.median(Dpos[Dpos > 0]), y[idx])

    print("\n" + pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
