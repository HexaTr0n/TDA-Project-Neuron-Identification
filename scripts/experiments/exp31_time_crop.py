"""Follow-up to exp30: one barcode per trial, no averaging, spike count allowed in.

Run from the repo root: python scripts/experiments/exp31_time_crop.py
  whole       every point of the trial
  time crop   a random amount of time cut off the start and the end, keeping 70% or 50% of the
              trial; one draw, every point of what is left (the user's suggestion)
and, for the whole cloud, more regions (8, 16, 32 anchors), since it has more points to fill them.
Delay cloud m = 3, H0 by minimum spanning tree, stable rank, L1. Retina, 60 neurons.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
src = (HERE / "exp30_cloud_sampling.py").read_text()
exec(src[:src.index("rows = []")])          # data, delay(), h0(), rank(), nn(), transfer(), halves


def crop(t, keep, dur):
    start = rng.uniform(0, dur * (1 - keep))
    return t[(t >= start) & (t < start + keep * dur)]


def sig(c, anchors):
    if anchors is None:
        return rank(h0(c))
    lab = np.linalg.norm(c[:, None] - anchors[None], axis=2).argmin(1)
    return np.concatenate([rank(h0(c[lab == r])) for r in range(len(anchors))])


def feats(times, dur, keep, anchors, shuffle):
    out = []
    for t in times:
        if keep < 1:
            t = crop(t, keep, dur)
        if len(t) < M + 2:
            out.append(np.zeros(len(GRID) * (1 if anchors is None else len(anchors)), np.float32)); continue
        if shuffle:
            t = t[0] + np.r_[0, np.cumsum(rng.permutation(np.diff(t)))]
        out.append(sig(delay(t), anchors))
    return np.stack(out)


ANCHORS = {8: ANCH}
for k in (16, 32):
    ANCHORS[k] = KMeans(k, n_init=3, random_state=0).fit(P[rng.choice(len(P), 8000, replace=False)]).cluster_centers_
rows = []
for label, anchors in (("delay Rips", None), ("delay regions, 8 anchors", ANCHORS[8]), ("delay regions, 16 anchors", ANCHORS[16]), ("delay regions, 32 anchors", ANCHORS[32])):
    for keep in (1.0, 0.7, 0.5):
        if keep < 1 and anchors is not None and len(anchors) != 8:
            continue
        for shuffle in (False, True):
            name = f"{label} H0, " + ("whole" if keep == 1 else f"time crop keeping {int(keep * 100)}%") + (", ISIs shuffled" if shuffle else "")
            F = feats(full, T, keep, anchors, shuffle); Fh = feats(halves, T / 2, keep, anchors, shuffle)
            D = cdist(F, F, "cityblock"); Dh = cdist(Fh, Fh, "cityblock")
            rows.append(dict(method=name, full_1nn=nn(D, y), full_cluster=rp.matched_accuracy(y, rp.k_medoids(D, y.max() + 1)),
                             cross_1nn=nn(Dh, hy, cross), cross_cluster=transfer(Dh, hy, half)))
            print({k: (round(float(v), 3) if k != "method" else v) for k, v in rows[-1].items()}, flush=True)
tab = pd.DataFrame(rows).round(3); tab.to_csv(HERE / "time_crop.csv", index=False)
