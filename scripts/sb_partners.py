"""Task L15: "with whom" on spontaneous activity (no stimulus to share).

(a) Partner profile per sample. A sample is one unit in one spontaneous window. Its profile is
    the Pearson correlation of its 20 ms binned train with each of P partner units (fixed for
    the whole session: up to --partners random good units with >= 1 Hz, the unit itself left
    out) in the same window. L1 between profiles. Also the profile with partner identities
    dropped: the correlations as bars [0, r+] -> stable rank in stablebear ("how much, not with
    whom"). Controls: partners jittered by +-50 ms (each spike moved uniformly), and partners
    taken from another spontaneous window (the next one). Neurons: curve_eval.select (>= 30 ISIs,
    60, seed 0); 1-NN and k-medoids. ISI fractions (stablebear) on the same samples as reference.
(b) Do neurons group by brain area? All good units with >= 1 Hz, whole spontaneous period,
    20 ms bins: correlation distance 1 - r, average linkage, cut into as many clusters as there
    are areas; ARI with the areas, against 200 shuffled area labellings. Also with coarse groups
    (visual cortex / hippocampus / thalamus / midbrain).
(c) Flow between areas. For every ordered pair, A[a, b] = spikes of b 0.5-5 ms after a spike of
    a (numba). Lead Y[a, b] = (A[a, b] - A[b, a]) / sqrt(A[a, b] + A[b, a]); area flow
    F[p, q] = mean of Y over units of p and q, antisymmetrised. Control: every spike jittered by
    +-25 ms. An area's net lead = mean of F over the other areas. Compared with the visual
    hierarchy (Siegle et al. 2021 order: LGd, VISp, VISl, VISrl, LP, VISal, VISpm, VISam), and
    a directed graph of flows above 5 times the largest control flow is checked for loops.

Run from the repo root:
    python scripts/sb_partners.py --data data/allen/allen829720705.parquet --session allen829720705_spont
    python scripts/sb_partners.py --data data/openscope/glo637908.parquet --session glo637908_spont
Writes results/partners_<session>_{a,b,c}.csv.
"""
import argparse
import re
import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curve_eval import kmedoids, l1_pdist, matched, nn_score, read, select
from sb_cross import isi_fraction_pcfs

RESULTS = Path(__file__).resolve().parent.parent / "results"
HIERARCHY = ["LGd", "VISp", "VISl", "VISrl", "LP", "VISal", "VISpm", "VISam"]


def base_area(a):
    a = str(a)
    m = re.match(r"^(VIS\w*?)(1|2/3|4|5|6a|6b)$", a)
    if m:
        return m.group(1)
    if a.startswith("DG"):
        return "DG"
    if a.startswith("LGd"):
        return "LGd"
    return a


def coarse(a):
    if a.startswith("VIS"):
        return "visual cortex"
    if a.startswith(("CA", "DG", "SUB", "ProS", "POST", "HPF")):
        return "hippocampus"
    if a in ("LGd", "LP", "MGd", "MGm", "TH", "PoT", "POL", "SGN", "LGv"):
        return "thalamus"
    if a in ("APN", "MB", "MRN", "SCig", "SCop", "NB"):
        return "midbrain"
    return "other"


def binned(trains, dur, bin_s):
    edges = np.arange(0, dur + bin_s / 2, bin_s)
    return np.stack([np.histogram(t, edges)[0] for t in trains]).astype(np.float32)


def corr_rows(X, Y):
    """Pearson correlation of every row of X with every row of Y (constant rows -> 0)."""
    Xc = X - X.mean(1, keepdims=True)
    Yc = Y - Y.mean(1, keepdims=True)
    nx = np.linalg.norm(Xc, axis=1, keepdims=True)
    ny = np.linalg.norm(Yc, axis=1, keepdims=True)
    return (Xc @ Yc.T) / np.maximum(nx * ny.T, 1e-12)


def score(D, y, k, seed):
    rng = np.random.default_rng(seed)
    nn = nn_score(D.copy(), y, rng)
    Dm = D.copy()
    np.fill_diagonal(Dm, 0)
    return nn, matched(y, kmedoids(Dm, k, rng), k)[0]


def part_a(full, args, rng):
    import stablebear as sb
    from stablebear import persistence
    from stablebear.persistence import Barcode, BarcodeTensor

    rows = select(full, args.neurons, args.min_isis, args.seed)
    targets = sorted(set(rows["neuron_id"]))
    dur = float(full["duration"].median())
    rate = full.groupby("neuron_id")["spike_times"].apply(lambda s: sum(len(x) for x in s)) / (dur * full["trial"].nunique())
    pool = sorted(rate.index[rate >= 1.0])
    partners = sorted(rng.choice(pool, min(args.partners, len(pool)), replace=False).tolist())
    trials = sorted(full["trial"].unique())
    by = {tr: g.set_index("neuron_id")["spike_times"] for tr, g in full.groupby("trial")}
    print(f"(a) {len(targets)} neurons, {len(rows)} samples; {len(partners)} partners (>= 1 Hz); "
          f"{len(trials)} windows of {dur:.0f} s", flush=True)

    def profiles(tr, partner_trial, jitter):
        P = []
        for p in partners:
            t = np.asarray(by[partner_trial].get(p, np.zeros(0)), float)
            if jitter:
                t = np.clip(t + rng.uniform(-jitter, jitter, len(t)), 0, dur - 1e-9)
            P.append(t)
        return binned(P, dur, 0.02)

    feats = {k: np.zeros((len(rows), len(partners)), np.float32) for k in ("real", "jittered", "other window")}
    for tr, g in rows.groupby("trial"):
        X = binned([np.asarray(t, float) for t in g["spike_times"]], dur, 0.02)
        nxt = trials[(trials.index(tr) + 1) % len(trials)]
        for name, Pm in (("real", profiles(tr, tr, 0)), ("jittered", profiles(tr, tr, 0.05)),
                         ("other window", profiles(tr, nxt, 0))):
            R = corr_rows(X, Pm)
            for j, nid in enumerate(g["neuron_id"]):
                if nid in partners:
                    R[j, partners.index(nid)] = 0.0  # never the unit itself
            feats[name][g.index.to_numpy()] = R
    y = np.unique(rows["neuron_id"], return_inverse=True)[1]
    k = y.max() + 1
    out = []

    def add(name, control, D):
        nn, km = score(D, y, k, args.seed)
        out.append(dict(session=args.session, signature=name, control=control, nn=nn, kmedoids=km,
                        neurons=k, samples=len(y), partners=len(partners)))
        print(f"  {name:<34} {control:<13} 1-NN {100 * nn:5.1f}  k-medoids {100 * km:5.1f}", flush=True)

    add("ISI fractions (reference)", "real",
        np.asarray(sb.pdist(sb.PcfTensor(isi_fraction_pcfs([np.asarray(t, float) for t in rows["spike_times"]])), p=1).to_dense(), float))
    for control, F in feats.items():
        add("partner profile", control, l1_pdist(F.astype(np.float64)))
        bars = [np.c_[np.zeros(len(r)), np.maximum(r, 0)] for r in F.astype(np.float64)]
        pcfs = persistence.barcode_to_stable_rank(BarcodeTensor([Barcode(b) for b in bars]))
        add("sorted correlations (who dropped)", control, np.asarray(sb.pdist(pcfs, p=1).to_dense(), float))
    pd.DataFrame(out).to_csv(RESULTS / f"partners_{args.session}_a.csv", index=False)


def spont_trains(full, units, gap=1.0):
    """Each unit's spikes on one clock: window w occupies [w (dur + gap), w (dur + gap) + dur)."""
    dur = float(full["duration"].median())
    trials = sorted(full["trial"].unique())
    off = {tr: i * (dur + gap) for i, tr in enumerate(trials)}
    g = full[full["neuron_id"].isin(units)]
    out = {u: [] for u in units}
    for r in g.itertuples(index=False):
        out[r.neuron_id].append(np.asarray(r.spike_times, float) + off[r.trial])
    return {u: np.sort(np.concatenate(v)) if v else np.zeros(0) for u, v in out.items()}, len(trials) * (dur + gap)


def part_b(full, units, areas, args, rng):
    trains, total = spont_trains(full, units)
    X = binned([trains[u] for u in units], total, 0.02)
    D = 1 - corr_rows(X, X).astype(np.float64)  # float32 inside: X is large
    np.fill_diagonal(D, 0)
    D = np.clip((D + D.T) / 2, 0, None)
    Z = linkage(squareform(D, checks=False), "average")
    out = []
    for name, lab in (("area", np.array([areas[u] for u in units])),
                      ("coarse group", np.array([coarse(areas[u]) for u in units]))):
        k = len(np.unique(lab))
        found = fcluster(Z, k, "maxclust")
        ari = adjusted_rand_score(lab, found)
        null = np.array([adjusted_rand_score(rng.permutation(lab), found) for _ in range(200)])
        sizes = np.sort(np.bincount(found))[::-1]
        out.append(dict(session=args.session, labels=name, groups=k, units=len(units), ari=ari,
                        null_mean=null.mean(), null_p95=np.quantile(null, 0.95),
                        largest_clusters=" ".join(map(str, sizes[:5]))))
        print(f"(b) {name}: {k} groups, {len(units)} units, ARI {ari:.3f} (shuffled {null.mean():.3f}, "
              f"95% {np.quantile(null, 0.95):.3f}); largest clusters {sizes[:5].tolist()}", flush=True)
        # cluster size is the usual failure of average linkage: report the k that gives the
        # best ARI as well, so a giant cluster does not hide structure.
        best = max(((adjusted_rand_score(lab, fcluster(Z, kk, "maxclust")), kk) for kk in range(2, 4 * k + 1)))
        out[-1].update(best_ari=best[0], best_k=best[1])
        print(f"    best ARI over 2..{4 * k} clusters: {best[0]:.3f} at {best[1]}", flush=True)
    pd.DataFrame(out).to_csv(RESULTS / f"partners_{args.session}_b.csv", index=False)


@numba.njit(cache=True)
def after_counts(t, u, n, lo, hi):
    """A[a, b] = number of (spike of a, spike of b) with lo < t_b - t_a <= hi."""
    A = np.zeros((n, n), np.int64)
    j0 = 0
    for i in range(len(t)):
        while j0 < len(t) and t[j0] <= t[i] + lo:
            j0 += 1
        j = j0
        while j < len(t) and t[j] <= t[i] + hi:
            A[u[i], u[j]] += 1
            j += 1
    return A


def lead_matrix(trains, units, jitter, rng):
    ts, us = [], []
    for i, un in enumerate(units):
        t = trains[un]
        if jitter:
            t = t + rng.uniform(-jitter, jitter, len(t))
        ts.append(t)
        us.append(np.full(len(t), i, np.int64))
    t = np.concatenate(ts)
    u = np.concatenate(us)
    o = np.argsort(t, kind="stable")
    A = after_counts(t[o], u[o], len(units), 0.0005, 0.005).astype(float)
    Y = (A - A.T) / np.sqrt(np.maximum(A + A.T, 1))
    np.fill_diagonal(Y, 0)
    return Y


def area_flow(Y, lab, names):
    F = np.array([[Y[np.ix_(lab == p, lab == q)].mean() for q in names] for p in names])
    return (F - F.T) / 2


def has_loop(edges, k):
    left = set(range(k))
    edges = set(edges)
    while True:
        free = [v for v in left if not any(b == v and a in left for a, b in edges)]
        if not free:
            return sorted(left)
        left -= set(free)


def part_c(full, units, areas, args, rng):
    trains, _ = spont_trains(full, units)
    lab = np.array([areas[u] for u in units])
    counts = pd.Series(lab).value_counts()
    names = sorted(counts.index[counts >= args.min_area_units])
    keep = np.isin(lab, names)
    units = [u for u, k in zip(units, keep) if k]
    lab = lab[keep]
    Y = lead_matrix(trains, units, 0, rng)
    Yc = lead_matrix(trains, units, 0.025, rng)
    F = area_flow(Y, lab, names)
    Fc = area_flow(Yc, lab, names)
    cut = 5 * np.abs(Fc).max()
    net = F.sum(1) / (len(names) - 1)
    net_c = Fc.sum(1) / (len(names) - 1)
    out = pd.DataFrame(dict(session=args.session, area=names, units=[int(counts[a]) for a in names],
                            net_lead=net, net_lead_control=net_c,
                            hierarchy_rank=[HIERARCHY.index(a) if a in HIERARCHY else np.nan for a in names]))
    out = out.sort_values("net_lead", ascending=False)
    print(f"(c) {len(units)} units in {len(names)} areas with >= {args.min_area_units} units; "
          f"control flow max {np.abs(Fc).max():.4f}", flush=True)
    print(out.round(4).to_string(index=False), flush=True)
    vis = out.dropna(subset=["hierarchy_rank"])
    if len(vis) >= 3:
        rho, p = spearmanr(vis["hierarchy_rank"], -vis["net_lead"])
        print(f"    visual areas: Spearman(hierarchy rank, -net lead) = {rho:.2f} (p = {p:.2f}, {len(vis)} areas)", flush=True)
    edges = [(a, b) for a in range(len(names)) for b in range(len(names)) if F[a, b] > cut]
    loop = has_loop(edges, len(names))
    print(f"    edges above 5 x control: {[(names[a], names[b]) for a, b in edges]}", flush=True)
    print(f"    areas left in a cycle: {[names[i] for i in loop] if loop else 'none'}", flush=True)
    out.to_csv(RESULTS / f"partners_{args.session}_c.csv", index=False)
    pd.DataFrame(F, index=names, columns=names).to_csv(RESULTS / f"partners_{args.session}_c_flow.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--parts", default="abc")
    ap.add_argument("--neurons", type=int, default=60)
    ap.add_argument("--partners", type=int, default=300)
    ap.add_argument("--min-isis", type=int, default=30)
    ap.add_argument("--min-rate", type=float, default=1.0)
    ap.add_argument("--min-area-units", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    full = read(args.data, [args.session])
    rng = np.random.default_rng(args.seed)
    if "a" in args.parts:
        part_a(full, args, rng)
    dur = float(full["duration"].median()) * full["trial"].nunique()
    rate = full.groupby("neuron_id")["spike_times"].apply(lambda s: sum(len(x) for x in s)) / dur
    units = sorted(rate.index[rate >= args.min_rate])
    areas = full.drop_duplicates("neuron_id").set_index("neuron_id")["area"].map(base_area).to_dict()
    if "b" in args.parts:
        part_b(full, units, areas, args, rng)
    if "c" in args.parts:
        part_c(full, units, areas, args, rng)


if __name__ == "__main__":
    main()
