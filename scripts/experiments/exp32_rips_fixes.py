"""Three fixes for what Rips cannot do. Retina, 60 neurons, whole trials (no subsampling).

Run from scripts/experiments: python exp32_rips_fixes.py
A. Time-filtered complex (from the proposal slides). Fix a Rips scale r in ISI-pattern space and
   let the delay points enter in spike order. A component is born when a pattern appears that is
   unlike every earlier one (further than r), and dies when a later point joins it to an older
   component. Bars live on the time axis, so this is merging homology that keeps time.
     Betti curve over time   = number of distinct pattern groups alive at each moment
     births over time        = how many new kinds of pattern have appeared so far
     lifetimes (stable rank) = how long groups stay separate, times dropped
B. H1 on the whole cloud inside regions (8 anchors), as a stable rank and as a Betti curve.
   The experiment Rips code is slow for H1, so a region with more than 80 points is cut to 80
   random points for H1 only (H0 always uses every point).
Scores: 1-NN and k-medoids on full trials; across movie halves. Shuffled control: ISIs permuted.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
src = (HERE / "exp30_cloud_sampling.py").read_text()
exec(src[:src.index("rows = []")])
sys.path.insert(0, str(HERE))
from tda import rips_matrix

TG = np.arange(0.01, T + 0.01, 0.02)


def time_filtered(t, r):
    """Bars (birth, death) of H0 when delay points enter in time order at Rips scale r."""
    c = delay(t); when = t[M:]                          # a pattern is complete at its last spike
    D = cdist(c, c) <= r
    parent = np.arange(len(c)); born = when.copy(); bars = []
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for k in range(len(c)):
        roots = {find(j) for j in np.flatnonzero(D[k, :k])}
        if not roots:
            continue
        old = min(roots, key=lambda q: born[q])
        parent[k] = old
        for q in roots - {old}:
            bars.append((born[q], when[k])); parent[q] = old
    alive = {find(i) for i in range(len(c))}
    return np.array(bars + [(born[q], np.inf) for q in alive]).reshape(-1, 2)


def tf_features(times, r, dur, shuffle):
    B, N, L = [], [], []
    lg = np.linspace(0, dur, 200, endpoint=False)
    for t in times:
        if shuffle:
            t = t[0] + np.r_[0, np.cumsum(rng.permutation(np.diff(t)))]
        b = time_filtered(t, r); g = TG[TG <= dur + 0.01]
        B.append(((b[:, 0][None] <= g[:, None]) & (g[:, None] < b[:, 1][None])).sum(1))
        N.append((b[:, 0][None] <= g[:, None]).sum(1))
        life = np.minimum(b[:, 1], dur) - b[:, 0]
        L.append((life[None, :] > lg[:, None]).sum(1))
    return dict(betti=np.stack(B).astype(np.float32), births=np.stack(N).astype(np.float32), lifetimes=np.stack(L).astype(np.float32))


BG = np.linspace(0, 3.0, 128, endpoint=False)
def region_h1(t, shuffle):
    if shuffle:
        t = t[0] + np.r_[0, np.cumsum(rng.permutation(np.diff(t)))]
    c = delay(t); lab = np.linalg.norm(c[:, None] - ANCH[None], axis=2).argmin(1)
    h0, sr, bc = [], [], []
    for r in range(K):
        p = c[lab == r]; h0.append(rank(h0_bars(p)))
        if len(p) > 80:
            p = p[rng.choice(len(p), 80, replace=False)]
        if len(p) >= 4:
            _, b, d = rips_matrix(cdist(p, p))
        else:
            b = d = np.zeros(0)
        sr.append(((d - b)[None, :] > BG[:, None]).sum(1)); bc.append(((b[None, :] <= BG[:, None]) & (BG[:, None] < d[None, :])).sum(1))
    return np.concatenate(h0), np.concatenate(sr).astype(np.float32), np.concatenate(bc).astype(np.float32)


h0_bars = h0
rows = []
def score(name, F, Fh):
    D = cdist(F, F, "cityblock"); Dh = cdist(Fh, Fh, "cityblock")
    rows.append(dict(method=name, full_1nn=nn(D, y), full_cluster=rp.matched_accuracy(y, rp.k_medoids(D, y.max() + 1)),
                     cross_1nn=nn(Dh, hy, cross), cross_cluster=transfer(Dh, hy, half)))
    print({k: (round(float(v), 3) if k != "method" else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).round(3).to_csv(HERE / "rips_fixes.csv", index=False)

def sc(X):
    i = rng.choice(len(X), 300, replace=False); return X / max(np.median(cdist(X[i], X[i], "cityblock")), 1e-9)

print("A. time-filtered complex", flush=True)
for r in (0.5, 1.0, 2.0):
    for shuffle in (False, True):
        F = tf_features(full, r, T, shuffle); Fh = tf_features(halves, r, T / 2, shuffle)
        for k in ("betti", "births", "lifetimes"):
            score(f"time-filtered r={r}: {k}" + (", ISIs shuffled" if shuffle else ""), F[k], Fh[k])
        if not shuffle:
            score(f"time-filtered r={r}: betti + births", np.hstack([sc(F["betti"]), sc(F["births"])]), np.hstack([sc(Fh["betti"]), sc(Fh["births"])]))
print("B. H1 inside regions, whole cloud", flush=True)
for shuffle in (False, True):
    tag = ", ISIs shuffled" if shuffle else ""
    R = [region_h1(t, shuffle) for t in full]; Rh = [region_h1(t, shuffle) for t in halves]
    S = lambda rs, i: np.stack([x[i] for x in rs])
    score("regions whole: H0" + tag, S(R, 0), S(Rh, 0))
    score("regions whole: H1 stable rank" + tag, S(R, 1), S(Rh, 1))
    score("regions whole: H1 Betti curve" + tag, S(R, 2), S(Rh, 2))
    score("regions whole: H0 + H1 stable rank" + tag, np.hstack([sc(S(R, 0)), sc(S(R, 1))]), np.hstack([sc(S(Rh, 0)), sc(S(Rh, 1))]))
    score("regions whole: H0 + H1 Betti curve" + tag, np.hstack([sc(S(R, 0)), sc(S(R, 2))]), np.hstack([sc(S(Rh, 0)), sc(S(Rh, 2))]))
