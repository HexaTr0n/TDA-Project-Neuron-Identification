"""Single uncoupled Izhikevich neurons. After a spike only u survives, so with constant
input the ISIs follow a 1D map. Claims tested:
 1. Without noise, H0 of the m=2 cloud counts the orbit's period.
 2. Noise and stimulus drive destroy that, and ISI memory falls with drive."""
from lab import *
import embeddings as em
from exp1_memory import gains
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

DT = 0.05  # ms

def simulate(a, b, c, d, I, seconds, noise=0.0, drive=None, seed=0, burn=1000.0):
    """All arguments are arrays over neurons, or scalars. Returns a list of spike-time arrays
    in seconds, measured after the burn-in."""
    n = len(a); rng = np.random.default_rng(seed)
    steps = int((seconds * 1000 + burn) / DT)
    v = np.full(n, -65.0); u = b * v
    out = [[] for _ in range(n)]
    for s in range(steps):
        i = I + (drive[s] if drive is not None else 0.0)
        if noise:
            i = i + noise * rng.normal(size=n) / np.sqrt(DT)
        v = v + DT * (0.04 * v ** 2 + 5 * v + 140 - u + i)
        u = u + DT * a * (b * v - u)
        f = v >= 30
        if f.any():
            if s * DT >= burn:
                for k in np.flatnonzero(f):
                    out[k].append((s * DT - burn) / 1000)
            v[f] = c[f]; u[f] += d[f]
    return [np.array(x) for x in out]

def period(isi, tol=0.015, pmax=12):
    x = isi[-60:]
    if len(x) < 40:
        return None
    for p in range(1, pmax + 1):
        if np.all(np.abs(x[p:] - x[:-p]) <= tol * x[:-p] + 2 * DT / 1000):
            return p
    return None

def components(isi, tau, m=2):
    if len(isi) < m + 2:
        return 0
    # H0 at scale tau is the number of connected components, so no boundary matrix is needed.
    # The full routine also built H1, which is very slow on a periodic orbit full of tied distances.
    c = em.isi_delay(isi, m=m)
    return int(connected_components(csr_matrix(cdist(c, c) <= tau), directed=False)[0])

TYPES = {"regular": (.02, .2, -65, 8), "intrinsic burst": (.02, .2, -55, 4), "chattering": (.02, .2, -50, 2),
         "fast": (.1, .2, -65, 2), "low threshold": (.02, .25, -65, 2)}

print("=== 1. noiseless: named types", flush=True)
P = np.array(list(TYPES.values())); zero = 10.0
tr = simulate(P[:, 0], P[:, 1], P[:, 2], P[:, 3], np.full(len(P), zero), 20)
for (name, _), t in zip(TYPES.items(), tr):
    i = np.diff(t)[-300:]
    print(f"  {name:16s} spikes {len(t)}  period {period(i)}  H0 components tau=.1 {components(i, .1)}  tau=.3 {components(i, .3)}", flush=True)

print("=== 1b. noiseless: random parameters", flush=True)
r = np.random.default_rng(3); N = 800
a, b = r.uniform(.01, .1, N), r.uniform(.2, .25, N)
c, d = r.uniform(-65, -48, N), r.uniform(.05, 8, N)
I = r.uniform(4, 20, N)
tr = simulate(a, b, c, d, I, 8, seed=1)
rows = []
for t in tr:
    if len(t) < 60:
        continue
    i = np.diff(t)[-300:]
    rows.append(dict(p=period(i), c1=components(i, .1), c3=components(i, .3)))
t = pd.DataFrame(rows)
print("  with enough spikes", len(t), " no period within 12:", int(t.p.isna().sum()), flush=True)
k = t.dropna()
print(k.groupby("p").agg(n=("p", "size"), exact_tau_1=("c1", lambda s: 0), ).assign(
    exact_tau_1=k.groupby("p").apply(lambda g: (g.c1 == g.name).mean(), include_groups=False).round(3),
    exact_tau_3=k.groupby("p").apply(lambda g: (g.c3 == g.name).mean(), include_groups=False).round(3)).to_string(), flush=True)
print("  overall exact match tau=.1", round((k.c1 == k.p).mean(), 3), " tau=.3", round((k.c3 == k.p).mean(), 3), flush=True)

print("=== 2. noise and stimulus drive, 10 s trials", flush=True)
def memory(t):
    if len(t) < 32:
        return np.nan
    x = np.log(np.maximum(np.diff(t), 1e-4))[:500]
    g = lambda z: (lambda res: res[("cond", 1, 0.5, 0.3)] - res[("marginal", 0, 0.0, 0.3)])(gains(z))
    return g(x) - g(np.random.default_rng(0).permutation(x))

steps = int((10 * 1000 + 1000) / DT)
out = []
for kind, levels in (("noise", (0.0, 0.5, 1, 2, 4, 8)), ("drive", (0.0, 1, 2, 4, 8, 16))):
    for lev in levels:
        rep = 12
        A = np.repeat(P, rep, axis=0); names = np.repeat(list(TYPES), rep)
        Iv = np.full(len(A), zero)
        drive = None
        if kind == "drive" and lev:
            g = np.random.default_rng(7); tau = 20.0
            drive = np.zeros(steps); e = g.normal(size=steps)
            for s in range(1, steps):
                drive[s] = drive[s - 1] * (1 - DT / tau) + np.sqrt(2 * DT / tau) * lev * e[s]
        tr = simulate(A[:, 0], A[:, 1], A[:, 2], A[:, 3], Iv, 10, noise=(lev if kind == "noise" else 0.15), drive=drive, seed=int(lev * 10) + 5)
        for nm, t in zip(names, tr):
            out.append(dict(kind=kind, level=lev, type=nm, spikes=len(t), comps=components(np.diff(t), .3) if len(t) > 3 else 0, memory=memory(t)))
        print(kind, lev, "done", flush=True)
o = pd.DataFrame(out); OUT.mkdir(exist_ok=True, parents=True); o.to_csv(OUT / "izhikevich.csv", index=False)
print(o.groupby(["kind", "level"]).agg(spikes=("spikes", "median"), components=("comps", "median"), memory=("memory", "mean")).round(3).to_string())
print(o.pivot_table(index=["kind", "level"], columns="type", values="memory").round(3).to_string())
print(o.pivot_table(index=["kind", "level"], columns="type", values="comps", aggfunc="median").round(1).to_string())
