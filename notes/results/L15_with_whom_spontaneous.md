# L15 With whom, on spontaneous activity

2026-11-12, local session. **Partly worked.**

- (a) **Worked on Allen.** A unit's correlations with 300 partners identify it at 30.4% 1-NN
  in a 30 s grey-screen window, against 14.9% with partners jittered by +-50 ms and 1.9% with
  partners from another window. On OpenScope's 10 s windows it gets 10.3%.
- (b) **Weak.** Clusters follow brain areas above shuffled (ARI 0.24 Allen, 0.08 OpenScope), but
  average linkage makes one giant cluster.
- (c) **Worked.** Millisecond lead-lag recovers known pathways: CA3 -> CA1, CA3 -> DG, CA1 -> SUB,
  VISp -> VISl -> VISam, VISp -> SCig. There are no loops.

## Data

- **Allen** FC session 829720705, spontaneous (grey screen):
  - 77 windows of 30 s (`allen829720705_spont`, 2389 s in all);
  - quality-good units: 903 with >= 1 Hz.
- **OpenScope** `glo637908`, grey intermissions:
  - 50 windows of 10 s (`glo637908_spont`, 500 s);
  - 1346 units with >= 1 Hz.
- Areas without layers (VISp5 -> VISp, DG-mo -> DG).
- Coarse groups: visual cortex, hippocampus (CA, DG, SUB, ProS, POST), thalamus, midbrain.

## Command

```sh
python scripts/sb_partners.py --data data/allen/allen829720705.parquet --session allen829720705_spont
python scripts/sb_partners.py --data data/openscope/glo637908.parquet --session glo637908_spont
```

Parts (a) and (b) come from the first run. Part (c) was rerun with 20 jitter controls
(`--parts c`), replacing a first version with one control.

## (a) Partner profile per sample

**Samples.**
- A sample is one unit in one window.
- Allen: 60 neurons (`curve_eval.select`, >= 30 ISIs, seed 0), 3745 samples.
- OpenScope: 60 neurons, 1757 samples.

**Profile.**
- Pearson correlation, 20 ms bins, of the unit with each of 300 partners in the same window.
- Partners are 300 random units with >= 1 Hz, the same set for the whole session; the unit itself is set to 0.
- Profiles are compared by L1.

**"Who dropped"** keeps only how strongly the unit correlates: the positive correlations become
bars [0, r], and stable ranks are computed in stablebear (`sb.pdist`).

**Scores.** 1-NN / k-medoids, %; chance 1.7%.

| | Allen 30 s | OpenScope 10 s |
|---|---|---|
| ISI fractions (reference, stablebear) | 38.5 / 24.5 | 32.5 / 22.2 |
| **partner profile** | **30.4 / 19.8** | 10.3 / 13.1 |
| partner profile, partners jittered +-50 ms | 14.9 / 14.3 | 4.6 / 10.0 |
| partner profile, partners from the next window | 1.9 / 4.9 | 1.7 / 7.6 |
| sorted correlations (who dropped) | 10.8 / 12.1 | 6.0 / 11.6 |
| sorted, jittered | 7.4 / 11.2 | 4.7 / 11.2 |
| sorted, next window | 2.6 / 6.8 | 2.0 / 8.7 |

## (b) Do neurons group by area?

**Method.**
- Distance 1 - r between units over the whole spontaneous period (20 ms bins).
- Average linkage, cut into as many clusters as there are areas.
- Null: 200 shuffled labellings.

| | units | groups | ARI | shuffled mean / 95% | largest clusters | best ARI (k) |
|---|---|---|---|---|---|---|
| Allen, area | 903 | 11 | **0.243** | 0.000 / 0.006 | 487, 240, 68, 40, 35 | 0.243 (13) |
| Allen, coarse group | 903 | 3 | 0.155 | 0.001 / 0.017 | 848, 44, 11 | 0.724 (9) |
| OpenScope, area | 1346 | 25 | 0.077 | 0.000 / 0.003 | 748, 437, 62, 25, 11 | 0.177 (100) |
| OpenScope, coarse group | 1346 | 4 | -0.002 | 0.000 / 0.002 | 1340, 3, 2, 1 | 0.033 (16) |

## (c) Flow between areas

**Lead between two units a and b.**
- A[a, b] is the number of spikes of b that fall 0.5-5 ms after a spike of a.
- The lead is Y = (A[a, b] - A[b, a]) / sqrt(A[a, b] + A[b, a]).

**Flow between areas.**
- Flow = Y averaged over all unit pairs between two areas, then antisymmetrised.
- An area's net lead is its mean flow to the other areas.
- Controls: 20 copies with every spike jittered by +-25 ms; z-scores are taken against those copies.
- Only areas with >= 10 units are used.

**Net lead** (positive = fires first). Allen, 900 units in 10 areas:

| area | units | net lead | z | | area | units | net lead | z |
|---|---|---|---|---|---|---|---|---|
| POST | 12 | 0.105 | 6.7 | | VISl | 117 | -0.014 | -2.3 |
| CA3 | 52 | 0.096 | 21.8 | | VISam | 113 | -0.018 | -4.9 |
| VISrl | 110 | 0.030 | 2.8 | | SCig | 43 | -0.032 | -4.9 |
| VISp | 75 | 0.019 | 5.3 | | CA1 | 181 | -0.077 | -13.9 |
| VISal | 110 | -0.006 | -3.4 | | DG | 87 | -0.103 | -12.0 |

**Edges.** Flows with z > 4, strongest first:

| Allen | OpenScope |
|---|---|
| CA3 -> DG (z 27.3) | CA1 -> SUB (z 65.8) |
| DG -> CA1 (23.1) | APN -> POST (20.6) |
| CA3 -> CA1 (22.4) | CA1 -> ProS (12.0) |
| VISp -> SCig (9.8) | ProS -> SUB (11.7) |
| VISrl -> VISal (8.8) | APN -> MB (8.0) |
| POST -> SCig (6.3) | CA3 -> DG (5.1) |
| VISl -> VISam (6.2) | SUB -> DG (4.9) |
| CA1 -> VISrl (5.6) | MGd -> MB, MGm -> MGd (4.7) |
| VISal -> VISam (4.4) | VISrl -> DG (4.7) |
| POST -> DG (4.1) | VISp -> VISpm (4.5) |
| VISp -> VISl (4.0) | APN -> CA1 (4.3), SCig -> POST (4.2) |

**Loops.** No cycle in either graph.

**Visual hierarchy** (Siegle et al. 2021), Spearman of rank against -net lead over the visual
areas present:
- Allen: 0.50 (p = 0.39, 5 areas). VISp, VISrl > VISal, VISl > VISam.
- OpenScope: 0.80 (p = 0.10, 5 areas). VISp, VISrl > VISl > VISam > VISpm.

## What to take from it

- **(a) "With whom" carries identity even without a stimulus**, and half of it is millisecond
  timing. On Allen, jittering the partners halves the score (30.4 -> 14.9), and partners from
  another window give chance (1.9). The structure is in which partners a unit fires with: with
  partner identities dropped (sorted correlations) it falls to 10.8%.
- **(a) Caveat, not controlled:** partners recorded on nearby channels share spikes through spike
  sorting and common local input, so part of the profile may be probe position. The parquet
  has no channel positions; a control would drop partners within ~100 um of the unit.
- **(a) on OpenScope (10 s windows)** is weak (10.3%), as for everything else on 10 s of cortex.
- **(b) Areas are visible in correlations, but average linkage chains them**: one cluster holds
  more than half the units. At 9 clusters the coarse groups reach ARI 0.72 on Allen.
  On OpenScope (25 small areas, 10 s windows) it is weaker.
- **(c) Lead-lag recovers known anatomy without any stimulus:**
  - CA3 -> CA1 (Schaffer collaterals), CA1 -> SUB and CA1 -> ProS (hippocampal output);
  - VISp -> VISl -> VISam and VISrl -> VISal -> VISam (feedforward visual hierarchy);
  - VISp -> SCig (cortex to superior colliculus).
  The visual-area ordering matches the hierarchy in sign on both sessions, but with 5 areas the
  rank correlation is not significant.
- **What failed or is odd:**
  - CA3 -> DG and POST -> DG are not classic forward pathways; DG mossy cells and back-projections
    exist, but cross-area lead-lag can also reflect shared drive with different latencies.
  - There are no loops above z = 4.
  - The OpenScope net-lead ranking puts midbrain (APN) and CA1 first, which needs a closer look.
