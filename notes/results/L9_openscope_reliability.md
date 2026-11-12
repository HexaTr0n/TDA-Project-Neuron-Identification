# L9 Reliability on OpenScope

2026-11-12, local session. **Worked, with a twist:** joined by condition, OpenScope trials show
a real repeat structure. Time-locked identity follows each unit's reliability (Spearman 0.47),
and the 60 most reliable units double their score (15.5% -> 32.8%). But at retina-level
reliability (0.26) they still reach only a third of the retina's 85%.

## Data

- OpenScope oddball session `glo637908`, 2031 good units.
- **Trials joined by condition.** Blocks repeat groups of five: A A A B (or A A A A), then a
  blank. A condition is (orientation, position in the group). Each block becomes one trial made of
  1 s windows (0.5 s grating + 0.5 s grey) in a fixed order: the first 20 presentations of A
  (135 degrees) at positions 0, 1 and 2, then the first 15 of B (45 degrees) at position 3,
  laid end to end. **Window m shows the same stimulus in every trial.** 26 blocks have enough of
  each (the blocks where 45 degrees is A, and blocks overlapping receptive field mapping or
  optotagging, are dropped). Trial length 75 s.
- Units with >= 10 trials of >= 30 ISIs: 1511; median 294 spikes per trial.
- **Reliability** per unit: mean pairwise correlation between its trials' PSTHs, 50 ms bins (as
  `build_isi_dataset.py`). OpenScope: median 0.004, quartiles 0.000 and 0.036, max 0.53.
  For comparison, same measure: **retina frozen noise median 0.248** (quartiles 0.139, 0.319;
  the 60 L1 neurons 0.261); Allen natural movie (L5) median 0.002 over all units, 0.032 in
  visual cortex.
- Square-root time bars in stablebear: time bars -> `barcode_to_betti_curve` -> square root of
  values -> `sb.Pcf`, `sb.pdist`; k-means with `sb.mean`, 3 seeds. ISI fractions as in L3.

## Command

```sh
python scripts/sb_reliability.py data/openscope/sub-637908_ses-1213341633_ogen.nwb
```

## Scores

60 units, 1-NN / k-means; chance 1.7%.

| Units | median reliability | sqrt time bars | ISIs shuffled | ISI fractions |
|---|---|---|---|---|
| 60 random | 0.009 | 15.5 / 10.9 | 3.8 / 6.4 | 53.1 / 31.0 |
| **60 most reliable** | 0.260 | **32.8 / 26.5** | 3.3 / 6.3 | 68.4 / 37.8 |
| retina, 60 neurons (L1/L2) | 0.261 | 84.9 / 66 | 7.7 / 11.9 | 54.3 / 42.4 |

Per unit, 400 random units together (chance 0.25%), each unit's 1-NN hit rate with square-root
time bars against its reliability: **Spearman 0.467, p = 5e-23.**

| reliability quintile | mean reliability | mean 1-NN |
|---|---|---|
| 1 | -0.001 | 1.1% |
| 2 | 0.001 | 0.1% |
| 3 | 0.009 | 2.0% |
| 4 | 0.037 | 5.4% |
| 5 | 0.162 | 19.9% |

## What to take from it

- **Reliability predicts time-locked identity within OpenScope too**, as W1 found across retina
  sessions: only the top fifth of units is identifiable by its timing at all.
- **Joining by condition helps:** 60 random units go from 7.2% (L1, raw 124 s blocks) to 15.5%
  here with the same signature.
- **Reliability is not the whole story.** The 60 most reliable OpenScope units are as reliable as
  retina cells (0.26 against 0.26) but reach 32.8% against 84.9%. Likely reason, not tested: a
  grating onset drives every unit at the same moments, so reliable units share the same PSTH
  shape and their time curves look alike. On the retina, frozen noise drives each cell with its
  own pattern through its own receptive field. A test: the correlation of PSTHs *between*
  different units, compared with within (see Ideas).
- The most reliable units are also more identifiable by ISI fractions (68.4% against 53.1%), so
  part of the gain is that they fire more.
- **What failed:** clustering stays low (26.5% for the most reliable); the shuffled control is at
  3-4% everywhere, so the time-locked signal is real but small.
