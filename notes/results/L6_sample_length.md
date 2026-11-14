# L6 Identity against sample length

2026-11-12, local session. **Worked:** identity from ISI fractions rises steadily with sample
length on both datasets, with no plateau by 40 s (retina) or 120 s (OpenScope). Spike count,
not seconds, sets the level.

## Data

- **Retina:** session `20180126_sr_le_sp`, unrepeated noise (L3 table,
  `load_retina_running.py`). The 4 pieces of 10 s before each frozen trial are joined into one
  40 s stretch: 30 stretches per neuron.
- **OpenScope:** session `glo637908_grating`, each 124 s grating block is one stretch: 46 per
  neuron.
- Neurons: the same 60 as in L1-L3 (seed 0, chosen on frozen trials or grating blocks with >= 30
  ISIs).
- Each stretch is cut into pieces of L seconds. Every neuron gets **20 random pieces at every
  length**, so the number of references does not change with L (1200 samples per length).
  Pieces with fewer than 2 ISIs stay in (empty barcode).
- ISI fractions in stablebear (`sb_cross.isi_fraction_pcfs`), `sb.pdist`, k-means with `sb.mean`
  (3 seeds). Rate only: |log(1 + spike count)|, 1-NN.

## Commands

```sh
python scripts/sb_length.py --data data/retina_running_20180126_sr_le_sp.parquet \
    --select-session 20180126_sr_le_sp --session 20180126_sr_le_sp_running --join-pieces
python scripts/sb_length.py --data data/openscope/glo637908.parquet \
    --select-session glo637908_grating --session glo637908_grating --lengths 2.5 5 10 20 40 120
```

## Scores

Chance 1.7%. "< 2 ISIs" is the share of pieces with an empty or near-empty barcode.

| Length | Retina spikes | < 2 ISIs | 1-NN | k-means (sd) | rate 1-NN | OpenScope spikes | < 2 ISIs | 1-NN | k-means (sd) | rate 1-NN |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.5 s | 38 | 1.9% | 22.2 | 25.8 (1.0) | 9.6 | 7 | 34.0% | 13.8 | 16.1 (0.4) | 4.0 |
| 5 s | 78 | 1.0% | 36.8 | 31.0 (0.3) | 10.6 | 13 | 23.9% | 16.3 | 18.4 (0.5) | 5.1 |
| 10 s | 152 | 0% | 46.2 | 39.2 (0.6) | 11.2 | 29 | 17.5% | 23.1 | 20.1 (0.1) | 5.8 |
| 20 s | 300 | 0% | 60.1 | 41.4 (0.4) | 10.3 | 54 | 12.2% | 29.2 | 23.6 (0.9) | 6.9 |
| 40 s | 595 | 0% | 67.2 | 46.0 (2.1) | 12.2 | 115 | 10.1% | 37.2 | 26.7 (0.1) | 6.5 |
| 120 s | | | | | | 391 | 6.2% | 50.5 | 30.9 (0.3) | 8.6 |

## What to take from it

- **No plateau.** Retina 1-NN goes 22 -> 37 -> 46 -> 60 -> 67% from 2.5 to 40 s; OpenScope
  14 -> 16 -> 23 -> 29 -> 37 -> 51% from 2.5 to 120 s. Clustering rises more slowly (retina
  26 -> 46%, OpenScope 16 -> 31%).
- **Spike count sets the level, roughly.** At about 150 spikes the retina (10 s) gives 46% and at
  about 115 spikes OpenScope (40 s) gives 37%; at 390 spikes OpenScope gives 51%, below the
  retina's 60% at 300. Cortex needs a little more spikes for the same identity.
- **How much recording a neuron needs:** for 50% 1-NN among 60 neurons, about 12-15 s of retina
  noise or about 2 minutes of cortex/hippocampus under gratings.
- At 2.5 s, k-means beats 1-NN (25.8 against 22.2 on the retina): mean curves average out the noise
  that single nearest neighbours suffer from.
- The firing rate alone stays at 4-12% at every length, so the gain comes from the ISI shape.
- **Do not compare across tasks:** with 20 references per neuron, OpenScope at 120 s gives 50.5%,
  below L1's 62.6% with all 46 blocks as references. 1-NN depends on how many references
  each neuron has.
- **What failed:** short OpenScope pieces are often near empty (34% of 2.5 s pieces have < 2 ISIs),
  so below 10 s OpenScope identity is mostly about whether a unit fires at all.
