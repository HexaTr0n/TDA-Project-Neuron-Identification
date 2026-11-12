# L11 Pdf per spike (predictive state) on long samples

2026-11-12, local session. **Worked only where it doesn't need topology.**

- The whole predictive-state cloud slightly beats ISI fractions on every dataset (+1.2 to +5.2
  points 1-NN), and shuffling costs it 2-7 points, so it sees some order.
- Its homology (H0, H0+H1) is weak: 13-28%, against 47-70% for ISI fractions.

## Data

Same datasets, selection and evaluation as L10 (`notes/results/L10_rips_long_samples.md`):
units with >= 10 samples of >= 103 ISIs, 60 at random (seed 0), so the neurons differ slightly
from L10.

| Dataset | samples | median spikes |
|---|---|---|
| OpenScope gratings, 124 s blocks | 2310 | 496 |
| Allen movie, 30 s repeats | 2299 | 211 |
| Allen spontaneous, 30 s windows | 2881 | 202 |
| Retina unrepeated noise, 40 s | 1775 | 742 |

**Embedding.** `predictive_state` from `scripts/experiments/legacy_scripts/embeddings.py`:
- one point per spike: the pdf of the next log ISI given the last L = 2 ISIs;
- kernel bandwidth 1;
- 32 Wasserstein-2 (quantile) coordinates.

**Summaries.**
- 100-point subsamples (20 per sample); Rips H0/H1 in stablebear, stable ranks averaged, L1.
- Position: mean point.
- Whole cloud: sliced Wasserstein-1, 64 directions.

**Control:** ISIs permuted within the sample before embedding. Scores are 1-NN / k-medoids.

## Command

```sh
python scripts/sb_rips_long.py --embedding predictive --data data/openscope/glo637908.parquet --session glo637908_grating
python scripts/sb_rips_long.py --embedding predictive --data data/allen/allen829720705.parquet --session allen829720705_movie
python scripts/sb_rips_long.py --embedding predictive --data data/allen/allen829720705.parquet --session allen829720705_spont
python scripts/sb_rips_long.py --embedding predictive --data data/retina_running_20180126_sr_le_sp.parquet \
    --session 20180126_sr_le_sp_running --join-pieces
```

## Scores

1-NN / k-medoids, %. Chance 1.7%.

| summary | OpenScope 124 s | shuf | Allen movie | shuf | Allen spont | shuf | Retina 40 s | shuf |
|---|---|---|---|---|---|---|---|---|
| ISI fractions (reference) | 64.2 / 32.4 | | 53.0 / 28.0 | | 47.3 / 24.7 | | 70.1 / 43.5 | |
| position | 62.4 / 33.4 | 63.8 / 33.2 | 52.9 / 29.1 | 50.9 / 29.1 | 44.9 / 25.3 | 45.3 / 23.8 | 70.8 / 44.3 | 70.5 / 44.2 |
| **whole cloud** | **69.4 / 34.7** | 62.8 / 34.7 | **57.5 / 29.9** | 51.9 / 28.8 | **50.3 / 26.4** | 44.1 / 24.2 | **72.7 / 46.5** | 70.4 / 47.0 |
| H0 | 17.8 / 16.6 | 15.0 / 17.2 | 16.0 / 16.9 | 13.5 / 16.5 | 13.2 / 14.1 | 11.5 / 13.6 | 23.8 / 23.8 | 20.6 / 24.1 |
| H1 | 9.7 / 14.7 | 9.4 / 14.6 | 9.5 / 14.7 | 6.7 / 13.2 | 8.2 / 12.9 | 7.2 / 12.4 | 11.7 / 18.4 | 7.9 / 16.6 |
| H0+H1 | 18.7 / 18.0 | 15.4 / 16.5 | 18.7 / 17.1 | 14.0 / 15.1 | 15.0 / 14.7 | 12.9 / 13.4 | 28.0 / 24.6 | 21.6 / 22.0 |

## What to take from it

- **Position is the ISI distribution.** Unlike the delay cloud (L10), each point here is a
  whole pdf of next ISIs, so the mean point is close to the overall ISI distribution. It scores
  like ISI fractions, and shuffling does not change it.
- **The whole cloud adds a little order:**
  - +5.2 points over ISI fractions on OpenScope (69.4 against 64.2), +4.5 on the Allen movie,
    +3.0 on spontaneous, +2.6 on the retina;
  - shuffled 2-7 points lower;
  - this matches L8: with hundreds of spikes, history dependence becomes visible.
- **Homology again throws it away.** H0+H1 is at 15-28%, below the delay-cloud version of L10
  on cortex, although shuffled controls stay a few points lower (order is slightly present).
- **What failed:**
  - Rips persistence of the predictive-state cloud, on all four datasets;
  - k-medoids is low for every summary.
