# L14 Rescaled-time ISI bars and delay regions

2026-11-12, local session. **Mixed.**

- **Population clock: did not help.** Rescaled time is 1-8 points worse than real time within
  a condition and does not improve movie <-> spontaneous transfer.
- **Delay regions k = 8: worked as a Rips signature.** It doubles plain m = 3 Rips (H0 23-41%
  against 12-20%) and keeps an order effect of 5-9 points, but stays below ISI fractions.

## Data

- **Neurons:** the 60 of `curve_eval.select` (>= 30 ISIs, seed 0), as in L1, L5 and L7.
- **OpenScope** `glo637908_grating`: 2397 blocks of 124 s, median 484 spikes.
- **Allen** `allen829720705_movie`: 2598 repeats of 30 s, median 127 spikes.
- **Allen** `allen829720705_spont`: 3745 windows of 30 s, median 119 spikes. For the cross test,
  3275 windows of 58 of the movie neurons.

**Population clock** (`scripts/sb_rescaled.py`, as exp21):
- On each trial a spike at t moves to (other units' spikes before t) / (all other units' spikes)
  x duration.
- The population is every good unit in the file except the unit itself: 1.2 M spikes per
  OpenScope block, about 150 k per Allen window.
- Then ISI fractions in stablebear; 1-NN and k-medoids.
- Control: ISIs permuted in real time before rescaling.

**Delay regions** (`run_pipeline.py`, configs `m3_fixed` and `regions_m3_k8`):
- Log-ISI delay cloud with m = 3, 25 points, 20 subsamples, Rips in stablebear.
- Regions: each cloud split by 8 fixed k-means anchors (fitted on other neurons' trials), with one
  barcode per region.
- Clustering columns are run_pipeline's own clustering (accuracy after Hungarian pairing).

## Commands

```sh
python scripts/sb_rescaled.py --data data/openscope/glo637908.parquet --session glo637908_grating
python scripts/sb_rescaled.py --data data/allen/allen829720705.parquet --session allen829720705_movie \
    --cross-session allen829720705_spont
python scripts/sb_rescaled.py --data data/allen/allen829720705.parquet --session allen829720705_spont
python scripts/run_pipeline.py m3_fixed m3_fixed_shuffled regions_m3_k8 regions_m3_k8_shuffled \
    --data data/openscope/glo637908.parquet --session glo637908_grating --neurons 60
python scripts/run_pipeline.py m3_fixed m3_fixed_shuffled regions_m3_k8 regions_m3_k8_shuffled \
    --data data/allen/allen829720705.parquet --session allen829720705_movie --neurons 60   # and _spont
```

## Scores: population clock

1-NN / k-medoids, %. Chance 1.7%.

| | OpenScope 124 s | Allen movie 30 s | Allen spont 30 s |
|---|---|---|---|
| ISI fractions, real time | 62.3 / 30.2 | 48.7 / 24.2 | 38.5 / 24.5 |
| ISI fractions, real time, shuffled | 62.3 / 30.2 | 48.7 / 24.2 | 38.5 / 24.5 |
| ISI fractions, rescaled time | 60.9 / 31.8 | 41.0 / 23.7 | 31.6 / 23.3 |
| rescaled, shuffled | 57.0 / 30.5 | 39.4 / 23.1 | 29.2 / 21.8 |

Across stimuli (Allen, 1-NN):

| | movie -> spont | spont -> movie |
|---|---|---|
| real time | 28.8 | 27.9 |
| rescaled time | 27.5 | 26.3 |
| rescaled time, shuffled | 23.2 | 23.1 |

## Scores: delay regions

1-NN / clustering accuracy, %.

| config | | OpenScope | shuffled | Allen movie | shuffled | Allen spont | shuffled |
|---|---|---|---|---|---|---|---|
| m3_fixed | H0 | 19.6 / 18.1 | 17.0 / 16.4 | 12.9 / 14.7 | 12.9 / 13.9 | 11.6 / 14.1 | 11.1 / 13.8 |
| m3_fixed | H0+H1 | 19.9 / 17.8 | 17.0 / 17.9 | 14.4 / 14.2 | 10.9 / 14.5 | 11.8 / 14.5 | 11.2 / 13.6 |
| **regions_m3_k8** | **H0** | **40.9 / 28.7** | 34.8 / 26.7 | **33.3 / 22.6** | 24.0 / 18.8 | **23.1 / 19.0** | 17.7 / 17.3 |
| regions_m3_k8 | H1 | 14.0 / 16.2 | 15.1 / 16.1 | 12.8 / 14.8 | 8.2 / 12.1 | 7.8 / 11.9 | 6.2 / 10.6 |
| regions_m3_k8 | H0+H1 | 34.1 / 22.4 | 30.0 / 23.0 | 29.7 / 20.4 | 17.0 / 16.4 | 17.9 / 17.0 | 14.0 / 14.9 |

## What to take from it

- **The population clock removes identity instead of noise.** Rescaling makes ISI fractions
  worse (Allen movie 48.7 -> 41.0, spontaneous 38.5 -> 31.6). Each unit's ISIs get mixed with the
  population's fluctuations, and on 30 s samples that adds more noise than it removes. It also
  does not help across stimuli (27.5 against 28.8).
- **Rescaled time sees a little order.** The shuffled control drops 2-4 points: shuffling moves
  spikes relative to population events, which the rescaling preserves.
- **Delay regions are the best Rips signature so far.** Splitting the cloud by fixed anchors lets
  the barcode keep where the points sit. H0 rises from 12-20% to 23-41%, close to the whole
  cloud's sliced Wasserstein of L10 (OpenScope 41 vs 50).
- **Regions see order.** The shuffled control is 5-9 points lower (Allen movie 33.3 vs 24.0).
- **What failed:**
  - the population clock;
  - H1 in every config (6-15%);
  - delay regions still stay below ISI fractions (41 vs 62 on OpenScope, 33 vs 49 on the Allen movie).
