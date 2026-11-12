# L10 Delay-embedding Rips on long samples

2026-11-12, local session. **Did not work as a topological signature:** on long samples Rips H0
and H1 of the ISI delay cloud stay at 12-39% 1-NN, far below ISI fractions (50-70%), and their
ISI-shuffled controls are nearly as high. The cloud itself carries more (sliced Wasserstein
34-51%), with some order effect, but homology throws most of it away.

## Data

60 neurons per dataset, seed 0, among units with >= 10 samples of >= 105 ISIs (enough for 100
points at m = 5). This is a different 60 from L1-L9 where the old set had too few spikes.

| Dataset | session | samples | median spikes | length |
|---|---|---|---|---|
| OpenScope gratings | `glo637908_grating` | 2255 | 490 | 124 s blocks |
| Allen natural movie | `allen829720705_movie` | 2426 | 220 | 30 s repeats |
| Allen spontaneous | `allen829720705_spont` | 2714 | 210 | 30 s grey windows |
| Retina unrepeated noise | `20180126_sr_le_sp_running` | 1774 | 742 | 40 s (4 x 10 s pieces joined, as L6) |

Method (`scripts/sb_rips_long.py`, all homology in stablebear):

- log ISIs, delay embedding m = 2, 3, 5, coordinates / sqrt(m);
- 20 random subsamples of exactly 100 points per sample;
- `persistence.compute_persistent_homology` (Rips, reduced, max_dim 1) on a `pcloud64` tensor;
- stable ranks averaged over the subsamples with `sb.mean`, L1 with `sb.pdist`;
- H0+H1 = sum of the two median-scaled matrices;
- position = mean point of the full cloud (Euclidean);
- whole cloud = sliced Wasserstein-1 between the first 100-point subsamples (64 directions);
- control: ISIs permuted within each sample before embedding.

Scores: 1-NN, and k-medoids on the same distances (3 restarts, clusters paired with neurons by
Hungarian matching). k-medoids, not k-means, because position and sliced Wasserstein are not
PCFs; numbers are therefore not directly comparable with the k-means columns of L1-L9.

## Commands

```sh
python scripts/sb_rips_long.py --data data/openscope/glo637908.parquet --session glo637908_grating
python scripts/sb_rips_long.py --data data/allen/allen829720705.parquet --session allen829720705_movie
python scripts/sb_rips_long.py --data data/allen/allen829720705.parquet --session allen829720705_spont
python scripts/sb_rips_long.py --data data/retina_running_20180126_sr_le_sp.parquet \
    --session 20180126_sr_le_sp_running --join-pieces
```

Tables in `results/rips_long_<session>.csv` (not committed). Each session took about 2 minutes.

## Scores

1-NN / k-medoids, %. Chance 1.7%. "shuf" = ISIs permuted within the sample.

| m | summary | OpenScope 124 s | shuf | Allen movie 30 s | shuf | Allen spont 30 s | shuf | Retina 40 s | shuf |
|---|---|---|---|---|---|---|---|---|---|
| - | **ISI fractions (reference)** | **68.7 / 35.1** | | **52.0 / 27.5** | | **49.9 / 26.5** | | **70.2 / 42.3** | |
| 2 | position | 9.3 / 13.3 | 9.6 / 13.0 | 8.4 / 12.7 | 6.8 / 12.5 | 7.8 / 13.0 | 8.5 / 13.2 | 9.8 / 18.9 | 8.8 / 18.8 |
| 2 | whole cloud | 50.8 / 30.3 | 46.7 / 28.8 | 42.3 / 25.9 | 33.8 / 22.0 | 39.0 / 23.1 | 32.2 / 22.0 | 43.6 / 29.8 | 39.6 / 30.2 |
| 2 | H0 | 25.1 / 21.5 | 23.1 / 20.5 | 16.2 / 18.3 | 16.6 / 17.1 | 19.1 / 17.8 | 18.0 / 16.8 | 24.2 / 25.9 | 24.5 / 25.5 |
| 2 | H1 | 12.9 / 16.6 | 11.4 / 16.0 | 11.3 / 14.0 | 10.1 / 14.1 | 11.8 / 14.4 | 11.8 / 14.4 | 13.7 / 20.5 | 14.3 / 20.3 |
| 2 | H0+H1 | 27.7 / 22.3 | 24.6 / 20.4 | 21.7 / 18.8 | 20.5 / 17.6 | 19.8 / 17.6 | 20.4 / 17.9 | 32.4 / 28.8 | 34.0 / 31.9 |
| 3 | position | 9.8 / 13.2 | 9.8 / 13.4 | 7.7 / 12.3 | 6.8 / 12.9 | 9.1 / 13.0 | 8.7 / 13.2 | 11.0 / 18.8 | 10.5 / 18.9 |
| 3 | whole cloud | 50.3 / 30.1 | 40.2 / 26.4 | 40.3 / 23.7 | 29.3 / 22.8 | 37.9 / 23.9 | 29.2 / 21.3 | 40.7 / 30.7 | 36.2 / 27.7 |
| 3 | H0 | 26.7 / 22.8 | 20.2 / 19.3 | 16.8 / 17.4 | 16.7 / 17.4 | 17.9 / 16.8 | 17.4 / 16.2 | 26.9 / 26.5 | 24.6 / 24.6 |
| 3 | H1 | 14.9 / 18.4 | 13.6 / 17.5 | 12.1 / 15.0 | 11.5 / 14.7 | 13.3 / 15.1 | 12.4 / 15.3 | 21.3 / 25.3 | 18.7 / 27.2 |
| 3 | H0+H1 | 30.3 / 23.6 | 25.7 / 20.6 | 20.1 / 18.7 | 19.8 / 18.0 | 21.3 / 18.3 | 20.4 / 16.9 | 37.5 / 33.5 | 36.0 / 31.3 |
| 5 | position | 12.1 / 13.2 | 10.6 / 13.3 | 7.6 / 12.7 | 7.0 / 12.9 | 9.5 / 13.1 | 9.0 / 13.2 | 11.6 / 18.5 | 12.6 / 18.9 |
| 5 | whole cloud | 46.6 / 28.9 | 33.5 / 25.2 | 35.2 / 23.2 | 25.4 / 21.4 | 34.0 / 23.5 | 24.9 / 20.6 | 39.1 / 29.9 | 34.2 / 28.3 |
| 5 | H0 | 23.2 / 21.3 | 18.3 / 19.2 | 16.2 / 16.8 | 13.8 / 16.2 | 18.6 / 15.8 | 16.2 / 15.7 | 28.9 / 27.8 | 28.9 / 27.8 |
| 5 | H1 | 17.4 / 18.0 | 13.2 / 18.1 | 11.7 / 15.5 | 10.1 / 14.0 | 14.4 / 14.8 | 13.5 / 15.5 | 22.3 / 27.1 | 20.3 / 25.3 |
| 5 | H0+H1 | 29.8 / 21.2 | 22.4 / 20.3 | 20.2 / 19.0 | 16.2 / 17.8 | 20.9 / 16.8 | 19.7 / 16.9 | 38.8 / 34.4 | 36.5 / 33.7 |

## What to take from it

- **Homology loses most of the identity.** The best Rips score (retina H0+H1 at m = 5: 38.8%)
  is about half of ISI fractions on the same samples (70.2%). In cortex and hippocampus H0+H1
  stays at 20-30%.
- **What homology keeps is mostly the ISI distribution, not the order.** Shuffled controls are
  within 0-7 points of the real score everywhere. The clearest order effect is OpenScope
  m = 5 H0+H1 (29.8 against 22.4) and H0 at m = 3 (26.7 against 20.2), the longest samples, in
  line with L8. On the retina the shuffled H0+H1 at m = 2 is even higher than the real one.
- **H1 is the weakest part** (10-22%): 100 points in 2-5 dimensions have few, short loops.
- **The whole cloud does better and shows order.** Sliced Wasserstein gives 34-51% 1-NN, and
  shuffling costs 4-13 points, more at larger m (OpenScope m = 5: 46.6 against 33.5; Allen
  movie m = 5: 35.2 against 25.4). The order information is in the cloud's shape, but Rips
  persistence, which is blind to where the cloud sits and to its density, does not keep it.
- **Position is near rate-only level** (8-12%): with log ISIs the mean point is just the mean
  log ISI.
- **What failed:** the original proposal (Rips barcodes of the delay cloud) does not identify
  neurons as well as a plain ISI histogram on any of the four datasets, even with 200-740 spikes
  per sample. k-medoids is low for everything (13-34%).
