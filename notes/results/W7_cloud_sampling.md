# W7 How to cut a delay cloud to size before Rips

2026-11-13, web chat. **Partly worked.** Using the whole cloud is clearly better than 25 points.
The consecutive stretch is worse than random draws, but has the largest order effect.

## Data and command

- Retina frozen noise, session `20180126_sr_le_sp`, 60 neurons (seed 0), 1645 trials; across =
  the two halves of the movie (5 s half-trials, 15 points where a size is fixed).
- Delay cloud m = 3, log ISIs. H0 only (edges of the minimum spanning tree), stable rank, L1.
  numpy and scipy, not stablebear. Delay regions: 8 anchors fitted on other neurons.
- random = 25 points at random, 20 draws averaged. stretch = 25 consecutive points after cutting
  a random number of spikes off the start and end, 20 draws averaged. whole = every point, once.
  "as fractions" = the whole cloud's stable rank divided by its number of points.
- `python scripts/experiments/exp30_cloud_sampling.py` (log: `cloud_sampling.log`). One run;
  k-medoids varies by about 2.5 points between starts.

## Scores

1-NN / k-medoids, %. Shuffled control (ISIs permuted) in brackets.

| Method | Full trials | Across halves |
|---|---|---|
| reference: spike count | 11.4 / 18.1 | 9.3 / 10.5 |
| reference: ISI fractions | 54.0 / 38.1 | 41.5 / 28.2 |
| delay Rips, random | 18.4 / 21.6 (16.0 / 20.9) | 14.1 / 13.3 (12.0 / 13.4) |
| delay Rips, stretch | 14.6 / 19.7 (12.1 / 18.2) | 10.7 / 9.8 (9.7 / 10.0) |
| delay Rips, whole | 33.1 / 29.0 (31.5 / 27.9) | 24.3 / 20.3 (23.4 / 20.9) |
| delay Rips, whole, as fractions | 19.1 / 20.2 (15.9 / 18.5) | 14.1 / 14.1 (13.1 / 12.4) |
| delay regions, random | 30.2 / 28.5 (25.8 / 26.3) | 20.8 / 18.4 (17.8 / 15.9) |
| delay regions, stretch | 26.5 / 26.4 (20.7 / 23.2) | 19.3 / 15.2 (14.6 / 13.2) |
| delay regions, whole | 43.2 / 34.2 (41.2 / 35.8) | 30.7 / 21.8 (28.9 / 24.5) |
| delay regions, whole, as fractions | 37.7 / 29.8 (33.2 / 26.0) | 26.6 / 17.7 (23.6 / 15.8) |

## What to take from it

- **The whole cloud is the best preparation.** Delay regions rise from 30.2 to 43.2 on 1-NN and
  from 28.5 to 34.2 in clustering; across halves from 20.8 to 30.7. Clustering now sits within
  4 points of ISI fractions (34.2 against 38.1).
- **Part of that gain is the spike count, but not all.** Divided by the number of points,
  delay regions still reach 37.7 (random draws: 30.2). Plain Rips falls back to 19.1, so for
  plain Rips the whole-cloud gain is almost entirely the count.
- **The consecutive stretch is worse than random draws** (regions 26.5 against 30.2). A stretch
  of 25 spikes covers only a part of the trial, so it samples the ISI distribution less evenly
  than 25 spikes spread over the whole trial.
- **But the stretch has the largest order effect:** 5.8 points over shuffled for regions (26.5
  against 20.7), against 4.4 for random draws and 2.0 for the whole cloud. Keeping neighbours
  together does keep more of the order; it just costs more than it brings on 10 s trials.
- **With the whole cloud the shuffled control is almost equal** (43.2 against 41.2; in
  clustering the shuffled one is higher). So the gain is distribution, not order.
