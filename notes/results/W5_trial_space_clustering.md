# W5 Topological clustering of the trial space

2026-11-12, web chat. **Did not work.** Density-peak clustering by persistence (ToMATo) is worse
than k-means with mean curves in 8 of 9 cells, and its barcode does not count the neurons.

## Data and command

- Retina frozen noise, three sessions, 60 random neurons each (seed 0), trials with >= 30 ISIs.
- Points are trials; distance is L1 between signatures. numpy only, coarse time grid (step s/4).
- ToMATo: log density from the mean distance to the k nearest trials, k-nearest-neighbour graph
  (k = 5, 8, 12), peaks merged in order of prominence until 60 clusters remain.
- "Suggested" = the number of clusters at the largest drop in peak prominence.
- `python scripts/experiments/exp28_trial_space_clustering.py` (log: `trial_space_clustering.log`).

## Scores

Share of trials in the right cluster, %. k-means is the mean of 3 seeds.

| Session | Signature | k-means | k-medoids | Average linkage | ToMATo k=5 / 8 / 12 | Suggested clusters |
|---|---|---|---|---|---|---|
| 20180126_sr_le_sp | multi-scale sqrt time bars | 68.1 | 55.1 | 36.3 | 38.8 / 31.8 / 26.3 | 18 / 15 / 14 |
| | ISI fractions | 40.9 | 38.1 | 25.2 | 33.1 / 28.0 / 26.0 | 8 / 7 / 8 |
| | time bars + ISI fractions | 62.8 | 63.2 | 52.5 | 67.5 / 56.4 / 49.1 | 13 / 16 / 16 |
| 20171215_sr_le_sp | multi-scale sqrt time bars | 67.7 | 56.4 | 8.9 | 45.7 / 31.2 / 13.0 | 8 / 7 / 7 |
| | ISI fractions | 30.0 | 29.9 | 28.8 | 20.8 / 18.0 / 12.8 | 7 / 7 / 14 |
| | time bars + ISI fractions | 61.5 | 59.7 | 34.0 | 44.8 / 32.5 / 19.9 | 17 / 12 / 8 |
| 20180405_sr_le | multi-scale sqrt time bars | 41.0 | 30.7 | 18.1 | 19.9 / 19.4 / 18.9 | 16 / 17 / 15 |
| | ISI fractions | 28.7 | 29.8 | 20.7 | 23.0 / 21.4 / 15.6 | 10 / 9 / 7 |
| | time bars + ISI fractions | 42.1 | 38.4 | 29.8 | 34.1 / 29.2 / 23.7 | 14 / 16 / 16 |

## What to take from it

- **k-means with mean curves stays the best clustering step.** ToMATo wins one cell (67.5 against
  62.8-63.2, at k = 5 only) and loses the other eight, usually by 10-30 points. It is also
  sensitive to k.
- **The trial space has far fewer density peaks than neurons:** the largest prominence drop
  suggests 7-18 clusters where there are 60. With about 27 trials per neuron in thousands of
  dimensions, most neurons do not form a density peak of their own.
- **So the bottleneck is not the choice of clustering algorithm.** A neuron's trials are too
  spread out relative to the gaps between neurons (W1's "margin"), which no clustering step
  repairs. What does help is in the signature: mean curves as centres, the square root, and
  longer samples.
- **Not tried:** Mapper, and ToMATo on a low-dimensional projection of the trial space.
