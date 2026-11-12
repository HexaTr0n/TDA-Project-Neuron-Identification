# W8 Fixes for what Rips cannot do: time-filtered complex, and H1 inside regions

2026-11-13, web chat. **Did not work.** The time-filtered complex is at the level of the spike
count, and adding H1 to H0 inside regions lowers the score.

## Data and command

- Retina frozen noise, session `20180126_sr_le_sp`, 60 neurons (seed 0), 1645 whole trials (no
  subsampling); across = the two halves of the movie. Delay cloud m = 3, log ISIs. numpy, scipy
  and the experiment Rips code, not stablebear. One run.
- `cd scripts/experiments && python exp32_rips_fixes.py` (log: `rips_fixes.log`).
- H1 inside regions: a region with more than 80 points is cut to 80 random points for H1 only,
  because the experiment Rips code is slow for H1 (53 s for one 300-point cloud).

## A. Time-filtered complex

Delay points enter in spike order at a fixed Rips scale r. 1-NN / k-medoids on full trials, %;
shuffled in brackets.

| Summary | r = 0.5 | r = 1.0 | r = 2.0 |
|---|---|---|---|
| Betti curve over time | 14.2 / 18.5 (12.6 / 19.6) | 9.0 / 15.1 (9.1 / 15.1) | 5.0 / 10.9 (5.5 / 10.5) |
| births so far over time | 16.0 / 19.5 (13.4 / 18.7) | 8.9 / 15.0 (7.7 / 15.0) | 4.3 / 10.7 (4.1 / 10.8) |
| lifetimes, times dropped | 12.2 / 16.9 (11.7 / 16.7) | 8.6 / 14.5 (8.7 / 15.6) | 5.1 / 10.3 (4.9 / 10.5) |
| Betti curve + births | 17.0 / 21.2 | 11.6 / 17.0 | 6.0 / 10.8 |

Across halves everything is at 4-14%. For reference: spike count 11.4 / 18.1, time bars about 85.

- New kinds of ISI pattern nearly all appear in the first second or two of a trial; after that
  every point joins an existing component, so the curves go flat and the rest of the trial
  leaves no trace. What is measured is the diversity of patterns, not when the neuron fires.
- Smaller r is better only because each spike then becomes its own pattern, which turns the
  births curve into a running spike count.
- The working form of "merging homology that keeps time" measures closeness in time instead:
  burst bars and peak matching (70-72%).

## B. H1 inside regions, whole cloud, 8 anchors

1-NN / k-medoids, %; shuffled in brackets.

| Summary | Full trials | Across halves |
|---|---|---|
| H0 | 43.2 / 34.2 (39.9 / 34.8) | 30.7 / 21.8 (28.2 / 22.5) |
| H1, stable rank | 18.6 / 18.8 (16.8 / 17.9) | 12.6 / 10.0 (12.7 / 10.5) |
| H1, Betti curve | 14.4 / 12.8 (13.4 / 14.4) | 7.9 / 6.4 (9.6 / 6.7) |
| H0 + H1 stable rank | 38.7 / 29.0 (34.2 / 29.5) | 24.3 / 17.9 (23.2 / 17.5) |
| H0 + H1 Betti curve | 37.1 / 28.3 (34.9 / 29.6) | 23.5 / 17.0 (23.0 / 16.6) |

- **Adding H1 lowers H0** by 4-6 points on full trials and 6-7 across halves, with either summary.
- **H1 alone is weak** (14-19%) and equal to its shuffled control.
- **The Betti curve is worse than the stable rank for H1** (14.4 against 18.6).
- Caveat: H1 here is capped at 80 points per region and computed inside regions only. H1 on the
  whole cloud without regions was not run here (too slow); L21 covers it on stablebear.
