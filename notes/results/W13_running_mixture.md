# W13 A mixture of floor(sqrt(n)) Gaussians fitted to the first n ISIs, one pdf per spike

2026-11-13, web chat. **Did not work on the retina.** The end of the path (the fit to the whole
trial) equals ISI fractions; everything read from the path itself is lower, and its Rips H0
equals the shuffled control.

## Method

- For every 4th spike from n = 8 on, and for the last spike: a mixture of floor(sqrt(n))
  Gaussians fitted by EM to the first n log ISIs of the trial (sd at least 0.1). Weights, means
  and widths are all refitted at every point.
- The point is the fitted pdf, stored as its survival curve on a 46-point log-ISI grid, so L1
  between points is Wasserstein-1 between fitted pdfs. A trial becomes a path of about 40 pdfs.
- Control: ISIs shuffled before fitting (same distribution, no order, no drift over the trial).
- Regions: 8 k-means anchors on 20,000 pooled points (no labels, same neurons).

## Data and command

- Retina frozen noise, session `20180126_sr_le_sp`, 60 neurons (seed 0), 1645 trials of 9.94 s,
  median 161 ISIs. numpy, scipy and scikit-learn, not stablebear; H0 from a minimum spanning
  tree. One run; k-means is the mean of 3 seeds. Chance 1.7%.
- `python scripts/pilots/running_mixture.py` (about
  30 min on 2 cores).
- Across halves: the path of a trial's first half matched among the paths of other trials'
  second halves.
- Reference, ISI fractions: 54.0 1-NN, 40.9 k-means, 38.8 across halves.

## Scores: real (shuffled), %

| Readout | Full trial, 1-NN | k-means | Halves, 1-NN |
|---|---|---|---|
| Last point: fit to the whole sample | 53.9 (53.9) | 40.2 (39.7) | 37.3 (38.4) |
| Mean point of the path | 40.5 (41.0) | 33.0 (33.9) | 28.1 (27.0) |
| Cloud without homology (quantiles over points) | 42.7 (43.6) | 34.6 (34.7) | 30.8 (29.4) |
| Path read at four fixed times | 48.4 (44.7) | 36.5 (37.6) | 30.8 (30.9) |
| Path minus its end point (drift only) | 8.7 (5.3) | 11.3 (9.8) | 5.5 (7.1) |
| Rips H0 of the path, whole cloud | 17.1 (16.7) | 24.5 (24.0) | 14.8 (12.8) |
| Rips H0, divided by point count | 12.6 (11.0) | 18.9 (19.0) | 12.2 (11.7) |
| Rips H0 in 8 regions | 23.5 (23.0) | 23.7 (23.5) | 19.9 (18.6) |
| Region counts (twin without bars) | 20.3 (20.5) | 26.3 (25.9) | 16.6 (16.4) |
| ISI fractions + path at fixed times | 54.6 (50.4) | 40.2 (39.3) | 34.3 (34.8) |
| ISI fractions + Rips H0 in 8 regions | 45.6 (44.1) | 31.3 (31.4) | 31.4 (30.5) |

## What to take from it

- **The path adds noise, not information.** Every earlier point is an estimate of the same
  distribution from fewer ISIs, so the mean point (40.5) and the cloud (42.7) are worse than the
  last point alone (53.9).
- **The drift over the trial is real but small:** the path read at fixed times scores 48.4
  against 44.7 shuffled, and the drift alone 8.7 against 5.3. It is gone across halves
  (30.8 against 30.9), as expected for something locked to the movie. Time bars capture the
  same thing far better (about 85%).
- **Rips sees nothing:** H0 of the path equals its shuffled control in every column. Its bars
  are the step sizes of the path, which shrink like 1/n for any train.
- With W11 and W12: on 10 s retina trials the whole-sample ISI distribution is the ceiling for
  every pdf design tried. A per-spike pdf either conditions on the past, which helps only if
  the past predicts the next ISI, or is a partial estimate, which is noisier than the whole.
- Not queued for the other datasets. L25 (the state model on long cortical samples) stays.
