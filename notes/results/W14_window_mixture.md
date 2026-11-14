# W14 A mixture of floor(sqrt(w)) Gaussians fitted to the last w ISIs, one pdf per spike

2026-11-13, web chat. **No new topology on the retina.** The cloud of local pdfs read without
homology equals ISI fractions; Rips H0 equals its shuffled control and H1 is at chance. The
path read at fixed times carries response timing, which is gone across halves.

## Method

- For every spike with at least w ISIs behind it: a mixture of K = floor(sqrt(w)) Gaussians
  fitted to the last w log ISIs. Own EM, all windows at once, 40 iterations, sd at least 0.1,
  means started at the window's quantiles (nothing random). w = 16, 32, 64 (K = 4, 5, 8).
- The point is the fitted pdf as its survival curve on a 46-point log-ISI grid; L1 between
  points is Wasserstein-1 between fitted pdfs. A trial becomes a path of 100-150 pdfs.
- Readouts: mean point; quantiles over points (cloud without homology); the path at four fixed
  times; sd per grid point; Rips H0 (minimum spanning tree) of the whole cloud; Rips H1 on 40
  landmarks (farthest-point sampling, experiment Rips code); Rips H0 inside 8 regions (k-means
  anchors on 20,000 pooled points, no labels, same neurons) and the region counts.
- Control: ISIs shuffled before fitting.
- This is the richer version of the proposal's "log-normal windows", which exp18 ran early on
  (last 20 ISIs, one Gaussian, retina 10 s: mean point 24.9% 1-NN, whole cloud 35.2%, H0 4.0%).

## Data and command

- Retina frozen noise, session `20180126_sr_le_sp`, 60 neurons (seed 0), 1645 trials of 9.94 s,
  median 161 ISIs. numpy and scipy, not stablebear. One run; k-medoids is the mean of 3 seeds.
  Chance 1.7%.
- `python scripts/pilots/window_mixture.py` (about 30 min; build `librips.so` first, see
  `scripts/README.md`).
- Across halves: first half of a trial among the second halves of other trials. Not run for
  w = 64 (a half has 81 ISIs).
- Reference, ISI fractions: 54.0 1-NN, 38.2 k-medoids, 38.8 across halves.

## Scores: real (shuffled), %. Each cell: full trial 1-NN; k-medoids; halves 1-NN

| Readout | last 16 ISIs | last 32 ISIs | last 64 ISIs (no halves) |
|---|---|---|---|
| Mean point | 51.8 (51.1); 36.7 (37.2); 35.9 (34.8) | 51.9 (50.6); 36.6 (37.4); 35.5 (34.2) | 52.2 (48.0); 35.9 (36.4) |
| Cloud without homology | 53.9 (51.7); 35.5 (36.3); 36.8 (33.6) | 53.1 (50.2); 37.2 (35.9); 35.0 (34.0) | 52.3 (47.8); 37.3 (36.0) |
| Path at fixed times | 34.1 (20.7); 26.0 (20.0); 19.4 (19.0) | 43.6 (32.5); 32.6 (27.9); 25.0 (26.9) | 52.3 (42.9); 36.8 (32.2) |
| Spread of the path | 33.7 (25.5); 28.0 (24.4); 19.0 (16.3) | 24.9 (22.1); 22.7 (20.8); 13.1 (12.8) | 17.9 (14.0); 19.5 (16.6) |
| Rips H0, whole cloud | 23.3 (25.0); 20.8 (20.1); 17.8 (15.7) | 22.3 (20.8); 20.0 (20.0); 15.2 (15.7) | 18.7 (20.9); 18.5 (19.7) |
| Rips H0, divided by point count | 12.4 (13.3); 17.4 (17.3); 9.5 (9.4) | 11.4 (13.9); 16.2 (16.3); 8.4 (8.6) | 9.0 (12.5); 16.0 (17.2) |
| Rips H1, 40 landmarks | 4.4 (4.2); 11.7 (11.5); 4.4 (3.8) | 4.0 (3.5); 12.0 (11.9); 3.6 (4.0) | 4.5 (4.0); 11.3 (11.5) |
| Rips H0 in 8 regions | 33.4 (33.3); 28.0 (25.8); 23.8 (22.9) | 32.0 (28.8); 24.4 (23.7); 21.6 (18.7) | 28.7 (28.1); 21.7 (21.7) |
| Region counts | 31.7 (30.6); 27.6 (25.0); 21.7 (21.9) | 28.3 (26.1); 25.0 (22.3); 20.0 (18.5) | 27.1 (26.1); 21.5 (20.8) |
| ISI fractions + path at fixed times | 51.2 (43.0); 35.5 (31.3); 28.5 (33.0) | 53.3 (47.1); 38.1 (35.2); 33.1 (34.3) | 56.7 (51.0); 39.6 (37.8) |
| ISI fractions + Rips H0 in 8 regions | 48.6 (47.3); 35.7 (33.6); 30.9 (30.9) | 46.9 (44.3); 31.8 (33.6); 31.1 (31.4) | 46.6 (45.2); 32.4 (31.0) |
| ISI fractions + Rips H0, whole cloud | 50.6 (49.8); 36.1 (35.9); 36.4 (35.9) | 50.7 (50.0); 36.2 (35.5); 36.7 (34.8) | 50.3 (50.4); 35.6 (35.3) |

## What to take from it

- **The cloud is the ISI distribution again:** read without homology it scores 52-54%, as ISI
  fractions do (54.0), with the shuffled control 2-5 points lower.
- **Where a local pdf sits in the trial is response timing:** the path at fixed times is 10-13
  points above shuffled within a stimulus (52.3 against 42.9 for 64 ISIs) and equal to shuffled
  across halves. Added to ISI fractions it gives 56.7 against 54.0 (64 ISIs), the only gain here.
- **Rips finds no shape:** H0 of the whole cloud equals its shuffled control for every window
  length, inside regions it is within 0-3 points, and H1 is at chance (4%). Consecutive windows
  share all but one ISI, so the path moves in small steps whose sizes depend on w, not on the
  neuron.
- Not done: long cortical samples; stablebear.
