# W12 One mixture-of-Gaussians pdf per spike, conditioned on the entire history

2026-11-13, web chat. **Did not work on the retina.** The fitted model finds almost no memory
between ISIs there, so the per-spike pdfs barely move and every readout is at or below ISI
fractions. The cloud's spread and its Rips H0 are above their shuffled controls with 3 states.

## Model

- Fitted to each trial separately: a hidden Markov model on x = log ISI. K hidden states, one
  Gaussian (mean, sd) per state, a K x K transition table, an initial distribution. EM
  (Baum-Welch), 60 iterations, sd at least 0.1, half a pseudo-count per transition; states
  sorted by mean. K = 2: 7 free parameters; K = 3: 14.
- After ISI number i the forward filter gives P(state of the next ISI | ISI_1 .. ISI_i). The
  predicted pdf of the next ISI is the mixture of the K Gaussians with those weights. It
  depends on the entire past through the filter.
- One point per spike: that pdf as its survival curve on a 46-point log-ISI grid. L1 between
  two points is Wasserstein-1 between two predicted pdfs.
- The parameters are fitted on the whole trial (also on later ISIs); only the weights use the
  past alone.

## Data and command

- Retina frozen noise, session `20180126_sr_le_sp`, 60 neurons (seed 0), 1645 trials of 9.94 s,
  median 161 ISIs. numpy, scipy and scikit-learn, not stablebear; H0 from a minimum spanning
  tree. One run; k-means is the mean of 3 seeds. Chance 1.7%.
- `python scripts/experiments/exp34_hmm_pdf_per_spike.py` (log: `hmm_pdf_per_spike.log`, 10 min).
- Across halves: first half of a trial matched among the second halves of other trials, with a
  model fitted to each half (81 ISIs).
- Control: ISIs shuffled before fitting. Regions: 8 k-means anchors on 20,000 pooled points
  (no labels used, but from the same neurons).
- Reference, ISI fractions: 54.0 1-NN, 40.9 k-means, 38.8 across halves.

## Scores: real (shuffled), %

| Readout | States | Full trial, 1-NN | k-means | Halves, 1-NN |
|---|---|---|---|---|
| Mean predicted pdf | 2 | 48.3 (50.6) | 38.6 (39.5) | 33.5 (35.0) |
| | 3 | 52.0 (51.1) | 40.8 (39.1) | 36.2 (36.4) |
| Model itself: next-ISI pdf after each state | 2 | 35.2 (35.7) | 20.4 (22.1) | 25.5 (23.8) |
| | 3 | 25.4 (20.8) | 18.0 (17.5) | 19.0 (15.8) |
| Cloud without homology (quantiles over spikes) | 2 | 48.0 (49.1) | 38.3 (40.0) | 32.8 (32.5) |
| | 3 | 49.1 (48.6) | 39.3 (40.2) | 34.5 (32.8) |
| Spread of the prediction only | 2 | 26.8 (23.8) | 18.0 (14.5) | 19.4 (15.9) |
| | 3 | 25.8 (16.8) | 21.8 (13.8) | 16.7 (12.9) |
| Rips H0, whole cloud | 2 | 17.9 (16.6) | 23.9 (21.4) | 15.1 (13.7) |
| | 3 | 17.6 (13.2) | 22.2 (19.6) | 13.7 (10.9) |
| Rips H0, divided by point count | 2 | 10.9 (12.0) | 14.1 (15.4) | 9.2 (9.7) |
| | 3 | 8.8 (7.4) | 14.9 (13.3) | 6.2 (5.6) |
| Rips H0 in 8 regions | 2 | 27.2 (23.2) | 24.8 (25.6) | 19.7 (20.2) |
| | 3 | 26.4 (22.3) | 25.2 (23.9) | 19.1 (16.1) |
| Region counts (twin without bars) | 2 | 25.9 (20.7) | 25.3 (25.2) | 19.2 (17.8) |
| | 3 | 25.7 (22.4) | 24.2 (25.1) | 20.2 (17.0) |
| ISI fractions + cloud without homology | 2 | 51.0 (52.7) | 39.2 (40.0) | 34.3 (34.3) |
| | 3 | 52.7 (51.6) | 42.0 (39.9) | 36.5 (33.9) |
| ISI fractions + Rips H0 in 8 regions | 2 | 45.3 (47.1) | 30.1 (29.6) | 28.4 (30.9) |
| | 3 | 43.6 (42.6) | 31.9 (29.8) | 29.0 (27.7) |

Mean probability of staying in the same state: 0.53 with 2 states (0.50 means no memory) and
0.38 with 3 (0.33 means no memory).

## What to take from it

- **The model finds almost no memory on the retina**, in line with W3 and L8: here ISI order is
  redundant with the ISI distribution. With no memory every spike gets nearly the same pdf.
- **Nothing beats ISI fractions.** The best row is the mean predicted pdf with 3 states (52.0),
  which is the per-trial mixture fit of W11 again.
- **History shows up weakly with 3 states:** the spread of the prediction scores 25.8 against
  16.8 shuffled, and Rips H0 of the cloud 17.6 against 13.2. Both are far below ISI fractions
  and add at most 1-2 points to them (42.0 against 40.9 in k-means, inside the seed noise).
- **The cloud lies on a simplex:** with K states every pdf is a mixture of the same K
  Gaussians, so the cloud of one trial sits on a segment (K = 2) or a triangle (K = 3). Rips
  on it can only see how the weights spread, which is why H0 says little.
- Not done: long cortical samples, where ISI order adds 9-11 points (L19x) and this model has
  something to find. Queued as L25.
