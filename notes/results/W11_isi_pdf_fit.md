# W11 Fit a pdf to the ISIs and compare the fitted pdfs

2026-11-13, web chat. **Did not beat the empirical distribution; one useful ablation.** A
two-parameter pdf (lognormal or gamma) keeps only half of the identification, so the identity
sits in the shape of the ISI distribution, not in its mean and spread.

## Data and command

- Retina frozen noise, session `20180126_sr_le_sp`, 60 neurons (seed 0), 1645 trials of 9.94 s;
  median 161 ISIs per trial, 81 per half trial. numpy, scipy and scikit-learn, not stablebear.
  One run; k-means is the mean of 3 seeds.
- `python scripts/pilots/isi_pdf_fit.py` (about 5 min on 2 cores).
- Each model is fitted to one trial's log ISIs by maximum likelihood (mixtures by EM,
  `reg_covar` 1e-3). The fitted pdf is turned into its survival curve on the grid of the ISI
  fractions, so L1 between curves is Wasserstein-1 between fitted pdfs. Hellinger is L2
  between square roots of the fitted densities (for the empirical row: a histogram).
- Across halves: the first half of a trial is matched among the second halves of all other
  trials.

## Scores (%, chance 1.7)

| Model | Parameters | Full trial, 1-NN, W1 | 1-NN, Hellinger | k-means, W1 (range) | Halves, 1-NN, W1 | Halves, Hellinger |
|---|---|---|---|---|---|---|
| Empirical (ISI fractions) | none | 54.0 | 51.9 | 40.9 (40.4-41.5) | 38.8 | 31.1 |
| One Gaussian on log ISI (lognormal) | 2 | 27.2 | 26.8 | 31.4 (31.3-31.5) | 16.7 | 17.3 |
| Gamma on the ISI | 2 | 27.7 | 26.9 | 31.8 (31.3-32.4) | 18.7 | 18.9 |
| Mixture of 2 Gaussians | 5 | 50.6 | 52.5 | 38.7 (38.4-39.3) | 34.6 | 36.0 |
| Mixture of 3 Gaussians | 8 | 52.2 | 53.2 | 39.5 (38.2-41.3) | 37.4 | 35.5 |
| Mixture of 4 Gaussians | 11 | 53.7 | 55.0 | 40.4 (39.6-41.2) | 37.8 | 35.4 |
| Mixture, K chosen per trial by BIC | 2-11 | 49.5 | 51.0 | 38.9 (37.1-39.9) | 32.2 | 33.1 |
| Kernel estimate (one Gaussian per ISI, Silverman width) | 1 width | 50.3 | 55.4 | 37.9 (37.1-38.8) | 34.5 | 37.0 |

BIC chose K = 1 in 29% of the fits, K = 2 in 62%, K = 3 in 8% and K = 4 in 1%.

## What to take from it

- **No fitted pdf beats the empirical distribution with Wasserstein-1.** W1 is already
  insensitive to sampling noise in the curve, so smoothing has nothing to add. The mixtures
  approach the empirical score from below as K grows.
- **Two parameters lose half:** 27% against 54%. A second mixture component recovers nearly
  all of it (50.6%), which fits the picture of two kinds of interval, inside a burst and
  between bursts.
- **Choosing K per trial hurts** (49.5 against 50.6 for a fixed K = 2), because K itself then
  varies between trials of the same neuron. If a mixture is used, fix K for all trials.
- **Fitting helps only density-based distances:** Hellinger rises from 51.9 (histogram) to
  55.0-55.4 (4 components, kernel). That is level with W1 on the empirical curve (54.0);
  1.4 points is about one standard error here.
- On half trials (81 ISIs) the empirical curve with W1 is still the best (38.8).
- Not done: other datasets, across real stimulus changes, stablebear. Given the above I would
  not queue it.
