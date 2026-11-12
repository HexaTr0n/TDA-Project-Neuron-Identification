# W3 Sharper versions of the ISI signature

2026-11-12, web chat. **Mostly did not work:** no other distance beats L1 on ISI fractions by
more than the seed noise. One useful finding: ISI order alone carries identity.

## Data and command

- Retina frozen noise, session `20180126_sr_le_sp`, 60 neurons (seed 0), 1645 trials of 9.94 s.
- 1-NN, and k-means with mean curves (mean of 3 seeds; range in brackets). numpy only.
- `python scripts/experiments/exp25_isi_distances.py` (log: `isi_distances.log`).

## Scores

| Signature | 1-NN | k-means |
|---|---|---|
| ISI fractions, L1 (reference) | 54.0 | 40.9 (40.4-41.5) |
| ISI fractions, L2 | 53.1 | 40.1 |
| ISI fractions, largest gap | 49.7 | 38.8 |
| Square root of the curve, L2 | 48.5 | 38.7 |
| 64 quantiles of log ISI, L2 (Wasserstein-2) | 54.1 | 42.7 (41.5-43.6) |
| Quantiles centred on the trial's mean log ISI | 42.8 | 38.6 |
| Joint fractions of consecutive ISI pairs (24 x 24 grid) | 52.8 | 38.8 |
| shuffled | 51.7 | 37.4 |
| **Joint minus product of its margins (order only)** | **27.7** | **24.0** |
| shuffled | 8.0 | 9.6 |
| ISI fractions + 0.5 x order-only part | 53.7 | 39.8 |

Joint fractions: share of consecutive pairs with `log ISI_k > a` and `log ISI_(k+1) > b`, a
two-parameter version of the stable rank. Order only: that table minus the product of the two
one-parameter curves, which is zero when consecutive ISIs are independent.

## What to take from it

- L1 on ISI fractions is as good as anything here. Wasserstein-2 on quantiles is 1-2 points
  better in clustering, inside the seed noise.
- Removing the mean log ISI costs 11 points in 1-NN, so the ISI scale is part of the identity.
- **ISI order alone identifies neurons at 27.7% against 8.0% shuffled.** This is the first clear
  order effect. It adds nothing on top of ISI fractions, so it is redundant with the
  distribution here, but it is not noise.
