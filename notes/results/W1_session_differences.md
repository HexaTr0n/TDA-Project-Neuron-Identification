# W1 Why retina sessions differ

2026-11-12, web chat. **Worked:** the difference between sessions is response reliability, not
firing rate or trial count.

## Data and command

- Retina frozen noise, all nine sessions, trials with >= 30 ISIs, 60 random neurons per session
  (seed 0; two sessions have only 53 and 38 usable neurons).
- Square-root time bars (67 ms), k-means with mean curves, mean of 3 seeds. numpy only.
- `python scripts/experiments/exp24_session_differences.py` (log: `session_differences.log`).
- Reliability of a neuron = correlation between its mean curve over even trials and over odd
  trials. Margin = distance from its mean curve to the nearest other neuron's, divided by the
  average distance of its own trials from its mean.

## Scores

| Session | Neurons | Clustering | Median rate (Hz) | Median reliability | Median margin | Trials per neuron |
|---|---|---|---|---|---|---|
| 20171215_sr_le_sp | 60 | 70.3 | 4.5 | 0.85 | 1.07 | 18 |
| 20180116_sr_le_fp | 60 | 68.1 | 6.5 | 0.87 | 1.00 | 29 |
| 20180126_sr_le_sp | 60 | 65.4 | 14.5 | 0.83 | 0.64 | 30 |
| 20180222_sr_le_sp | 53 | 53.5 | 7.0 | 0.78 | 0.55 | 30 |
| 20180316_sr_le_sp | 60 | 42.5 | 5.7 | 0.68 | 0.49 | 29 |
| 20180405_sr_le | 60 | 41.0 | 8.7 | 0.64 | 0.47 | 30 |
| 20180222_fr_re | 60 | 39.7 | 11.2 | 0.72 | 0.47 | 29 |
| 20180405_fr_re | 57 | 37.1 | 12.3 | 0.63 | 0.45 | 30 |
| 20180316_sr_le_fp | 38 | 35.2 | 7.7 | 0.48 | 0.93 | 4 |

Spearman correlation with clustering across the nine sessions: reliability +0.93, margin +0.65,
neurons +0.55, trials per neuron +0.01, rate -0.48.

Per neuron (508 neurons), share of its trials in its main cluster: margin +0.49 within sessions,
reliability +0.14 within sessions (+0.30 pooled), rate -0.35. Neurons in the top reliability
quarter keep 86% of their trials together, the bottom half 64-68%.

## What to take from it

- A time-locked signature can only be as good as the neuron's response is repeatable. The three
  good sessions have median reliability 0.83-0.87, the weak ones 0.48-0.78.
- Within a session, what matters is how far a neuron's mean curve is from its nearest
  neighbour's, relative to its own trial-to-trial spread. Higher-rate neurons are kept together
  less well, even after the square root.
- Trial count does not explain it, except for `20180316_sr_le_fp`, where only 4 trials per
  neuron pass the 30-ISI filter.
- **Not checked:** whether the stimulus-free ISI signature has the same session dependence (W4).
