# W4 ISI signature by retina session

2026-11-12, web chat. **Worked:** the ISI signature varies much less between sessions than time
bars do, but it follows the same sessions and neurons.

## Data and command

- Retina frozen noise, all nine sessions, trials with >= 30 ISIs, 60 random neurons per session
  (seed 0). ISI fractions, L1, k-means with mean curves (mean of 3 seeds). numpy only.
- `python scripts/experiments/exp26_isi_by_session.py` (log: `isi_by_session.log`).
- Reliability and margin as in W1 (from square-root time bars).

## Scores

| Session | ISI fractions k-means | ISI fractions 1-NN | Time bars k-means (W1) | Reliability |
|---|---|---|---|---|
| 20180126_sr_le_sp | 40.9 | 54.0 | 65.4 | 0.83 |
| 20180116_sr_le_fp | 31.7 | 39.9 | 68.1 | 0.87 |
| 20171215_sr_le_sp | 30.0 | 33.7 | 70.3 | 0.85 |
| 20180222_fr_re | 29.1 | 38.5 | 39.7 | 0.72 |
| 20180405_sr_le | 28.7 | 34.0 | 41.0 | 0.64 |
| 20180222_sr_le_sp | 28.2 | 36.7 | 53.5 | 0.78 |
| 20180405_fr_re | 27.0 | 33.1 | 37.1 | 0.63 |
| 20180316_sr_le_fp | 25.6 | 19.1 | 35.2 | 0.48 |
| 20180316_sr_le_sp | 24.3 | 23.1 | 42.5 | 0.68 |

Spearman with ISI clustering across sessions: reliability +0.78, time-bars clustering +0.67,
rate +0.18. Per neuron within sessions: margin +0.68, reliability +0.33, rate +0.16.

## What to take from it

- ISI clustering ranges 24-41% across sessions, time bars 35-70%. The intrinsic signature is the
  steadier of the two.
- The session used for most earlier work (20180126) is the best one for ISI fractions, by 9
  points. Averages over sessions are about 30% clustering and 35% 1-NN, not 41% and 54%.
- Sessions with reliable responses also have more distinctive ISI distributions. Both may reflect
  recording quality (cleaner spike sorting), which this analysis cannot separate.
