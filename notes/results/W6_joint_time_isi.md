# W6 A joint "when x how" signature against adding the two up

2026-11-12, web chat. **Did not work:** the joint signature is equal to plain time bars within
the noise, and below the simple sum on 1-NN.

## Data and command

- Retina frozen noise, three sessions, 60 random neurons each (seed 0), trials with >= 30 ISIs.
- Joint signature: N(t, a) = square root of the number of spikes in the last s seconds whose
  preceding ISI exceeds a, for a = 0, 5, 20, 80 ms. a = 0 is ordinary time bars; summed over t it
  is the ISI stable rank. Each threshold's curve is divided by its median distance, then all
  are concatenated. numpy only, coarse time grid (step s/4).
- 1-NN, and k-means with mean curves (mean of 3 seeds; seeds differ by 2-7 points).
- `python scripts/experiments/exp29_joint_time_isi.py` (log: `joint_time_isi.log`).

## Scores

1-NN / k-means, %, for sessions 20171215_sr_le_sp, 20180126_sr_le_sp, 20180405_sr_le.

| Signature | 20171215 | 20180126 | 20180405 |
|---|---|---|---|
| sqrt time bars 67 ms | 87.7 / 69.3 | 84.1 / 66.1 | 53.5 / 41.1 |
| multi-scale sqrt time bars | 88.4 / 68.3 | 88.0 / 68.9 | 55.6 / 42.2 |
| sqrt time bars + ISI fractions (sum) | 91.1 / 63.6 | 90.9 / 62.5 | 66.0 / 44.1 |
| joint, 67 ms | 81.2 / 67.0 | 82.7 / 63.6 | 51.1 / 39.9 |
| joint, multi-scale | 82.9 / 65.6 | 86.5 / 72.4 | 56.3 / 43.5 |
| joint multi-scale + ISI fractions | 89.1 / 57.0 | 89.8 / 62.7 | 64.7 / 41.6 |
| burst spikes and isolated spikes as two curves | 87.6 / 61.7 | 77.8 / 56.0 | 38.9 / 26.4 |
| joint, ISIs shuffled | 3.8 / 7.5 | 7.1 / 10.3 | 4.8 / 9.9 |

## What to take from it

- **Joint does not beat sum.** On 1-NN the plain sum (time bars + ISI fractions) is best in all
  three sessions; the joint version is 1-8 points lower.
- **For clustering the joint multi-scale version equals multi-scale time bars** (65.6 / 72.4 /
  43.5 against 68.3 / 68.9 / 42.2). The 72.4 on one session is inside the spread between seeds.
- **Splitting spikes by their preceding ISI spreads the same spikes over more curves,** each
  noisier. On 10 s trials that costs as much as the interaction adds.
- Together with L19's interim result (the joint spike-feature cloud does not beat ISI fractions
  on short samples), this says the qualities overlap: combinations land near the best
  ingredient for the task.
