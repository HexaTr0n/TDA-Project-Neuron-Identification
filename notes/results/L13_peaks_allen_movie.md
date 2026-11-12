# L13 Peak matching on the Allen movie

2026-11-12, local session. **Worked for reliable units only.** On the 60 most reliable units,
rate peaks matched in time (sigma 10 ms) give 38.6% 1-NN (shuffled 1.7-2.5%), close to
square-root time bars (42.3%). On 60 random units peak matching is near chance (6-11%).
Peaks with time removed give 22-32% for both sets, but their shuffled control is nearly as high.

## Data

- Allen FC session 829720705, natural movie, 60 repeats of 30 s (`allen829720705_movie`).
- 991 units with >= 10 repeats of >= 30 ISIs.
- Reliability is the mean pairwise PSTH correlation, 50 ms bins (as L9): median 0.012, max 0.434.
- **60 random units** (seed 0): 3156 repeats, median 129 spikes, median reliability 0.003.
- **60 most reliable:** 2975 repeats, median 133 spikes, median reliability 0.215.

**Peaks** (`barcodes.peak_events`):
- H0 of the superlevel sets of the rate smoothed with sigma 10, 20, 50 ms;
- mass = prominence x sigma x sqrt(2 pi), peaks with mass < 0.2 spikes dropped;
- about 90 / 75 / 50 peaks per repeat.

**Distances.**
- *Matched in time:* `spike_distance.matching_matrix`, q = 10 per s.
- *Time removed:* bars [0, mass], stable rank in stablebear, L1.
- References: square-root time bars (stablebear Betti curves, L9) and ISI fractions.

**Control:** ISIs permuted within each repeat (computed at sigma 20 ms only). 1-NN / k-medoids, chance 1.7%.

## Command

```sh
python scripts/sb_peaks.py --data data/allen/allen829720705.parquet --session allen829720705_movie
```

## Scores

| signature | 60 random | shuffled | 60 most reliable | shuffled |
|---|---|---|---|---|
| ISI fractions | 44.9 / 26.1 | 44.9 / 26.1 | 41.7 / 24.0 | 41.7 / 24.0 |
| sqrt time bars | 11.1 / 11.4 | 4.6 / 8.8 | **42.3 / 29.7** | 2.5 / 8.0 |
| peaks matched in time, 10 ms | 11.3 / 13.3 | | **38.6 / 23.8** | |
| peaks matched in time, 20 ms | 8.8 / 9.8 | 2.3 / 8.1 | 29.1 / 16.1 | 1.7 / 7.5 |
| peaks matched in time, 50 ms | 6.4 / 6.9 | | 17.5 / 11.2 | |
| peaks, time removed, 10 ms | 31.7 / 21.6 | | 30.7 / 18.0 | |
| peaks, time removed, 20 ms | 29.0 / 20.7 | 25.1 / 20.3 | 27.1 / 20.6 | 22.2 / 19.5 |
| peaks, time removed, 50 ms | 22.4 / 20.7 | | 23.6 / 20.6 | |

## What to take from it

- **Time-locked identity on the movie needs reliable units, as on OpenScope (L9).**
  - Random units: square-root time bars 11.1% and peak matching 6-11%.
  - The 60 most reliable: 42.3% and 38.6%.
  - The shuffled control falls to 2%, so this really is timing.
  - The most reliable Allen units (median reliability 0.21) are close to the retina's level (0.25),
    but still reach only half the retina's 85%.
- **Finer is better:** peak matching improves as sigma drops from 50 to 10 ms (17.5 -> 29.1 ->
  38.6%). At sigma 10 ms it nearly matches time bars, which compare raw spikes.
- **Peaks with time removed measure burstiness, not timing.**
  - They score 22-32% whether or not the unit is reliable.
  - The shuffled control stays 4-5 points below.
  - They do worse than ISI fractions (42-45%), which carry the same information more directly.
- **What failed:** peak matching on typical units; and peak matching never beats square-root
  time bars.
