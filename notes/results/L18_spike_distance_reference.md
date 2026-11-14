# L18 The spike distance as the reference, on the cells of L17

2026-11-12, local session. **Worked as a test. The answer: the spike distance is not the gold
standard here.**

- **Within the driven stimulus:**
  - Retina frozen noise: best 80.8% 1-NN at q = 10/s, below multiscale square-root time bars
    (87.7%).
  - Allen movie: 27.3% at q = 1/s. That beats time bars (13.9%) but stays below ISI fractions
    (44.8%).
  - OpenScope gratings: 12.1% at q = 3/s, below ISI fractions (26.1%).
- **Without stimulus locking it is weak everywhere:** 5-13% within unrepeated noise, grey screen
  and spontaneous activity.
- **Across a stimulus change it sits near its shuffled control:**
  - 1-NN 6-12% against 5-12% shuffled, while ISI fractions keep 21-49%.
  - Both expectations held. At large q it falls to chance (0.9-2.1% at q = 100). At small q it
    becomes the spike-count difference (q = 0: 4-12%).
- **Cost:** about 100x the time of square-root time bars plus ISI fractions, even with a compiled
  kernel. The repo's numpy version would take about 0.6-1.8 h per matrix.

## Data

Same neurons and samples as L17 (`sb_full_table.py`'s `samples()` and `choose()`, seed 0):
- 60 neurons among those with >= 5 samples of >= 30 ISIs on both sides;
- at most 30 random samples per neuron and side.

| cell | A <-> B | candidates | samples | median spikes |
|---|---|---|---|---|
| retina 10 s, piece 2 dropped | `20180126_sr_le_sp` frozen trials <-> unrepeated 10 s pieces | 143 | 1701 + 1788 | 168 / 165 |
| OpenScope 10 s | `glo637908` grating blocks cut into 10 s <-> 10 s grey screen | 1243 | 1769 + 1490 | 62 / 74 |
| Allen 30 s | 829720705 natural movie repeats <-> 30 s spontaneous windows | 990 | 1783 + 1734 | 116 / 88 |

**Distances** (`scripts/sb_spike_distance.py`).

- **Spike distance (Victor-Purpura)** at q = 0, 1, 3, 10, 30, 100 per s.
  - q = 0 is the spike-count difference; 1/q is the time scale.
  - The matrices come from `spike_distance.vp_distance_matrix_fast`, which is new: the same
    recursion as `vp_distance_matrix`, compiled with numba. On a 500-sample subset of every cell
    the two agree to 7e-13 (table below).
- **van Rossum** at sigma = 10, 20, 50 ms: L2 between Gaussian-smoothed trains, 1 ms grid.
- **Spike distance + 0.5 x ISI fractions**, each median-scaled. This is computed at the best q
  of each cell, where the best q is picked by 1-NN on all neurons, so it is optimistic.
- **Shuffled control** at the same q values: ISIs permuted within every sample.

**Scores.**
- Within a stimulus: 1-NN / k-medoids.
- Across: 1-NN of A's samples among B's. Cross clustering means k-medoids on A, clusters paired
  with neurons, B's samples assigned to the nearest medoid (as in L17).
- No mean exists for these distances, so there is no k-means.
- Chance 1.7%.

**Honest q** (as `compare.py`):
- q is chosen by 1-NN on 30 random neurons and scored on the other 30, over 50 splits.
- These numbers are on 30 neurons: chance 3.3%, and they read higher than the 60-neuron
  columns.
- Next to each is the best q picked on the scored half itself. The gap between the two is the
  cost of choosing, and it is at most 1.5 points.

## Command

```sh
NUMBA_NUM_THREADS=8 python scripts/sb_spike_distance.py --dataset retina10 --drop-piece 2 --jobs 8
NUMBA_NUM_THREADS=8 python scripts/sb_spike_distance.py --dataset openscope10 --jobs 8
NUMBA_NUM_THREADS=8 python scripts/sb_spike_distance.py --dataset allen30 --jobs 8
```

Each cell is 1-NN / k-medoids, %. The **bold** rows are the q chosen honestly (held-out). The L17
rows are copied from `results/full_table_*.csv` (same samples).

## Retina 10 s, frozen <-> unrepeated (piece 2 dropped)

| method | setting | within frozen | within unrepeated | frozen -> unrepeated | unrepeated -> frozen |
|---|---|---|---|---|---|
| spike distance | q=0 | 11.0 / 19.7 | 8.9 / 17.8 | 11.7 / 12.2 | 10.3 / 11.9 |
| spike distance | q=1 | 37.7 / 24.2 | 9.7 / 18.3 | 10.1 / 12.6 | 10.1 / 15.6 |
| spike distance | q=3 | 61.8 / 36.0 | 9.8 / 16.8 | 9.1 / 10.2 | 9.6 / 12.6 |
| spike distance | q=10 | 80.8 / 44.3 | 6.8 / 14.7 | 6.2 / 7.5 | 6.9 / 7.9 |
| spike distance | q=30 | 73.3 / 34.6 | 2.0 / 10.3 | 1.5 / 4.2 | 2.1 / 5.8 |
| spike distance | q=100 | 29.5 / 11.9 | 1.6 / 6.2 | 0.9 / 2.5 | 1.8 / 2.4 |
| **spike distance, held-out q (30 neurons)** | | **84.5** (same half 84.5), q=10 in 50/50 | **18.3** (19.2), q=1 in 36/50 | **19.5** (21.0), q=0 or 1 | **18.7** (19.9), q=0, 1 or 3 |
| spike distance, shuffled | q=0 | 11.0 / 19.7 | 8.9 / 17.8 | 11.7 / 12.2 | 10.3 / 11.9 |
| spike distance, shuffled | q=3 | 9.6 / 16.5 | 7.9 / 16.6 | 8.6 / 11.6 | 8.6 / 12.3 |
| spike distance, shuffled | q=10 | 6.2 / 13.9 | 5.8 / 14.3 | 6.4 / 8.3 | 7.1 / 9.8 |
| van Rossum | 10 ms | 70.3 / 34.5 | 1.8 / 9.7 | 0.9 / 2.9 | 2.0 / 5.5 |
| van Rossum | 20 ms | 80.0 / 38.3 | 2.6 / 11.1 | 2.3 / 4.6 | 3.4 / 5.0 |
| van Rossum | 50 ms | 75.0 / 39.4 | 6.7 / 13.2 | 5.6 / 7.4 | 7.6 / 7.3 |
| **van Rossum, held-out sigma (30 neurons)** | | **83.6** (83.9), 20 ms | **12.0** (12.0), 50 ms | **10.9** (10.9), 50 ms | **12.8** (12.8), 50 ms |
| spike distance + 0.5 ISI fractions | q=10 (best for frozen) | 86.1 / 49.0 | 29.1 / 25.2 | 29.5 / 23.8 | 28.1 / 24.2 |
| spike distance + 0.5 ISI fractions | q=3 (best for unrepeated) | 72.8 / 43.4 | 32.4 / 26.0 | 33.5 / 25.7 | 30.2 / 22.8 |
| spike distance + 0.5 ISI fractions | q=0 (best across) | 42.7 / 31.0 | 36.9 / 26.8 | 42.7 / 22.8 | 39.3 / 22.6 |
| L17: multiscale sqrt time bars | | 87.7 / 48.2 | 5.0 / 12.7 | 5.9 / 6.7 | 6.1 / 7.7 |
| L17: multiscale sqrt time bars, shuffled | | 6.8 / 13.8 | 5.0 / 12.9 | 5.6 / 7.5 | 6.3 / 9.2 |
| L17: sqrt time bars + ISI fractions | | 91.1 / 53.3 | 29.5 / 25.7 | 30.4 / 22.7 | 28.4 / 19.8 |
| L17: ISI fractions (shuffled identical) | | 50.9 / 37.6 | 43.5 / 33.6 | 49.0 / 32.7 | 46.2 / 30.4 |
| L17: burst table | | 52.6 / 34.7 | 45.6 / 34.6 | 47.4 / 30.4 | 46.1 / 32.9 |
| L17: burst table, shuffled | | 44.3 / 37.1 | 43.3 / 33.8 | 44.8 / 29.9 | 41.6 / 31.3 |

## OpenScope 10 s, gratings <-> grey screen

| method | setting | within gratings | within grey | gratings -> grey | grey -> gratings |
|---|---|---|---|---|---|
| spike distance | q=0 | 4.4 / 12.5 | 4.6 / 11.9 | 4.1 / 5.9 | 4.2 / 4.2 |
| spike distance | q=1 | 9.2 / 12.7 | 13.0 / 15.4 | 6.6 / 6.1 | 5.9 / 5.0 |
| spike distance | q=3 | 12.1 / 13.8 | 13.1 / 15.1 | 6.0 / 5.5 | 6.4 / 5.5 |
| spike distance | q=10 | 11.9 / 13.9 | 11.3 / 14.8 | 5.5 / 5.4 | 4.9 / 5.5 |
| spike distance | q=30 | 8.3 / 11.3 | 6.7 / 10.9 | 3.8 / 4.0 | 3.2 / 5.1 |
| spike distance | q=100 | 5.0 / 7.4 | 3.6 / 8.5 | 2.1 / 1.5 | 1.3 / 2.0 |
| **spike distance, held-out q (30 neurons)** | | **17.2** (17.8), q=3 in 37/50 | **18.7** (19.3), q=1 or 3 | **10.9** (11.7), q=1 in 32/50 | **10.8** (11.5), q=3 in 32/50 |
| spike distance, shuffled | q=1 | 6.0 / 13.2 | 7.4 / 13.0 | 5.0 / 6.4 | 6.3 / 5.6 |
| spike distance, shuffled | q=3 | 5.5 / 11.7 | 7.0 / 13.0 | 4.7 / 4.6 | 5.1 / 5.5 |
| van Rossum | 10 ms | 6.2 / 9.0 | 5.0 / 10.5 | 3.0 / 3.7 | 1.9 / 4.3 |
| van Rossum | 20 ms | 9.4 / 10.8 | 8.3 / 13.2 | 3.7 / 4.2 | 3.7 / 4.5 |
| van Rossum | 50 ms | 10.1 / 12.1 | 10.5 / 14.2 | 3.8 / 4.3 | 4.8 / 4.7 |
| **van Rossum, held-out sigma (30 neurons)** | | **15.3** (15.4), 50 ms | **15.2** (15.3), 50 ms | **8.1** (8.2), 50 ms | **8.9** (8.9), 50 ms |
| spike distance + 0.5 ISI fractions | q=3 (best within, grey -> gratings) | 24.0 / 19.0 | 26.6 / 21.0 | 15.7 / 9.9 | 15.2 / 9.3 |
| spike distance + 0.5 ISI fractions | q=1 (best gratings -> grey) | 22.3 / 17.8 | 26.2 / 19.5 | 15.3 / 8.9 | 16.0 / 8.2 |
| L17: multiscale sqrt time bars | | 11.2 / 12.8 | 9.3 / 14.1 | 4.4 / 4.3 | 4.6 / 5.0 |
| L17: multiscale sqrt time bars, shuffled | | 4.6 / 11.4 | 6.5 / 11.5 | 4.0 / 4.6 | 4.8 / 4.0 |
| L17: sqrt time bars + ISI fractions | | 22.7 / 20.2 | 23.0 / 19.7 | 14.8 / 11.1 | 13.1 / 11.3 |
| L17: ISI fractions (shuffled identical) | | 26.1 / 21.9 | 29.4 / 22.4 | 21.0 / 14.5 | 24.0 / 12.3 |
| L17: burst table | | 31.4 / 22.4 | 34.8 / 23.6 | 25.4 / 14.8 | 28.1 / 13.1 |
| L17: burst table, shuffled | | 22.6 / 21.8 | 27.7 / 21.4 | 20.1 / 15.0 | 23.1 / 14.7 |

## Allen 30 s, movie <-> spontaneous

| method | setting | within movie | within spontaneous | movie -> spontaneous | spontaneous -> movie |
|---|---|---|---|---|---|
| spike distance | q=0 | 8.0 / 13.6 | 4.1 / 12.9 | 4.2 / 3.4 | 3.2 / 5.3 |
| spike distance | q=1 | 27.3 / 21.2 | 6.5 / 13.0 | 7.2 / 4.8 | 5.7 / 6.1 |
| spike distance | q=3 | 26.5 / 20.9 | 5.1 / 12.1 | 5.5 / 4.5 | 4.0 / 5.2 |
| spike distance | q=10 | 22.2 / 18.0 | 3.1 / 9.6 | 4.0 / 3.7 | 3.2 / 5.3 |
| spike distance | q=30 | 12.5 / 12.1 | 2.0 / 7.4 | 3.5 / 3.0 | 2.2 / 3.3 |
| spike distance | q=100 | 5.7 / 6.3 | 2.1 / 5.4 | 1.3 / 1.9 | 2.0 / 2.0 |
| **spike distance, held-out q (30 neurons)** | | **33.8** (34.2), q=1 in 45/50 | **11.3** (11.6), q=1 | **10.9** (11.0), q=1 | **9.7** (9.7), q=1 |
| spike distance, shuffled | q=1 | 8.3 / 14.1 | 6.4 / 12.8 | 5.6 / 5.7 | 4.1 / 7.1 |
| van Rossum | 10 ms | 10.8 / 9.5 | 2.3 / 7.2 | 3.4 / 3.7 | 2.4 / 3.8 |
| van Rossum | 20 ms | 16.0 / 13.8 | 3.2 / 7.7 | 3.9 / 3.2 | 3.1 / 3.8 |
| van Rossum | 50 ms | 23.1 / 16.4 | 3.3 / 9.7 | 4.4 / 3.5 | 3.3 / 4.3 |
| **van Rossum, held-out sigma (30 neurons)** | | **27.7** (27.7), 50 ms | **5.4** (5.6), 50 ms | **6.5** (6.6), 50 ms | **5.3** (5.8), 50 ms |
| spike distance + 0.5 ISI fractions | q=1 (best in every cell) | 43.2 / 26.2 | 21.5 / 20.6 | 15.5 / 8.0 | 13.6 / 9.6 |
| L17: multiscale sqrt time bars | | 13.9 / 12.1 | 2.5 / 8.1 | 3.9 / 2.7 | 2.2 / 4.5 |
| L17: multiscale sqrt time bars, shuffled | | 3.7 / 9.1 | 3.1 / 8.0 | 3.8 / 3.6 | 2.5 / 4.3 |
| L17: sqrt time bars + ISI fractions | | 29.4 / 21.2 | 13.1 / 20.1 | 11.2 / 8.1 | 10.2 / 10.7 |
| L17: ISI fractions (shuffled identical) | | 44.8 / 27.2 | 35.8 / 26.0 | 25.4 / 12.5 | 24.5 / 14.8 |
| L17: burst table | | 47.7 / 28.2 | 37.9 / 28.8 | 30.6 / 13.5 | 27.9 / 14.3 |
| L17: burst table, shuffled | | 39.6 / 24.6 | 31.5 / 24.7 | 22.8 / 12.9 | 22.4 / 12.2 |

## Run time per distance matrix

12-core laptop. Spike distance with 8 numba threads; the numpy version with 8 processes.

| | retina 10 s (3489 samples, ~166 spikes) | Allen 30 s (3517, ~100) | OpenScope 10 s (3259, ~68) |
|---|---|---|---|
| ISI fractions (stablebear, curves + pdist) | 1.3 s | 1 s | 0.8 s |
| multiscale sqrt time bars (3 scales, stablebear) | 8.0 s | 8 s | 3.7 s |
| van Rossum, sigma 10 / 20 / 50 ms | 1.0 / 1.7 / 4.2 s | similar | 1.2 / 1.7 / 4.3 s |
| spike distance, numba, one q | 87-91 s | 79 s (shuffled q=1) | 17-20 s |
| spike distance, repo numpy, 500-sample subset | 135 s | 106-121 s | 50-58 s |
| spike distance, repo numpy, full matrix (n^2 extrapolation) | about 1.8 h | about 1.5-1.7 h | about 0.6-0.7 h |

The numba kernel is new in `spike_distance.py`. On the 500-sample subsets its largest deviation
from the numpy version is 7e-13. For OpenScope and Allen the full-matrix times are from the
first run, before an edit added the "same half" column; the matrices were cached, so the rerun
did not time them again.

## What to take from it

- **The spike distance is not the ceiling on any cell.**
  - Retina frozen noise: multiscale square-root time bars beat it (87.7 vs 80.8 at the best q),
    with or without ISI fractions added (91.1 vs 86.1).
  - Allen movie: at q = 1/s (a 1 s time scale, so the rate profile, not spike timing) it reaches
    27.3%, double the time bars' 13.9%. It is still below ISI fractions (44.8%), and adding it
    to ISI fractions does not help (43.2%).
  - OpenScope gratings: below ISI fractions at every q.
  - Choosing q honestly costs at most 1.5 points: the best q is stable (retina q = 10, Allen
    q = 1, OpenScope q = 3 in most splits).
- **Both expectations held.**
  - *Large q is locked to the stimulus.* At q = 30-100 every cross cell is 0.9-3.5%, and so is
    every unrepeated or spontaneous cell (1.6-3.6%).
  - *Small q becomes a rate comparison.* q = 0 (the count difference) gives 4-12%, the same
    whether ISIs are shuffled or not. The best cross cells use q = 0-3 (6-12%), only 0-2 points
    above shuffled.
  - The spike distance has no regime between "time-locked" and "count" that carries identity
    the way the ISI distribution does.
- **On the retina, the count carries what crosses.** Across stimuli, q = 0 (count only) gives
  10.3-11.7%, and q = 1 does no better (10.1%). Firing rates differ a lot between retinal cells,
  but timing at any scale adds nothing across stimuli.
- **van Rossum tracks the spike distance at the matching time scale.** It is slightly lower
  everywhere (retina 80.0 at 20 ms against 80.8 at q = 10), and about 20-90x faster.
- **What failed:** the spike distance as a stimulus-free identifier, and as a better
  time-locked one than square-root time bars. Its cost is about 100x that of the time bars plus
  ISI fractions.
