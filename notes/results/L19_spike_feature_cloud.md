# L19 The spike-feature cloud

2026-11-13, local session. **Mostly failed; one partial success.**

- **The joint 7-coordinate cloud never beats ISI fractions or the burst table.** Its best readout
  (sliced W1 of the whole cloud) roughly ties ISI fractions on OpenScope and Allen, and is
  12-16 points lower on the retina.
- **The population coordinates (6-7) hurt the joint cloud.** Dropping them raises it on 4 of 5
  datasets (retina 10 s sliced W1 35 -> 41; retina 40 s 52 -> 60).
  - On their own they identify almost nothing (2-12% 1-NN).
  - Their signal is real where synapses exist: on Allen and OpenScope 124 s, 6-7 alone gives
    7-12% against 1-2% with ISIs shuffled or other units from another trial; on the retina,
    2-4% in every condition.
- **Partial success: on long samples the regional homology beats the region counts and ISI
  fractions.**
  - OpenScope 124 s, coordinates 1-2 only (delay regions, m = 2, K = 16): H0 67.2% 1-NN, against
    region counts 59.3 and ISI fractions 64.2. Shuffled: 57.3.
  - Retina 40 s: 64.3 against counts 63.6 and ISI fractions 64.3 (a tie).
  - The burst table is still higher on both (71.8 and 68.3).
- **The expectation held.** On short samples (25 points) the regional bars add 0-2 points over
  the counts. On long samples (100 points) they add 1-8 points.
- **The time coordinate (8) does not help.** The one-stimulus cloud's H0 is 1-9 points below the
  stimulus-free one, and weight 4 lowers it further.

## Data

Same neurons and samples as L17 (`sb_full_table.samples()` and `choose()`, seed 0). The script
recomputes ISI fractions, the burst table and multiscale square-root time bars on the same samples
and matches L17 exactly.

| dataset | A <-> B | candidates | samples | median spikes | points per cloud |
|---|---|---|---|---|---|
| retina 10 s, piece 2 dropped | `20180126_sr_le_sp` frozen trials <-> unrepeated 10 s pieces | 143 | 1701 + 1788 | 168 / 165 | 25 |
| OpenScope 10 s | `glo637908` gratings cut into 10 s <-> 10 s grey screen | 1243 | 1769 + 1490 | 62 / 74 | 25 |
| Allen 30 s | 829720705 natural movie <-> 30 s spontaneous | 990 | 1783 + 1734 | 116 / 88 | 25 |
| retina 40 s (piece 2 included, flagged) | 4 frozen trials joined <-> 4 unrepeated pieces joined | 148 | 420 + 1796 | 609 / 629 | 100 (100 samples have fewer points and are drawn with replacement) |
| OpenScope 124 s | grating blocks, within only; units with >= 10 blocks of >= 103 ISIs (as L11) | 1513 | 2310 | 496 | 100 |

**Population ("other units").** Every unit of the session on the same trial or time window,
minus the unit itself: 156 retinal cells, 2031 OpenScope units, 1344 Allen units.

**Method** (`scripts/sb_feature_cloud.py`):
- **Coordinates:** one point per spike, for spikes with a neighbour on both sides, with the eight
  coordinates of the board task.
  - The other-unit counts are pop minus own in every window. Pop is every unit's spikes on that
    trial; own is the unit's original train. This keeps the ISI-shuffled control honest.
- **Scaling and regions:**
  - Coordinates are standardised with the mean and sd of spikes from up to 300 samples per side
    of the neurons outside the comparison. The one-stimulus version uses side A only.
  - Regions come from k-means anchors on 20,000 of those scaled spikes.
- **Readouts:** 20 subsamples of 25 or 100 points.
  - Rips H0 and H0+H1 inside each region (stablebear). Stable ranks are averaged over
    subsamples and summed over regions. A region with fewer than 2 points gets two coincident
    points, which gives an empty barcode.
  - Region counts, averaged over subsamples, L1.
  - Sliced W1: the 20 pooled subsamples projected on 64 fixed directions, 50 quantiles each, L1.
  - Mean point, L2.
- **Controls:**
  - (a) ISIs permuted, then every coordinate recomputed.
  - (b) Other units taken from the next trial or window of the same session. This is skipped for
    sets without 6-7, where it would be identical to the real run.
- **Scores (as L17):** 1-NN / k-medoids / k-means with `sb.mean` centres.
  - Within each stimulus and across (A -> B: 1-NN of A among B; clustering: cluster A, assign B).
  - Chance 1.7%.
  - Ablations at K = 16 only. Sliced W1 and the mean point do not depend on K.

## Commands

```sh
python scripts/sb_feature_cloud.py --dataset retina10 --drop-piece 2
python scripts/sb_feature_cloud.py --dataset openscope10
python scripts/sb_feature_cloud.py --dataset allen30
python scripts/sb_feature_cloud.py --dataset retina40
python scripts/sb_feature_cloud.py --dataset openscope124
```

## Summary: 1-NN, within stimulus A / A -> B (%), K = 16

OpenScope 124 s has within only. Shuffled = ISIs permuted. "Other trial" = other units from
another trial. The full tables, with k-medoids, k-means and both directions, are at the end.

| signature | retina 10 s | OpenScope 10 s | Allen 30 s | retina 40 s | OpenScope 124 s |
|---|---|---|---|---|---|
| ISI fractions (shuffled identical) | 50.9 / 49.0 | 26.1 / 21.0 | 44.8 / 25.4 | 64.3 / 73.6 | 64.2 |
| burst table | 52.6 / 47.4 | 31.4 / 25.4 | 47.7 / 30.6 | 68.3 / 74.8 | 71.8 |
| multiscale sqrt time bars | 87.7 / 5.9 | 11.2 / 4.4 | 13.9 / 3.9 | 80.7 / 34.3 | 13.7 |
| L17 delay regions k=8, H0 (25 points) | 30.4 / 28.8 | 16.7 / 13.7 | 30.0 / 21.5 | 35.5 / 44.8 | - |
| **all 1-7: sliced W1** | 35.2 / 36.4 | 27.1 / 21.6 | 45.5 / 25.7 | 52.4 / 63.3 | 63.4 |
| shuffled | 27.7 / 30.3 | 19.4 / 16.8 | 31.5 / 16.9 | 41.7 / 49.3 | 45.5 |
| **all 1-7: H0** | 23.0 / 24.0 | 21.8 / 16.8 | 32.9 / 20.7 | 45.2 / 55.2 | 62.4 |
| shuffled | 21.9 / 21.1 | 15.7 / 13.4 | 23.7 / 13.6 | 36.4 / 47.9 | 41.3 |
| other trial | 22.8 / 25.4 | 14.4 / 12.8 | 26.1 / 18.3 | 38.8 / 52.1 | 44.7 |
| all 1-7: region counts | 22.3 / 24.7 | 20.8 / 16.8 | 32.8 / 20.8 | 42.9 / 50.2 | 56.2 |
| all 1-7: H0+H1 | 18.2 / 19.3 | 14.9 / 13.5 | 24.4 / 15.3 | 40.2 / 50.2 | 56.9 |
| all 1-7: mean point | 31.3 / 31.5 | 19.6 / 17.8 | 36.5 / 20.5 | 42.1 / 54.5 | 51.6 |
| all 1-7, K = 64: H0 | 21.6 / 20.7 | 16.8 / 15.0 | 30.6 / 19.1 | 44.0 / 53.8 | 64.6 |
| all 1-7, K = 64: region counts | 24.8 / 24.0 | 22.4 / 21.0 | 38.6 / 23.7 | 47.4 / 56.7 | 64.9 |
| without 6-7: H0 | 33.2 / 32.5 | 20.1 / 17.9 | 32.1 / 22.6 | 55.2 / 61.0 | 64.7 |
| without 6-7: sliced W1 | 41.2 / 42.0 | 24.6 / 21.1 | 40.3 / 26.8 | 59.8 / 66.7 | 64.1 |
| without 3-5: H0 | 13.3 / 11.9 | 18.5 / 15.8 | 31.0 / 18.3 | 23.6 / 30.5 | 60.4 |
| without 1-2: H0 | 19.0 / 18.2 | 19.9 / 16.2 | 31.4 / 19.9 | 33.8 / 46.0 | 54.1 |
| **only 1-2 (delay regions m = 2): H0** | 38.6 / 36.1 | 22.7 / 17.8 | 34.7 / 24.3 | 64.3 / 68.1 | **67.2** |
| shuffled | 33.5 / 32.6 | 17.1 / 15.9 | 26.7 / 17.4 | 62.1 / 63.8 | 57.3 |
| only 1-2: region counts | 36.9 / 37.3 | 24.2 / 18.3 | 32.8 / 22.2 | 63.6 / 64.8 | 59.3 |
| only 1-2: sliced W1 | 45.0 / 43.5 | 26.7 / 23.1 | 41.6 / 27.3 | 60.2 / 68.1 | 64.8 |
| only 3-5: sliced W1 | 37.9 / 35.5 | 21.4 / 17.8 | 33.7 / 22.3 | 55.5 / 61.2 | 50.6 |
| only 6-7: sliced W1 | 2.6 / 1.9 | 4.7 / 4.0 | 10.0 / 6.3 | 2.4 / 2.1 | 11.9 |
| shuffled | 2.5 / 2.4 | 3.1 / 2.3 | 2.2 / 2.1 | 1.2 / 3.3 | 0.9 |
| one-stimulus 1-8: H0 | 21.2 | 17.2 | 32.1 | 36.4 | 54.5 |
| one-stimulus 1-8: sliced W1 | 35.6 | 24.0 | 44.8 | 45.7 | 57.1 |
| one-stimulus 1-8, time x4: H0 | 18.8 | 11.0 | 26.6 | 25.5 | 35.2 |

## What to take from it

- **One joint cloud with equal weights dilutes the best coordinates.**
  - The ISI coordinates (1-2) carry the most identity. The own-count coordinates (3-5) come
    second.
  - Coordinates 6-7 and 8 add noise. Each coordinate has the same weight after standardising,
    so seven coordinates make the useful ones a small part of every distance.
  - No weights were tuned with labels, as the task required. A fair next step would be to
    learn the scaling on outside neurons.
- **Population coupling is real but weak.** Coordinates 6-7 alone reach 7-12% on Allen and
  OpenScope 124 s, and the "other trial" control drops them to 1-2%. In the joint cloud the same
  control lowers H0 by 7-18 points on those datasets (Allen 32.9 -> 26.1, OpenScope 124 s
  62.4 -> 44.7), so the coupling does carry identity. It also lowers H0 by 6 points on retina
  40 s and 7 on OpenScope 10 s, so not all of it is synaptic. It is
  not enough to pay for the noise it adds on short samples.
- **The homology readout helps only where clouds are large.**
  - 25 points: H0 is at most 2 points above the region counts, and below them for coordinates
    3-5 alone (Allen 22.0 against 29.0). H0+H1 is below H0 everywhere.
  - 100 points: H0 adds 1-8 points over the counts at K = 16, with the largest gain on
    OpenScope 124 s.
  - At K = 64 the counts catch up (OpenScope 124 s: counts 64.9, H0 64.6), because each region
    then holds too few points for its bars to mean much.
- **Order shows up through the cloud.** Shuffling ISIs costs the full cloud 1-21 points (H0 on
  OpenScope 124 s 62.4 -> 41.3), much more than any earlier Rips method. Part of this comes from
  coordinates 3-7, which change when the spikes move. For coordinates 1-2 alone it is 2-10 points.
- **What failed:**
  - the joint cloud as a better identifier than ISI fractions;
  - H0+H1 (below H0 in every cell);
  - K = 64 regions on short samples;
  - the time coordinate.

## Full tables

Each cell is 1-NN / k-medoids / k-means, %. The ISI fractions, burst table and time-bar rows
are recomputed on the same samples and are identical to L17; the delay-regions row is copied
from L17.

## retina10_without_piece2

60 neurons, 1701 + 1788 samples; clouds needing replacement: 0

### References (same samples)

| signature | control | within frozen | within unrepeated | frozen -> unrepeated | unrepeated -> frozen |
|---|---|---|---|---|---|
| ISI fractions |  | 50.9 / 37.6 / 40.4 | 43.5 / 33.6 / 37.0 | 49.0 / 32.7 / 32.8 | 46.2 / 30.4 / 35.2 |
| burst table |  | 52.6 / 34.7 / 40.8 | 45.6 / 34.6 / 39.3 | 47.4 / 30.4 / 35.3 | 46.1 / 32.9 / 37.9 |
| multiscale sqrt time bars |  | 87.7 / 48.2 / 70.2 | 5.0 / 12.7 / 13.6 | 5.9 / 6.7 / 8.1 | 6.1 / 7.7 / 8.3 |
| ISI fractions | shuffled | 50.9 / 37.6 / 40.4 | 43.5 / 33.6 / 37.0 | 49.0 / 32.7 / 32.8 | 46.2 / 30.4 / 35.2 |
| burst table | shuffled | 44.3 / 37.1 / 37.4 | 43.3 / 33.8 / 35.6 | 44.8 / 29.9 / 32.3 | 41.6 / 31.3 / 35.4 |
| multiscale sqrt time bars | shuffled | 6.8 / 13.8 / 14.1 | 5.0 / 12.9 / 13.8 | 5.6 / 7.5 / 7.5 | 6.3 / 9.2 / 9.1 |
| L17: delay regions k=8, H0 |  | 30.4 / 26.4 / 31.0 | 26.5 / 24.4 / 28.4 | 28.8 / 21.0 / 24.9 | 26.6 / 20.4 / 24.5 |
| L17: delay regions k=8, H0 | shuffled | 25.5 / 25.5 / 27.7 | 22.9 / 23.8 / 26.5 | 23.3 / 20.0 / 20.6 | 23.2 / 20.0 / 22.1 |

### Stimulus-free cloud, coordinates 1-7

| signature | control | within frozen | within unrepeated | frozen -> unrepeated | unrepeated -> frozen |
|---|---|---|---|---|---|
| H0, K = 16 |  | 23.0 / 23.2 / 25.1 | 22.9 / 24.8 / 26.2 | 24.0 / 17.7 / 20.3 | 24.4 / 19.9 / 20.3 |
| H0, K = 16 | shuffled | 21.9 / 22.4 / 24.2 | 17.0 / 20.7 / 23.9 | 21.1 / 17.8 / 19.5 | 20.5 / 17.2 / 19.0 |
| H0, K = 16 | others from another trial | 22.8 / 23.5 / 25.8 | 24.8 / 23.4 / 24.9 | 25.4 / 17.7 / 20.9 | 23.3 / 19.6 / 20.9 |
| H0+H1, K = 16 |  | 18.2 / 21.2 / 21.3 | 20.0 / 19.7 / 20.7 | 19.3 / 15.9 / 15.2 | 18.3 / 15.9 / 12.7 |
| H0+H1, K = 16 | shuffled | 17.6 / 19.5 / 20.6 | 14.4 / 18.2 / 19.7 | 16.9 / 13.6 / 13.8 | 16.2 / 13.3 / 16.0 |
| H0+H1, K = 16 | others from another trial | 18.9 / 20.7 / 22.8 | 18.2 / 21.0 / 21.3 | 18.6 / 14.8 / 15.0 | 18.7 / 14.5 / 18.3 |
| region counts, K = 16 |  | 22.3 / 24.0 / 26.3 | 23.4 / 24.8 / 26.0 | 24.7 / 18.0 / 19.0 | 24.6 / 20.7 / 21.5 |
| region counts, K = 16 | shuffled | 21.2 / 22.7 / 26.0 | 17.3 / 22.6 / 24.2 | 20.6 / 16.9 / 17.6 | 19.8 / 17.6 / 20.8 |
| region counts, K = 16 | others from another trial | 22.9 / 25.6 / 26.6 | 23.7 / 23.3 / 26.4 | 23.2 / 18.2 / 20.0 | 21.6 / 19.5 / 21.5 |
| sliced W1 |  | 35.2 / 27.7 / 28.5 | 32.2 / 25.9 / 27.7 | 36.4 / 20.2 / 22.4 | 33.3 / 23.2 / 23.9 |
| sliced W1 | shuffled | 27.7 / 26.2 / 26.2 | 27.7 / 24.8 / 24.4 | 30.3 / 21.5 / 18.9 | 27.3 / 19.7 / 19.1 |
| sliced W1 | others from another trial | 38.6 / 26.5 / 28.5 | 32.1 / 26.8 / 28.4 | 35.8 / 19.0 / 22.0 | 32.2 / 24.2 / 24.0 |
| mean point |  | 31.3 / 26.0 / 27.4 | 28.3 / 25.7 / 26.2 | 31.5 / 19.2 / 19.9 | 28.4 / 22.0 / 22.3 |
| mean point | shuffled | 27.6 / 24.2 / 24.3 | 24.8 / 23.5 / 24.4 | 28.3 / 19.5 / 19.2 | 25.8 / 20.0 / 19.5 |
| mean point | others from another trial | 29.7 / 24.9 / 25.6 | 28.2 / 25.1 / 25.9 | 30.5 / 17.0 / 19.2 | 26.5 / 21.3 / 22.3 |
| H0, K = 64 |  | 21.6 / 22.0 / 25.2 | 19.7 / 20.9 / 25.2 | 20.7 / 14.4 / 19.7 | 19.9 / 16.6 / 21.0 |
| H0, K = 64 | shuffled | 18.0 / 19.3 / 22.8 | 16.8 / 19.7 / 21.6 | 18.7 / 13.4 / 17.2 | 16.3 / 13.1 / 16.8 |
| H0, K = 64 | others from another trial | 22.5 / 19.5 / 24.8 | 19.1 / 19.4 / 23.4 | 21.1 / 11.4 / 17.3 | 18.1 / 12.1 / 17.5 |
| H0+H1, K = 64 |  | 17.6 / 18.9 / 11.0 | 17.8 / 18.0 / 8.3 | 17.5 / 12.9 / 6.6 | 16.2 / 14.4 / 5.2 |
| H0+H1, K = 64 | shuffled | 13.9 / 17.9 / 10.6 | 13.6 / 16.4 / 9.4 | 15.5 / 11.6 / 7.2 | 13.0 / 11.6 / 6.5 |
| H0+H1, K = 64 | others from another trial | 18.8 / 18.8 / 11.5 | 16.9 / 17.8 / 8.8 | 17.5 / 12.7 / 7.3 | 14.9 / 11.5 / 5.3 |
| region counts, K = 64 |  | 24.8 / 23.9 / 27.2 | 23.3 / 23.3 / 27.5 | 24.0 / 17.3 / 19.6 | 23.0 / 18.9 / 21.3 |
| region counts, K = 64 | shuffled | 20.2 / 20.8 / 25.0 | 19.1 / 21.3 / 24.3 | 21.5 / 14.8 / 21.1 | 19.9 / 16.2 / 20.6 |
| region counts, K = 64 | others from another trial | 24.5 / 22.6 / 26.9 | 21.9 / 21.5 / 26.5 | 24.9 / 16.2 / 21.0 | 23.0 / 15.2 / 22.3 |

### Ablations, K = 16 (stimulus-free)

| signature | control | within frozen | within unrepeated | frozen -> unrepeated | unrepeated -> frozen |
|---|---|---|---|---|---|
| without 1-2: H0 |  | 19.0 / 21.0 / 24.2 | 16.9 / 21.5 / 22.7 | 18.2 / 15.0 / 17.6 | 18.5 / 14.7 / 17.2 |
| without 1-2: H0 | shuffled | 15.2 / 19.5 / 21.6 | 16.7 / 20.5 / 22.3 | 17.7 / 14.1 / 16.4 | 15.5 / 15.5 / 16.1 |
| without 1-2: H0 | others from another trial | 17.9 / 21.5 / 22.6 | 17.2 / 21.4 / 22.7 | 18.6 / 15.3 / 16.2 | 18.2 / 17.1 / 17.4 |
| without 1-2: region counts |  | 20.4 / 22.7 / 23.6 | 18.2 / 20.8 / 24.2 | 19.3 / 15.3 / 16.5 | 18.6 / 14.7 / 19.5 |
| without 1-2: region counts | shuffled | 16.8 / 21.3 / 21.9 | 15.7 / 20.4 / 21.6 | 19.2 / 13.0 / 16.1 | 16.9 / 17.0 / 15.8 |
| without 1-2: region counts | others from another trial | 18.0 / 22.5 / 23.7 | 18.0 / 21.3 / 23.8 | 20.0 / 15.5 / 15.5 | 17.7 / 14.5 / 18.4 |
| without 1-2: sliced W1 |  | 32.7 / 27.0 / 27.7 | 28.0 / 26.6 / 27.6 | 32.3 / 19.7 / 22.9 | 31.3 / 21.2 / 23.2 |
| without 1-2: sliced W1 | shuffled | 24.0 / 24.2 / 25.1 | 24.7 / 22.6 / 24.4 | 27.2 / 18.8 / 18.8 | 24.5 / 18.2 / 18.6 |
| without 1-2: sliced W1 | others from another trial | 35.0 / 25.2 / 27.6 | 27.9 / 25.8 / 27.4 | 31.4 / 19.3 / 20.6 | 28.9 / 20.0 / 24.9 |
| without 1-2: mean point |  | 30.4 / 26.5 / 26.3 | 26.1 / 24.6 / 25.6 | 28.9 / 20.1 / 21.1 | 27.8 / 21.6 / 22.8 |
| without 1-2: mean point | shuffled | 26.3 / 24.3 / 23.5 | 23.7 / 23.2 / 25.6 | 25.6 / 18.2 / 17.8 | 23.7 / 18.4 / 20.9 |
| without 1-2: mean point | others from another trial | 28.2 / 25.3 / 26.6 | 26.6 / 24.3 / 25.4 | 29.4 / 18.5 / 19.6 | 25.6 / 18.8 / 21.3 |
| without 1-2: H0+H1 |  | 16.8 / 18.2 / 19.5 | 13.9 / 17.6 / 19.5 | 16.0 / 10.9 / 12.3 | 15.4 / 13.3 / 14.3 |
| without 1-2: H0+H1 | shuffled | 13.0 / 17.2 / 18.3 | 13.0 / 17.7 / 18.3 | 13.9 / 10.1 / 12.0 | 14.1 / 11.7 / 13.3 |
| without 1-2: H0+H1 | others from another trial | 16.6 / 17.8 / 19.3 | 15.3 / 18.1 / 18.7 | 14.3 / 12.8 / 11.9 | 14.2 / 12.6 / 13.9 |
| without 3-5: H0 |  | 13.3 / 17.1 / 17.8 | 12.8 / 16.1 / 17.6 | 11.9 / 9.9 / 10.3 | 12.3 / 9.3 / 10.8 |
| without 3-5: H0 | shuffled | 10.6 / 15.6 / 16.9 | 11.2 / 15.6 / 16.7 | 11.9 / 8.1 / 9.1 | 12.0 / 8.1 / 10.3 |
| without 3-5: H0 | others from another trial | 12.0 / 16.1 / 17.3 | 12.2 / 15.3 / 17.1 | 12.5 / 9.0 / 10.2 | 12.7 / 9.6 / 10.3 |
| without 3-5: region counts |  | 10.0 / 15.7 / 17.5 | 9.8 / 16.4 / 16.5 | 9.6 / 10.3 / 11.0 | 9.4 / 9.1 / 10.4 |
| without 3-5: region counts | shuffled | 9.6 / 15.6 / 16.2 | 8.6 / 15.4 / 16.6 | 10.2 / 9.6 / 10.0 | 10.7 / 9.5 / 9.2 |
| without 3-5: region counts | others from another trial | 10.2 / 16.1 / 16.9 | 10.4 / 15.9 / 17.3 | 9.3 / 9.5 / 8.6 | 9.6 / 8.3 / 10.5 |
| without 3-5: sliced W1 |  | 26.6 / 22.8 / 22.2 | 22.5 / 21.3 / 23.0 | 25.2 / 16.6 / 15.5 | 23.9 / 16.1 / 16.8 |
| without 3-5: sliced W1 | shuffled | 22.9 / 21.3 / 21.5 | 21.5 / 20.2 / 21.0 | 24.9 / 15.0 / 14.6 | 22.4 / 15.6 / 15.6 |
| without 3-5: sliced W1 | others from another trial | 25.5 / 21.3 / 21.6 | 21.2 / 21.0 / 22.3 | 25.5 / 16.2 / 13.5 | 21.1 / 17.5 / 16.0 |
| without 3-5: mean point |  | 10.1 / 17.2 / 17.9 | 10.1 / 16.9 / 17.2 | 9.9 / 9.5 / 11.3 | 10.6 / 12.8 / 10.1 |
| without 3-5: mean point | shuffled | 10.1 / 16.5 / 17.0 | 9.7 / 17.1 / 16.6 | 11.3 / 9.8 / 10.1 | 11.0 / 11.1 / 11.3 |
| without 3-5: mean point | others from another trial | 10.3 / 17.0 / 17.2 | 8.6 / 16.5 / 17.6 | 10.7 / 9.5 / 9.5 | 9.1 / 10.8 / 12.3 |
| without 3-5: H0+H1 |  | 8.6 / 14.1 / 13.1 | 9.0 / 13.5 / 13.5 | 9.2 / 8.7 / 7.2 | 10.2 / 7.0 / 8.1 |
| without 3-5: H0+H1 | shuffled | 7.8 / 13.4 / 12.8 | 9.5 / 13.4 / 12.9 | 9.3 / 6.5 / 5.8 | 9.1 / 7.2 / 6.3 |
| without 3-5: H0+H1 | others from another trial | 10.9 / 13.3 / 13.0 | 8.5 / 14.0 / 12.5 | 9.8 / 6.7 / 6.0 | 9.1 / 8.3 / 5.9 |
| without 6-7: H0 |  | 33.2 / 29.9 / 31.4 | 31.1 / 27.5 / 32.2 | 32.5 / 24.2 / 28.0 | 30.9 / 26.0 / 27.9 |
| without 6-7: H0 | shuffled | 26.2 / 27.3 / 27.7 | 23.8 / 25.3 / 27.9 | 24.5 / 19.5 / 22.4 | 24.0 / 21.2 / 27.3 |
| without 6-7: region counts |  | 31.2 / 30.3 / 31.7 | 28.1 / 28.5 / 31.8 | 30.8 / 23.9 / 27.7 | 29.1 / 26.9 / 29.3 |
| without 6-7: region counts | shuffled | 26.1 / 25.0 / 28.2 | 21.8 / 24.9 / 26.3 | 23.8 / 20.9 / 21.6 | 23.7 / 21.9 / 22.5 |
| without 6-7: sliced W1 |  | 41.2 / 30.6 / 33.6 | 36.9 / 30.4 / 31.8 | 42.0 / 25.6 / 28.0 | 39.2 / 26.9 / 29.3 |
| without 6-7: sliced W1 | shuffled | 35.0 / 30.2 / 32.5 | 31.8 / 27.7 / 29.0 | 31.3 / 24.2 / 23.9 | 31.3 / 24.6 / 24.6 |
| without 6-7: mean point |  | 33.9 / 28.4 / 29.9 | 32.7 / 27.3 / 29.1 | 33.0 / 20.6 / 24.5 | 31.6 / 24.0 / 24.5 |
| without 6-7: mean point | shuffled | 32.6 / 29.6 / 30.4 | 29.3 / 27.0 / 26.1 | 29.3 / 22.7 / 25.1 | 28.1 / 24.8 / 22.1 |
| without 6-7: H0+H1 |  | 24.2 / 22.6 / 24.4 | 24.5 / 23.8 / 23.4 | 24.5 / 16.9 / 17.7 | 24.7 / 18.5 / 18.4 |
| without 6-7: H0+H1 | shuffled | 22.6 / 20.7 / 23.5 | 18.2 / 21.9 / 22.1 | 18.8 / 14.9 / 18.5 | 18.7 / 15.5 / 17.4 |
| only 1-2: H0 |  | 38.6 / 30.2 / 34.9 | 36.1 / 29.9 / 36.7 | 36.1 / 24.5 / 26.5 | 35.8 / 28.2 / 34.9 |
| only 1-2: H0 | shuffled | 33.5 / 31.7 / 32.7 | 30.0 / 28.9 / 34.5 | 32.6 / 24.8 / 27.9 | 31.4 / 25.7 / 31.5 |
| only 1-2: region counts |  | 36.9 / 31.9 / 35.6 | 36.0 / 31.3 / 37.1 | 37.3 / 26.3 / 31.6 | 35.3 / 26.6 / 33.9 |
| only 1-2: region counts | shuffled | 33.3 / 29.7 / 34.0 | 29.8 / 27.8 / 32.7 | 31.1 / 23.3 / 26.7 | 30.6 / 25.1 / 29.7 |
| only 1-2: sliced W1 |  | 45.0 / 33.2 / 34.0 | 42.8 / 32.4 / 33.2 | 43.5 / 27.0 / 27.8 | 42.1 / 28.0 / 32.3 |
| only 1-2: sliced W1 | shuffled | 40.4 / 31.0 / 33.6 | 34.9 / 29.8 / 31.7 | 37.7 / 23.9 / 28.9 | 35.1 / 25.6 / 28.9 |
| only 1-2: mean point |  | 10.7 / 17.5 / 18.0 | 9.1 / 18.9 / 19.3 | 9.3 / 11.6 / 13.0 | 9.8 / 10.8 / 11.2 |
| only 1-2: mean point | shuffled | 10.5 / 18.5 / 18.6 | 9.9 / 18.7 / 18.7 | 9.2 / 11.8 / 11.7 | 9.2 / 12.1 / 12.0 |
| only 1-2: H0+H1 |  | 25.7 / 21.2 / 19.8 | 24.0 / 21.5 / 19.9 | 26.3 / 16.1 / 14.9 | 25.1 / 15.2 / 16.5 |
| only 1-2: H0+H1 | shuffled | 22.2 / 19.6 / 19.4 | 21.5 / 19.0 / 19.0 | 23.5 / 12.0 / 13.5 | 21.1 / 15.1 / 14.2 |
| only 3-5: H0 |  | 26.6 / 26.6 / 27.0 | 23.2 / 26.2 / 27.3 | 25.7 / 22.7 / 21.3 | 26.7 / 21.0 / 24.0 |
| only 3-5: H0 | shuffled | 21.9 / 23.4 / 23.3 | 20.7 / 21.6 / 22.8 | 21.0 / 16.7 / 18.5 | 19.5 / 16.9 / 20.0 |
| only 3-5: region counts |  | 28.9 / 27.1 / 30.5 | 25.8 / 26.0 / 29.5 | 28.8 / 21.8 / 25.1 | 27.7 / 22.3 / 27.4 |
| only 3-5: region counts | shuffled | 26.2 / 24.6 / 28.2 | 21.6 / 23.2 / 24.1 | 23.0 / 18.8 / 19.1 | 22.2 / 17.6 / 22.2 |
| only 3-5: sliced W1 |  | 37.9 / 30.6 / 32.7 | 33.3 / 30.8 / 31.9 | 35.5 / 24.9 / 27.6 | 35.7 / 26.7 / 28.9 |
| only 3-5: sliced W1 | shuffled | 29.6 / 29.1 / 30.1 | 28.0 / 25.7 / 28.3 | 27.5 / 20.2 / 22.9 | 26.6 / 24.0 / 25.8 |
| only 3-5: mean point |  | 32.6 / 29.2 / 29.6 | 30.1 / 27.6 / 29.4 | 32.6 / 22.9 / 23.0 | 30.9 / 24.9 / 23.8 |
| only 3-5: mean point | shuffled | 30.1 / 29.7 / 29.5 | 26.7 / 27.0 / 26.8 | 26.8 / 20.5 / 21.9 | 26.3 / 24.3 / 23.3 |
| only 3-5: H0+H1 |  | 21.6 / 22.1 / 20.0 | 18.5 / 20.8 / 17.0 | 20.2 / 15.0 / 12.5 | 20.8 / 16.0 / 11.9 |
| only 3-5: H0+H1 | shuffled | 18.2 / 20.2 / 17.3 | 15.2 / 18.6 / 16.0 | 16.5 / 12.8 / 11.4 | 15.0 / 12.9 / 11.1 |
| only 6-7: H0 |  | 2.6 / 8.6 / 8.7 | 2.1 / 7.9 / 8.4 | 2.1 / 2.8 / 2.5 | 2.5 / 2.1 / 2.8 |
| only 6-7: H0 | shuffled | 1.9 / 8.5 / 8.8 | 2.1 / 8.5 / 8.8 | 2.8 / 2.3 / 2.6 | 2.2 / 3.0 / 2.1 |
| only 6-7: H0 | others from another trial | 2.6 / 9.0 / 9.0 | 2.2 / 8.2 / 8.4 | 2.1 / 2.6 / 1.7 | 1.6 / 1.6 / 2.1 |
| only 6-7: region counts |  | 2.6 / 8.8 / 8.9 | 2.1 / 8.2 / 8.7 | 2.1 / 1.7 / 2.1 | 2.3 / 2.4 / 2.3 |
| only 6-7: region counts | shuffled | 1.9 / 8.4 / 9.0 | 2.1 / 8.6 / 8.4 | 2.4 / 2.1 / 2.4 | 2.5 / 2.0 / 2.1 |
| only 6-7: region counts | others from another trial | 2.9 / 8.6 / 9.1 | 1.9 / 7.8 / 8.3 | 3.0 / 2.1 / 1.8 | 2.5 / 2.0 / 1.6 |
| only 6-7: sliced W1 |  | 2.6 / 8.8 / 9.2 | 2.3 / 8.0 / 9.0 | 1.9 / 2.5 / 2.2 | 2.1 / 2.3 / 2.8 |
| only 6-7: sliced W1 | shuffled | 2.5 / 8.8 / 8.8 | 2.1 / 8.3 / 8.0 | 2.4 / 2.2 / 2.0 | 2.0 / 2.4 / 2.4 |
| only 6-7: sliced W1 | others from another trial | 2.4 / 9.0 / 9.3 | 2.3 / 7.6 / 8.3 | 2.4 / 2.4 / 1.4 | 2.1 / 2.8 / 2.2 |
| only 6-7: mean point |  | 1.7 / 9.1 / 9.4 | 2.1 / 8.9 / 8.4 | 2.7 / 1.9 / 2.4 | 2.6 / 2.5 / 2.2 |
| only 6-7: mean point | shuffled | 2.3 / 8.8 / 8.9 | 1.6 / 8.9 / 8.6 | 1.8 / 2.2 / 1.8 | 2.2 / 2.2 / 2.4 |
| only 6-7: mean point | others from another trial | 1.8 / 9.4 / 9.3 | 1.7 / 8.6 / 9.1 | 1.8 / 1.8 / 1.8 | 1.8 / 2.4 / 2.2 |
| only 6-7: H0+H1 |  | 2.2 / 8.6 / 7.7 | 2.1 / 8.3 / 7.8 | 2.8 / 2.5 / 1.6 | 2.3 / 2.8 / 1.8 |
| only 6-7: H0+H1 | shuffled | 1.6 / 8.1 / 7.9 | 1.5 / 7.7 / 7.3 | 2.4 / 2.4 / 1.9 | 2.6 / 2.2 / 1.8 |
| only 6-7: H0+H1 | others from another trial | 2.1 / 8.2 / 7.9 | 2.1 / 7.9 / 7.8 | 1.7 / 1.9 / 2.1 | 2.0 / 1.2 / 2.2 |

### One-stimulus cloud, coordinates 1-8, within frozen

| signature | control | within frozen |
|---|---|---|
| all 1-8: H0, K = 16 |  | 21.2 / 22.4 / 23.9 |
| all 1-8: H0, K = 16 | shuffled | 15.9 / 20.6 / 21.9 |
| all 1-8: H0, K = 16 | others from another trial | 20.9 / 22.6 / 24.8 |
| all 1-8: H0+H1, K = 16 |  | 14.1 / 18.8 / 20.0 |
| all 1-8: H0+H1, K = 16 | shuffled | 14.6 / 18.2 / 18.2 |
| all 1-8: H0+H1, K = 16 | others from another trial | 16.7 / 17.9 / 19.5 |
| all 1-8: region counts, K = 16 |  | 20.3 / 23.0 / 26.0 |
| all 1-8: region counts, K = 16 | shuffled | 16.3 / 20.9 / 22.0 |
| all 1-8: region counts, K = 16 | others from another trial | 20.8 / 22.8 / 24.8 |
| all 1-8: sliced W1 |  | 35.6 / 27.4 / 27.3 |
| all 1-8: sliced W1 | shuffled | 26.7 / 23.2 / 24.4 |
| all 1-8: sliced W1 | others from another trial | 37.7 / 27.2 / 27.4 |
| all 1-8: mean point |  | 29.9 / 25.8 / 27.2 |
| all 1-8: mean point | shuffled | 26.9 / 23.8 / 25.0 |
| all 1-8: mean point | others from another trial | 32.4 / 26.9 / 25.6 |
| all 1-8: H0, K = 64 |  | 19.7 / 19.1 / 23.9 |
| all 1-8: H0, K = 64 | shuffled | 15.9 / 18.1 / 21.3 |
| all 1-8: H0, K = 64 | others from another trial | 18.6 / 18.6 / 22.5 |
| all 1-8: H0+H1, K = 64 |  | 16.3 / 18.3 / 8.8 |
| all 1-8: H0+H1, K = 64 | shuffled | 12.7 / 16.2 / 8.8 |
| all 1-8: H0+H1, K = 64 | others from another trial | 16.1 / 17.3 / 10.5 |
| all 1-8: region counts, K = 64 |  | 23.8 / 22.7 / 26.2 |
| all 1-8: region counts, K = 64 | shuffled | 18.6 / 20.0 / 22.9 |
| all 1-8: region counts, K = 64 | others from another trial | 23.3 / 21.6 / 25.8 |
| all 1-8, time x4: H0, K = 16 |  | 18.8 / 20.8 / 21.9 |
| all 1-8, time x4: H0, K = 16 | shuffled | 10.6 / 16.9 / 17.5 |
| all 1-8, time x4: H0, K = 16 | others from another trial | 19.8 / 18.9 / 22.4 |
| all 1-8, time x4: H0+H1, K = 16 |  | 14.1 / 15.2 / 16.4 |
| all 1-8, time x4: H0+H1, K = 16 | shuffled | 9.8 / 13.9 / 13.8 |
| all 1-8, time x4: H0+H1, K = 16 | others from another trial | 14.8 / 16.4 / 14.6 |
| all 1-8, time x4: region counts, K = 16 |  | 20.1 / 19.9 / 22.5 |
| all 1-8, time x4: region counts, K = 16 | shuffled | 11.9 / 16.6 / 18.1 |
| all 1-8, time x4: region counts, K = 16 | others from another trial | 19.5 / 19.8 / 23.3 |
| all 1-8, time x4: sliced W1 |  | 34.3 / 25.6 / 25.7 |
| all 1-8, time x4: sliced W1 | shuffled | 18.2 / 18.9 / 20.1 |
| all 1-8, time x4: sliced W1 | others from another trial | 34.1 / 24.9 / 25.5 |
| all 1-8, time x4: mean point |  | 27.6 / 24.1 / 25.4 |
| all 1-8, time x4: mean point | shuffled | 23.7 / 20.2 / 20.5 |
| all 1-8, time x4: mean point | others from another trial | 29.0 / 23.5 / 24.8 |
| all 1-8, time x4: H0, K = 64 |  | 19.0 / 19.1 / 23.3 |
| all 1-8, time x4: H0, K = 64 | shuffled | 11.7 / 15.3 / 19.1 |
| all 1-8, time x4: H0, K = 64 | others from another trial | 18.6 / 16.8 / 22.5 |
| all 1-8, time x4: H0+H1, K = 64 |  | 18.0 / 17.3 / 10.8 |
| all 1-8, time x4: H0+H1, K = 64 | shuffled | 9.8 / 14.5 / 7.7 |
| all 1-8, time x4: H0+H1, K = 64 | others from another trial | 16.8 / 16.0 / 9.3 |
| all 1-8, time x4: region counts, K = 64 |  | 23.0 / 20.0 / 25.7 |
| all 1-8, time x4: region counts, K = 64 | shuffled | 14.1 / 17.8 / 21.4 |
| all 1-8, time x4: region counts, K = 64 | others from another trial | 23.9 / 19.1 / 25.6 |

## openscope10

60 neurons, 1769 + 1490 samples; clouds needing replacement: 0

### References (same samples)

| signature | control | within gratings | within grey | gratings -> grey | grey -> gratings |
|---|---|---|---|---|---|
| ISI fractions |  | 26.1 / 21.9 / 22.4 | 29.4 / 22.4 / 22.3 | 21.0 / 14.5 / 15.8 | 24.0 / 12.3 / 12.4 |
| burst table |  | 31.4 / 22.4 / 23.1 | 34.8 / 23.6 / 23.2 | 25.4 / 14.8 / 16.6 | 28.1 / 13.1 / 15.1 |
| multiscale sqrt time bars |  | 11.2 / 12.8 / 10.5 | 9.3 / 14.1 / 13.2 | 4.4 / 4.3 / 3.1 | 4.6 / 5.0 / 3.6 |
| ISI fractions | shuffled | 26.1 / 21.9 / 22.4 | 29.4 / 22.4 / 22.3 | 21.0 / 14.5 / 15.8 | 24.0 / 12.3 / 12.4 |
| burst table | shuffled | 22.6 / 21.8 / 22.2 | 27.7 / 21.4 / 21.3 | 20.1 / 15.0 / 15.3 | 23.1 / 14.7 / 15.0 |
| multiscale sqrt time bars | shuffled | 4.6 / 11.4 / 9.6 | 6.5 / 11.5 / 10.5 | 4.0 / 4.6 / 4.6 | 4.8 / 4.0 / 5.1 |
| L17: delay regions k=8, H0 |  | 16.7 / 16.7 / 16.7 | 19.6 / 20.2 / 19.7 | 13.7 / 11.6 / 11.6 | 15.4 / 13.1 / 11.6 |
| L17: delay regions k=8, H0 | shuffled | 14.5 / 16.2 / 17.0 | 13.5 / 17.8 / 18.0 | 11.1 / 10.6 / 9.7 | 12.3 / 8.6 / 10.3 |

### Stimulus-free cloud, coordinates 1-7

| signature | control | within gratings | within grey | gratings -> grey | grey -> gratings |
|---|---|---|---|---|---|
| H0, K = 16 |  | 21.8 / 20.6 / 20.1 | 22.3 / 20.8 / 21.1 | 16.8 / 12.3 / 12.3 | 18.9 / 11.8 / 12.4 |
| H0, K = 16 | shuffled | 15.7 / 17.7 / 17.9 | 17.0 / 18.9 / 19.9 | 13.4 / 12.0 / 12.2 | 15.2 / 12.1 / 12.0 |
| H0, K = 16 | others from another trial | 14.4 / 18.0 / 18.2 | 18.8 / 16.9 / 18.1 | 12.8 / 8.7 / 11.3 | 15.2 / 8.0 / 10.8 |
| H0+H1, K = 16 |  | 14.9 / 15.7 / 15.8 | 15.3 / 17.2 / 16.3 | 13.5 / 9.3 / 8.5 | 16.6 / 8.3 / 8.8 |
| H0+H1, K = 16 | shuffled | 12.4 / 16.1 / 15.8 | 13.1 / 16.8 / 16.3 | 9.8 / 9.3 / 8.5 | 12.0 / 8.5 / 9.5 |
| H0+H1, K = 16 | others from another trial | 12.7 / 14.9 / 14.8 | 14.5 / 15.6 / 16.3 | 9.7 / 7.5 / 7.1 | 11.7 / 7.7 / 8.8 |
| region counts, K = 16 |  | 20.8 / 20.5 / 21.8 | 22.4 / 21.1 / 21.7 | 16.8 / 13.5 / 13.6 | 19.1 / 11.6 / 12.1 |
| region counts, K = 16 | shuffled | 14.6 / 18.7 / 18.9 | 16.4 / 19.3 / 20.0 | 13.7 / 11.9 / 10.0 | 14.6 / 10.5 / 11.9 |
| region counts, K = 16 | others from another trial | 14.4 / 17.8 / 19.4 | 18.9 / 17.4 / 19.1 | 13.6 / 11.5 / 14.0 | 16.3 / 10.2 / 11.1 |
| sliced W1 |  | 27.1 / 20.0 / 21.1 | 32.6 / 21.9 / 20.9 | 21.6 / 11.6 / 14.6 | 25.8 / 10.9 / 12.5 |
| sliced W1 | shuffled | 19.4 / 18.9 / 18.9 | 19.7 / 19.0 / 19.2 | 16.8 / 12.1 / 12.1 | 19.3 / 10.3 / 9.5 |
| sliced W1 | others from another trial | 17.8 / 16.3 / 16.6 | 20.1 / 17.0 / 17.4 | 14.6 / 9.6 / 9.4 | 17.0 / 9.7 / 8.4 |
| mean point |  | 19.6 / 18.9 / 18.6 | 24.2 / 20.9 / 20.2 | 17.8 / 11.9 / 12.0 | 19.9 / 12.2 / 10.6 |
| mean point | shuffled | 17.1 / 16.8 / 18.1 | 19.6 / 18.2 / 17.2 | 16.6 / 10.1 / 11.8 | 18.7 / 9.4 / 11.3 |
| mean point | others from another trial | 13.7 / 16.2 / 16.6 | 15.7 / 18.0 / 16.1 | 12.5 / 9.8 / 8.3 | 14.2 / 10.2 / 8.9 |
| H0, K = 64 |  | 16.8 / 17.4 / 19.1 | 19.2 / 18.5 / 19.4 | 15.0 / 10.5 / 13.0 | 18.3 / 11.6 / 9.7 |
| H0, K = 64 | shuffled | 14.0 / 15.4 / 16.4 | 15.5 / 16.4 / 17.3 | 12.5 / 10.3 / 10.1 | 13.8 / 5.9 / 9.0 |
| H0, K = 64 | others from another trial | 13.5 / 14.3 / 15.9 | 14.8 / 15.0 / 15.6 | 11.9 / 7.6 / 9.1 | 13.4 / 7.1 / 8.5 |
| H0+H1, K = 64 |  | 14.2 / 15.0 / 9.5 | 14.8 / 16.2 / 12.1 | 12.7 / 9.6 / 5.8 | 12.2 / 7.5 / 5.9 |
| H0+H1, K = 64 | shuffled | 9.9 / 14.8 / 10.5 | 12.1 / 15.1 / 11.7 | 10.9 / 8.9 / 6.6 | 11.2 / 5.8 / 4.6 |
| H0+H1, K = 64 | others from another trial | 9.9 / 13.0 / 8.4 | 11.3 / 13.1 / 10.6 | 9.4 / 6.8 / 3.5 | 9.3 / 5.6 / 4.6 |
| region counts, K = 64 |  | 22.4 / 19.8 / 21.2 | 26.4 / 20.3 / 21.9 | 21.0 / 13.7 / 13.5 | 21.6 / 10.5 / 12.9 |
| region counts, K = 64 | shuffled | 16.1 / 17.4 / 17.4 | 17.3 / 18.2 / 20.3 | 15.3 / 11.6 / 13.8 | 15.8 / 9.2 / 11.0 |
| region counts, K = 64 | others from another trial | 15.5 / 16.1 / 17.1 | 17.1 / 16.2 / 17.3 | 14.5 / 8.9 / 8.9 | 15.9 / 8.6 / 11.0 |

### Ablations, K = 16 (stimulus-free)

| signature | control | within gratings | within grey | gratings -> grey | grey -> gratings |
|---|---|---|---|---|---|
| without 1-2: H0 |  | 19.9 / 18.8 / 18.5 | 22.1 / 20.1 / 21.4 | 16.2 / 11.7 / 12.2 | 18.4 / 10.6 / 12.0 |
| without 1-2: H0 | shuffled | 14.5 / 16.7 / 17.4 | 14.1 / 16.9 / 18.4 | 13.6 / 10.8 / 10.3 | 14.8 / 8.6 / 9.8 |
| without 1-2: H0 | others from another trial | 12.1 / 15.8 / 16.1 | 14.5 / 16.6 / 16.1 | 11.4 / 10.1 / 8.5 | 13.0 / 8.5 / 6.2 |
| without 1-2: region counts |  | 19.7 / 18.4 / 19.5 | 23.5 / 20.2 / 21.1 | 15.7 / 11.5 / 12.3 | 19.1 / 10.5 / 11.9 |
| without 1-2: region counts | shuffled | 14.1 / 17.8 / 18.2 | 14.9 / 19.1 / 19.2 | 13.6 / 11.1 / 10.9 | 13.7 / 10.6 / 10.6 |
| without 1-2: region counts | others from another trial | 13.9 / 16.6 / 17.2 | 16.4 / 16.7 / 17.2 | 11.4 / 8.7 / 9.9 | 13.7 / 9.2 / 8.6 |
| without 1-2: sliced W1 |  | 24.5 / 18.1 / 20.0 | 29.5 / 21.3 / 21.8 | 20.9 / 13.2 / 13.8 | 24.7 / 11.4 / 11.6 |
| without 1-2: sliced W1 | shuffled | 16.4 / 18.5 / 18.2 | 18.2 / 17.9 / 18.1 | 16.5 / 10.3 / 12.3 | 17.5 / 10.6 / 10.5 |
| without 1-2: sliced W1 | others from another trial | 16.2 / 16.2 / 16.1 | 18.9 / 17.3 / 15.4 | 14.4 / 8.3 / 9.2 | 15.4 / 8.6 / 8.5 |
| without 1-2: mean point |  | 17.5 / 18.9 / 18.6 | 21.6 / 20.5 / 21.1 | 16.9 / 12.3 / 12.7 | 18.9 / 11.7 / 10.3 |
| without 1-2: mean point | shuffled | 15.5 / 17.2 / 18.1 | 19.0 / 18.3 / 17.2 | 15.4 / 10.5 / 9.7 | 16.5 / 9.9 / 9.6 |
| without 1-2: mean point | others from another trial | 11.8 / 16.2 / 16.8 | 15.0 / 17.0 / 15.6 | 11.6 / 8.5 / 10.3 | 12.4 / 8.8 / 7.3 |
| without 1-2: H0+H1 |  | 15.4 / 15.6 / 14.9 | 17.0 / 17.3 / 15.9 | 13.9 / 10.3 / 9.2 | 14.4 / 9.7 / 8.9 |
| without 1-2: H0+H1 | shuffled | 10.9 / 14.5 / 14.5 | 11.5 / 16.3 / 15.8 | 9.7 / 9.3 / 9.7 | 12.7 / 7.7 / 8.1 |
| without 1-2: H0+H1 | others from another trial | 9.6 / 13.6 / 12.9 | 11.5 / 14.6 / 13.8 | 9.3 / 6.6 / 6.0 | 10.2 / 7.7 / 7.1 |
| without 3-5: H0 |  | 18.5 / 17.6 / 17.5 | 20.4 / 17.9 / 20.3 | 15.8 / 12.1 / 11.5 | 17.0 / 8.9 / 11.6 |
| without 3-5: H0 | shuffled | 14.0 / 15.5 / 15.8 | 14.2 / 16.8 / 17.7 | 12.5 / 10.3 / 10.4 | 13.1 / 9.6 / 9.6 |
| without 3-5: H0 | others from another trial | 12.9 / 15.1 / 15.0 | 12.3 / 14.7 / 15.4 | 12.4 / 7.2 / 7.9 | 12.1 / 8.3 / 8.8 |
| without 3-5: region counts |  | 18.9 / 16.9 / 18.5 | 19.1 / 18.7 / 20.6 | 15.5 / 10.4 / 11.2 | 16.6 / 9.4 / 11.8 |
| without 3-5: region counts | shuffled | 11.4 / 15.7 / 16.5 | 12.7 / 17.7 / 19.1 | 11.5 / 9.5 / 10.8 | 12.3 / 9.0 / 10.9 |
| without 3-5: region counts | others from another trial | 12.5 / 14.6 / 16.2 | 11.7 / 14.6 / 15.5 | 12.5 / 7.0 / 7.8 | 12.6 / 7.3 / 8.5 |
| without 3-5: sliced W1 |  | 23.5 / 16.8 / 16.8 | 24.4 / 17.9 / 19.0 | 18.5 / 9.9 / 10.0 | 21.5 / 9.0 / 8.7 |
| without 3-5: sliced W1 | shuffled | 15.3 / 15.5 / 14.7 | 15.7 / 15.4 / 16.6 | 12.6 / 9.8 / 9.5 | 14.2 / 7.7 / 8.0 |
| without 3-5: sliced W1 | others from another trial | 13.7 / 14.4 / 14.5 | 12.1 / 13.9 / 13.6 | 11.3 / 6.4 / 7.2 | 12.9 / 6.5 / 6.6 |
| without 3-5: mean point |  | 8.1 / 15.1 / 14.6 | 9.0 / 15.6 / 15.4 | 7.9 / 7.0 / 7.7 | 8.1 / 7.2 / 6.8 |
| without 3-5: mean point | shuffled | 6.4 / 12.7 / 13.2 | 7.0 / 13.8 / 14.0 | 5.9 / 5.7 / 7.7 | 6.4 / 5.3 / 7.6 |
| without 3-5: mean point | others from another trial | 5.2 / 12.8 / 12.7 | 4.5 / 13.4 / 12.7 | 4.7 / 5.6 / 5.8 | 5.3 / 5.7 / 5.3 |
| without 3-5: H0+H1 |  | 12.8 / 15.3 / 14.9 | 15.4 / 14.4 / 15.6 | 11.9 / 8.9 / 7.7 | 10.9 / 6.4 / 9.0 |
| without 3-5: H0+H1 | shuffled | 10.1 / 13.5 / 13.9 | 10.9 / 14.4 / 14.6 | 9.0 / 7.0 / 8.4 | 9.8 / 7.0 / 6.6 |
| without 3-5: H0+H1 | others from another trial | 10.2 / 13.5 / 12.9 | 9.7 / 13.4 / 12.8 | 8.8 / 6.5 / 6.6 | 9.7 / 6.2 / 4.4 |
| without 6-7: H0 |  | 20.1 / 20.2 / 20.9 | 25.2 / 20.9 / 21.7 | 17.9 / 12.0 / 15.1 | 17.6 / 12.5 / 13.6 |
| without 6-7: H0 | shuffled | 17.4 / 20.6 / 20.3 | 17.4 / 21.5 / 20.5 | 12.5 / 12.1 / 12.4 | 16.2 / 11.2 / 11.0 |
| without 6-7: region counts |  | 20.2 / 21.1 / 21.6 | 23.6 / 22.2 / 24.1 | 16.7 / 13.2 / 13.5 | 18.7 / 14.1 / 14.4 |
| without 6-7: region counts | shuffled | 16.7 / 19.8 / 20.8 | 16.7 / 20.0 / 21.4 | 12.5 / 12.3 / 11.0 | 15.4 / 11.4 / 13.2 |
| without 6-7: sliced W1 |  | 24.6 / 21.0 / 20.3 | 29.9 / 23.0 / 23.5 | 21.1 / 13.4 / 14.2 | 23.0 / 13.0 / 12.6 |
| without 6-7: sliced W1 | shuffled | 18.8 / 21.0 / 19.5 | 21.5 / 20.7 / 20.7 | 16.2 / 13.6 / 12.2 | 17.7 / 11.4 / 11.7 |
| without 6-7: mean point |  | 18.5 / 19.6 / 20.0 | 21.2 / 20.6 / 21.2 | 16.6 / 13.2 / 11.3 | 16.8 / 10.6 / 10.5 |
| without 6-7: mean point | shuffled | 17.1 / 19.6 / 18.9 | 18.3 / 19.1 / 18.9 | 14.6 / 13.0 / 11.7 | 15.0 / 10.1 / 10.5 |
| without 6-7: H0+H1 |  | 14.8 / 17.2 / 16.6 | 17.5 / 18.3 / 17.4 | 12.8 / 10.8 / 9.6 | 15.2 / 9.8 / 9.7 |
| without 6-7: H0+H1 | shuffled | 12.8 / 17.6 / 15.1 | 14.5 / 17.2 / 17.8 | 11.2 / 11.0 / 8.0 | 12.9 / 9.8 / 9.1 |
| only 1-2: H0 |  | 22.7 / 20.5 / 20.4 | 23.6 / 19.6 / 22.6 | 17.8 / 14.9 / 13.1 | 18.7 / 10.1 / 14.0 |
| only 1-2: H0 | shuffled | 17.1 / 18.4 / 18.9 | 18.9 / 19.6 / 20.3 | 15.9 / 12.3 / 13.3 | 16.6 / 11.2 / 12.1 |
| only 1-2: region counts |  | 24.2 / 20.5 / 22.1 | 23.1 / 20.3 / 23.6 | 18.3 / 14.8 / 14.8 | 20.0 / 14.0 / 14.5 |
| only 1-2: region counts | shuffled | 16.6 / 19.1 / 20.9 | 18.1 / 21.1 / 22.6 | 15.7 / 12.4 / 13.3 | 16.4 / 14.0 / 14.6 |
| only 1-2: sliced W1 |  | 26.7 / 21.4 / 22.8 | 31.9 / 22.6 / 21.8 | 23.1 / 13.8 / 14.4 | 25.7 / 13.8 / 12.2 |
| only 1-2: sliced W1 | shuffled | 19.9 / 19.1 / 20.6 | 23.2 / 20.9 / 20.6 | 19.2 / 12.2 / 13.2 | 21.3 / 12.5 / 11.5 |
| only 1-2: mean point |  | 5.2 / 12.7 / 13.1 | 5.3 / 12.5 / 14.0 | 4.8 / 5.6 / 5.6 | 5.6 / 4.8 / 4.6 |
| only 1-2: mean point | shuffled | 4.9 / 12.3 / 12.5 | 6.0 / 12.9 / 13.0 | 4.5 / 4.9 / 4.5 | 5.1 / 4.2 / 4.5 |
| only 1-2: H0+H1 |  | 15.9 / 16.7 / 16.4 | 17.2 / 17.0 / 17.0 | 14.4 / 9.9 / 10.6 | 15.6 / 10.5 / 10.1 |
| only 1-2: H0+H1 | shuffled | 13.8 / 16.2 / 16.3 | 14.4 / 16.8 / 18.0 | 11.6 / 11.3 / 9.5 | 12.6 / 9.8 / 8.9 |
| only 3-5: H0 |  | 15.1 / 17.2 / 14.8 | 16.9 / 18.2 / 17.0 | 12.3 / 8.3 / 8.8 | 12.7 / 9.2 / 8.5 |
| only 3-5: H0 | shuffled | 11.8 / 16.1 / 13.3 | 12.2 / 16.6 / 15.4 | 10.2 / 8.6 / 8.8 | 11.0 / 7.7 / 7.7 |
| only 3-5: region counts |  | 17.9 / 20.7 / 20.9 | 22.3 / 21.4 / 22.8 | 16.6 / 13.8 / 15.0 | 16.8 / 12.5 / 12.4 |
| only 3-5: region counts | shuffled | 13.4 / 18.0 / 18.9 | 14.4 / 19.5 / 19.5 | 13.3 / 10.9 / 11.2 | 14.6 / 9.7 / 10.2 |
| only 3-5: sliced W1 |  | 21.4 / 21.3 / 22.0 | 26.5 / 22.3 / 22.1 | 17.8 / 13.3 / 14.6 | 18.3 / 14.0 / 12.5 |
| only 3-5: sliced W1 | shuffled | 15.0 / 19.3 / 18.6 | 16.8 / 19.5 / 19.7 | 13.0 / 12.8 / 13.3 | 13.9 / 12.2 / 11.2 |
| only 3-5: mean point |  | 14.8 / 19.4 / 18.9 | 18.5 / 21.5 / 19.9 | 13.2 / 13.4 / 11.6 | 13.6 / 12.2 / 11.7 |
| only 3-5: mean point | shuffled | 13.9 / 19.3 / 19.1 | 15.6 / 19.1 / 19.1 | 12.1 / 11.9 / 12.4 | 12.6 / 10.3 / 11.7 |
| only 3-5: H0+H1 |  | 13.6 / 15.5 / 12.2 | 14.4 / 17.4 / 13.6 | 11.2 / 7.9 / 6.2 | 11.7 / 8.1 / 6.0 |
| only 3-5: H0+H1 | shuffled | 10.3 / 14.8 / 11.3 | 9.6 / 15.8 / 13.2 | 9.9 / 7.9 / 5.8 | 10.7 / 7.3 / 6.2 |
| only 6-7: H0 |  | 4.2 / 9.9 / 9.7 | 4.7 / 10.3 / 9.5 | 4.6 / 2.7 / 3.8 | 4.4 / 3.6 / 3.1 |
| only 6-7: H0 | shuffled | 2.9 / 8.6 / 8.6 | 1.7 / 9.1 / 9.2 | 3.3 / 1.9 / 2.4 | 2.6 / 3.2 / 2.0 |
| only 6-7: H0 | others from another trial | 3.0 / 8.5 / 8.4 | 2.3 / 9.0 / 8.7 | 2.3 / 2.6 / 1.7 | 2.7 / 2.3 / 2.3 |
| only 6-7: region counts |  | 3.8 / 9.4 / 10.3 | 3.8 / 10.5 / 10.9 | 3.8 / 3.6 / 2.9 | 4.3 / 3.0 / 2.7 |
| only 6-7: region counts | shuffled | 2.2 / 8.6 / 9.7 | 2.8 / 8.9 / 9.2 | 2.5 / 1.5 / 1.6 | 3.0 / 2.1 / 1.9 |
| only 6-7: region counts | others from another trial | 2.4 / 8.5 / 8.6 | 1.7 / 8.7 / 8.7 | 2.8 / 1.5 / 1.8 | 2.8 / 1.8 / 2.2 |
| only 6-7: sliced W1 |  | 4.7 / 10.6 / 10.5 | 4.8 / 11.1 / 11.3 | 4.0 / 3.6 / 4.1 | 4.6 / 3.2 / 3.7 |
| only 6-7: sliced W1 | shuffled | 3.1 / 8.6 / 8.7 | 1.7 / 8.8 / 9.0 | 2.3 / 2.2 / 2.2 | 2.8 / 1.5 / 2.4 |
| only 6-7: sliced W1 | others from another trial | 2.9 / 8.5 / 8.8 | 1.6 / 8.8 / 8.8 | 2.3 / 2.0 / 2.1 | 2.4 / 2.2 / 2.0 |
| only 6-7: mean point |  | 2.3 / 9.6 / 9.5 | 2.7 / 10.4 / 10.5 | 2.6 / 4.0 / 3.4 | 3.2 / 3.6 / 3.3 |
| only 6-7: mean point | shuffled | 3.0 / 9.2 / 9.0 | 1.0 / 9.2 / 9.7 | 1.7 / 2.0 / 2.8 | 1.7 / 2.3 / 1.5 |
| only 6-7: mean point | others from another trial | 2.3 / 8.8 / 8.7 | 1.7 / 9.4 / 9.3 | 1.9 / 1.5 / 2.2 | 1.6 / 1.9 / 1.8 |
| only 6-7: H0+H1 |  | 3.3 / 9.6 / 8.7 | 4.5 / 10.5 / 10.1 | 3.6 / 3.5 / 3.0 | 3.9 / 2.8 / 2.9 |
| only 6-7: H0+H1 | shuffled | 2.3 / 8.8 / 8.1 | 2.1 / 9.6 / 9.0 | 2.4 / 2.1 / 2.4 | 2.8 / 1.9 / 3.1 |
| only 6-7: H0+H1 | others from another trial | 2.7 / 8.4 / 7.6 | 2.3 / 9.0 / 8.3 | 2.0 / 2.2 / 2.5 | 2.4 / 1.5 / 1.5 |

### One-stimulus cloud, coordinates 1-8, within gratings

| signature | control | within gratings |
|---|---|---|
| all 1-8: H0, K = 16 |  | 17.2 / 17.4 / 18.9 |
| all 1-8: H0, K = 16 | shuffled | 13.2 / 17.9 / 18.1 |
| all 1-8: H0, K = 16 | others from another trial | 12.8 / 16.2 / 16.3 |
| all 1-8: H0+H1, K = 16 |  | 12.9 / 14.8 / 15.0 |
| all 1-8: H0+H1, K = 16 | shuffled | 11.9 / 15.8 / 14.1 |
| all 1-8: H0+H1, K = 16 | others from another trial | 10.1 / 13.8 / 13.4 |
| all 1-8: region counts, K = 16 |  | 17.9 / 17.4 / 20.2 |
| all 1-8: region counts, K = 16 | shuffled | 12.7 / 17.0 / 19.1 |
| all 1-8: region counts, K = 16 | others from another trial | 13.6 / 17.1 / 17.6 |
| all 1-8: sliced W1 |  | 24.0 / 19.1 / 18.9 |
| all 1-8: sliced W1 | shuffled | 18.9 / 17.8 / 18.3 |
| all 1-8: sliced W1 | others from another trial | 17.1 / 16.6 / 16.4 |
| all 1-8: mean point |  | 20.6 / 18.7 / 19.7 |
| all 1-8: mean point | shuffled | 18.5 / 17.0 / 17.3 |
| all 1-8: mean point | others from another trial | 14.4 / 16.9 / 16.9 |
| all 1-8: H0, K = 64 |  | 15.7 / 15.0 / 17.4 |
| all 1-8: H0, K = 64 | shuffled | 13.1 / 15.3 / 16.4 |
| all 1-8: H0, K = 64 | others from another trial | 11.8 / 12.9 / 15.2 |
| all 1-8: H0+H1, K = 64 |  | 11.6 / 14.3 / 8.3 |
| all 1-8: H0+H1, K = 64 | shuffled | 8.8 / 12.4 / 8.3 |
| all 1-8: H0+H1, K = 64 | others from another trial | 8.6 / 12.3 / 8.6 |
| all 1-8: region counts, K = 64 |  | 20.2 / 17.1 / 19.1 |
| all 1-8: region counts, K = 64 | shuffled | 16.1 / 17.3 / 18.8 |
| all 1-8: region counts, K = 64 | others from another trial | 15.7 / 14.8 / 16.8 |
| all 1-8, time x4: H0, K = 16 |  | 11.0 / 13.3 / 14.0 |
| all 1-8, time x4: H0, K = 16 | shuffled | 9.2 / 13.3 / 14.1 |
| all 1-8, time x4: H0, K = 16 | others from another trial | 9.9 / 13.1 / 13.9 |
| all 1-8, time x4: H0+H1, K = 16 |  | 8.9 / 12.8 / 11.2 |
| all 1-8, time x4: H0+H1, K = 16 | shuffled | 7.6 / 11.1 / 11.0 |
| all 1-8, time x4: H0+H1, K = 16 | others from another trial | 7.5 / 12.2 / 10.8 |
| all 1-8, time x4: region counts, K = 16 |  | 10.1 / 13.3 / 14.6 |
| all 1-8, time x4: region counts, K = 16 | shuffled | 8.9 / 13.6 / 14.8 |
| all 1-8, time x4: region counts, K = 16 | others from another trial | 9.9 / 13.0 / 13.5 |
| all 1-8, time x4: sliced W1 |  | 19.2 / 15.3 / 16.1 |
| all 1-8, time x4: sliced W1 | shuffled | 12.5 / 14.9 / 16.4 |
| all 1-8, time x4: sliced W1 | others from another trial | 12.7 / 13.8 / 13.8 |
| all 1-8, time x4: mean point |  | 18.1 / 16.0 / 16.4 |
| all 1-8, time x4: mean point | shuffled | 14.8 / 15.9 / 16.2 |
| all 1-8, time x4: mean point | others from another trial | 10.7 / 14.1 / 14.2 |
| all 1-8, time x4: H0, K = 64 |  | 13.1 / 13.6 / 16.7 |
| all 1-8, time x4: H0, K = 64 | shuffled | 10.3 / 12.2 / 14.4 |
| all 1-8, time x4: H0, K = 64 | others from another trial | 10.2 / 13.1 / 14.7 |
| all 1-8, time x4: H0+H1, K = 64 |  | 9.8 / 10.9 / 8.4 |
| all 1-8, time x4: H0+H1, K = 64 | shuffled | 7.7 / 11.5 / 8.0 |
| all 1-8, time x4: H0+H1, K = 64 | others from another trial | 7.9 / 11.1 / 8.3 |
| all 1-8, time x4: region counts, K = 64 |  | 17.2 / 16.9 / 17.5 |
| all 1-8, time x4: region counts, K = 64 | shuffled | 12.3 / 14.4 / 16.8 |
| all 1-8, time x4: region counts, K = 64 | others from another trial | 13.5 / 14.3 / 16.3 |

## allen30

60 neurons, 1783 + 1734 samples; clouds needing replacement: 0

### References (same samples)

| signature | control | within movie | within spontaneous | movie -> spontaneous | spontaneous -> movie |
|---|---|---|---|---|---|
| ISI fractions |  | 44.8 / 27.2 / 28.0 | 35.8 / 26.0 / 25.8 | 25.4 / 12.5 / 11.8 | 24.5 / 14.8 / 13.8 |
| burst table |  | 47.7 / 28.2 / 29.4 | 37.9 / 28.8 / 28.7 | 30.6 / 13.5 / 14.0 | 27.9 / 14.3 / 15.0 |
| multiscale sqrt time bars |  | 13.9 / 12.1 / 9.7 | 2.5 / 8.1 / 5.1 | 3.9 / 2.7 / 2.7 | 2.2 / 4.5 / 3.1 |
| ISI fractions | shuffled | 44.8 / 27.2 / 28.0 | 35.8 / 26.0 / 25.8 | 25.4 / 12.5 / 11.8 | 24.5 / 14.8 / 13.8 |
| burst table | shuffled | 39.6 / 24.6 / 25.1 | 31.5 / 24.7 / 25.1 | 22.8 / 12.9 / 11.4 | 22.4 / 12.2 / 14.5 |
| multiscale sqrt time bars | shuffled | 3.7 / 9.1 / 6.7 | 3.1 / 8.0 / 5.0 | 3.8 / 3.6 / 2.7 | 2.5 / 4.3 / 2.6 |
| L17: delay regions k=8, H0 |  | 27.4 / 24.4 / 24.1 | 22.5 / 20.5 / 20.8 | 17.0 / 10.7 / 12.6 | 15.6 / 10.9 / 12.6 |
| L17: delay regions k=8, H0 | shuffled | 20.1 / 19.9 / 21.2 | 16.0 / 18.2 / 17.8 | 14.1 / 8.0 / 10.8 | 12.5 / 8.1 / 9.3 |

### Stimulus-free cloud, coordinates 1-7

| signature | control | within movie | within spontaneous | movie -> spontaneous | spontaneous -> movie |
|---|---|---|---|---|---|
| H0, K = 16 |  | 32.9 / 25.2 / 26.3 | 25.1 / 21.0 / 23.1 | 20.7 / 14.0 / 13.7 | 19.1 / 15.0 / 14.9 |
| H0, K = 16 | shuffled | 23.7 / 19.4 / 21.4 | 19.1 / 18.6 / 19.4 | 13.6 / 9.3 / 9.8 | 13.0 / 10.3 / 9.9 |
| H0, K = 16 | others from another trial | 26.1 / 20.6 / 23.3 | 19.3 / 18.7 / 20.8 | 18.3 / 10.4 / 10.3 | 17.4 / 12.3 / 12.8 |
| H0+H1, K = 16 |  | 24.4 / 19.6 / 20.4 | 18.1 / 19.2 / 19.6 | 15.3 / 11.6 / 9.6 | 14.8 / 10.8 / 11.3 |
| H0+H1, K = 16 | shuffled | 16.3 / 16.9 / 16.9 | 13.0 / 16.3 / 15.5 | 10.3 / 7.8 / 7.4 | 8.8 / 8.2 / 9.0 |
| H0+H1, K = 16 | others from another trial | 20.4 / 18.2 / 17.6 | 15.7 / 16.5 / 15.3 | 14.3 / 7.7 / 9.5 | 12.1 / 9.7 / 9.9 |
| region counts, K = 16 |  | 32.8 / 23.1 / 27.0 | 26.4 / 23.1 / 24.4 | 20.8 / 13.7 / 14.2 | 19.7 / 14.8 / 15.4 |
| region counts, K = 16 | shuffled | 21.8 / 21.9 / 21.3 | 17.2 / 18.1 / 19.6 | 14.5 / 9.8 / 10.7 | 12.8 / 10.0 / 11.0 |
| region counts, K = 16 | others from another trial | 25.7 / 24.6 / 25.7 | 20.0 / 19.4 / 21.8 | 19.5 / 13.6 / 13.2 | 16.2 / 11.1 / 13.2 |
| sliced W1 |  | 45.5 / 26.3 / 25.6 | 34.9 / 24.5 / 24.3 | 25.7 / 12.9 / 12.6 | 25.1 / 14.1 / 15.4 |
| sliced W1 | shuffled | 31.5 / 22.2 / 23.3 | 20.4 / 19.8 / 19.6 | 16.9 / 8.7 / 10.1 | 16.4 / 12.0 / 9.0 |
| sliced W1 | others from another trial | 33.3 / 21.8 / 22.8 | 24.2 / 20.2 / 19.7 | 19.8 / 9.2 / 10.0 | 19.3 / 9.4 / 7.9 |
| mean point |  | 36.5 / 24.6 / 25.0 | 27.5 / 23.2 / 23.7 | 20.5 / 11.9 / 13.4 | 20.9 / 13.8 / 15.0 |
| mean point | shuffled | 27.0 / 21.1 / 21.4 | 17.9 / 19.5 / 19.2 | 15.7 / 8.4 / 8.8 | 14.8 / 11.7 / 9.1 |
| mean point | others from another trial | 25.5 / 20.9 / 21.3 | 20.5 / 18.6 / 19.6 | 15.8 / 9.3 / 9.2 | 16.3 / 9.2 / 9.0 |
| H0, K = 64 |  | 30.6 / 22.7 / 25.8 | 23.9 / 20.0 / 22.0 | 19.1 / 11.9 / 14.6 | 19.1 / 11.2 / 14.5 |
| H0, K = 64 | shuffled | 19.3 / 19.2 / 20.3 | 13.8 / 15.8 / 16.4 | 11.5 / 7.4 / 9.5 | 10.5 / 8.5 / 9.5 |
| H0, K = 64 | others from another trial | 22.7 / 18.9 / 21.3 | 16.4 / 15.6 / 18.4 | 16.3 / 7.9 / 10.6 | 15.1 / 9.7 / 10.4 |
| H0+H1, K = 64 |  | 23.3 / 17.3 / 13.0 | 16.8 / 15.1 / 10.4 | 15.1 / 9.3 / 6.2 | 15.5 / 8.4 / 5.1 |
| H0+H1, K = 64 | shuffled | 15.9 / 16.3 / 12.0 | 11.6 / 13.4 / 10.0 | 10.2 / 6.5 / 5.5 | 9.4 / 7.4 / 5.4 |
| H0+H1, K = 64 | others from another trial | 15.9 / 17.3 / 10.8 | 13.0 / 13.6 / 9.7 | 13.2 / 8.0 / 4.7 | 11.6 / 7.2 / 6.2 |
| region counts, K = 64 |  | 38.6 / 23.9 / 26.8 | 28.7 / 22.0 / 26.0 | 23.7 / 12.9 / 16.6 | 22.8 / 13.6 / 16.3 |
| region counts, K = 64 | shuffled | 24.3 / 18.7 / 21.6 | 17.1 / 17.0 / 20.1 | 13.3 / 9.3 / 8.1 | 12.9 / 11.1 / 10.2 |
| region counts, K = 64 | others from another trial | 27.7 / 18.8 / 24.3 | 21.0 / 18.0 / 19.8 | 18.9 / 11.2 / 13.8 | 18.1 / 9.5 / 13.5 |

### Ablations, K = 16 (stimulus-free)

| signature | control | within movie | within spontaneous | movie -> spontaneous | spontaneous -> movie |
|---|---|---|---|---|---|
| without 1-2: H0 |  | 31.4 / 22.9 / 24.7 | 22.6 / 20.5 / 22.0 | 19.9 / 14.2 / 11.7 | 18.0 / 14.6 / 11.6 |
| without 1-2: H0 | shuffled | 19.5 / 18.4 / 18.7 | 13.7 / 16.3 / 16.4 | 11.0 / 7.6 / 7.6 | 10.9 / 10.0 / 9.0 |
| without 1-2: H0 | others from another trial | 22.7 / 19.3 / 20.0 | 16.2 / 16.8 / 17.8 | 13.6 / 7.4 / 10.1 | 12.6 / 8.6 / 10.6 |
| without 1-2: region counts |  | 31.3 / 23.2 / 25.7 | 24.9 / 21.1 / 22.9 | 20.8 / 12.2 / 12.9 | 17.6 / 11.4 / 13.7 |
| without 1-2: region counts | shuffled | 20.5 / 19.1 / 20.0 | 12.9 / 16.5 / 17.9 | 11.8 / 9.1 / 7.1 | 11.5 / 9.1 / 9.9 |
| without 1-2: region counts | others from another trial | 23.6 / 20.9 / 21.9 | 17.2 / 17.6 / 19.0 | 14.9 / 10.6 / 10.4 | 14.0 / 12.4 / 9.5 |
| without 1-2: sliced W1 |  | 39.8 / 24.8 / 25.5 | 30.0 / 21.0 / 23.4 | 24.1 / 14.0 / 13.7 | 23.2 / 14.9 / 13.4 |
| without 1-2: sliced W1 | shuffled | 25.1 / 20.0 / 21.3 | 17.1 / 17.6 / 17.9 | 13.9 / 8.1 / 8.4 | 13.0 / 10.5 / 8.1 |
| without 1-2: sliced W1 | others from another trial | 28.9 / 19.8 / 21.1 | 20.2 / 17.4 / 17.7 | 17.6 / 9.1 / 10.0 | 16.7 / 9.1 / 9.3 |
| without 1-2: mean point |  | 31.8 / 25.1 / 25.7 | 23.6 / 21.7 / 22.4 | 17.7 / 13.7 / 13.9 | 18.7 / 12.4 / 11.8 |
| without 1-2: mean point | shuffled | 22.5 / 20.9 / 20.9 | 15.1 / 18.5 / 18.2 | 12.8 / 8.7 / 8.8 | 12.3 / 10.7 / 7.4 |
| without 1-2: mean point | others from another trial | 19.5 / 20.1 / 20.5 | 17.8 / 18.7 / 18.7 | 12.8 / 8.5 / 9.6 | 12.9 / 8.6 / 9.9 |
| without 1-2: H0+H1 |  | 24.2 / 20.4 / 20.8 | 17.9 / 17.5 / 16.7 | 16.3 / 11.4 / 10.7 | 13.6 / 11.0 / 9.0 |
| without 1-2: H0+H1 | shuffled | 15.5 / 16.1 / 16.6 | 11.2 / 14.7 / 14.6 | 9.0 / 8.7 / 4.8 | 8.8 / 6.8 / 7.5 |
| without 1-2: H0+H1 | others from another trial | 16.0 / 16.0 / 17.3 | 11.4 / 14.9 / 14.6 | 10.4 / 6.9 / 5.9 | 9.5 / 6.7 / 6.8 |
| without 3-5: H0 |  | 31.0 / 23.7 / 25.4 | 22.6 / 22.1 / 22.0 | 18.3 / 13.6 / 15.1 | 17.0 / 15.8 / 12.2 |
| without 3-5: H0 | shuffled | 18.6 / 17.2 / 17.6 | 14.8 / 16.2 / 16.3 | 9.8 / 8.6 / 7.7 | 9.6 / 9.4 / 8.6 |
| without 3-5: H0 | others from another trial | 20.9 / 17.9 / 18.5 | 13.8 / 16.8 / 17.3 | 11.9 / 7.4 / 9.3 | 11.9 / 9.4 / 8.8 |
| without 3-5: region counts |  | 29.3 / 22.7 / 25.7 | 20.8 / 19.8 / 22.3 | 15.8 / 11.0 / 14.1 | 17.4 / 13.9 / 12.3 |
| without 3-5: region counts | shuffled | 15.9 / 17.5 / 18.2 | 12.6 / 16.6 / 17.1 | 8.7 / 8.6 / 9.0 | 9.9 / 7.9 / 8.9 |
| without 3-5: region counts | others from another trial | 19.7 / 18.9 / 19.7 | 13.1 / 17.0 / 18.1 | 11.3 / 7.0 / 9.3 | 12.2 / 8.2 / 10.1 |
| without 3-5: sliced W1 |  | 40.6 / 23.3 / 26.1 | 29.0 / 20.0 / 20.2 | 20.4 / 11.8 / 13.6 | 22.0 / 11.4 / 11.9 |
| without 3-5: sliced W1 | shuffled | 24.3 / 18.1 / 19.2 | 15.8 / 15.6 / 16.1 | 12.4 / 8.1 / 6.8 | 12.4 / 7.0 / 7.0 |
| without 3-5: sliced W1 | others from another trial | 23.7 / 17.9 / 18.2 | 16.1 / 14.3 / 16.1 | 13.8 / 8.3 / 6.5 | 13.9 / 6.4 / 7.2 |
| without 3-5: mean point |  | 14.5 / 18.8 / 18.5 | 10.7 / 15.9 / 16.6 | 7.6 / 7.4 / 7.8 | 7.2 / 7.3 / 7.1 |
| without 3-5: mean point | shuffled | 8.5 / 14.0 / 14.3 | 5.2 / 13.1 / 13.0 | 4.0 / 4.7 / 5.0 | 4.6 / 4.3 / 4.3 |
| without 3-5: mean point | others from another trial | 7.0 / 13.7 / 13.3 | 5.4 / 12.9 / 13.8 | 3.9 / 4.2 / 5.0 | 4.6 / 3.9 / 4.9 |
| without 3-5: H0+H1 |  | 22.4 / 17.9 / 18.6 | 15.5 / 18.2 / 17.8 | 13.2 / 9.4 / 10.0 | 14.5 / 10.8 / 10.2 |
| without 3-5: H0+H1 | shuffled | 13.0 / 14.6 / 14.7 | 11.4 / 13.8 / 13.1 | 8.6 / 5.7 / 6.3 | 7.8 / 5.3 / 6.1 |
| without 3-5: H0+H1 | others from another trial | 13.6 / 15.8 / 16.3 | 10.8 / 13.8 / 14.5 | 8.1 / 7.0 / 6.2 | 8.4 / 6.9 / 7.3 |
| without 6-7: H0 |  | 32.1 / 26.2 / 25.8 | 28.3 / 25.0 / 25.9 | 22.6 / 16.5 / 15.6 | 21.1 / 17.8 / 16.3 |
| without 6-7: H0 | shuffled | 23.1 / 24.3 / 23.9 | 21.2 / 21.0 / 23.2 | 17.0 / 13.7 / 14.3 | 16.3 / 12.0 / 13.9 |
| without 6-7: region counts |  | 31.5 / 27.3 / 28.5 | 29.5 / 26.8 / 27.1 | 22.5 / 15.6 / 17.0 | 22.6 / 18.1 / 17.4 |
| without 6-7: region counts | shuffled | 22.0 / 22.2 / 23.9 | 19.7 / 22.7 / 22.7 | 14.4 / 14.0 / 13.0 | 15.3 / 11.0 / 14.8 |
| without 6-7: sliced W1 |  | 40.3 / 27.0 / 27.9 | 34.3 / 25.0 / 27.0 | 26.8 / 13.8 / 13.8 | 26.3 / 11.9 / 13.7 |
| without 6-7: sliced W1 | shuffled | 31.8 / 23.7 / 24.5 | 26.7 / 22.5 / 23.1 | 18.3 / 11.8 / 11.9 | 18.7 / 12.3 / 11.7 |
| without 6-7: mean point |  | 29.3 / 23.2 / 24.5 | 25.7 / 22.3 / 23.0 | 18.5 / 11.5 / 11.5 | 18.1 / 11.8 / 11.4 |
| without 6-7: mean point | shuffled | 26.3 / 23.4 / 24.2 | 22.5 / 22.3 / 22.4 | 15.8 / 11.9 / 12.5 | 16.7 / 12.1 / 12.2 |
| without 6-7: H0+H1 |  | 22.2 / 20.5 / 19.3 | 20.2 / 18.9 / 16.3 | 15.3 / 9.5 / 10.0 | 14.6 / 9.2 / 10.4 |
| without 6-7: H0+H1 | shuffled | 16.5 / 17.4 / 17.4 | 15.1 / 17.2 / 16.1 | 11.0 / 7.3 / 7.3 | 11.1 / 9.2 / 10.8 |
| only 1-2: H0 |  | 34.7 / 25.9 / 26.9 | 30.3 / 25.0 / 24.7 | 24.3 / 17.2 / 14.6 | 21.1 / 15.8 / 12.5 |
| only 1-2: H0 | shuffled | 26.7 / 23.3 / 24.0 | 21.9 / 21.6 / 23.6 | 17.4 / 11.1 / 10.8 | 17.4 / 11.8 / 13.1 |
| only 1-2: region counts |  | 32.8 / 26.7 / 29.7 | 27.2 / 23.8 / 27.1 | 22.2 / 10.7 / 13.0 | 19.8 / 13.8 / 13.7 |
| only 1-2: region counts | shuffled | 24.6 / 24.8 / 25.4 | 19.3 / 21.0 / 23.0 | 16.4 / 12.9 / 12.2 | 14.3 / 11.3 / 11.8 |
| only 1-2: sliced W1 |  | 41.6 / 27.0 / 27.1 | 35.4 / 25.1 / 25.8 | 27.3 / 12.2 / 14.9 | 25.3 / 13.4 / 15.1 |
| only 1-2: sliced W1 | shuffled | 33.1 / 24.2 / 25.4 | 27.2 / 23.4 / 23.4 | 19.5 / 11.1 / 11.5 | 19.3 / 12.5 / 11.7 |
| only 1-2: mean point |  | 6.2 / 14.0 / 13.9 | 4.8 / 12.9 / 12.9 | 4.0 / 3.1 / 3.5 | 3.3 / 3.8 / 4.3 |
| only 1-2: mean point | shuffled | 6.3 / 14.1 / 13.6 | 4.9 / 13.6 / 13.0 | 4.0 / 4.6 / 3.4 | 3.9 / 4.8 / 3.7 |
| only 1-2: H0+H1 |  | 22.0 / 18.4 / 18.3 | 18.9 / 18.7 / 17.3 | 15.4 / 10.5 / 8.7 | 13.3 / 11.1 / 9.4 |
| only 1-2: H0+H1 | shuffled | 16.9 / 17.4 / 18.1 | 14.8 / 15.8 / 16.0 | 11.9 / 8.6 / 8.7 | 9.8 / 9.5 / 9.9 |
| only 3-5: H0 |  | 22.0 / 21.2 / 20.0 | 17.9 / 17.6 / 17.2 | 13.7 / 10.8 / 9.7 | 15.2 / 12.5 / 9.1 |
| only 3-5: H0 | shuffled | 16.5 / 20.4 / 17.0 | 13.8 / 17.1 / 15.4 | 11.3 / 8.9 / 7.6 | 11.0 / 10.4 / 9.0 |
| only 3-5: region counts |  | 29.0 / 25.9 / 28.4 | 26.1 / 24.5 / 25.1 | 20.4 / 14.9 / 17.6 | 19.7 / 14.9 / 16.7 |
| only 3-5: region counts | shuffled | 19.7 / 21.2 / 22.8 | 17.0 / 21.0 / 21.6 | 13.2 / 12.0 / 10.2 | 13.7 / 11.8 / 14.1 |
| only 3-5: sliced W1 |  | 33.7 / 27.0 / 26.4 | 29.1 / 23.7 / 26.5 | 22.3 / 12.8 / 14.9 | 22.3 / 14.6 / 16.2 |
| only 3-5: sliced W1 | shuffled | 24.2 / 21.2 / 23.2 | 17.8 / 21.9 / 23.1 | 14.0 / 11.1 / 11.5 | 14.5 / 11.3 / 11.5 |
| only 3-5: mean point |  | 20.8 / 23.2 / 22.5 | 19.7 / 23.1 / 22.7 | 15.2 / 12.6 / 12.6 | 15.1 / 13.0 / 11.4 |
| only 3-5: mean point | shuffled | 20.9 / 21.1 / 22.1 | 15.9 / 22.3 / 22.0 | 13.8 / 9.9 / 10.0 | 14.2 / 12.6 / 10.7 |
| only 3-5: H0+H1 |  | 18.6 / 18.6 / 12.2 | 15.0 / 16.4 / 11.4 | 12.3 / 9.8 / 6.9 | 13.4 / 11.2 / 9.1 |
| only 3-5: H0+H1 | shuffled | 14.1 / 17.6 / 11.9 | 11.8 / 16.5 / 11.2 | 11.0 / 8.8 / 6.8 | 9.3 / 8.0 / 6.6 |
| only 6-7: H0 |  | 7.2 / 11.6 / 11.8 | 5.8 / 11.2 / 11.2 | 5.3 / 4.4 / 4.5 | 5.6 / 4.5 / 3.8 |
| only 6-7: H0 | shuffled | 2.3 / 8.1 / 8.5 | 1.9 / 7.9 / 8.1 | 2.8 / 1.6 / 1.9 | 2.7 / 1.8 / 1.9 |
| only 6-7: H0 | others from another trial | 2.2 / 8.4 / 8.8 | 2.2 / 8.0 / 8.3 | 2.1 / 1.4 / 2.1 | 1.8 / 2.2 / 2.1 |
| only 6-7: region counts |  | 7.7 / 10.8 / 12.5 | 5.7 / 11.5 / 11.4 | 5.0 / 4.3 / 5.1 | 6.7 / 3.9 / 3.2 |
| only 6-7: region counts | shuffled | 1.7 / 8.6 / 8.9 | 2.1 / 8.3 / 8.4 | 3.0 / 2.2 / 2.0 | 2.4 / 1.6 / 1.8 |
| only 6-7: region counts | others from another trial | 2.0 / 8.6 / 9.0 | 2.0 / 8.4 / 8.8 | 2.4 / 1.8 / 1.6 | 2.1 / 2.9 / 2.3 |
| only 6-7: sliced W1 |  | 10.0 / 12.7 / 12.7 | 7.3 / 11.6 / 12.1 | 6.3 / 4.8 / 4.7 | 7.5 / 4.8 / 4.1 |
| only 6-7: sliced W1 | shuffled | 2.2 / 8.3 / 8.6 | 1.8 / 7.8 / 8.4 | 2.1 / 1.0 / 2.0 | 1.6 / 1.2 / 2.1 |
| only 6-7: sliced W1 | others from another trial | 1.3 / 8.5 / 8.7 | 1.8 / 8.1 / 8.1 | 2.7 / 1.7 / 1.0 | 2.7 / 2.4 / 2.4 |
| only 6-7: mean point |  | 3.9 / 12.2 / 11.6 | 3.8 / 11.6 / 11.0 | 3.8 / 4.8 / 3.7 | 4.7 / 4.3 / 2.9 |
| only 6-7: mean point | shuffled | 1.8 / 8.7 / 8.8 | 2.2 / 8.4 / 8.8 | 2.1 / 2.0 / 1.6 | 1.7 / 1.5 / 1.9 |
| only 6-7: mean point | others from another trial | 2.1 / 8.4 / 8.7 | 1.8 / 8.7 / 8.9 | 2.3 / 1.4 / 1.4 | 2.8 / 2.0 / 1.5 |
| only 6-7: H0+H1 |  | 5.5 / 10.4 / 9.9 | 4.4 / 9.7 / 8.9 | 4.5 / 3.3 / 3.6 | 5.1 / 3.0 / 3.0 |
| only 6-7: H0+H1 | shuffled | 1.7 / 8.8 / 8.0 | 2.1 / 8.2 / 8.0 | 2.4 / 1.7 / 1.6 | 2.5 / 2.1 / 1.8 |
| only 6-7: H0+H1 | others from another trial | 2.2 / 8.6 / 7.9 | 2.2 / 8.4 / 8.1 | 2.3 / 1.6 / 1.8 | 2.5 / 1.9 / 1.3 |

### One-stimulus cloud, coordinates 1-8, within movie

| signature | control | within movie |
|---|---|---|
| all 1-8: H0, K = 16 |  | 32.1 / 23.6 / 25.5 |
| all 1-8: H0, K = 16 | shuffled | 18.1 / 18.1 / 20.7 |
| all 1-8: H0, K = 16 | others from another trial | 25.0 / 21.1 / 21.3 |
| all 1-8: H0+H1, K = 16 |  | 22.6 / 18.6 / 19.6 |
| all 1-8: H0+H1, K = 16 | shuffled | 12.9 / 16.0 / 16.3 |
| all 1-8: H0+H1, K = 16 | others from another trial | 18.7 / 16.0 / 18.2 |
| all 1-8: region counts, K = 16 |  | 31.0 / 24.1 / 25.5 |
| all 1-8: region counts, K = 16 | shuffled | 17.9 / 17.7 / 19.7 |
| all 1-8: region counts, K = 16 | others from another trial | 24.2 / 21.6 / 23.2 |
| all 1-8: sliced W1 |  | 44.8 / 25.1 / 24.9 |
| all 1-8: sliced W1 | shuffled | 28.3 / 20.4 / 20.9 |
| all 1-8: sliced W1 | others from another trial | 32.0 / 21.3 / 21.9 |
| all 1-8: mean point |  | 36.6 / 24.1 / 24.6 |
| all 1-8: mean point | shuffled | 26.4 / 20.6 / 20.8 |
| all 1-8: mean point | others from another trial | 28.2 / 21.5 / 20.8 |
| all 1-8: H0, K = 64 |  | 30.1 / 21.4 / 23.8 |
| all 1-8: H0, K = 64 | shuffled | 16.0 / 17.8 / 19.6 |
| all 1-8: H0, K = 64 | others from another trial | 22.7 / 18.1 / 20.3 |
| all 1-8: H0+H1, K = 64 |  | 22.5 / 17.7 / 9.3 |
| all 1-8: H0+H1, K = 64 | shuffled | 13.2 / 14.8 / 9.7 |
| all 1-8: H0+H1, K = 64 | others from another trial | 15.1 / 15.1 / 9.3 |
| all 1-8: region counts, K = 64 |  | 38.2 / 21.5 / 26.6 |
| all 1-8: region counts, K = 64 | shuffled | 21.0 / 19.0 / 20.6 |
| all 1-8: region counts, K = 64 | others from another trial | 27.9 / 20.2 / 24.6 |
| all 1-8, time x4: H0, K = 16 |  | 26.6 / 21.8 / 25.1 |
| all 1-8, time x4: H0, K = 16 | shuffled | 16.1 / 16.0 / 16.2 |
| all 1-8, time x4: H0, K = 16 | others from another trial | 26.7 / 20.8 / 22.9 |
| all 1-8, time x4: H0+H1, K = 16 |  | 19.7 / 16.1 / 17.9 |
| all 1-8, time x4: H0+H1, K = 16 | shuffled | 11.2 / 13.6 / 14.1 |
| all 1-8, time x4: H0+H1, K = 16 | others from another trial | 18.1 / 16.1 / 15.5 |
| all 1-8, time x4: region counts, K = 16 |  | 25.7 / 21.7 / 23.5 |
| all 1-8, time x4: region counts, K = 16 | shuffled | 14.3 / 15.7 / 17.8 |
| all 1-8, time x4: region counts, K = 16 | others from another trial | 24.5 / 21.4 / 24.1 |
| all 1-8, time x4: sliced W1 |  | 35.4 / 22.4 / 22.3 |
| all 1-8, time x4: sliced W1 | shuffled | 21.1 / 16.9 / 18.2 |
| all 1-8, time x4: sliced W1 | others from another trial | 27.9 / 20.1 / 22.4 |
| all 1-8, time x4: mean point |  | 29.3 / 21.8 / 22.1 |
| all 1-8, time x4: mean point | shuffled | 22.0 / 18.8 / 18.6 |
| all 1-8, time x4: mean point | others from another trial | 22.8 / 19.1 / 19.0 |
| all 1-8, time x4: H0, K = 64 |  | 29.0 / 20.2 / 26.4 |
| all 1-8, time x4: H0, K = 64 | shuffled | 11.8 / 14.9 / 16.0 |
| all 1-8, time x4: H0, K = 64 | others from another trial | 23.6 / 18.6 / 22.8 |
| all 1-8, time x4: H0+H1, K = 64 |  | 21.4 / 16.3 / 11.3 |
| all 1-8, time x4: H0+H1, K = 64 | shuffled | 11.1 / 13.2 / 9.6 |
| all 1-8, time x4: H0+H1, K = 64 | others from another trial | 16.4 / 15.5 / 10.6 |
| all 1-8, time x4: region counts, K = 64 |  | 35.8 / 23.6 / 29.4 |
| all 1-8, time x4: region counts, K = 64 | shuffled | 14.9 / 16.5 / 18.8 |
| all 1-8, time x4: region counts, K = 64 | others from another trial | 28.9 / 20.6 / 27.2 |

## retina40

60 neurons, 420 + 1796 samples; clouds needing replacement: 100

### References (same samples)

| signature | control | within frozen x4 | within unrepeated x4 | frozen x4 -> unrepeated x4 | unrepeated x4 -> frozen x4 |
|---|---|---|---|---|---|
| ISI fractions |  | 64.3 / 48.8 / 48.1 | 67.4 / 42.3 / 47.0 | 73.6 / 40.0 / 39.1 | 63.5 / 46.0 / 49.0 |
| burst table |  | 68.3 / 53.3 / 49.0 | 70.9 / 46.1 / 48.7 | 74.8 / 42.0 / 41.3 | 66.5 / 50.2 / 49.0 |
| multiscale sqrt time bars |  | 80.7 / 62.1 / 58.6 | 33.9 / 21.7 / 33.7 | 34.3 / 19.7 / 32.7 | 31.5 / 19.3 / 36.2 |
| ISI fractions | shuffled | 64.3 / 48.8 / 48.1 | 67.4 / 42.3 / 47.0 | 73.6 / 40.0 / 39.1 | 63.5 / 46.0 / 49.0 |
| burst table | shuffled | 65.7 / 49.8 / 49.8 | 67.1 / 44.0 / 47.4 | 71.0 / 38.6 / 42.1 | 61.0 / 46.4 / 48.8 |
| multiscale sqrt time bars | shuffled | 3.6 / 21.7 / 21.9 | 5.2 / 14.8 / 14.7 | 5.0 / 9.5 / 7.9 | 6.0 / 12.9 / 10.0 |
| L17: delay regions k=8, H0 |  | 35.5 / 40.2 / 43.3 | 38.1 / 33.7 / 36.2 | 44.8 / 29.4 / 29.3 | 37.6 / 31.2 / 37.9 |
| L17: delay regions k=8, H0 | shuffled | 29.8 / 38.3 / 37.9 | 33.6 / 30.5 / 34.6 | 38.6 / 26.3 / 25.0 | 32.1 / 24.3 / 32.4 |

### Stimulus-free cloud, coordinates 1-7

| signature | control | within frozen x4 | within unrepeated x4 | frozen x4 -> unrepeated x4 | unrepeated x4 -> frozen x4 |
|---|---|---|---|---|---|
| H0, K = 16 |  | 45.2 / 44.8 / 44.3 | 49.4 / 35.5 / 43.7 | 55.2 / 34.1 / 35.5 | 45.2 / 35.2 / 44.8 |
| H0, K = 16 | shuffled | 36.4 / 42.6 / 43.8 | 42.9 / 35.1 / 39.6 | 47.9 / 31.5 / 34.9 | 42.6 / 33.3 / 41.7 |
| H0, K = 16 | others from another trial | 38.8 / 42.9 / 46.7 | 47.1 / 37.0 / 39.0 | 52.1 / 32.3 / 34.5 | 44.6 / 37.6 / 41.9 |
| H0+H1, K = 16 |  | 40.2 / 43.3 / 43.1 | 43.3 / 34.8 / 37.2 | 50.2 / 30.7 / 34.1 | 41.5 / 34.8 / 37.1 |
| H0+H1, K = 16 | shuffled | 37.1 / 40.5 / 42.6 | 40.8 / 33.1 / 35.5 | 43.1 / 26.3 / 31.9 | 39.4 / 31.9 / 36.4 |
| H0+H1, K = 16 | others from another trial | 40.0 / 41.4 / 40.7 | 41.8 / 33.6 / 37.8 | 47.1 / 30.1 / 28.2 | 39.6 / 32.1 / 37.4 |
| region counts, K = 16 |  | 42.9 / 44.5 / 47.6 | 47.6 / 35.2 / 39.5 | 50.2 / 32.6 / 36.9 | 44.6 / 35.2 / 41.2 |
| region counts, K = 16 | shuffled | 35.0 / 41.7 / 41.0 | 41.5 / 36.7 / 36.5 | 45.5 / 30.7 / 30.4 | 38.0 / 41.4 / 39.3 |
| region counts, K = 16 | others from another trial | 38.1 / 44.3 / 43.1 | 45.7 / 34.1 / 36.7 | 50.5 / 35.3 / 33.9 | 44.0 / 34.5 / 38.3 |
| sliced W1 |  | 52.4 / 44.8 / 42.6 | 57.5 / 35.4 / 38.8 | 63.3 / 30.8 / 32.5 | 52.4 / 35.2 / 37.1 |
| sliced W1 | shuffled | 41.7 / 38.6 / 40.2 | 48.9 / 34.1 / 35.7 | 49.3 / 29.9 / 29.2 | 43.2 / 36.9 / 35.7 |
| sliced W1 | others from another trial | 47.9 / 40.5 / 40.2 | 51.8 / 32.3 / 35.5 | 61.2 / 27.9 / 29.8 | 49.3 / 32.6 / 34.5 |
| mean point |  | 42.1 / 41.7 / 40.0 | 48.6 / 34.2 / 34.1 | 54.5 / 27.5 / 28.8 | 44.8 / 31.0 / 31.4 |
| mean point | shuffled | 34.8 / 37.9 / 38.1 | 44.7 / 30.5 / 33.4 | 46.2 / 25.0 / 26.9 | 41.3 / 30.2 / 30.5 |
| mean point | others from another trial | 37.9 / 39.3 / 39.5 | 44.2 / 30.5 / 31.6 | 48.8 / 28.3 / 29.1 | 41.3 / 30.5 / 32.9 |
| H0, K = 64 |  | 44.0 / 38.8 / 44.5 | 46.3 / 31.2 / 40.3 | 53.8 / 28.0 / 34.2 | 43.0 / 32.1 / 41.7 |
| H0, K = 64 | shuffled | 35.0 / 37.4 / 43.1 | 35.7 / 30.0 / 38.3 | 41.9 / 27.2 / 33.2 | 36.4 / 28.8 / 42.1 |
| H0, K = 64 | others from another trial | 41.0 / 42.9 / 43.8 | 44.0 / 32.2 / 36.0 | 46.9 / 31.5 / 34.4 | 42.3 / 30.0 / 34.0 |
| H0+H1, K = 64 |  | 37.9 / 36.2 / 37.9 | 37.5 / 28.1 / 32.1 | 41.0 / 23.0 / 27.6 | 36.4 / 26.2 / 30.2 |
| H0+H1, K = 64 | shuffled | 25.7 / 33.1 / 36.2 | 29.4 / 26.1 / 29.7 | 33.1 / 22.0 / 24.6 | 29.2 / 25.7 / 29.5 |
| H0+H1, K = 64 | others from another trial | 31.0 / 36.7 / 36.9 | 37.6 / 28.7 / 30.6 | 39.8 / 23.5 / 26.8 | 34.2 / 26.4 / 30.2 |
| region counts, K = 64 |  | 47.4 / 44.3 / 41.0 | 47.7 / 33.7 / 39.5 | 56.7 / 34.4 / 33.8 | 44.2 / 31.9 / 41.2 |
| region counts, K = 64 | shuffled | 30.5 / 36.2 / 43.1 | 35.7 / 33.4 / 38.3 | 45.0 / 26.4 / 31.5 | 35.6 / 30.7 / 41.2 |
| region counts, K = 64 | others from another trial | 42.6 / 41.2 / 43.6 | 45.3 / 33.2 / 39.3 | 51.0 / 29.0 / 33.6 | 44.5 / 34.0 / 39.8 |

### Ablations, K = 16 (stimulus-free)

| signature | control | within frozen x4 | within unrepeated x4 | frozen x4 -> unrepeated x4 | unrepeated x4 -> frozen x4 |
|---|---|---|---|---|---|
| without 1-2: H0 |  | 33.8 / 39.0 / 38.8 | 36.8 / 31.8 / 35.5 | 46.0 / 26.6 / 27.8 | 39.0 / 29.0 / 37.1 |
| without 1-2: H0 | shuffled | 30.7 / 36.4 / 39.0 | 30.4 / 29.4 / 32.0 | 37.9 / 24.0 / 25.3 | 32.9 / 30.2 / 30.2 |
| without 1-2: H0 | others from another trial | 33.8 / 37.6 / 39.0 | 33.0 / 29.1 / 33.7 | 39.8 / 26.0 / 28.7 | 35.4 / 28.3 / 31.7 |
| without 1-2: region counts |  | 31.9 / 37.4 / 37.4 | 36.6 / 32.5 / 36.5 | 42.4 / 29.5 / 24.8 | 37.7 / 29.8 / 37.9 |
| without 1-2: region counts | shuffled | 28.3 / 38.1 / 40.0 | 30.6 / 28.2 / 31.2 | 34.3 / 22.1 / 25.3 | 29.9 / 24.5 / 29.8 |
| without 1-2: region counts | others from another trial | 33.3 / 38.3 / 36.9 | 33.2 / 29.6 / 33.0 | 37.1 / 24.2 / 24.0 | 33.9 / 26.0 / 32.4 |
| without 1-2: sliced W1 |  | 50.5 / 44.5 / 44.8 | 54.1 / 33.9 / 36.9 | 58.8 / 32.5 / 30.3 | 50.1 / 32.1 / 39.5 |
| without 1-2: sliced W1 | shuffled | 36.4 / 37.6 / 37.9 | 41.6 / 31.5 / 32.4 | 46.0 / 26.3 / 28.0 | 39.5 / 31.7 / 31.2 |
| without 1-2: sliced W1 | others from another trial | 44.3 / 39.8 / 40.0 | 47.2 / 33.4 / 33.1 | 58.8 / 28.2 / 28.6 | 45.9 / 31.9 / 35.5 |
| without 1-2: mean point |  | 42.1 / 43.1 / 42.9 | 45.7 / 32.5 / 35.4 | 52.1 / 30.5 / 29.8 | 43.7 / 30.7 / 35.5 |
| without 1-2: mean point | shuffled | 34.8 / 38.8 / 38.1 | 41.2 / 30.6 / 33.5 | 43.1 / 28.2 / 26.4 | 39.8 / 30.5 / 34.3 |
| without 1-2: mean point | others from another trial | 36.4 / 39.8 / 39.3 | 41.1 / 30.6 / 31.1 | 49.0 / 28.0 / 27.4 | 40.2 / 29.3 / 32.4 |
| without 1-2: H0+H1 |  | 30.5 / 39.3 / 36.9 | 33.7 / 30.3 / 30.5 | 42.9 / 26.4 / 25.5 | 35.4 / 31.7 / 33.3 |
| without 1-2: H0+H1 | shuffled | 30.7 / 35.2 / 37.9 | 27.6 / 27.1 / 31.0 | 35.0 / 19.6 / 23.6 | 30.4 / 27.9 / 26.9 |
| without 1-2: H0+H1 | others from another trial | 30.5 / 38.3 / 36.0 | 29.3 / 26.9 / 31.1 | 36.2 / 23.3 / 23.9 | 31.1 / 24.5 / 27.1 |
| without 3-5: H0 |  | 23.6 / 30.2 / 34.5 | 27.6 / 21.5 / 28.0 | 30.5 / 14.5 / 20.8 | 25.6 / 19.8 / 26.4 |
| without 3-5: H0 | shuffled | 22.4 / 29.8 / 30.2 | 22.9 / 20.2 / 24.5 | 29.0 / 17.0 / 19.8 | 22.4 / 14.0 / 24.8 |
| without 3-5: H0 | others from another trial | 21.4 / 32.4 / 30.2 | 25.1 / 20.5 / 24.7 | 28.8 / 17.4 / 15.5 | 23.8 / 15.2 / 21.2 |
| without 3-5: region counts |  | 16.9 / 28.3 / 30.7 | 18.8 / 19.7 / 21.8 | 21.9 / 12.9 / 15.6 | 18.7 / 14.8 / 19.0 |
| without 3-5: region counts | shuffled | 15.7 / 26.7 / 28.6 | 15.6 / 19.0 / 20.0 | 16.9 / 11.2 / 13.1 | 14.5 / 13.1 / 13.6 |
| without 3-5: region counts | others from another trial | 15.5 / 28.6 / 28.3 | 16.8 / 19.7 / 20.7 | 18.3 / 12.2 / 13.0 | 15.5 / 15.7 / 15.0 |
| without 3-5: sliced W1 |  | 37.6 / 37.6 / 36.7 | 45.3 / 27.8 / 27.4 | 53.1 / 24.6 / 24.8 | 42.1 / 27.9 / 26.4 |
| without 3-5: sliced W1 | shuffled | 32.4 / 35.2 / 35.5 | 41.0 / 25.9 / 26.0 | 47.6 / 21.8 / 23.2 | 35.0 / 23.1 / 22.6 |
| without 3-5: sliced W1 | others from another trial | 34.0 / 33.1 / 36.2 | 39.5 / 27.0 / 26.9 | 49.8 / 22.0 / 22.5 | 36.4 / 24.0 / 23.6 |
| without 3-5: mean point |  | 11.0 / 30.2 / 30.2 | 15.1 / 21.6 / 21.0 | 16.9 / 15.8 / 15.1 | 15.0 / 19.0 / 18.8 |
| without 3-5: mean point | shuffled | 13.1 / 27.1 / 28.1 | 13.4 / 19.5 / 20.6 | 16.2 / 13.2 / 12.5 | 14.5 / 14.3 / 16.9 |
| without 3-5: mean point | others from another trial | 12.1 / 28.1 / 27.4 | 12.6 / 19.4 / 20.2 | 13.1 / 12.5 / 12.8 | 12.7 / 16.7 / 14.3 |
| without 3-5: H0+H1 |  | 19.5 / 29.3 / 30.2 | 24.9 / 21.4 / 22.7 | 26.9 / 15.8 / 16.7 | 22.6 / 16.7 / 19.5 |
| without 3-5: H0+H1 | shuffled | 18.1 / 27.9 / 29.8 | 19.3 / 21.2 / 23.2 | 20.2 / 13.1 / 15.2 | 18.4 / 18.3 / 20.7 |
| without 3-5: H0+H1 | others from another trial | 19.3 / 31.0 / 30.2 | 21.2 / 21.1 / 20.5 | 23.6 / 14.4 / 14.4 | 18.8 / 14.3 / 13.6 |
| without 6-7: H0 |  | 55.2 / 51.0 / 52.4 | 58.9 / 42.6 / 48.9 | 61.0 / 41.9 / 44.9 | 54.9 / 43.6 / 50.5 |
| without 6-7: H0 | shuffled | 45.7 / 43.6 / 46.0 | 48.3 / 41.3 / 44.1 | 51.0 / 35.2 / 36.9 | 48.1 / 37.4 / 45.5 |
| without 6-7: region counts |  | 50.5 / 49.8 / 49.8 | 53.3 / 41.7 / 45.9 | 57.1 / 37.1 / 38.8 | 50.8 / 48.8 / 46.2 |
| without 6-7: region counts | shuffled | 35.5 / 43.6 / 43.8 | 42.0 / 38.7 / 41.6 | 46.2 / 33.1 / 35.3 | 41.8 / 38.8 / 41.0 |
| without 6-7: sliced W1 |  | 59.8 / 43.6 / 45.5 | 61.6 / 40.5 / 42.0 | 66.7 / 33.0 / 36.2 | 58.1 / 38.8 / 39.5 |
| without 6-7: sliced W1 | shuffled | 50.5 / 43.1 / 42.9 | 53.7 / 39.7 / 39.5 | 61.2 / 29.5 / 32.7 | 50.6 / 38.8 / 36.9 |
| without 6-7: mean point |  | 48.1 / 41.2 / 44.3 | 52.1 / 35.3 / 35.3 | 56.9 / 28.6 / 31.1 | 47.8 / 31.9 / 32.4 |
| without 6-7: mean point | shuffled | 42.1 / 37.1 / 39.5 | 48.3 / 35.8 / 37.1 | 52.1 / 27.4 / 28.8 | 45.7 / 37.1 / 35.2 |
| without 6-7: H0+H1 |  | 52.9 / 50.7 / 50.5 | 54.0 / 39.5 / 45.4 | 59.5 / 40.5 / 38.9 | 52.2 / 40.0 / 46.7 |
| without 6-7: H0+H1 | shuffled | 43.8 / 42.1 / 42.9 | 44.3 / 38.2 / 40.9 | 47.4 / 30.8 / 32.5 | 41.8 / 36.7 / 43.1 |
| only 1-2: H0 |  | 64.3 / 57.9 / 59.5 | 62.6 / 45.9 / 51.2 | 68.1 / 49.2 / 50.8 | 63.3 / 48.6 / 55.0 |
| only 1-2: H0 | shuffled | 62.1 / 53.8 / 56.4 | 58.8 / 42.1 / 52.1 | 63.8 / 45.3 / 47.4 | 58.5 / 47.1 / 56.2 |
| only 1-2: region counts |  | 63.6 / 55.0 / 53.1 | 59.8 / 46.1 / 49.9 | 64.8 / 45.9 / 43.3 | 57.9 / 51.7 / 54.3 |
| only 1-2: region counts | shuffled | 55.5 / 48.8 / 51.4 | 53.7 / 44.4 / 47.3 | 57.9 / 38.1 / 43.0 | 52.3 / 43.8 / 51.4 |
| only 1-2: sliced W1 |  | 60.2 / 47.1 / 47.6 | 64.5 / 41.6 / 42.4 | 68.1 / 39.0 / 38.5 | 59.8 / 42.9 / 41.4 |
| only 1-2: sliced W1 | shuffled | 56.0 / 46.0 / 45.0 | 59.7 / 42.7 / 41.9 | 63.1 / 35.1 / 33.6 | 55.2 / 43.1 / 38.3 |
| only 1-2: mean point |  | 12.4 / 29.5 / 29.5 | 13.8 / 20.2 / 20.7 | 13.6 / 14.0 / 13.9 | 13.3 / 14.0 / 16.2 |
| only 1-2: mean point | shuffled | 9.8 / 28.3 / 29.5 | 12.6 / 21.1 / 20.7 | 12.1 / 13.0 / 14.3 | 12.9 / 14.3 / 13.6 |
| only 1-2: H0+H1 |  | 59.0 / 46.7 / 47.6 | 58.1 / 42.0 / 48.9 | 61.0 / 34.4 / 40.3 | 55.7 / 42.9 / 53.8 |
| only 1-2: H0+H1 | shuffled | 54.8 / 53.3 / 50.7 | 53.9 / 41.8 / 48.9 | 57.6 / 43.4 / 43.0 | 52.7 / 37.6 / 51.9 |
| only 3-5: H0 |  | 39.3 / 42.6 / 44.5 | 45.7 / 36.6 / 40.8 | 47.4 / 31.4 / 33.7 | 42.3 / 31.9 / 40.0 |
| only 3-5: H0 | shuffled | 37.6 / 39.0 / 41.0 | 37.9 / 33.7 / 34.5 | 39.0 / 26.2 / 28.6 | 35.6 / 32.6 / 33.1 |
| only 3-5: region counts |  | 49.0 / 43.3 / 46.9 | 53.5 / 39.5 / 43.9 | 56.4 / 34.5 / 35.5 | 52.0 / 36.9 / 46.4 |
| only 3-5: region counts | shuffled | 39.5 / 41.7 / 46.9 | 40.5 / 36.5 / 37.9 | 42.9 / 31.3 / 36.4 | 41.4 / 36.2 / 36.9 |
| only 3-5: sliced W1 |  | 55.5 / 42.9 / 48.3 | 57.5 / 40.9 / 42.4 | 61.2 / 31.5 / 36.4 | 54.7 / 40.2 / 38.6 |
| only 3-5: sliced W1 | shuffled | 42.9 / 41.4 / 43.3 | 46.0 / 37.3 / 39.0 | 49.5 / 29.1 / 31.3 | 44.7 / 37.1 / 37.9 |
| only 3-5: mean point |  | 45.2 / 44.0 / 43.6 | 47.5 / 36.6 / 37.4 | 55.0 / 30.5 / 31.1 | 47.9 / 32.9 / 35.5 |
| only 3-5: mean point | shuffled | 41.4 / 39.3 / 42.4 | 43.9 / 36.1 / 37.5 | 46.9 / 29.7 / 30.2 | 42.9 / 32.9 / 36.4 |
| only 3-5: H0+H1 |  | 38.6 / 41.4 / 40.5 | 40.9 / 36.3 / 34.9 | 46.0 / 29.7 / 30.2 | 39.3 / 33.6 / 31.0 |
| only 3-5: H0+H1 | shuffled | 31.9 / 38.6 / 38.3 | 33.7 / 30.2 / 30.8 | 37.6 / 26.2 / 27.3 | 31.5 / 30.0 / 28.1 |
| only 6-7: H0 |  | 3.6 / 17.6 / 19.3 | 4.2 / 8.9 / 9.6 | 3.6 / 3.5 / 3.6 | 4.2 / 3.3 / 3.3 |
| only 6-7: H0 | shuffled | 2.1 / 15.5 / 17.4 | 3.1 / 8.8 / 9.5 | 3.6 / 3.0 / 3.2 | 4.2 / 4.8 / 4.5 |
| only 6-7: H0 | others from another trial | 3.3 / 17.9 / 17.4 | 2.1 / 8.1 / 8.7 | 2.6 / 2.9 / 2.4 | 3.2 / 2.9 / 4.0 |
| only 6-7: region counts |  | 3.1 / 18.1 / 18.3 | 2.5 / 8.0 / 8.5 | 2.1 / 2.8 / 2.4 | 3.0 / 2.1 / 3.6 |
| only 6-7: region counts | shuffled | 2.9 / 16.7 / 17.1 | 1.5 / 7.3 / 8.1 | 4.0 / 2.2 / 2.2 | 3.2 / 2.1 / 2.9 |
| only 6-7: region counts | others from another trial | 3.1 / 17.1 / 17.4 | 2.3 / 7.6 / 7.9 | 1.4 / 2.2 / 1.8 | 2.0 / 1.7 / 1.9 |
| only 6-7: sliced W1 |  | 2.4 / 16.4 / 17.4 | 1.8 / 8.6 / 8.7 | 2.1 / 2.4 / 2.3 | 2.6 / 3.6 / 2.9 |
| only 6-7: sliced W1 | shuffled | 1.2 / 16.0 / 16.4 | 1.8 / 8.0 / 8.1 | 3.3 / 1.5 / 2.2 | 2.6 / 2.9 / 2.6 |
| only 6-7: sliced W1 | others from another trial | 1.4 / 17.4 / 17.9 | 1.4 / 7.6 / 7.9 | 2.6 / 1.5 / 2.1 | 1.8 / 1.7 / 1.2 |
| only 6-7: mean point |  | 2.6 / 18.6 / 18.6 | 2.4 / 9.1 / 9.1 | 1.4 / 1.9 / 2.2 | 2.6 / 2.9 / 2.1 |
| only 6-7: mean point | shuffled | 1.7 / 17.9 / 17.9 | 2.6 / 8.4 / 8.2 | 3.1 / 2.3 / 2.2 | 2.2 / 2.6 / 2.9 |
| only 6-7: mean point | others from another trial | 1.2 / 18.8 / 18.8 | 1.4 / 8.2 / 8.4 | 1.4 / 1.7 / 1.4 | 1.8 / 1.9 / 2.1 |
| only 6-7: H0+H1 |  | 3.3 / 18.6 / 18.6 | 3.3 / 9.0 / 9.6 | 3.1 / 3.3 / 4.5 | 4.0 / 4.3 / 5.7 |
| only 6-7: H0+H1 | shuffled | 2.1 / 16.2 / 16.7 | 2.5 / 8.3 / 9.1 | 2.9 / 2.1 / 2.6 | 3.7 / 2.1 / 2.6 |
| only 6-7: H0+H1 | others from another trial | 3.1 / 17.9 / 16.9 | 1.7 / 8.6 / 8.7 | 3.3 / 3.1 / 2.8 | 2.3 / 1.7 / 4.3 |

### One-stimulus cloud, coordinates 1-8, within frozen x4

| signature | control | within frozen x4 |
|---|---|---|
| all 1-8: H0, K = 16 |  | 36.4 / 41.0 / 41.2 |
| all 1-8: H0, K = 16 | shuffled | 29.0 / 40.7 / 39.8 |
| all 1-8: H0, K = 16 | others from another trial | 35.0 / 40.7 / 41.0 |
| all 1-8: H0+H1, K = 16 |  | 33.8 / 40.2 / 38.8 |
| all 1-8: H0+H1, K = 16 | shuffled | 26.2 / 38.8 / 37.6 |
| all 1-8: H0+H1, K = 16 | others from another trial | 30.7 / 38.1 / 36.2 |
| all 1-8: region counts, K = 16 |  | 36.0 / 38.1 / 42.4 |
| all 1-8: region counts, K = 16 | shuffled | 26.2 / 38.1 / 40.5 |
| all 1-8: region counts, K = 16 | others from another trial | 30.7 / 35.0 / 41.0 |
| all 1-8: sliced W1 |  | 45.7 / 39.5 / 40.7 |
| all 1-8: sliced W1 | shuffled | 36.2 / 36.2 / 38.3 |
| all 1-8: sliced W1 | others from another trial | 42.9 / 39.5 / 42.6 |
| all 1-8: mean point |  | 40.0 / 39.8 / 39.5 |
| all 1-8: mean point | shuffled | 31.7 / 36.4 / 37.9 |
| all 1-8: mean point | others from another trial | 39.8 / 37.6 / 38.1 |
| all 1-8: H0, K = 64 |  | 31.9 / 38.6 / 37.4 |
| all 1-8: H0, K = 64 | shuffled | 32.4 / 34.8 / 37.1 |
| all 1-8: H0, K = 64 | others from another trial | 36.2 / 43.8 / 38.6 |
| all 1-8: H0+H1, K = 64 |  | 27.1 / 30.0 / 33.6 |
| all 1-8: H0+H1, K = 64 | shuffled | 23.6 / 28.3 / 31.7 |
| all 1-8: H0+H1, K = 64 | others from another trial | 29.3 / 31.9 / 33.8 |
| all 1-8: region counts, K = 64 |  | 36.4 / 40.5 / 38.6 |
| all 1-8: region counts, K = 64 | shuffled | 31.0 / 36.2 / 38.6 |
| all 1-8: region counts, K = 64 | others from another trial | 36.7 / 38.1 / 41.9 |
| all 1-8, time x4: H0, K = 16 |  | 25.5 / 34.3 / 33.8 |
| all 1-8, time x4: H0, K = 16 | shuffled | 18.1 / 30.5 / 32.1 |
| all 1-8, time x4: H0, K = 16 | others from another trial | 25.2 / 35.5 / 34.0 |
| all 1-8, time x4: H0+H1, K = 16 |  | 20.0 / 30.7 / 29.8 |
| all 1-8, time x4: H0+H1, K = 16 | shuffled | 19.8 / 31.0 / 31.2 |
| all 1-8, time x4: H0+H1, K = 16 | others from another trial | 20.5 / 31.9 / 31.0 |
| all 1-8, time x4: region counts, K = 16 |  | 19.5 / 34.3 / 31.4 |
| all 1-8, time x4: region counts, K = 16 | shuffled | 15.7 / 31.2 / 30.7 |
| all 1-8, time x4: region counts, K = 16 | others from another trial | 21.0 / 33.6 / 32.1 |
| all 1-8, time x4: sliced W1 |  | 30.0 / 33.3 / 33.3 |
| all 1-8, time x4: sliced W1 | shuffled | 27.4 / 32.9 / 33.1 |
| all 1-8, time x4: sliced W1 | others from another trial | 31.4 / 32.1 / 31.7 |
| all 1-8, time x4: mean point |  | 29.5 / 33.8 / 35.0 |
| all 1-8, time x4: mean point | shuffled | 27.1 / 34.0 / 35.5 |
| all 1-8, time x4: mean point | others from another trial | 27.4 / 33.6 / 35.0 |
| all 1-8, time x4: H0, K = 64 |  | 27.9 / 34.8 / 34.3 |
| all 1-8, time x4: H0, K = 64 | shuffled | 24.5 / 34.8 / 35.2 |
| all 1-8, time x4: H0, K = 64 | others from another trial | 32.4 / 33.6 / 35.7 |
| all 1-8, time x4: H0+H1, K = 64 |  | 19.5 / 26.4 / 26.9 |
| all 1-8, time x4: H0+H1, K = 64 | shuffled | 17.1 / 27.6 / 30.5 |
| all 1-8, time x4: H0+H1, K = 64 | others from another trial | 23.6 / 28.1 / 31.2 |
| all 1-8, time x4: region counts, K = 64 |  | 29.0 / 31.7 / 33.1 |
| all 1-8, time x4: region counts, K = 64 | shuffled | 25.0 / 31.4 / 36.0 |
| all 1-8, time x4: region counts, K = 64 | others from another trial | 32.6 / 36.0 / 36.4 |

## openscope124

60 neurons, 2310 + 0 samples; clouds needing replacement: 0

### References (same samples)

| signature | control | within gratings 124 s |
|---|---|---|
| ISI fractions |  | 64.2 / 31.5 / 32.4 |
| burst table |  | 71.8 / 35.2 / 35.8 |
| multiscale sqrt time bars |  | 13.7 / 13.0 / 10.6 |
| ISI fractions | shuffled | 64.2 / 31.5 / 32.4 |
| burst table | shuffled | 60.8 / 33.4 / 30.5 |
| multiscale sqrt time bars | shuffled | 5.0 / 10.6 / 7.9 |

### Stimulus-free cloud, coordinates 1-7

| signature | control | within gratings 124 s |
|---|---|---|
| H0, K = 16 |  | 62.4 / 32.2 / 34.8 |
| H0, K = 16 | shuffled | 41.3 / 26.2 / 27.5 |
| H0, K = 16 | others from another trial | 44.7 / 27.9 / 28.8 |
| H0+H1, K = 16 |  | 56.9 / 27.5 / 31.3 |
| H0+H1, K = 16 | shuffled | 38.8 / 23.9 / 25.2 |
| H0+H1, K = 16 | others from another trial | 40.6 / 25.5 / 25.8 |
| region counts, K = 16 |  | 56.2 / 31.8 / 32.8 |
| region counts, K = 16 | shuffled | 36.1 / 23.1 / 25.9 |
| region counts, K = 16 | others from another trial | 40.2 / 24.8 / 27.4 |
| sliced W1 |  | 63.4 / 29.9 / 29.7 |
| sliced W1 | shuffled | 45.5 / 24.2 / 24.4 |
| sliced W1 | others from another trial | 45.4 / 24.1 / 24.4 |
| mean point |  | 51.6 / 26.8 / 27.2 |
| mean point | shuffled | 42.2 / 23.5 / 23.9 |
| mean point | others from another trial | 39.4 / 22.0 / 22.6 |
| H0, K = 64 |  | 64.6 / 30.0 / 31.9 |
| H0, K = 64 | shuffled | 40.3 / 24.5 / 26.5 |
| H0, K = 64 | others from another trial | 46.8 / 24.4 / 26.3 |
| H0+H1, K = 64 |  | 53.5 / 28.6 / 29.1 |
| H0+H1, K = 64 | shuffled | 35.1 / 21.5 / 23.1 |
| H0+H1, K = 64 | others from another trial | 40.0 / 22.0 / 23.2 |
| region counts, K = 64 |  | 64.9 / 33.7 / 33.4 |
| region counts, K = 64 | shuffled | 40.3 / 24.9 / 25.8 |
| region counts, K = 64 | others from another trial | 47.6 / 24.1 / 28.1 |

### Ablations, K = 16 (stimulus-free)

| signature | control | within gratings 124 s |
|---|---|---|
| without 1-2: H0 |  | 54.1 / 28.7 / 28.7 |
| without 1-2: H0 | shuffled | 31.4 / 22.5 / 22.2 |
| without 1-2: H0 | others from another trial | 38.4 / 22.3 / 23.2 |
| without 1-2: region counts |  | 51.7 / 28.5 / 27.7 |
| without 1-2: region counts | shuffled | 31.4 / 22.9 / 22.9 |
| without 1-2: region counts | others from another trial | 36.5 / 22.8 / 23.5 |
| without 1-2: sliced W1 |  | 57.8 / 26.4 / 27.2 |
| without 1-2: sliced W1 | shuffled | 32.6 / 22.2 / 21.7 |
| without 1-2: sliced W1 | others from another trial | 38.7 / 21.6 / 21.9 |
| without 1-2: mean point |  | 42.8 / 26.5 / 25.5 |
| without 1-2: mean point | shuffled | 33.4 / 21.3 / 22.0 |
| without 1-2: mean point | others from another trial | 30.6 / 21.1 / 21.4 |
| without 1-2: H0+H1 |  | 51.2 / 26.8 / 27.4 |
| without 1-2: H0+H1 | shuffled | 28.4 / 21.2 / 22.3 |
| without 1-2: H0+H1 | others from another trial | 34.7 / 21.4 / 21.8 |
| without 3-5: H0 |  | 60.4 / 29.0 / 30.0 |
| without 3-5: H0 | shuffled | 37.7 / 22.5 / 22.2 |
| without 3-5: H0 | others from another trial | 40.4 / 22.6 / 23.6 |
| without 3-5: region counts |  | 53.2 / 29.7 / 29.4 |
| without 3-5: region counts | shuffled | 29.6 / 20.0 / 21.9 |
| without 3-5: region counts | others from another trial | 31.3 / 20.8 / 22.8 |
| without 3-5: sliced W1 |  | 58.8 / 25.3 / 26.4 |
| without 3-5: sliced W1 | shuffled | 34.9 / 19.7 / 20.1 |
| without 3-5: sliced W1 | others from another trial | 35.1 / 19.0 / 18.7 |
| without 3-5: mean point |  | 17.0 / 18.9 / 17.8 |
| without 3-5: mean point | shuffled | 9.9 / 14.6 / 14.2 |
| without 3-5: mean point | others from another trial | 8.8 / 14.2 / 14.4 |
| without 3-5: H0+H1 |  | 52.2 / 27.7 / 28.0 |
| without 3-5: H0+H1 | shuffled | 33.3 / 20.6 / 21.2 |
| without 3-5: H0+H1 | others from another trial | 35.3 / 21.0 / 22.1 |
| without 6-7: H0 |  | 64.7 / 38.2 / 38.9 |
| without 6-7: H0 | shuffled | 53.5 / 34.6 / 33.9 |
| without 6-7: region counts |  | 56.7 / 35.0 / 38.3 |
| without 6-7: region counts | shuffled | 45.0 / 31.0 / 34.2 |
| without 6-7: sliced W1 |  | 64.1 / 32.6 / 31.2 |
| without 6-7: sliced W1 | shuffled | 53.6 / 29.7 / 31.7 |
| without 6-7: mean point |  | 47.0 / 30.6 / 30.3 |
| without 6-7: mean point | shuffled | 45.3 / 28.8 / 28.5 |
| without 6-7: H0+H1 |  | 58.7 / 33.3 / 35.2 |
| without 6-7: H0+H1 | shuffled | 48.3 / 32.9 / 32.2 |
| only 1-2: H0 |  | 67.2 / 39.1 / 38.4 |
| only 1-2: H0 | shuffled | 57.3 / 35.8 / 35.3 |
| only 1-2: region counts |  | 59.3 / 34.5 / 37.4 |
| only 1-2: region counts | shuffled | 46.6 / 34.0 / 34.5 |
| only 1-2: sliced W1 |  | 64.8 / 32.2 / 31.0 |
| only 1-2: sliced W1 | shuffled | 54.8 / 29.7 / 29.5 |
| only 1-2: mean point |  | 9.1 / 15.4 / 15.6 |
| only 1-2: mean point | shuffled | 8.6 / 14.8 / 15.1 |
| only 1-2: H0+H1 |  | 60.5 / 37.4 / 37.3 |
| only 1-2: H0+H1 | shuffled | 50.6 / 32.2 / 33.4 |
| only 3-5: H0 |  | 38.3 / 28.2 / 27.2 |
| only 3-5: H0 | shuffled | 26.4 / 23.4 / 22.1 |
| only 3-5: region counts |  | 50.8 / 31.9 / 35.4 |
| only 3-5: region counts | shuffled | 35.5 / 29.6 / 29.3 |
| only 3-5: sliced W1 |  | 50.6 / 29.9 / 30.3 |
| only 3-5: sliced W1 | shuffled | 35.5 / 28.4 / 27.8 |
| only 3-5: mean point |  | 34.7 / 28.6 / 27.4 |
| only 3-5: mean point | shuffled | 35.0 / 26.9 / 26.3 |
| only 3-5: H0+H1 |  | 35.0 / 24.7 / 19.8 |
| only 3-5: H0+H1 | shuffled | 26.0 / 23.6 / 19.9 |
| only 6-7: H0 |  | 9.7 / 11.5 / 12.0 |
| only 6-7: H0 | shuffled | 1.1 / 5.9 / 6.3 |
| only 6-7: H0 | others from another trial | 1.6 / 6.7 / 7.1 |
| only 6-7: region counts |  | 8.9 / 11.6 / 12.0 |
| only 6-7: region counts | shuffled | 1.1 / 6.5 / 6.3 |
| only 6-7: region counts | others from another trial | 1.8 / 7.3 / 7.4 |
| only 6-7: sliced W1 |  | 11.9 / 12.2 / 12.5 |
| only 6-7: sliced W1 | shuffled | 0.9 / 6.3 / 6.5 |
| only 6-7: sliced W1 | others from another trial | 1.7 / 7.4 / 7.2 |
| only 6-7: mean point |  | 4.3 / 10.7 / 10.5 |
| only 6-7: mean point | shuffled | 2.1 / 7.1 / 7.4 |
| only 6-7: mean point | others from another trial | 2.4 / 7.7 / 7.8 |
| only 6-7: H0+H1 |  | 8.3 / 11.3 / 11.6 |
| only 6-7: H0+H1 | shuffled | 1.3 / 6.4 / 6.5 |
| only 6-7: H0+H1 | others from another trial | 2.1 / 7.0 / 7.2 |

### One-stimulus cloud, coordinates 1-8, within gratings 124 s

| signature | control | within gratings 124 s |
|---|---|---|
| all 1-8: H0, K = 16 |  | 54.5 / 25.3 / 28.0 |
| all 1-8: H0, K = 16 | shuffled | 33.9 / 22.5 / 23.9 |
| all 1-8: H0, K = 16 | others from another trial | 38.4 / 23.1 / 25.1 |
| all 1-8: H0+H1, K = 16 |  | 46.8 / 23.7 / 25.8 |
| all 1-8: H0+H1, K = 16 | shuffled | 30.3 / 20.1 / 23.1 |
| all 1-8: H0+H1, K = 16 | others from another trial | 32.6 / 19.9 / 21.3 |
| all 1-8: region counts, K = 16 |  | 47.1 / 25.5 / 27.3 |
| all 1-8: region counts, K = 16 | shuffled | 29.2 / 21.9 / 23.2 |
| all 1-8: region counts, K = 16 | others from another trial | 32.9 / 22.6 / 24.0 |
| all 1-8: sliced W1 |  | 57.1 / 27.4 / 27.6 |
| all 1-8: sliced W1 | shuffled | 40.1 / 24.4 / 23.5 |
| all 1-8: sliced W1 | others from another trial | 39.5 / 23.9 / 24.6 |
| all 1-8: mean point |  | 49.4 / 25.3 / 26.4 |
| all 1-8: mean point | shuffled | 40.1 / 23.9 / 22.8 |
| all 1-8: mean point | others from another trial | 36.1 / 22.9 / 22.3 |
| all 1-8: H0, K = 64 |  | 56.8 / 25.1 / 26.9 |
| all 1-8: H0, K = 64 | shuffled | 35.3 / 22.0 / 22.7 |
| all 1-8: H0, K = 64 | others from another trial | 38.4 / 19.9 / 23.4 |
| all 1-8: H0+H1, K = 64 |  | 47.6 / 23.7 / 24.3 |
| all 1-8: H0+H1, K = 64 | shuffled | 30.4 / 18.1 / 18.7 |
| all 1-8: H0+H1, K = 64 | others from another trial | 33.6 / 19.5 / 20.0 |
| all 1-8: region counts, K = 64 |  | 56.9 / 26.9 / 28.1 |
| all 1-8: region counts, K = 64 | shuffled | 35.7 / 21.9 / 23.0 |
| all 1-8: region counts, K = 64 | others from another trial | 40.8 / 20.2 / 23.4 |
| all 1-8, time x4: H0, K = 16 |  | 35.2 / 21.2 / 21.4 |
| all 1-8, time x4: H0, K = 16 | shuffled | 27.4 / 22.0 / 24.5 |
| all 1-8, time x4: H0, K = 16 | others from another trial | 29.0 / 20.2 / 20.4 |
| all 1-8, time x4: H0+H1, K = 16 |  | 29.7 / 19.1 / 20.2 |
| all 1-8, time x4: H0+H1, K = 16 | shuffled | 25.0 / 20.4 / 22.5 |
| all 1-8, time x4: H0+H1, K = 16 | others from another trial | 24.8 / 18.1 / 20.6 |
| all 1-8, time x4: region counts, K = 16 |  | 27.6 / 19.8 / 19.4 |
| all 1-8, time x4: region counts, K = 16 | shuffled | 22.7 / 20.3 / 22.4 |
| all 1-8, time x4: region counts, K = 16 | others from another trial | 25.9 / 18.7 / 20.7 |
| all 1-8, time x4: sliced W1 |  | 42.9 / 19.8 / 20.2 |
| all 1-8, time x4: sliced W1 | shuffled | 28.5 / 20.5 / 20.4 |
| all 1-8, time x4: sliced W1 | others from another trial | 30.1 / 18.6 / 19.9 |
| all 1-8, time x4: mean point |  | 40.1 / 20.2 / 19.9 |
| all 1-8, time x4: mean point | shuffled | 33.7 / 19.9 / 21.3 |
| all 1-8, time x4: mean point | others from another trial | 31.9 / 19.7 / 18.3 |
| all 1-8, time x4: H0, K = 64 |  | 39.5 / 22.1 / 22.7 |
| all 1-8, time x4: H0, K = 64 | shuffled | 25.1 / 19.1 / 20.5 |
| all 1-8, time x4: H0, K = 64 | others from another trial | 26.2 / 19.1 / 20.6 |
| all 1-8, time x4: H0+H1, K = 64 |  | 32.9 / 19.6 / 20.7 |
| all 1-8, time x4: H0+H1, K = 64 | shuffled | 20.4 / 16.8 / 18.1 |
| all 1-8, time x4: H0+H1, K = 64 | others from another trial | 21.9 / 16.7 / 18.0 |
| all 1-8, time x4: region counts, K = 64 |  | 39.7 / 23.5 / 24.8 |
| all 1-8, time x4: region counts, K = 64 | shuffled | 25.8 / 19.7 / 21.7 |
| all 1-8, time x4: region counts, K = 64 | others from another trial | 27.8 / 19.8 / 20.0 |
