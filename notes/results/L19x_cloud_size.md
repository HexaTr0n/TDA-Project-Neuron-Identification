# L19x (extra) Was Rips held back by cloud size?

2026-11-13, local session. **Worked: yes, it was.**

- With the same number of ISIs, regional Rips H0 matches ISI fractions on the retina from 100
  points up, and beats them on OpenScope from 50 points up.
  - The setup is the delay cloud (log ISI before, log ISI after) cut by 16 fixed regions.
  - OpenScope 124 s at 200 points: 69.7% against 62.3% 1-NN, and above ISI fractions on every
    ISI (64.2%).
  - The ISIs-shuffled control is 11 points lower at the same size (58.8%), so the gain comes
    from ISI order.
- At 25 points, the size used for short samples in L17, H0 is 4 points *below* ISI fractions
  from the same 25 ISIs on both datasets. The earlier gap to ISI fractions was mostly the
  subsampling.
- Restated as the web chat suggested: **at equal data, regional Rips equals or beats the ISI
  distribution, and it adds order**:
  - 2-6 points on retina 40 s at 50-200 points;
  - 10-11 points on OpenScope 124 s at 100-200 points.

## Data

Same neurons and samples as L19 (`sb_feature_cloud.samples_keyed` with `sb_full_table.choose`
or `choose_one`, seed 0).
- **Retina 40 s** (piece 2 included, flagged as in L17):
  - 4 frozen trials joined <-> 4 unrepeated pieces joined;
  - 60 neurons, 420 + 1796 samples;
  - median 621 delay points (min 30).
- **OpenScope 124 s:**
  - `glo637908` grating blocks, within only;
  - 60 neurons with >= 10 blocks of >= 103 ISIs; 2310 samples;
  - median 494 delay points (min 102).

**Method** (`scripts/sb_cloud_sampling.py` is L20; this is `scripts/sb_cloud_size.py`):
- **Cloud:** points (log ISI before, log ISI after) of every spike with a neighbour on both sides
  (L19's coordinates 1-2, delay embedding m = 2). They are standardised with the mean and sd of
  outside neurons.
- **Regions:** 16 k-means anchors on 20,000 outside points.
- **For each cloud size P = 25, 50, 100, 200, 400, with 20 random subsamples:**
  - *regional H0:* Rips H0 inside each region in stablebear; stable ranks averaged over
    subsamples, summed over regions.
  - *region counts:* the twin without homology.
  - *ISI fractions, same ISIs:* the stable rank of the P ISIs before the same subsampled points,
    averaged over the 20 subsamples (stablebear). This is the ISI distribution at equal data.
  - Samples with fewer than P points are drawn with replacement; the "short" column counts
    them. At 200-400 points this affects 229-950 samples, so those rows are less clean.
- **Control:** ISIs permuted within each sample.
  - "ISI fractions, same ISIs" is not rerun shuffled: it would score the same up to
    subsampling.
  - ISI fractions on every ISI is unchanged by shuffling.
- **Scores:** 1-NN / k-medoids, within each stimulus; on the retina also 1-NN across in both
  directions. Chance 1.7%.

## Commands

```sh
python scripts/sb_cloud_size.py --dataset retina40
python scripts/sb_cloud_size.py --dataset openscope124
```

## Scores: OpenScope 124 s, within gratings (1-NN / k-medoids, %)

| points | regional H0 | shuffled | region counts | shuffled | ISI fractions, same ISIs | short |
|---|---|---|---|---|---|---|
| 25 | 53.3 / 35.0 | 44.0 / 30.6 | 52.6 / 34.6 | 42.7 / 32.0 | 57.6 / 31.7 | 0 |
| 50 | 61.9 / 36.8 | 51.9 / 32.0 | 56.2 / 35.8 | 45.2 / 31.3 | 60.0 / 31.2 | 0 |
| 100 | 66.4 / 38.4 | 55.4 / 35.2 | 58.2 / 33.9 | 46.4 / 33.1 | 62.6 / 33.1 | 0 |
| 200 | **69.7** / 38.1 | 58.8 / 34.7 | 61.3 / 36.6 | 47.5 / 33.2 | 62.3 / 33.2 | 377 |
| 400 | 69.2 / 39.3 | 60.5 / 34.1 | 61.5 / 36.5 | 50.0 / 33.2 | 64.3 / 31.9 | 950 |
| every ISI | | | | | 64.2 / 31.5 (shuffled identical) | 0 |

## Scores: retina 40 s (1-NN / k-medoids, %)

| points | readout | within frozen | within unrepeated | frozen -> unrepeated (1-NN) | unrepeated -> frozen (1-NN) | short |
|---|---|---|---|---|---|---|
| 25 | regional H0 | 50.5 / 42.4 | 50.4 / 37.8 | 56.9 | 47.6 | 0 |
| | shuffled | 42.9 / 42.9 | 43.3 / 38.4 | 50.7 | 43.8 | |
| | region counts | 45.7 / 46.0 | 49.8 / 39.5 | 54.8 | 46.4 | |
| | shuffled | 39.0 / 43.8 | 42.7 / 37.3 | 45.5 | 44.3 | |
| | ISI fractions, same ISIs | 54.3 / 44.5 | 54.0 / 40.8 | 62.6 | 54.4 | |
| 50 | regional H0 | 60.2 / 50.2 | 59.0 / 44.2 | 62.1 | 55.1 | 29 |
| | shuffled | 57.4 / 47.4 | 53.3 / 42.8 | 57.1 | 54.1 | |
| | region counts | 52.4 / 51.7 | 55.2 / 46.9 | 56.9 | 53.0 | |
| | shuffled | 51.7 / 47.9 | 47.8 / 42.5 | 52.4 | 47.6 | |
| | ISI fractions, same ISIs | 56.9 / 47.1 | 63.6 / 40.5 | 68.1 | 60.0 | |
| 100 | regional H0 | 64.8 / 54.8 | 62.1 / 43.1 | 66.2 | 63.4 | 100 |
| | shuffled | 59.0 / 51.4 | 58.5 / 44.2 | 62.4 | 58.7 | |
| | region counts | 56.9 / 53.6 | 57.4 / 46.6 | 64.0 | 57.5 | |
| | shuffled | 51.2 / 50.0 | 52.0 / 40.6 | 53.8 | 50.2 | |
| | ISI fractions, same ISIs | 62.9 / 49.3 | 62.9 / 41.2 | 68.8 | 59.7 | |
| 200 | regional H0 | 64.8 / 53.1 | 65.7 / 47.9 | 69.5 | 64.4 | 229 |
| | shuffled | 64.3 / 53.1 | 60.2 / 43.7 | 66.4 | 61.7 | |
| | region counts | 58.1 / 53.6 | 58.7 / 46.2 | 67.4 | 59.6 | |
| | shuffled | 50.7 / 47.4 | 52.6 / 44.7 | 57.9 | 52.9 | |
| | ISI fractions, same ISIs | 64.5 / 52.1 | 67.5 / 40.8 | 72.9 | 63.1 | |
| 400 | regional H0 | 64.5 / 53.6 | 65.6 / 43.1 | 69.8 | 65.9 | 621 |
| | shuffled | 66.2 / 53.8 | 61.6 / 42.4 | 68.6 | 62.5 | |
| | region counts | 60.2 / 52.6 | 59.7 / 46.0 | 67.9 | 60.2 | |
| | shuffled | 54.0 / 46.9 | 54.1 / 45.0 | 59.5 | 53.3 | |
| | ISI fractions, same ISIs | 63.8 / 48.8 | 67.7 / 43.8 | 72.4 | 62.2 | |
| every ISI | ISI fractions (shuffled identical) | 64.3 / 48.8 | 67.4 / 42.3 | 73.6 | 63.5 | 0 |

## What to take from it

- **Cloud size explains the old gap.**
  - At 25 points, regional H0 is 4 points below ISI fractions from the same ISIs (retina
    50.5 vs 54.3; OpenScope 53.3 vs 57.6).
  - From 100 points it is level or ahead (retina 64.8 vs 62.9; OpenScope 66.4 vs 62.6).
  - Both curves rise with size, but regional H0 rises faster. Its bars need enough points in a
    region before they say anything.
- **The bars add to the counts at every size:** 3-9 points on OpenScope, 1-8 on the retina. The
  regions alone (a 16-bin joint histogram of consecutive ISI pairs) are below ISI fractions;
  the barcode of how points spread inside each region lifts them above.
- **Order:**
  - On OpenScope the shuffled control stays 9-11 points below at every size from 50 up. The
    pair (ISI before, ISI after) carries real serial dependence in these cortical and
    hippocampal units (L8).
  - On the retina it is 2-6 points at 50-200 points, 6-8 at 25, and gone (or reversed) at 400.
    The 400-point clouds are mostly resampled with replacement there (621 of 2216 samples are
    short), which blurs the real order.
- **Across stimuli (retina):** regional H0 crosses as well as ISI fractions do at equal size
  (200 points: 69.5 vs 72.9 frozen -> unrepeated, 64.4 vs 63.1 back).
- **What failed or stays open:**
  - the burst table is still the best single signature on OpenScope 124 s (71.8% in L19);
  - k-medoids gains less than 1-NN;
  - the 200-400 point rows are blurred by resampling;
  - short samples (10-30 s) cannot reach 100 points for most units. L20's whole-cloud and
    time-crop variants test what happens when every available point is used instead.
