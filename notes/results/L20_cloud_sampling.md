# L20 Cutting a delay cloud to size: whole cloud, time crop, random draw, consecutive stretch

2026-11-13, local session. **Worked: the whole cloud is the best way to feed Rips, and with 32
regions it is level with ISI fractions on long samples.**

- **Every point, one barcode** beats every averaged or cropped version on all five datasets:
  - delay regions k = 8 H0 against 25 or 100 random points: retina 10 s 42.4 vs 27.9, Allen
    34.4 vs 27.2, retina 40 s 56.7 vs 45.2 (1-NN within the first stimulus);
  - order: whole > crop 70% > random draws ~ crop 50% > consecutive stretch.
- **The number of regions matters.** 32 is best or within 1 point of best on every dataset:
  - OpenScope 124 s 67.2 against ISI fractions 64.2;
  - retina 40 s 67.4 against 64.3;
  - Allen 39.0 against 44.8; retina 10 s 43.7 against 50.9; OpenScope 10 s 21.5 against 26.1.
  - Fixed for L22: **k = 32**.
- **The spike count barely helps.** Dividing the stable rank by the number of points changes
  scores by -2 to +1 point. The bars carry the identity, not the count.
- **ISI order is in it.** The shuffled control is 3-11 points lower for the whole cloud with
  16-32 regions (OpenScope 124 s 67.2 vs 56.4).
- **What failed:**
  - consecutive stretches (the lowest everywhere, despite the largest order effect on the
    retina);
  - H0+H1, below H0 for regions on every dataset (-6 to -12 points at k = 32);
  - plain delay Rips without regions, even on whole clouds (13-34%).

## Data

Same neurons and samples as L17, and L19 for OpenScope 124 s (`sb_full_table.samples/choose`,
`sb_feature_cloud.samples_keyed/choose_one`, seed 0); 60 neurons each.
- **Retina 10 s:** piece 2 dropped, 1701 + 1788 samples.
- **OpenScope 10 s:** 1769 + 1490 samples.
- **Allen 30 s:** 1783 + 1734 samples.
- **Retina 40 s:** **old data with the repeating 5 s chunk** (L24), 420 + 1796 samples. These are
  stimulus-free rows, so the effect is at most ~3 points (L24); L22 uses `retina40clean`.
- **OpenScope 124 s:** 2310 samples, within gratings only.

**Clouds:** delay embedding m = 3 of log ISIs (`sb_full_table.delay_cloud`). Regions are k-means
anchors fitted on 8000 points of up to 300 samples of outside neurons. All Rips, stable ranks,
means and distances are in stablebear.

**Sampling** (`scripts/sb_cloud_sampling.py`):
- **whole:** every point, one barcode per sample. Reported raw and divided by the number of
  points. Regions k = 8, 16, 32, 64.
- **crop70 / crop50:** a random amount of time cut off the start and the end, so that 70% / 50%
  of the sample remains. Every point of it, one barcode, no averaging. Regions k = 8.
- **random:** P points drawn at random, 20 draws, stable ranks averaged. P = 25 on short samples
  and 100 on long ones (retina 40 s, OpenScope 124 s). Regions k = 8.
- **consecutive:** P consecutive points from a random start, 20 draws averaged. Regions k = 8.
- Samples shorter than P use every point (counted in the tables).

**Signatures:**
- delay Rips m = 3, H0 and H0+H1;
- delay regions, H0 and H0+H1 (regions summed raw; H0 and H1 groups median-scaled and added).

**Control:** ISIs permuted within every sample.

**Scores as L17:** 1-NN / k-medoids / k-means with `sb.mean` centres, within each stimulus and
across. Chance is 1.7%.

## Commands

```sh
python scripts/sb_cloud_sampling.py --dataset retina10 --drop-piece 2
python scripts/sb_cloud_sampling.py --dataset openscope10
python scripts/sb_cloud_sampling.py --dataset allen30
python scripts/sb_cloud_sampling.py --dataset retina40
python scripts/sb_cloud_sampling.py --dataset openscope124
```

## Summary: delay regions H0, 1-NN within A / A -> B (in brackets: within A, ISIs shuffled)

References on the same samples, 1-NN within A / A -> B:

| | retina 10 s | OpenScope 10 s | Allen 30 s | retina 40 s | OpenScope 124 s |
|---|---|---|---|---|---|
| ISI fractions | 50.9 / 49.0 | 26.1 / 21.0 | 44.8 / 25.4 | 64.3 / 73.6 | 64.2 |
| burst table | 52.6 / 47.4 | 31.4 / 25.4 | 47.7 / 30.6 | 68.3 / 74.8 | 71.8 |

| dataset | whole k=8 | whole k=16 | whole k=32 | whole k=64 | crop 70% k=8 | crop 50% k=8 | random k=8 | consecutive k=8 |
|---|---|---|---|---|---|---|---|---|
| retina10_without_piece2 | 42.4/41.3 (37.6) | 43.3/40.7 (38.9) | 43.7/40.0 (35.3) | 36.5/35.9 (32.9) | 34.3/33.8 (29.7) | 31.2/28.3 (23.2) | 27.9/28.4 (25.2) | 27.5/26.0 (19.6) |
| openscope10 | 19.0/16.5 (16.4) | 23.0/20.6 (17.2) | 21.5/19.3 (17.5) | 20.0/17.4 (15.9) | 17.4/14.6 (13.8) | 13.9/11.6 (11.4) | 16.4/12.6 (14.1) | 14.4/12.9 (12.7) |
| allen30 | 34.4/22.7 (26.5) | 38.4/25.4 (29.8) | 39.0/27.9 (30.0) | 35.8/26.0 (28.2) | 29.2/18.0 (20.6) | 21.6/15.5 (18.4) | 27.2/17.2 (18.7) | 22.7/14.6 (17.9) |
| retina40 | 56.7/63.3 (56.0) | 63.8/68.3 (60.5) | 67.4/70.7 (60.5) | 65.7/69.5 (62.4) | 54.0/59.3 (48.6) | 45.5/48.3 (43.6) | 45.2/50.2 (37.4) | 41.0/45.0 (36.7) |
| openscope124 | 59.2 (49.6) | 63.5 (53.5) | 67.2 (56.4) | 67.9 (57.4) | 53.2 (44.7) | 46.4 (42.3) | 57.1 (46.8) | 53.3 (43.3) |

Whole cloud at k = 32, other readouts (1-NN within A / A -> B):

| | retina 10 s | OpenScope 10 s | Allen 30 s | retina 40 s | OpenScope 124 s |
|---|---|---|---|---|---|
| H0 | 43.7 / 40.0 | 21.5 / 19.3 | 39.0 / 27.9 | 67.4 / 70.7 | 67.2 |
| H0, / points | 42.1 / 40.9 | 22.0 / 18.8 | 39.8 / 26.8 | 66.9 / 71.0 | 67.3 |
| H0+H1 | 32.0 / 31.0 | 15.4 / 14.3 | 28.0 / 20.1 | 61.2 / 65.7 | 61.1 |
| H0, k-means within A | 35.9 | 20.7 | 28.2 | 53.8 | 36.9 |
| plain delay Rips (no regions), H0 | 31.0 / 28.0 | 12.8 / 11.3 | 20.7 / 12.1 | 33.8 / 45.7 | 34.0 |
| plain delay Rips, H0+H1 | 31.9 / 28.7 | 13.0 / 10.6 | 21.2 / 12.6 | 39.0 / 50.0 | 36.4 |

## What to take from it

- **Using every point is what made Rips work.** L19x showed that the gap to ISI fractions was
  cloud size; L20 shows the simplest fix is to use the whole cloud.
  - Long samples: delay regions (k = 32) are level with or above ISI fractions (67.2 vs 64.2 on
    OpenScope 124 s; 67.4 vs 64.3 on retina 40 s), about 10 points above their shuffled control.
  - Short samples (10-30 s): they stay 5-7 points behind ISI fractions (Allen 39.0 vs 44.8,
    retina 43.7 vs 50.9), 3-9 points above shuffled.
  - The burst table is still the best single signature everywhere.
- **Not averaging is better than averaging.** A 70% crop with one barcode beats 20 averaged
  draws of 25 points on the short samples, and loses to the whole cloud by 2-8 points (k = 8). The
  barcode improves with every point it gets.
- **Consecutive stretches are worst.** They see the most order on the retina (27.5 vs 19.6
  shuffled) but too little of the ISI distribution.
- **Count leakage is negligible.** "/ points" is within 2 points of the raw version everywhere.
- **Across stimuli the whole cloud keeps most of its score** (retina 43.7 -> 40.0, OpenScope 10
  s 21.5 -> 19.3), like ISI fractions. On Allen the across drop (39.0 -> 27.9) is mostly drift
  (L24).
- **Clustering lags 1-NN** (k-means 20-54% where 1-NN is 21-67%). On OpenScope 124 s, k-means is
  36.9 against 1-NN 67.2: many tight but small clusters.
- **What failed:** H0+H1 (H1 lowers regional H0 everywhere: see L21 and L23); consecutive
  stretches; 64 regions on short samples (too few points per region).

## Full tables

Each cell is 1-NN / k-medoids / k-means, %.

### retina10_without_piece2

60 neurons, 1701 + 1788 samples, P = 25. Samples shorter than P (random / consecutive): 0 / 0.

| sampling | signature | control | within frozen | within unrepeated | frozen -> unrepeated | unrepeated -> frozen |
|---|---|---|---|---|---|---|
| whole | delay Rips m=3, H0 |  | 31.0 / 28.7 / 30.9 | 25.9 / 27.3 / 29.5 | 28.0 / 21.7 / 24.2 | 27.1 / 25.7 / 25.2 |
| whole | delay Rips m=3, H0 | shuffled | 28.8 / 28.0 / 29.0 | 24.8 / 26.0 / 27.6 | 28.2 / 19.9 / 23.4 | 26.1 / 24.6 / 24.3 |
| whole | delay Rips m=3, H0+H1 |  | 31.9 / 27.7 / 29.3 | 29.1 / 27.7 / 27.6 | 28.7 / 22.8 / 24.3 | 28.7 / 25.7 / 23.1 |
| whole | delay Rips m=3, H0+H1 | shuffled | 29.1 / 26.6 / 28.8 | 24.6 / 26.2 / 26.5 | 28.1 / 21.1 / 23.4 | 26.5 / 22.3 / 23.8 |
| whole | delay regions k=8, H0 |  | 42.4 / 37.0 / 40.4 | 38.9 / 31.9 / 38.7 | 41.3 / 30.0 / 35.1 | 39.5 / 27.7 / 36.9 |
| whole | delay regions k=8, H0 | shuffled | 37.6 / 34.3 / 36.6 | 35.3 / 28.5 / 34.8 | 36.2 / 28.1 / 30.4 | 34.1 / 22.7 / 32.7 |
| whole | delay regions k=8, H0+H1 |  | 36.9 / 30.3 / 33.5 | 32.9 / 28.7 / 30.5 | 35.4 / 25.7 / 27.1 | 34.8 / 24.6 / 27.0 |
| whole | delay regions k=8, H0+H1 | shuffled | 34.2 / 30.6 / 34.9 | 30.5 / 25.7 / 31.0 | 32.6 / 24.6 / 30.9 | 30.1 / 20.1 / 28.2 |
| whole | delay regions k=16, H0 |  | 43.3 / 32.7 / 38.2 | 39.5 / 31.2 / 38.0 | 40.7 / 27.0 / 33.2 | 39.7 / 26.5 / 34.6 |
| whole | delay regions k=16, H0 | shuffled | 38.9 / 33.2 / 40.0 | 31.9 / 30.0 / 33.3 | 35.4 / 25.3 / 31.7 | 35.0 / 30.6 / 33.6 |
| whole | delay regions k=16, H0+H1 |  | 33.6 / 31.5 / 32.4 | 33.6 / 27.0 / 28.8 | 32.3 / 24.0 / 27.0 | 33.4 / 22.6 / 24.8 |
| whole | delay regions k=16, H0+H1 | shuffled | 32.2 / 29.7 / 28.6 | 28.2 / 25.8 / 27.2 | 28.2 / 22.1 / 22.9 | 29.2 / 22.0 / 24.5 |
| whole | delay regions k=32, H0 |  | 43.7 / 33.9 / 35.9 | 36.7 / 30.2 / 35.7 | 40.0 / 26.9 / 31.7 | 39.6 / 28.3 / 31.5 |
| whole | delay regions k=32, H0 | shuffled | 35.3 / 32.0 / 37.6 | 32.4 / 29.7 / 35.1 | 33.6 / 26.3 / 32.0 | 33.9 / 26.3 / 33.7 |
| whole | delay regions k=32, H0+H1 |  | 32.0 / 26.0 / 25.5 | 29.8 / 24.0 / 23.0 | 31.0 / 22.7 / 21.5 | 30.5 / 20.8 / 21.5 |
| whole | delay regions k=32, H0+H1 | shuffled | 26.8 / 26.6 / 22.5 | 24.6 / 25.4 / 22.1 | 24.9 / 20.2 / 17.0 | 26.5 / 19.7 / 18.5 |
| whole | delay regions k=64, H0 |  | 36.5 / 31.2 / 34.4 | 34.6 / 28.5 / 31.6 | 35.9 / 26.8 / 31.6 | 34.5 / 24.5 / 28.5 |
| whole | delay regions k=64, H0 | shuffled | 32.9 / 28.6 / 32.0 | 29.4 / 28.7 / 31.0 | 30.7 / 22.4 / 26.6 | 32.2 / 24.2 / 27.9 |
| whole | delay regions k=64, H0+H1 |  | 23.4 / 21.4 / 15.9 | 22.8 / 20.6 / 15.2 | 23.7 / 16.6 / 12.5 | 24.0 / 16.2 / 13.6 |
| whole | delay regions k=64, H0+H1 | shuffled | 20.5 / 20.0 / 16.0 | 17.5 / 19.5 / 14.3 | 17.8 / 14.7 / 12.1 | 21.7 / 17.5 / 13.5 |
| whole | delay Rips m=3, H0, / points |  | 19.3 / 23.3 / 22.9 | 15.3 / 20.9 / 20.4 | 17.8 / 16.7 / 16.4 | 17.7 / 16.3 / 15.7 |
| whole | delay Rips m=3, H0, / points | shuffled | 17.9 / 21.2 / 21.2 | 13.6 / 20.0 / 19.7 | 16.9 / 13.3 / 13.3 | 15.9 / 14.3 / 14.9 |
| whole | delay Rips m=3, H0+H1, / points |  | 21.6 / 22.6 / 24.8 | 18.7 / 23.5 / 23.0 | 19.8 / 17.3 / 17.7 | 19.4 / 18.4 / 17.4 |
| whole | delay Rips m=3, H0+H1, / points | shuffled | 21.8 / 23.8 / 25.2 | 19.5 / 22.1 / 23.1 | 19.9 / 15.7 / 18.8 | 20.0 / 19.2 / 18.1 |
| whole | delay regions k=8, H0, / points |  | 38.6 / 28.7 / 32.6 | 35.2 / 28.6 / 35.3 | 36.7 / 24.2 / 27.7 | 35.1 / 23.6 / 31.0 |
| whole | delay regions k=8, H0, / points | shuffled | 31.0 / 27.2 / 31.3 | 30.1 / 25.7 / 29.1 | 29.0 / 22.4 / 25.7 | 28.7 / 20.2 / 24.0 |
| whole | delay regions k=8, H0+H1, / points |  | 32.9 / 26.6 / 29.6 | 29.9 / 28.8 / 30.5 | 33.2 / 20.3 / 23.9 | 31.8 / 23.9 / 25.9 |
| whole | delay regions k=8, H0+H1, / points | shuffled | 29.2 / 26.5 / 27.4 | 26.7 / 25.2 / 27.5 | 28.3 / 20.6 / 22.5 | 24.7 / 20.7 / 22.2 |
| whole | delay regions k=16, H0, / points |  | 39.7 / 34.6 / 35.9 | 37.4 / 31.4 / 37.5 | 40.3 / 28.1 / 30.6 | 36.3 / 26.6 / 36.4 |
| whole | delay regions k=16, H0, / points | shuffled | 34.7 / 27.8 / 32.7 | 29.8 / 27.6 / 29.3 | 32.6 / 22.0 / 24.6 | 31.3 / 24.2 / 28.0 |
| whole | delay regions k=16, H0+H1, / points |  | 32.5 / 29.2 / 30.9 | 31.9 / 26.8 / 30.2 | 32.3 / 22.4 / 24.4 | 31.1 / 24.6 / 25.5 |
| whole | delay regions k=16, H0+H1, / points | shuffled | 30.2 / 26.9 / 28.4 | 26.6 / 24.4 / 28.7 | 28.9 / 20.9 / 23.3 | 27.3 / 20.0 / 23.3 |
| whole | delay regions k=32, H0, / points |  | 42.1 / 31.9 / 41.1 | 37.6 / 32.8 / 36.2 | 40.9 / 25.6 / 35.2 | 37.8 / 30.5 / 36.4 |
| whole | delay regions k=32, H0, / points | shuffled | 34.2 / 30.3 / 34.4 | 30.3 / 28.1 / 31.2 | 34.3 / 25.2 / 27.3 | 31.5 / 25.6 / 31.7 |
| whole | delay regions k=32, H0+H1, / points |  | 33.6 / 29.6 / 29.3 | 31.5 / 26.8 / 30.1 | 33.5 / 24.4 / 23.4 | 31.0 / 22.6 / 26.3 |
| whole | delay regions k=32, H0+H1, / points | shuffled | 27.3 / 23.2 / 28.6 | 25.7 / 24.9 / 27.5 | 26.9 / 18.6 / 21.6 | 25.4 / 19.8 / 21.6 |
| whole | delay regions k=64, H0, / points |  | 38.6 / 31.2 / 32.2 | 35.7 / 32.2 / 31.9 | 39.4 / 25.6 / 26.4 | 34.7 / 28.4 / 26.4 |
| whole | delay regions k=64, H0, / points | shuffled | 33.3 / 30.3 / 33.0 | 30.5 / 28.1 / 32.9 | 32.5 / 23.5 / 26.2 | 31.3 / 23.4 / 30.3 |
| whole | delay regions k=64, H0+H1, / points |  | 30.0 / 25.8 / 26.3 | 29.5 / 24.6 / 23.8 | 32.3 / 19.6 / 21.8 | 28.5 / 20.7 / 20.3 |
| whole | delay regions k=64, H0+H1, / points | shuffled | 26.7 / 23.0 / 25.3 | 22.4 / 22.1 / 22.4 | 25.0 / 17.2 / 19.8 | 26.2 / 18.2 / 18.3 |
| crop70 | delay Rips m=3, H0 |  | 26.4 / 26.6 / 27.7 | 24.2 / 25.8 / 27.2 | 25.3 / 20.6 / 22.8 | 23.9 / 22.3 / 22.7 |
| crop70 | delay Rips m=3, H0 | shuffled | 22.5 / 25.9 / 27.2 | 20.6 / 24.5 / 25.5 | 20.6 / 17.7 / 18.1 | 21.0 / 18.7 / 19.6 |
| crop70 | delay Rips m=3, H0+H1 |  | 27.5 / 25.9 / 27.3 | 24.7 / 26.1 / 26.3 | 24.7 / 20.1 / 21.3 | 23.9 / 20.9 / 21.7 |
| crop70 | delay Rips m=3, H0+H1 | shuffled | 22.6 / 24.8 / 24.0 | 21.5 / 24.2 / 23.8 | 21.7 / 19.6 / 18.1 | 20.9 / 20.0 / 19.5 |
| crop70 | delay regions k=8, H0 |  | 34.3 / 28.5 / 34.9 | 30.6 / 29.4 / 33.1 | 33.8 / 21.5 / 28.4 | 33.3 / 24.2 / 29.9 |
| crop70 | delay regions k=8, H0 | shuffled | 29.7 / 30.0 / 32.3 | 30.3 / 28.6 / 32.6 | 30.0 / 24.8 / 27.2 | 29.4 / 26.3 / 28.7 |
| crop70 | delay regions k=8, H0+H1 |  | 28.5 / 28.0 / 27.9 | 27.3 / 27.1 / 27.9 | 28.6 / 23.2 / 24.6 | 27.9 / 22.6 / 21.4 |
| crop70 | delay regions k=8, H0+H1 | shuffled | 25.9 / 24.6 / 25.7 | 24.2 / 24.2 / 24.4 | 26.2 / 17.6 / 21.2 | 24.6 / 20.3 / 20.7 |
| crop50 | delay Rips m=3, H0 |  | 23.1 / 25.0 / 25.3 | 19.7 / 24.7 / 25.0 | 21.6 / 18.2 / 18.6 | 20.1 / 19.8 / 20.9 |
| crop50 | delay Rips m=3, H0 | shuffled | 18.6 / 23.3 / 24.1 | 19.2 / 20.7 / 21.9 | 18.5 / 16.7 / 17.4 | 18.2 / 13.5 / 17.9 |
| crop50 | delay Rips m=3, H0+H1 |  | 22.5 / 22.7 / 22.8 | 18.8 / 23.7 / 23.8 | 20.2 / 17.2 / 15.9 | 19.8 / 19.5 / 18.8 |
| crop50 | delay Rips m=3, H0+H1 | shuffled | 17.6 / 21.6 / 21.7 | 17.4 / 20.3 / 21.4 | 17.1 / 16.1 / 15.6 | 17.4 / 14.6 / 15.9 |
| crop50 | delay regions k=8, H0 |  | 31.2 / 28.2 / 31.4 | 29.8 / 25.3 / 30.0 | 28.3 / 21.8 / 24.6 | 28.5 / 21.1 / 24.3 |
| crop50 | delay regions k=8, H0 | shuffled | 23.2 / 25.3 / 28.0 | 24.6 / 26.6 / 26.0 | 25.8 / 18.7 / 21.5 | 25.4 / 20.3 / 22.2 |
| crop50 | delay regions k=8, H0+H1 |  | 24.1 / 24.4 / 23.8 | 22.2 / 23.0 / 21.5 | 21.3 / 18.3 / 18.8 | 25.0 / 17.4 / 17.8 |
| crop50 | delay regions k=8, H0+H1 | shuffled | 19.5 / 21.6 / 20.9 | 20.9 / 21.3 / 22.1 | 21.8 / 14.8 / 16.3 | 20.6 / 15.0 / 15.8 |
| random | delay Rips m=3, H0 |  | 16.1 / 21.9 / 21.5 | 14.6 / 19.6 / 21.0 | 15.5 / 15.2 / 14.8 | 16.1 / 16.0 / 16.9 |
| random | delay Rips m=3, H0 | shuffled | 15.4 / 20.3 / 20.4 | 12.5 / 19.4 / 18.6 | 16.0 / 13.0 / 13.4 | 14.7 / 14.3 / 15.8 |
| random | delay Rips m=3, H0+H1 |  | 17.5 / 20.8 / 21.6 | 17.2 / 20.2 / 21.4 | 17.7 / 14.9 / 15.8 | 16.9 / 16.6 / 13.6 |
| random | delay Rips m=3, H0+H1 | shuffled | 15.3 / 20.2 / 22.2 | 13.5 / 19.1 / 19.5 | 16.8 / 13.6 / 14.1 | 16.1 / 15.6 / 15.1 |
| random | delay regions k=8, H0 |  | 27.9 / 28.5 / 29.9 | 26.6 / 27.5 / 31.2 | 28.4 / 23.2 / 22.4 | 27.0 / 21.6 / 26.1 |
| random | delay regions k=8, H0 | shuffled | 25.2 / 23.9 / 27.4 | 24.4 / 22.9 / 26.1 | 23.3 / 15.9 / 19.5 | 22.4 / 18.0 / 21.4 |
| random | delay regions k=8, H0+H1 |  | 24.2 / 21.5 / 24.6 | 22.7 / 22.1 / 23.3 | 22.4 / 15.0 / 17.6 | 20.9 / 17.8 / 18.7 |
| random | delay regions k=8, H0+H1 | shuffled | 20.5 / 19.9 / 21.1 | 18.0 / 21.1 / 21.4 | 19.1 / 13.0 / 15.9 | 18.5 / 15.6 / 14.9 |
| consecutive | delay Rips m=3, H0 |  | 14.1 / 19.4 / 18.6 | 12.8 / 17.0 / 16.8 | 14.1 / 13.3 / 11.1 | 13.4 / 13.6 / 13.8 |
| consecutive | delay Rips m=3, H0 | shuffled | 12.2 / 17.8 / 16.8 | 10.2 / 16.8 / 17.4 | 12.7 / 11.5 / 11.0 | 11.1 / 11.8 / 11.5 |
| consecutive | delay Rips m=3, H0+H1 |  | 12.6 / 18.6 / 19.4 | 13.9 / 16.8 / 18.0 | 13.1 / 9.3 / 13.1 | 14.5 / 12.0 / 14.3 |
| consecutive | delay Rips m=3, H0+H1 | shuffled | 12.9 / 17.6 / 17.0 | 11.5 / 16.3 / 17.6 | 12.2 / 13.1 / 11.7 | 12.2 / 11.0 / 11.9 |
| consecutive | delay regions k=8, H0 |  | 27.5 / 26.3 / 26.5 | 26.0 / 24.1 / 27.2 | 26.0 / 19.0 / 20.9 | 26.2 / 20.9 / 20.9 |
| consecutive | delay regions k=8, H0 | shuffled | 19.6 / 22.9 / 24.5 | 19.0 / 21.1 / 24.0 | 20.7 / 17.1 / 15.3 | 20.6 / 14.1 / 17.6 |
| consecutive | delay regions k=8, H0+H1 |  | 20.5 / 20.2 / 21.9 | 20.0 / 19.6 / 19.8 | 22.2 / 15.1 / 14.9 | 20.2 / 14.9 / 15.0 |
| consecutive | delay regions k=8, H0+H1 | shuffled | 17.8 / 20.0 / 19.8 | 17.8 / 18.1 / 18.9 | 17.2 / 13.2 / 13.2 | 16.6 / 13.3 / 12.6 |

### openscope10

60 neurons, 1769 + 1490 samples, P = 25. Samples shorter than P (random / consecutive): 0 / 0.

| sampling | signature | control | within gratings | within grey | gratings -> grey | grey -> gratings |
|---|---|---|---|---|---|---|
| whole | delay Rips m=3, H0 |  | 12.8 / 17.6 / 18.1 | 15.0 / 18.1 / 17.7 | 11.3 / 10.9 / 9.2 | 12.5 / 8.5 / 10.5 |
| whole | delay Rips m=3, H0 | shuffled | 13.5 / 17.6 / 17.5 | 13.6 / 17.7 / 18.2 | 9.8 / 8.6 / 9.4 | 13.0 / 8.9 / 10.3 |
| whole | delay Rips m=3, H0+H1 |  | 13.0 / 18.3 / 17.2 | 14.5 / 17.7 / 17.8 | 10.6 / 10.2 / 9.7 | 12.3 / 9.9 / 8.2 |
| whole | delay Rips m=3, H0+H1 | shuffled | 12.7 / 17.5 / 17.4 | 13.6 / 18.5 / 18.6 | 11.0 / 6.9 / 10.5 | 13.0 / 9.0 / 9.0 |
| whole | delay regions k=8, H0 |  | 19.0 / 19.7 / 20.1 | 22.8 / 20.5 / 20.6 | 16.5 / 11.2 / 13.2 | 18.6 / 12.5 / 12.4 |
| whole | delay regions k=8, H0 | shuffled | 16.4 / 17.3 / 18.1 | 17.2 / 17.7 / 18.7 | 13.6 / 11.0 / 11.1 | 14.9 / 10.1 / 10.3 |
| whole | delay regions k=8, H0+H1 |  | 13.4 / 16.5 / 16.3 | 16.6 / 16.8 / 16.8 | 13.2 / 11.1 / 10.0 | 13.5 / 9.4 / 8.8 |
| whole | delay regions k=8, H0+H1 | shuffled | 13.5 / 16.9 / 17.0 | 14.6 / 16.1 / 17.6 | 11.4 / 9.1 / 9.1 | 11.2 / 8.0 / 10.6 |
| whole | delay regions k=16, H0 |  | 23.0 / 21.0 / 22.1 | 26.3 / 22.5 / 23.2 | 20.6 / 15.6 / 17.2 | 21.5 / 12.0 / 14.6 |
| whole | delay regions k=16, H0 | shuffled | 17.2 / 18.7 / 19.5 | 22.6 / 19.4 / 20.1 | 15.9 / 10.9 / 11.4 | 17.9 / 12.2 / 11.6 |
| whole | delay regions k=16, H0+H1 |  | 17.5 / 17.4 / 15.2 | 20.0 / 17.9 / 15.5 | 15.5 / 10.4 / 11.3 | 17.8 / 10.2 / 8.5 |
| whole | delay regions k=16, H0+H1 | shuffled | 14.6 / 16.2 / 14.6 | 16.5 / 18.3 / 15.5 | 11.8 / 9.2 / 10.7 | 13.4 / 10.2 / 9.0 |
| whole | delay regions k=32, H0 |  | 21.5 / 19.1 / 20.7 | 26.6 / 20.5 / 20.9 | 19.3 / 12.7 / 15.0 | 20.0 / 11.6 / 12.6 |
| whole | delay regions k=32, H0 | shuffled | 17.5 / 18.1 / 19.8 | 20.2 / 18.2 / 19.1 | 15.9 / 11.5 / 14.6 | 16.8 / 11.4 / 12.9 |
| whole | delay regions k=32, H0+H1 |  | 15.4 / 16.8 / 11.2 | 20.1 / 16.8 / 13.3 | 14.3 / 10.3 / 7.8 | 13.9 / 8.4 / 7.4 |
| whole | delay regions k=32, H0+H1 | shuffled | 14.1 / 16.5 / 11.8 | 13.5 / 15.2 / 11.3 | 12.0 / 10.3 / 6.6 | 12.6 / 7.7 / 5.5 |
| whole | delay regions k=64, H0 |  | 20.0 / 17.0 / 17.0 | 21.6 / 17.8 / 18.5 | 17.4 / 10.7 / 13.9 | 17.9 / 10.2 / 12.3 |
| whole | delay regions k=64, H0 | shuffled | 15.9 / 15.8 / 16.6 | 18.7 / 16.1 / 17.7 | 13.2 / 9.5 / 12.9 | 14.5 / 9.9 / 10.1 |
| whole | delay regions k=64, H0+H1 |  | 16.1 / 15.7 / 8.5 | 17.4 / 16.9 / 10.1 | 13.0 / 9.5 / 6.4 | 14.0 / 9.4 / 5.0 |
| whole | delay regions k=64, H0+H1 | shuffled | 11.8 / 14.6 / 7.3 | 14.1 / 15.6 / 9.9 | 9.9 / 9.1 / 5.2 | 10.6 / 7.5 / 4.9 |
| whole | delay Rips m=3, H0, / points |  | 7.1 / 13.2 / 13.4 | 6.4 / 13.9 / 14.3 | 6.1 / 5.3 / 6.0 | 6.3 / 5.7 / 6.2 |
| whole | delay Rips m=3, H0, / points | shuffled | 6.6 / 12.9 / 13.4 | 5.6 / 13.0 / 13.2 | 4.8 / 6.0 / 5.4 | 5.6 / 5.7 / 6.6 |
| whole | delay Rips m=3, H0+H1, / points |  | 9.3 / 14.8 / 14.8 | 10.8 / 14.6 / 15.8 | 6.9 / 6.8 / 6.8 | 7.3 / 7.9 / 6.9 |
| whole | delay Rips m=3, H0+H1, / points | shuffled | 9.9 / 14.8 / 14.2 | 10.4 / 15.4 / 15.1 | 8.6 / 8.9 / 6.6 | 8.6 / 6.8 / 7.1 |
| whole | delay regions k=8, H0, / points |  | 18.3 / 18.8 / 15.8 | 21.5 / 19.2 / 19.3 | 14.8 / 11.1 / 9.8 | 16.2 / 11.0 / 9.7 |
| whole | delay regions k=8, H0, / points | shuffled | 15.4 / 16.8 / 15.9 | 16.0 / 17.7 / 18.1 | 12.2 / 11.3 / 9.9 | 14.3 / 11.1 / 8.8 |
| whole | delay regions k=8, H0+H1, / points |  | 13.3 / 16.3 / 15.4 | 17.7 / 17.7 / 18.0 | 13.5 / 10.6 / 9.6 | 12.6 / 9.9 / 8.8 |
| whole | delay regions k=8, H0+H1, / points | shuffled | 13.2 / 17.2 / 16.3 | 14.3 / 16.4 / 16.7 | 10.5 / 9.0 / 9.1 | 10.5 / 8.3 / 9.9 |
| whole | delay regions k=16, H0, / points |  | 23.0 / 19.1 / 18.3 | 25.0 / 21.7 / 21.7 | 20.4 / 14.0 / 12.0 | 21.3 / 13.6 / 13.8 |
| whole | delay regions k=16, H0, / points | shuffled | 17.9 / 19.2 / 18.6 | 22.1 / 19.8 / 19.8 | 15.1 / 11.9 / 12.1 | 17.4 / 11.1 / 11.4 |
| whole | delay regions k=16, H0+H1, / points |  | 17.2 / 18.3 / 18.6 | 21.2 / 19.5 / 20.3 | 15.0 / 11.4 / 12.3 | 17.4 / 11.1 / 11.4 |
| whole | delay regions k=16, H0+H1, / points | shuffled | 14.9 / 17.1 / 17.1 | 16.6 / 19.3 / 16.6 | 11.5 / 10.5 / 10.3 | 13.4 / 10.6 / 9.2 |
| whole | delay regions k=32, H0, / points |  | 22.0 / 18.8 / 19.2 | 26.4 / 19.9 / 23.2 | 18.8 / 13.2 / 13.5 | 20.1 / 13.1 / 13.6 |
| whole | delay regions k=32, H0, / points | shuffled | 17.6 / 18.0 / 16.4 | 20.5 / 19.7 / 18.8 | 17.5 / 12.5 / 12.1 | 17.4 / 12.2 / 10.2 |
| whole | delay regions k=32, H0+H1, / points |  | 16.4 / 17.5 / 15.5 | 19.1 / 19.0 / 17.4 | 12.7 / 10.5 / 9.8 | 15.2 / 8.6 / 8.9 |
| whole | delay regions k=32, H0+H1, / points | shuffled | 14.4 / 15.8 / 14.0 | 14.6 / 16.0 / 15.0 | 12.6 / 9.3 / 9.1 | 12.8 / 7.8 / 7.7 |
| whole | delay regions k=64, H0, / points |  | 21.6 / 18.9 / 18.8 | 24.8 / 20.5 / 19.0 | 19.2 / 11.9 / 12.8 | 19.6 / 12.7 / 11.7 |
| whole | delay regions k=64, H0, / points | shuffled | 16.6 / 17.8 / 16.2 | 21.5 / 17.3 / 20.3 | 14.8 / 10.9 / 11.1 | 15.9 / 10.0 / 12.6 |
| whole | delay regions k=64, H0+H1, / points |  | 17.5 / 17.0 / 12.3 | 19.0 / 17.8 / 14.6 | 13.1 / 11.3 / 9.5 | 16.0 / 8.5 / 7.2 |
| whole | delay regions k=64, H0+H1, / points | shuffled | 13.1 / 15.1 / 10.5 | 15.2 / 16.8 / 12.6 | 11.3 / 9.6 / 6.6 | 12.6 / 9.4 / 6.3 |
| crop70 | delay Rips m=3, H0 |  | 11.0 / 16.0 / 16.8 | 11.3 / 16.4 / 16.4 | 9.4 / 8.3 / 9.2 | 10.8 / 9.5 / 10.3 |
| crop70 | delay Rips m=3, H0 | shuffled | 11.8 / 16.1 / 17.0 | 12.3 / 15.6 / 16.7 | 8.3 / 8.0 / 9.7 | 11.4 / 7.8 / 7.3 |
| crop70 | delay Rips m=3, H0+H1 |  | 11.4 / 15.8 / 16.3 | 10.7 / 15.4 / 16.0 | 9.6 / 8.6 / 9.1 | 9.8 / 8.6 / 8.8 |
| crop70 | delay Rips m=3, H0+H1 | shuffled | 12.7 / 15.7 / 16.5 | 11.7 / 17.0 / 16.1 | 8.7 / 7.9 / 9.9 | 10.5 / 7.1 / 7.9 |
| crop70 | delay regions k=8, H0 |  | 17.4 / 16.7 / 16.6 | 17.2 / 17.2 / 17.9 | 14.6 / 11.1 / 11.2 | 15.4 / 9.7 / 10.6 |
| crop70 | delay regions k=8, H0 | shuffled | 13.8 / 16.1 / 17.3 | 15.2 / 16.8 / 18.2 | 11.5 / 8.2 / 11.2 | 12.8 / 9.3 / 8.4 |
| crop70 | delay regions k=8, H0+H1 |  | 12.1 / 15.3 / 14.2 | 11.9 / 15.0 / 13.7 | 10.1 / 8.3 / 7.8 | 11.5 / 7.9 / 6.5 |
| crop70 | delay regions k=8, H0+H1 | shuffled | 10.5 / 15.0 / 14.1 | 12.7 / 14.5 / 14.8 | 10.0 / 8.0 / 7.9 | 9.9 / 6.7 / 6.4 |
| crop50 | delay Rips m=3, H0 |  | 10.1 / 15.0 / 15.3 | 9.5 / 16.5 / 16.1 | 8.7 / 8.3 / 9.5 | 11.1 / 8.9 / 9.2 |
| crop50 | delay Rips m=3, H0 | shuffled | 8.9 / 15.0 / 15.4 | 10.3 / 15.7 / 15.4 | 7.3 / 9.1 / 7.5 | 9.1 / 7.9 / 8.0 |
| crop50 | delay Rips m=3, H0+H1 |  | 10.1 / 15.5 / 14.3 | 9.5 / 16.0 / 15.6 | 7.1 / 7.9 / 8.5 | 9.7 / 9.3 / 7.9 |
| crop50 | delay Rips m=3, H0+H1 | shuffled | 9.3 / 15.2 / 14.9 | 11.1 / 15.7 / 15.9 | 8.9 / 6.2 / 9.5 | 8.7 / 6.4 / 7.8 |
| crop50 | delay regions k=8, H0 |  | 13.9 / 15.5 / 16.1 | 14.7 / 15.5 / 16.8 | 11.6 / 9.0 / 9.6 | 12.5 / 8.9 / 9.8 |
| crop50 | delay regions k=8, H0 | shuffled | 11.4 / 14.7 / 16.1 | 11.6 / 15.9 / 16.9 | 8.6 / 8.3 / 11.1 | 11.1 / 8.0 / 9.2 |
| crop50 | delay regions k=8, H0+H1 |  | 9.6 / 13.0 / 11.5 | 10.7 / 15.4 / 12.6 | 8.0 / 7.4 / 6.8 | 9.9 / 6.7 / 5.4 |
| crop50 | delay regions k=8, H0+H1 | shuffled | 9.3 / 13.8 / 12.5 | 8.3 / 14.6 / 12.8 | 7.7 / 6.5 / 5.6 | 9.0 / 6.5 / 6.4 |
| random | delay Rips m=3, H0 |  | 7.5 / 12.8 / 12.6 | 9.7 / 14.6 / 14.8 | 6.8 / 5.9 / 7.0 | 7.7 / 5.7 / 6.0 |
| random | delay Rips m=3, H0 | shuffled | 7.9 / 14.4 / 12.7 | 8.8 / 13.5 / 13.6 | 5.5 / 5.6 / 5.4 | 8.3 / 4.6 / 5.4 |
| random | delay Rips m=3, H0+H1 |  | 7.4 / 13.1 / 12.9 | 9.7 / 15.0 / 14.9 | 7.0 / 6.4 / 6.4 | 7.8 / 6.1 / 7.1 |
| random | delay Rips m=3, H0+H1 | shuffled | 7.3 / 12.9 / 12.8 | 8.0 / 14.2 / 14.3 | 6.5 / 7.0 / 6.6 | 7.8 / 6.1 / 5.9 |
| random | delay regions k=8, H0 |  | 16.4 / 17.2 / 16.6 | 19.4 / 18.3 / 19.4 | 12.6 / 11.2 / 9.9 | 15.8 / 9.7 / 9.0 |
| random | delay regions k=8, H0 | shuffled | 14.1 / 16.7 / 16.7 | 16.2 / 16.9 / 17.2 | 11.4 / 10.6 / 10.4 | 13.3 / 8.0 / 10.0 |
| random | delay regions k=8, H0+H1 |  | 11.2 / 13.8 / 13.8 | 15.5 / 15.7 / 15.1 | 10.9 / 7.9 / 8.1 | 12.8 / 7.3 / 8.9 |
| random | delay regions k=8, H0+H1 | shuffled | 12.2 / 14.1 / 15.7 | 13.3 / 15.4 / 15.5 | 9.4 / 6.4 / 7.7 | 11.3 / 8.0 / 6.7 |
| consecutive | delay Rips m=3, H0 |  | 7.8 / 12.5 / 13.6 | 8.9 / 14.7 / 13.7 | 5.6 / 6.4 / 6.3 | 7.4 / 5.8 / 5.0 |
| consecutive | delay Rips m=3, H0 | shuffled | 6.6 / 12.2 / 12.7 | 6.9 / 12.9 / 13.2 | 5.4 / 5.2 / 4.6 | 5.3 / 4.7 / 5.5 |
| consecutive | delay Rips m=3, H0+H1 |  | 7.7 / 12.6 / 13.1 | 7.5 / 13.6 / 13.7 | 6.0 / 5.7 / 5.4 | 7.0 / 6.4 / 5.4 |
| consecutive | delay Rips m=3, H0+H1 | shuffled | 6.7 / 12.4 / 12.3 | 6.6 / 13.8 / 13.7 | 5.5 / 4.9 / 5.5 | 6.8 / 4.5 / 4.5 |
| consecutive | delay regions k=8, H0 |  | 14.4 / 15.9 / 15.6 | 16.2 / 17.2 / 16.8 | 12.9 / 11.2 / 9.9 | 15.7 / 9.9 / 9.3 |
| consecutive | delay regions k=8, H0 | shuffled | 12.7 / 16.4 / 15.3 | 14.6 / 16.7 / 18.1 | 10.5 / 10.5 / 9.7 | 12.8 / 7.9 / 8.5 |
| consecutive | delay regions k=8, H0+H1 |  | 11.5 / 13.4 / 13.3 | 13.2 / 15.7 / 15.2 | 9.8 / 6.9 / 7.7 | 10.9 / 8.2 / 7.5 |
| consecutive | delay regions k=8, H0+H1 | shuffled | 9.5 / 13.8 / 13.3 | 12.6 / 14.8 / 15.5 | 8.9 / 8.1 / 6.9 | 9.9 / 7.6 / 7.7 |

### allen30

60 neurons, 1783 + 1734 samples, P = 25. Samples shorter than P (random / consecutive): 0 / 0.

| sampling | signature | control | within movie | within spontaneous | movie -> spontaneous | spontaneous -> movie |
|---|---|---|---|---|---|---|
| whole | delay Rips m=3, H0 |  | 20.7 / 19.0 / 19.9 | 17.4 / 18.6 / 18.9 | 12.1 / 6.5 / 8.1 | 11.2 / 9.8 / 9.1 |
| whole | delay Rips m=3, H0 | shuffled | 19.0 / 19.2 / 19.0 | 15.4 / 18.1 / 18.9 | 11.9 / 7.7 / 7.3 | 10.3 / 8.4 / 10.6 |
| whole | delay Rips m=3, H0+H1 |  | 21.2 / 19.9 / 20.0 | 17.2 / 18.3 / 18.3 | 12.6 / 9.1 / 8.8 | 11.8 / 8.2 / 10.3 |
| whole | delay Rips m=3, H0+H1 | shuffled | 20.1 / 19.6 / 19.2 | 16.8 / 17.4 / 18.0 | 12.6 / 8.0 / 8.2 | 10.0 / 9.3 / 9.1 |
| whole | delay regions k=8, H0 |  | 34.4 / 25.7 / 26.9 | 27.9 / 23.6 / 24.7 | 22.7 / 9.7 / 11.5 | 20.2 / 12.7 / 13.2 |
| whole | delay regions k=8, H0 | shuffled | 26.5 / 22.2 / 22.0 | 20.8 / 20.9 / 22.5 | 15.4 / 9.9 / 9.3 | 13.7 / 10.9 / 12.9 |
| whole | delay regions k=8, H0+H1 |  | 27.3 / 23.6 / 22.7 | 23.4 / 21.5 / 21.6 | 18.3 / 11.2 / 9.3 | 16.6 / 13.6 / 12.3 |
| whole | delay regions k=8, H0+H1 | shuffled | 23.3 / 20.9 / 19.2 | 18.5 / 19.9 / 20.2 | 12.8 / 8.5 / 8.5 | 11.6 / 8.9 / 8.1 |
| whole | delay regions k=16, H0 |  | 38.4 / 27.3 / 28.2 | 30.9 / 24.0 / 24.7 | 25.4 / 12.6 / 12.3 | 22.4 / 13.2 / 16.4 |
| whole | delay regions k=16, H0 | shuffled | 29.8 / 25.0 / 24.7 | 22.4 / 21.5 / 22.3 | 17.9 / 12.5 / 11.0 | 16.6 / 15.0 / 12.7 |
| whole | delay regions k=16, H0+H1 |  | 29.1 / 22.5 / 21.4 | 24.6 / 22.4 / 18.6 | 22.0 / 10.3 / 8.5 | 18.2 / 11.9 / 10.3 |
| whole | delay regions k=16, H0+H1 | shuffled | 24.1 / 21.6 / 19.7 | 18.7 / 20.1 / 18.5 | 14.1 / 8.2 / 8.2 | 13.4 / 10.9 / 9.5 |
| whole | delay regions k=32, H0 |  | 39.0 / 26.6 / 28.2 | 31.1 / 22.9 / 23.4 | 27.9 / 14.0 / 15.9 | 23.1 / 16.6 / 14.8 |
| whole | delay regions k=32, H0 | shuffled | 30.0 / 22.2 / 24.6 | 23.1 / 21.5 / 22.0 | 18.2 / 13.4 / 11.9 | 17.1 / 12.5 / 13.3 |
| whole | delay regions k=32, H0+H1 |  | 28.0 / 21.3 / 17.3 | 23.7 / 18.8 / 13.8 | 20.1 / 10.4 / 8.4 | 17.5 / 11.2 / 10.0 |
| whole | delay regions k=32, H0+H1 | shuffled | 21.1 / 18.3 / 15.9 | 17.4 / 17.8 / 13.0 | 14.5 / 7.8 / 5.4 | 13.0 / 10.2 / 9.1 |
| whole | delay regions k=64, H0 |  | 35.8 / 25.1 / 23.8 | 29.5 / 21.1 / 20.1 | 26.0 / 13.4 / 12.5 | 20.6 / 13.7 / 14.5 |
| whole | delay regions k=64, H0 | shuffled | 28.2 / 21.5 / 21.6 | 21.4 / 19.6 / 20.0 | 19.6 / 11.1 / 8.8 | 16.1 / 10.2 / 11.0 |
| whole | delay regions k=64, H0+H1 |  | 23.2 / 19.3 / 12.8 | 19.6 / 16.5 / 9.9 | 18.3 / 8.1 / 7.3 | 14.5 / 9.9 / 6.3 |
| whole | delay regions k=64, H0+H1 | shuffled | 17.2 / 18.1 / 11.7 | 13.3 / 16.7 / 10.0 | 13.5 / 7.5 / 3.4 | 10.7 / 10.4 / 6.0 |
| whole | delay Rips m=3, H0, / points |  | 11.8 / 15.3 / 15.4 | 12.4 / 14.9 / 15.0 | 8.4 / 5.9 / 5.4 | 7.2 / 6.3 / 6.1 |
| whole | delay Rips m=3, H0, / points | shuffled | 10.7 / 15.8 / 15.1 | 9.1 / 14.3 / 15.1 | 7.5 / 5.2 / 5.9 | 7.0 / 5.3 / 5.8 |
| whole | delay Rips m=3, H0+H1, / points |  | 14.6 / 16.3 / 16.7 | 12.6 / 16.0 / 15.6 | 9.5 / 6.3 / 5.9 | 8.8 / 7.6 / 7.3 |
| whole | delay Rips m=3, H0+H1, / points | shuffled | 14.5 / 16.8 / 16.7 | 12.7 / 15.9 / 16.1 | 9.9 / 5.8 / 6.9 | 7.5 / 7.6 / 7.6 |
| whole | delay regions k=8, H0, / points |  | 34.2 / 22.9 / 22.0 | 26.1 / 22.4 / 19.7 | 20.3 / 10.4 / 10.9 | 17.5 / 11.3 / 11.3 |
| whole | delay regions k=8, H0, / points | shuffled | 24.0 / 20.5 / 19.3 | 19.1 / 19.7 / 18.7 | 14.4 / 6.3 / 9.6 | 12.1 / 11.1 / 9.1 |
| whole | delay regions k=8, H0+H1, / points |  | 27.2 / 22.5 / 23.1 | 21.0 / 18.5 / 21.0 | 16.3 / 8.7 / 9.9 | 15.9 / 10.4 / 11.0 |
| whole | delay regions k=8, H0+H1, / points | shuffled | 22.7 / 19.5 / 19.4 | 17.8 / 18.1 / 18.6 | 13.0 / 9.1 / 8.7 | 11.1 / 9.7 / 9.5 |
| whole | delay regions k=16, H0, / points |  | 39.1 / 28.7 / 27.5 | 30.9 / 24.4 / 23.2 | 24.7 / 14.8 / 14.6 | 23.8 / 15.6 / 12.0 |
| whole | delay regions k=16, H0, / points | shuffled | 29.3 / 23.6 / 23.4 | 21.6 / 21.6 / 19.3 | 17.2 / 9.6 / 12.5 | 16.4 / 12.1 / 10.5 |
| whole | delay regions k=16, H0+H1, / points |  | 30.7 / 22.7 / 25.5 | 25.4 / 21.6 / 22.4 | 20.6 / 10.4 / 14.1 | 18.6 / 12.8 / 14.2 |
| whole | delay regions k=16, H0+H1, / points | shuffled | 24.3 / 20.6 / 21.8 | 20.6 / 20.0 / 19.3 | 13.9 / 10.7 / 8.2 | 13.3 / 11.0 / 9.9 |
| whole | delay regions k=32, H0, / points |  | 39.8 / 25.2 / 27.7 | 33.3 / 23.0 / 26.0 | 26.8 / 13.3 / 15.3 | 24.8 / 13.0 / 14.4 |
| whole | delay regions k=32, H0, / points | shuffled | 29.1 / 22.9 / 22.4 | 25.3 / 20.0 / 23.1 | 18.7 / 11.1 / 12.5 | 18.9 / 12.2 / 13.3 |
| whole | delay regions k=32, H0+H1, / points |  | 30.6 / 21.3 / 22.9 | 26.0 / 20.6 / 21.6 | 20.9 / 11.9 / 12.2 | 19.6 / 13.1 / 12.9 |
| whole | delay regions k=32, H0+H1, / points | shuffled | 22.9 / 19.7 / 21.3 | 18.3 / 18.7 / 19.0 | 14.8 / 8.5 / 9.2 | 13.7 / 10.4 / 10.8 |
| whole | delay regions k=64, H0, / points |  | 38.1 / 25.7 / 28.4 | 31.9 / 22.7 / 22.0 | 25.2 / 13.9 / 17.9 | 24.2 / 13.0 / 13.7 |
| whole | delay regions k=64, H0, / points | shuffled | 29.4 / 21.4 / 22.2 | 23.1 / 20.3 / 21.5 | 19.2 / 13.2 / 12.9 | 18.3 / 13.4 / 13.5 |
| whole | delay regions k=64, H0+H1, / points |  | 29.3 / 22.8 / 20.0 | 23.2 / 19.1 / 18.2 | 20.2 / 12.5 / 10.7 | 17.7 / 11.1 / 13.0 |
| whole | delay regions k=64, H0+H1, / points | shuffled | 21.4 / 19.3 / 17.6 | 16.1 / 18.6 / 16.0 | 15.4 / 9.1 / 6.7 | 13.6 / 10.3 / 8.2 |
| crop70 | delay Rips m=3, H0 |  | 16.8 / 18.1 / 18.1 | 13.0 / 17.2 / 17.6 | 9.6 / 5.3 / 5.9 | 9.0 / 7.8 / 6.9 |
| crop70 | delay Rips m=3, H0 | shuffled | 13.9 / 18.6 / 18.1 | 15.2 / 18.0 / 18.1 | 10.2 / 6.9 / 5.9 | 9.2 / 7.3 / 8.7 |
| crop70 | delay Rips m=3, H0+H1 |  | 17.6 / 17.7 / 17.4 | 14.0 / 16.3 / 15.9 | 9.7 / 7.8 / 6.2 | 9.7 / 8.9 / 7.7 |
| crop70 | delay Rips m=3, H0+H1 | shuffled | 14.4 / 18.6 / 18.8 | 14.2 / 17.5 / 17.3 | 11.1 / 6.7 / 7.6 | 8.9 / 8.4 / 10.2 |
| crop70 | delay regions k=8, H0 |  | 29.2 / 21.7 / 23.7 | 21.5 / 21.2 / 20.1 | 18.0 / 9.2 / 10.2 | 17.0 / 12.5 / 9.4 |
| crop70 | delay regions k=8, H0 | shuffled | 20.6 / 19.8 / 20.8 | 18.3 / 19.4 / 20.1 | 14.3 / 7.3 / 7.0 | 11.7 / 10.2 / 11.7 |
| crop70 | delay regions k=8, H0+H1 |  | 22.9 / 20.3 / 20.1 | 17.9 / 18.5 / 17.0 | 13.7 / 9.3 / 7.8 | 13.2 / 11.5 / 8.8 |
| crop70 | delay regions k=8, H0+H1 | shuffled | 15.6 / 18.6 / 17.9 | 14.9 / 17.4 / 17.7 | 10.7 / 7.2 / 5.7 | 10.7 / 8.7 / 8.9 |
| crop50 | delay Rips m=3, H0 |  | 14.8 / 16.4 / 17.0 | 12.4 / 15.9 / 16.1 | 9.0 / 4.8 / 5.9 | 7.3 / 9.3 / 7.9 |
| crop50 | delay Rips m=3, H0 | shuffled | 13.4 / 17.6 / 17.7 | 11.0 / 16.3 / 16.7 | 11.5 / 5.7 / 5.5 | 9.2 / 8.6 / 8.7 |
| crop50 | delay Rips m=3, H0+H1 |  | 14.2 / 17.8 / 17.1 | 11.2 / 14.9 / 15.0 | 9.5 / 6.1 / 6.3 | 7.6 / 7.7 / 8.4 |
| crop50 | delay Rips m=3, H0+H1 | shuffled | 12.8 / 16.9 / 17.2 | 11.1 / 16.4 / 16.8 | 9.8 / 5.4 / 6.9 | 7.6 / 7.0 / 7.8 |
| crop50 | delay regions k=8, H0 |  | 21.6 / 21.4 / 22.0 | 19.2 / 18.8 / 18.1 | 15.5 / 9.1 / 10.1 | 14.5 / 10.5 / 9.1 |
| crop50 | delay regions k=8, H0 | shuffled | 18.4 / 18.8 / 18.5 | 14.2 / 17.5 / 18.6 | 12.2 / 8.5 / 8.1 | 10.2 / 10.2 / 9.1 |
| crop50 | delay regions k=8, H0+H1 |  | 18.3 / 18.1 / 16.4 | 15.0 / 17.3 / 15.7 | 10.9 / 7.4 / 6.1 | 11.4 / 7.6 / 7.3 |
| crop50 | delay regions k=8, H0+H1 | shuffled | 16.1 / 16.9 / 15.6 | 12.5 / 15.5 / 14.0 | 10.3 / 7.8 / 6.0 | 8.1 / 7.9 / 7.0 |
| random | delay Rips m=3, H0 |  | 13.4 / 15.4 / 16.0 | 10.5 / 15.6 / 15.3 | 8.0 / 6.7 / 6.5 | 7.2 / 6.2 / 7.1 |
| random | delay Rips m=3, H0 | shuffled | 11.5 / 15.9 / 16.0 | 10.4 / 15.5 / 15.2 | 7.9 / 6.1 / 5.8 | 7.6 / 6.1 / 6.0 |
| random | delay Rips m=3, H0+H1 |  | 11.7 / 16.8 / 16.9 | 11.8 / 16.3 / 15.7 | 8.0 / 6.9 / 6.6 | 7.9 / 7.3 / 5.8 |
| random | delay Rips m=3, H0+H1 | shuffled | 10.3 / 15.9 / 16.7 | 10.0 / 16.1 / 15.9 | 8.3 / 7.2 / 6.5 | 7.8 / 6.6 / 7.6 |
| random | delay regions k=8, H0 |  | 27.2 / 22.9 / 24.0 | 20.8 / 20.1 / 19.2 | 17.2 / 11.4 / 11.1 | 17.0 / 9.3 / 12.1 |
| random | delay regions k=8, H0 | shuffled | 18.7 / 20.2 / 20.1 | 16.0 / 18.3 / 17.6 | 13.2 / 9.2 / 8.5 | 12.2 / 9.8 / 10.5 |
| random | delay regions k=8, H0+H1 |  | 22.0 / 20.6 / 19.8 | 17.8 / 18.3 / 17.6 | 15.2 / 8.4 / 9.3 | 13.7 / 10.4 / 9.6 |
| random | delay regions k=8, H0+H1 | shuffled | 15.1 / 16.9 / 16.9 | 13.7 / 16.2 / 16.2 | 11.3 / 7.0 / 6.6 | 12.0 / 8.8 / 8.7 |
| consecutive | delay Rips m=3, H0 |  | 10.3 / 14.9 / 15.3 | 9.4 / 15.2 / 14.9 | 7.1 / 5.9 / 6.4 | 7.6 / 6.3 / 7.7 |
| consecutive | delay Rips m=3, H0 | shuffled | 8.5 / 15.5 / 14.4 | 8.9 / 14.4 / 14.0 | 6.2 / 6.3 / 5.5 | 6.6 / 5.0 / 5.9 |
| consecutive | delay Rips m=3, H0+H1 |  | 10.2 / 15.1 / 14.9 | 10.7 / 15.0 / 14.5 | 8.1 / 6.7 / 6.3 | 7.3 / 5.8 / 6.2 |
| consecutive | delay Rips m=3, H0+H1 | shuffled | 8.4 / 13.9 / 13.9 | 8.8 / 15.1 / 14.6 | 6.5 / 5.8 / 6.2 | 7.0 / 5.6 / 6.2 |
| consecutive | delay regions k=8, H0 |  | 22.7 / 21.1 / 22.1 | 19.5 / 17.8 / 20.5 | 14.6 / 11.0 / 12.6 | 14.1 / 10.2 / 12.6 |
| consecutive | delay regions k=8, H0 | shuffled | 17.9 / 19.0 / 18.6 | 12.6 / 16.0 / 16.8 | 11.7 / 8.8 / 7.5 | 11.0 / 9.0 / 8.6 |
| consecutive | delay regions k=8, H0+H1 |  | 18.5 / 17.4 / 18.2 | 14.9 / 16.7 / 16.6 | 13.1 / 8.5 / 8.0 | 12.9 / 9.6 / 7.3 |
| consecutive | delay regions k=8, H0+H1 | shuffled | 13.3 / 16.8 / 15.6 | 11.0 / 15.5 / 14.8 | 11.3 / 6.1 / 6.2 | 9.4 / 5.9 / 8.2 |

### retina40

60 neurons, 420 + 1796 samples, P = 100. Samples shorter than P (random / consecutive): 100 / 101.

| sampling | signature | control | within frozen x4 | within unrepeated x4 (piece 2 in) | frozen x4 -> unrepeated x4 (piece 2 in) | unrepeated x4 (piece 2 in) -> frozen x4 |
|---|---|---|---|---|---|---|
| whole | delay Rips m=3, H0 |  | 33.8 / 40.0 / 40.0 | 41.9 / 33.5 / 34.7 | 45.7 / 26.9 / 28.2 | 38.0 / 33.3 / 30.5 |
| whole | delay Rips m=3, H0 | shuffled | 36.0 / 41.4 / 41.9 | 41.4 / 32.1 / 34.1 | 44.5 / 27.2 / 29.6 | 37.3 / 34.5 / 27.4 |
| whole | delay Rips m=3, H0+H1 |  | 39.0 / 41.2 / 42.4 | 44.4 / 33.1 / 33.2 | 50.0 / 29.5 / 29.9 | 42.0 / 29.3 / 31.0 |
| whole | delay Rips m=3, H0+H1 | shuffled | 44.5 / 40.7 / 42.9 | 45.2 / 33.5 / 35.1 | 51.4 / 29.8 / 29.7 | 43.2 / 31.4 / 33.3 |
| whole | delay regions k=8, H0 |  | 56.7 / 48.3 / 47.9 | 58.1 / 38.0 / 46.1 | 63.3 / 38.6 / 37.6 | 56.1 / 36.0 / 48.8 |
| whole | delay regions k=8, H0 | shuffled | 56.0 / 49.5 / 49.3 | 54.0 / 38.9 / 45.0 | 59.3 / 39.2 / 38.6 | 51.9 / 40.0 / 46.9 |
| whole | delay regions k=8, H0+H1 |  | 49.0 / 46.2 / 46.0 | 54.0 / 35.9 / 43.0 | 56.7 / 34.7 / 35.2 | 50.5 / 38.1 / 42.1 |
| whole | delay regions k=8, H0+H1 | shuffled | 50.0 / 45.2 / 44.8 | 49.2 / 34.1 / 42.1 | 55.0 / 35.7 / 36.2 | 49.2 / 35.0 / 43.3 |
| whole | delay regions k=16, H0 |  | 63.8 / 52.4 / 53.6 | 64.3 / 44.3 / 46.2 | 68.3 / 43.0 / 43.9 | 61.0 / 44.3 / 47.4 |
| whole | delay regions k=16, H0 | shuffled | 60.5 / 50.7 / 52.9 | 58.7 / 43.1 / 49.3 | 59.8 / 41.5 / 43.4 | 54.4 / 39.0 / 49.3 |
| whole | delay regions k=16, H0+H1 |  | 58.1 / 51.9 / 52.4 | 58.7 / 43.6 / 45.0 | 64.0 / 41.5 / 43.4 | 57.4 / 47.1 / 46.4 |
| whole | delay regions k=16, H0+H1 | shuffled | 55.7 / 46.4 / 49.3 | 52.6 / 39.9 / 47.6 | 59.0 / 36.9 / 37.9 | 52.2 / 37.9 / 47.4 |
| whole | delay regions k=32, H0 |  | 67.4 / 53.8 / 53.8 | 64.7 / 46.5 / 48.3 | 70.7 / 45.5 / 43.2 | 65.4 / 50.0 / 52.1 |
| whole | delay regions k=32, H0 | shuffled | 60.5 / 50.2 / 51.2 | 58.6 / 42.5 / 48.1 | 63.1 / 42.4 / 44.8 | 57.6 / 43.6 / 51.0 |
| whole | delay regions k=32, H0+H1 |  | 61.2 / 49.0 / 49.8 | 60.9 / 40.2 / 48.2 | 65.7 / 41.1 / 42.1 | 59.7 / 42.9 / 52.4 |
| whole | delay regions k=32, H0+H1 | shuffled | 53.1 / 47.6 / 49.3 | 55.0 / 40.9 / 48.0 | 60.7 / 39.9 / 41.5 | 53.5 / 43.3 / 48.1 |
| whole | delay regions k=64, H0 |  | 65.7 / 54.3 / 51.2 | 64.4 / 46.7 / 49.0 | 69.5 / 44.0 / 45.1 | 63.4 / 49.5 / 51.9 |
| whole | delay regions k=64, H0 | shuffled | 62.4 / 50.5 / 51.9 | 58.1 / 40.9 / 48.7 | 61.9 / 43.2 / 43.7 | 58.7 / 42.4 / 54.3 |
| whole | delay regions k=64, H0+H1 |  | 57.9 / 49.8 / 50.2 | 57.7 / 41.3 / 47.2 | 64.5 / 40.3 / 41.9 | 57.6 / 39.3 / 52.4 |
| whole | delay regions k=64, H0+H1 | shuffled | 54.0 / 45.5 / 47.6 | 52.6 / 38.5 / 45.4 | 58.8 / 33.4 / 39.0 | 53.0 / 37.4 / 49.0 |
| whole | delay Rips m=3, H0, / points |  | 23.1 / 33.1 / 33.6 | 26.2 / 25.4 / 24.6 | 28.1 / 18.8 / 19.0 | 24.3 / 21.4 / 20.2 |
| whole | delay Rips m=3, H0, / points | shuffled | 24.5 / 34.0 / 35.0 | 25.1 / 25.2 / 26.0 | 26.7 / 17.5 / 18.4 | 21.5 / 22.6 / 23.6 |
| whole | delay Rips m=3, H0+H1, / points |  | 30.5 / 36.7 / 37.1 | 33.5 / 29.3 / 29.5 | 35.7 / 21.3 / 23.8 | 31.5 / 30.2 / 30.0 |
| whole | delay Rips m=3, H0+H1, / points | shuffled | 32.4 / 37.4 / 38.1 | 32.1 / 29.1 / 28.6 | 37.9 / 24.1 / 24.1 | 30.6 / 30.2 / 30.2 |
| whole | delay regions k=8, H0, / points |  | 47.6 / 41.0 / 43.3 | 51.8 / 38.0 / 39.3 | 53.1 / 30.0 / 32.2 | 48.9 / 35.7 / 39.5 |
| whole | delay regions k=8, H0, / points | shuffled | 46.0 / 40.2 / 41.0 | 46.2 / 36.6 / 38.2 | 52.4 / 27.3 / 28.3 | 43.0 / 37.4 / 36.9 |
| whole | delay regions k=8, H0+H1, / points |  | 40.5 / 39.5 / 42.1 | 46.2 / 36.0 / 38.8 | 52.1 / 29.7 / 32.8 | 43.7 / 36.4 / 37.6 |
| whole | delay regions k=8, H0+H1, / points | shuffled | 41.9 / 38.6 / 40.7 | 41.8 / 34.7 / 37.9 | 48.1 / 28.8 / 27.5 | 39.8 / 32.1 / 36.7 |
| whole | delay regions k=16, H0, / points |  | 62.4 / 51.2 / 50.0 | 61.9 / 44.0 / 48.6 | 65.7 / 42.8 / 41.0 | 58.0 / 45.2 / 52.6 |
| whole | delay regions k=16, H0, / points | shuffled | 54.8 / 48.1 / 46.4 | 56.7 / 40.3 / 45.4 | 57.9 / 35.2 / 36.0 | 49.3 / 41.4 / 43.1 |
| whole | delay regions k=16, H0+H1, / points |  | 55.0 / 47.6 / 51.0 | 56.9 / 39.8 / 41.9 | 60.0 / 34.9 / 39.2 | 55.8 / 39.3 / 42.1 |
| whole | delay regions k=16, H0+H1, / points | shuffled | 50.5 / 45.5 / 46.2 | 49.8 / 39.0 / 42.1 | 59.3 / 36.5 / 36.4 | 48.3 / 38.8 / 45.2 |
| whole | delay regions k=32, H0, / points |  | 66.9 / 50.7 / 52.4 | 64.8 / 43.3 / 49.4 | 71.0 / 43.2 / 45.5 | 64.1 / 48.8 / 54.3 |
| whole | delay regions k=32, H0, / points | shuffled | 59.0 / 47.9 / 49.8 | 58.8 / 42.3 / 47.5 | 63.3 / 39.5 / 42.7 | 54.7 / 42.9 / 50.7 |
| whole | delay regions k=32, H0+H1, / points |  | 61.0 / 48.3 / 48.1 | 59.2 / 40.6 / 44.6 | 67.1 / 37.9 / 38.1 | 58.2 / 41.4 / 48.8 |
| whole | delay regions k=32, H0+H1, / points | shuffled | 51.7 / 44.8 / 44.8 | 53.1 / 42.0 / 45.4 | 61.0 / 34.9 / 32.6 | 50.9 / 42.4 / 46.7 |
| whole | delay regions k=64, H0, / points |  | 65.7 / 50.0 / 50.5 | 65.6 / 45.9 / 50.2 | 73.3 / 40.1 / 41.5 | 63.3 / 45.2 / 54.3 |
| whole | delay regions k=64, H0, / points | shuffled | 59.8 / 46.2 / 49.8 | 57.7 / 42.9 / 49.5 | 64.5 / 35.9 / 39.1 | 56.3 / 42.4 / 55.5 |
| whole | delay regions k=64, H0+H1, / points |  | 61.0 / 45.7 / 48.1 | 58.5 / 43.9 / 50.8 | 64.0 / 34.4 / 39.1 | 56.4 / 47.6 / 52.6 |
| whole | delay regions k=64, H0+H1, / points | shuffled | 51.9 / 45.7 / 46.7 | 52.8 / 38.0 / 50.4 | 62.1 / 34.6 / 37.0 | 51.3 / 37.4 / 51.7 |
| crop70 | delay Rips m=3, H0 |  | 34.3 / 39.0 / 41.2 | 36.6 / 30.1 / 34.2 | 43.6 / 26.1 / 27.7 | 36.6 / 29.3 / 33.8 |
| crop70 | delay Rips m=3, H0 | shuffled | 33.3 / 37.4 / 40.2 | 35.9 / 30.5 / 31.8 | 37.1 / 25.6 / 27.6 | 35.3 / 28.6 / 30.5 |
| crop70 | delay Rips m=3, H0+H1 |  | 36.9 / 40.2 / 40.7 | 38.0 / 30.3 / 33.6 | 42.9 / 25.4 / 28.3 | 38.9 / 30.2 / 35.2 |
| crop70 | delay Rips m=3, H0+H1 | shuffled | 37.6 / 38.3 / 39.0 | 40.9 / 32.0 / 34.0 | 49.5 / 26.1 / 26.2 | 39.2 / 31.0 / 34.0 |
| crop70 | delay regions k=8, H0 |  | 54.0 / 48.3 / 46.4 | 51.7 / 38.3 / 42.4 | 59.3 / 37.0 / 36.8 | 51.4 / 42.1 / 49.0 |
| crop70 | delay regions k=8, H0 | shuffled | 48.6 / 44.3 / 46.2 | 48.4 / 37.2 / 40.4 | 54.8 / 34.7 / 34.9 | 47.9 / 35.7 / 39.8 |
| crop70 | delay regions k=8, H0+H1 |  | 46.4 / 44.3 / 43.6 | 46.5 / 38.0 / 41.9 | 52.6 / 31.0 / 30.6 | 48.3 / 36.7 / 47.1 |
| crop70 | delay regions k=8, H0+H1 | shuffled | 48.1 / 40.5 / 43.1 | 43.4 / 34.5 / 37.8 | 46.7 / 29.5 / 34.1 | 44.0 / 34.8 / 34.8 |
| crop50 | delay Rips m=3, H0 |  | 34.3 / 40.0 / 40.0 | 32.6 / 28.5 / 30.1 | 41.0 / 25.7 / 25.8 | 34.7 / 27.6 / 29.3 |
| crop50 | delay Rips m=3, H0 | shuffled | 31.9 / 37.1 / 38.6 | 32.9 / 27.5 / 30.0 | 39.8 / 21.5 / 23.8 | 32.6 / 27.1 / 32.4 |
| crop50 | delay Rips m=3, H0+H1 |  | 35.7 / 38.3 / 39.5 | 35.5 / 30.6 / 30.7 | 40.5 / 22.7 / 26.9 | 36.0 / 33.6 / 30.7 |
| crop50 | delay Rips m=3, H0+H1 | shuffled | 35.7 / 37.4 / 36.7 | 37.4 / 29.3 / 32.4 | 39.0 / 24.1 / 24.5 | 34.7 / 26.9 / 29.3 |
| crop50 | delay regions k=8, H0 |  | 45.5 / 43.8 / 44.3 | 46.1 / 34.4 / 38.0 | 48.3 / 30.8 / 34.0 | 46.2 / 36.4 / 37.4 |
| crop50 | delay regions k=8, H0 | shuffled | 43.6 / 43.8 / 44.5 | 44.2 / 35.2 / 36.0 | 46.9 / 31.0 / 32.8 | 42.1 / 41.0 / 35.7 |
| crop50 | delay regions k=8, H0+H1 |  | 44.0 / 41.4 / 38.6 | 40.3 / 32.6 / 38.4 | 47.1 / 29.0 / 29.5 | 39.6 / 30.2 / 39.8 |
| crop50 | delay regions k=8, H0+H1 | shuffled | 38.3 / 41.0 / 42.4 | 38.6 / 31.7 / 34.0 | 36.4 / 29.4 / 30.2 | 35.3 / 31.2 / 33.6 |
| random | delay Rips m=3, H0 |  | 21.7 / 31.4 / 30.5 | 24.1 / 24.0 / 25.3 | 27.9 / 18.1 / 17.1 | 23.6 / 24.5 / 22.9 |
| random | delay Rips m=3, H0 | shuffled | 16.7 / 31.4 / 31.4 | 25.1 / 25.9 / 26.8 | 23.8 / 16.5 / 18.0 | 20.5 / 18.3 / 20.2 |
| random | delay Rips m=3, H0+H1 |  | 28.8 / 36.0 / 36.7 | 32.2 / 32.7 / 32.6 | 37.1 / 23.8 / 26.2 | 29.0 / 27.6 / 26.7 |
| random | delay Rips m=3, H0+H1 | shuffled | 31.7 / 35.5 / 36.9 | 30.5 / 32.6 / 33.4 | 32.4 / 23.5 / 25.1 | 30.1 / 29.5 / 32.9 |
| random | delay regions k=8, H0 |  | 45.2 / 38.6 / 41.4 | 45.8 / 36.2 / 38.9 | 50.2 / 26.3 / 28.2 | 43.1 / 35.7 / 38.6 |
| random | delay regions k=8, H0 | shuffled | 37.4 / 38.3 / 38.3 | 40.3 / 34.0 / 35.5 | 41.0 / 27.0 / 26.7 | 38.5 / 30.0 / 34.5 |
| random | delay regions k=8, H0+H1 |  | 37.6 / 38.1 / 38.6 | 40.4 / 29.0 / 35.8 | 44.0 / 25.9 / 26.9 | 39.7 / 27.1 / 34.3 |
| random | delay regions k=8, H0+H1 | shuffled | 31.0 / 36.2 / 36.9 | 38.3 / 32.3 / 33.2 | 38.6 / 23.5 / 22.3 | 34.1 / 28.8 / 32.9 |
| consecutive | delay Rips m=3, H0 |  | 15.2 / 31.2 / 29.8 | 20.1 / 21.8 / 21.8 | 19.5 / 17.9 / 15.4 | 18.6 / 18.3 / 17.1 |
| consecutive | delay Rips m=3, H0 | shuffled | 16.4 / 30.0 / 30.2 | 18.4 / 22.2 / 21.0 | 17.6 / 12.3 / 13.0 | 16.8 / 15.7 / 18.1 |
| consecutive | delay Rips m=3, H0+H1 |  | 27.1 / 35.2 / 35.2 | 28.1 / 28.2 / 29.1 | 31.7 / 23.9 / 22.3 | 26.7 / 21.7 / 25.7 |
| consecutive | delay Rips m=3, H0+H1 | shuffled | 22.1 / 32.9 / 35.7 | 26.9 / 27.3 / 28.6 | 30.2 / 21.7 / 23.0 | 26.2 / 23.8 / 28.8 |
| consecutive | delay regions k=8, H0 |  | 41.0 / 37.6 / 39.5 | 42.5 / 34.9 / 36.5 | 45.0 / 26.0 / 29.6 | 38.0 / 31.7 / 37.1 |
| consecutive | delay regions k=8, H0 | shuffled | 36.7 / 39.5 / 39.0 | 35.6 / 31.5 / 34.0 | 39.0 / 25.1 / 26.9 | 35.4 / 32.1 / 33.3 |
| consecutive | delay regions k=8, H0+H1 |  | 32.6 / 37.6 / 37.4 | 36.6 / 26.9 / 33.7 | 38.3 / 23.2 / 24.2 | 33.4 / 24.3 / 34.5 |
| consecutive | delay regions k=8, H0+H1 | shuffled | 28.6 / 34.5 / 36.7 | 30.3 / 28.7 / 31.0 | 34.0 / 21.9 / 21.8 | 29.8 / 26.0 / 31.4 |

### openscope124

60 neurons, 2310 + 0 samples, P = 100. Samples shorter than P (random / consecutive): 0 / 0.

| sampling | signature | control | within gratings 124 s |
|---|---|---|---|
| whole | delay Rips m=3, H0 |  | 34.0 / 23.0 / 22.9 |
| whole | delay Rips m=3, H0 | shuffled | 29.2 / 21.6 / 21.5 |
| whole | delay Rips m=3, H0+H1 |  | 36.4 / 22.3 / 23.2 |
| whole | delay Rips m=3, H0+H1 | shuffled | 33.7 / 21.4 / 21.5 |
| whole | delay regions k=8, H0 |  | 59.2 / 29.0 / 33.2 |
| whole | delay regions k=8, H0 | shuffled | 49.6 / 28.9 / 29.4 |
| whole | delay regions k=8, H0+H1 |  | 55.3 / 29.8 / 29.7 |
| whole | delay regions k=8, H0+H1 | shuffled | 45.5 / 29.0 / 28.7 |
| whole | delay regions k=16, H0 |  | 63.5 / 31.8 / 35.6 |
| whole | delay regions k=16, H0 | shuffled | 53.5 / 32.0 / 30.0 |
| whole | delay regions k=16, H0+H1 |  | 58.5 / 31.0 / 31.7 |
| whole | delay regions k=16, H0+H1 | shuffled | 49.1 / 29.1 / 30.0 |
| whole | delay regions k=32, H0 |  | 67.2 / 36.1 / 36.9 |
| whole | delay regions k=32, H0 | shuffled | 56.4 / 33.1 / 32.3 |
| whole | delay regions k=32, H0+H1 |  | 61.1 / 34.5 / 33.1 |
| whole | delay regions k=32, H0+H1 | shuffled | 50.7 / 33.8 / 31.4 |
| whole | delay regions k=64, H0 |  | 67.9 / 35.8 / 34.8 |
| whole | delay regions k=64, H0 | shuffled | 57.4 / 34.6 / 33.5 |
| whole | delay regions k=64, H0+H1 |  | 61.5 / 35.0 / 34.4 |
| whole | delay regions k=64, H0+H1 | shuffled | 51.0 / 32.3 / 30.6 |
| whole | delay Rips m=3, H0, / points |  | 19.7 / 16.8 / 16.6 |
| whole | delay Rips m=3, H0, / points | shuffled | 17.1 / 15.8 / 15.5 |
| whole | delay Rips m=3, H0+H1, / points |  | 28.7 / 18.5 / 18.6 |
| whole | delay Rips m=3, H0+H1, / points | shuffled | 24.5 / 17.4 / 17.5 |
| whole | delay regions k=8, H0, / points |  | 58.5 / 31.6 / 28.6 |
| whole | delay regions k=8, H0, / points | shuffled | 48.1 / 28.5 / 28.5 |
| whole | delay regions k=8, H0+H1, / points |  | 55.7 / 31.2 / 30.5 |
| whole | delay regions k=8, H0+H1, / points | shuffled | 44.7 / 28.9 / 28.9 |
| whole | delay regions k=16, H0, / points |  | 63.7 / 32.0 / 32.5 |
| whole | delay regions k=16, H0, / points | shuffled | 52.3 / 31.3 / 28.2 |
| whole | delay regions k=16, H0+H1, / points |  | 57.3 / 35.0 / 32.3 |
| whole | delay regions k=16, H0+H1, / points | shuffled | 47.1 / 31.3 / 31.1 |
| whole | delay regions k=32, H0, / points |  | 67.3 / 36.7 / 34.2 |
| whole | delay regions k=32, H0, / points | shuffled | 55.7 / 34.0 / 31.5 |
| whole | delay regions k=32, H0+H1, / points |  | 62.1 / 35.2 / 33.7 |
| whole | delay regions k=32, H0+H1, / points | shuffled | 49.3 / 34.3 / 32.7 |
| whole | delay regions k=64, H0, / points |  | 69.3 / 37.7 / 37.2 |
| whole | delay regions k=64, H0, / points | shuffled | 56.7 / 35.4 / 33.9 |
| whole | delay regions k=64, H0+H1, / points |  | 61.0 / 37.7 / 38.9 |
| whole | delay regions k=64, H0+H1, / points | shuffled | 51.6 / 33.4 / 34.9 |
| crop70 | delay Rips m=3, H0 |  | 30.0 / 21.0 / 21.5 |
| crop70 | delay Rips m=3, H0 | shuffled | 28.0 / 21.5 / 21.0 |
| crop70 | delay Rips m=3, H0+H1 |  | 31.9 / 20.6 / 22.0 |
| crop70 | delay Rips m=3, H0+H1 | shuffled | 30.6 / 21.3 / 21.3 |
| crop70 | delay regions k=8, H0 |  | 53.2 / 28.1 / 27.7 |
| crop70 | delay regions k=8, H0 | shuffled | 44.7 / 29.1 / 28.1 |
| crop70 | delay regions k=8, H0+H1 |  | 48.0 / 27.3 / 26.4 |
| crop70 | delay regions k=8, H0+H1 | shuffled | 42.8 / 28.7 / 26.4 |
| crop50 | delay Rips m=3, H0 |  | 26.4 / 21.0 / 21.0 |
| crop50 | delay Rips m=3, H0 | shuffled | 27.4 / 20.1 / 21.7 |
| crop50 | delay Rips m=3, H0+H1 |  | 28.5 / 20.2 / 21.3 |
| crop50 | delay Rips m=3, H0+H1 | shuffled | 27.9 / 20.5 / 21.5 |
| crop50 | delay regions k=8, H0 |  | 46.4 / 27.2 / 28.3 |
| crop50 | delay regions k=8, H0 | shuffled | 42.3 / 27.1 / 27.3 |
| crop50 | delay regions k=8, H0+H1 |  | 41.9 / 24.6 / 26.2 |
| crop50 | delay regions k=8, H0+H1 | shuffled | 38.7 / 24.7 / 25.3 |
| random | delay Rips m=3, H0 |  | 23.0 / 19.8 / 19.8 |
| random | delay Rips m=3, H0 | shuffled | 19.1 / 18.0 / 17.9 |
| random | delay Rips m=3, H0+H1 |  | 28.2 / 21.6 / 21.5 |
| random | delay Rips m=3, H0+H1 | shuffled | 24.5 / 20.7 / 20.2 |
| random | delay regions k=8, H0 |  | 57.1 / 32.5 / 32.4 |
| random | delay regions k=8, H0 | shuffled | 46.8 / 30.2 / 30.3 |
| random | delay regions k=8, H0+H1 |  | 51.5 / 30.6 / 31.3 |
| random | delay regions k=8, H0+H1 | shuffled | 41.2 / 27.7 / 27.7 |
| consecutive | delay Rips m=3, H0 |  | 20.0 / 19.4 / 19.5 |
| consecutive | delay Rips m=3, H0 | shuffled | 18.6 / 17.6 / 17.8 |
| consecutive | delay Rips m=3, H0+H1 |  | 24.7 / 20.6 / 20.0 |
| consecutive | delay Rips m=3, H0+H1 | shuffled | 23.6 / 19.3 / 19.1 |
| consecutive | delay regions k=8, H0 |  | 53.3 / 30.9 / 33.3 |
| consecutive | delay regions k=8, H0 | shuffled | 43.3 / 27.2 / 30.7 |
| consecutive | delay regions k=8, H0+H1 |  | 48.2 / 28.4 / 28.3 |
| consecutive | delay regions k=8, H0+H1 | shuffled | 38.6 / 26.6 / 27.4 |
