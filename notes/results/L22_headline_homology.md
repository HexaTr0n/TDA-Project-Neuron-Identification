# L22 Headline homology numbers (barcode-only rows, with their counting twins)

2026-11-13, local session. **Worked, with an honest caveat: most of the whole-cloud gain is a fine
joint histogram of consecutive ISIs.**

**Four-line summary** (1-NN, seed 0; in brackets: ISIs shuffled; every row is built from barcodes
only):
- **Retina, within a repeated stimulus:** delay regions + ISI bars + burst bars 82.3% (39.3);
  82.4% on the clean 40 s data. Square-root time bars give 87.7.
- **Retina, across stimuli (frozen -> unrepeated):**
  - 10 s: delay regions + ISI bars 41.4% (37.4), against ISI fractions 49.0.
  - 40 s clean: 68.6% (65.7), against ISI fractions 75.0.
- **Cortex, within a stimulus:**
  - OpenScope 124 s: delay regions 69.1% (57.7), against ISI fractions 64.2 and the burst
    table 71.8.
  - Allen movie, adjacent pair: delay regions + ISI bars + burst bars 46.9% (38.9), against ISI
    fractions 50.7.
- **Cortex, across stimuli:** Allen movie -> spontaneous just before it, delay regions 37.4%
  (27.4), against ISI fractions 38.7.

**What the bars add beyond counting** (regional H0 minus the region-count twin, 1-NN within A;
extra run `sb_twins.py`, same samples):

| | k = 8 | k = 16 | k = 32 |
|---|---|---|---|
| retina 10 s | +5.4 | +5.0 | -0.8 |
| retina 40 s clean | +10.0 | +6.0 | -0.2 |
| OpenScope 10 s | +4.1 | +1.4 | -1.5 |
| Allen adjacent pair | +10.2 | +4.0 | -0.8 |
| OpenScope 124 s | +14.1 | +5.6 | +2.5 (5 seeds: +2.5, range of each 60.9-72.2) |

- With few regions the bars carry real extra information (4-14 points). As k grows, the counts
  catch up while regional H0 stays about level.
- So at k = 32 the "delay regions" signature is mostly a 32-bin joint histogram of three
  consecutive log ISIs. The bars add 2.5 points only on the long cortical samples.
- **The order effect is real and shared by both:** shuffling ISIs costs regional H0 and region
  counts alike, 1-14 points.

## Data

Neurons and samples as L17 (`sb_full_table.choose`, seeds 0-4); L19's selection for OpenScope
124 s; 60 neurons unless stated. Chance 1.7%.

| table | samples | notes |
|---|---|---|
| retina 10 s, frozen <-> unrepeated, piece 2 dropped | 1701 + 1788 | |
| retina 40 s **clean** (`retina40clean`, L24), frozen x4 <-> unrepeated x4 | 420 + 1795 | |
| OpenScope 10 s, gratings <-> grey | 1769 + 1490 | |
| OpenScope 124 s, gratings, within only | 2310 | |
| Allen 30 s, movie <-> spontaneous (L17 pair) | 1783 + 1734 | |
| **Allen 30 s, movie block 1 <-> spontaneous 0-38 min (adjacent, L24 correction)** | 1551 + 781, 55 neurons | |
| nine retina sessions, frozen trials, within | 208-1701 trials, 18-60 neurons | outside neurons from the other eight sessions |

**Rows** (`scripts/sb_headline.py`; stablebear for every barcode, PCF and distance):
- **Delay regions (k = 32, H0):** the whole delay cloud (m = 3 log ISIs, every point, one
  barcode). Its 32 k-means anchors are fitted on 8000 points of 300 outside-neuron samples. Rips
  H0 per region; stable ranks summed over regions; point count left in. k = 32 is fixed from
  L20 for all datasets.
  - *Twin:* points per region.
- **ISI bars:** the stable rank of the bars [0, log(ISI / 1e-4)], not divided by the count.
  - *Twin:* spike count.
- **Delay regions + ISI bars:** the sum of the two distance matrices, each divided by its median
  over pairs of 300 outside-neuron samples.
  - *Twin:* region counts + spike count, scaled the same way.
- **+ burst bars:** the Betti curve of `barcodes.burst_bars` (s = 67 ms), an H0 that keeps time.
  Only meaningful within a repeated stimulus: read its "within A" column.
  - *Twin:* burst count.
- **References on the same samples:** ISI fractions, burst table, multiscale square-root time
  bars.

**Control:** ISIs permuted within every sample.

**Scores:** 1-NN / k-medoids / k-means with `sb.mean`, within each stimulus and across, as in
L17. k-means is skipped for seeds 1-4 and the nine sessions.

## Commands

```sh
python scripts/sb_headline.py --dataset retina10 --drop-piece 2          # also retina40clean, openscope10,
python scripts/sb_headline.py --dataset allen30adj                       # allen30, openscope124
python scripts/sb_headline.py --dataset retina10 --drop-piece 2 --seed 1 --no-kmeans   # seeds 1-4; also openscope124
python scripts/sb_headline.py --dataset retina:20171215_sr_le_sp --no-kmeans           # each of the nine sessions
python scripts/sb_twins.py --dataset retina10 --drop-piece 2             # k = 8, 16, 32 twins; also retina40clean,
                                                                         # openscope10, allen30adj, openscope124
```

## How stable is it

- **Seeds 0-4** (different 60 neurons each time), 1-NN within A, mean [range]:

  | | delay regions | region counts | regions + ISI bars | + burst bars | ISI fractions | burst table |
  |---|---|---|---|---|---|---|
  | retina 10 s | 43.5 [39.7-47.6] | 43.2 [37.6-48.2] | 46.5 [41.7-50.3] | 83.2 [81.2-85.7] | 52.5 [47.5-55.5] | 54.5 [49.5-57.9] |
  | OpenScope 124 s | 67.1 [63.5-72.2] | 64.6 [60.9-68.7] | 62.4 [58.8-66.6] | 65.4 [62.6-68.8] | 62.6 [59.2-64.2] | 69.6 [67.2-72.5] |

  The neuron draw moves every row by about +-4 points. The order of the rows holds in every
  seed.
- **Nine retina sessions** (within frozen trials, 1-NN):
  - delay regions 18.4-40.7 (mean 27.8);
  - + ISI bars + burst bars 45.6-88.3 (mean 65.6);
  - ISI fractions mean 35.0; time bars mean 70.2.
  - The session used everywhere else (20180126) is the best for ISI-based rows and third best
    for timing rows.
  - One session (20180316_sr_le_fp) has only 18 usable cells (chance 5.6%).

## What to take from it

- **Good homology numbers exist, and they lean on counting.**
  - Within a repeated stimulus, barcodes alone reach 82-88% on the retina; the burst bars carry
    it.
  - Across stimuli, the whole-cloud delay regions reach 21-69%. That is level with ISI fractions
    on cortex (OpenScope 10 s 21.3 vs 21.0; Allen adjacent pair 37.4 vs 38.7) and 6-9 points
    under them on the retina (10 s 39.6 vs 49.0; 40 s 68.1 vs 75.0).
  - On long cortical samples they are above ISI fractions (OpenScope 124 s: 67.1 vs 62.6 over
    five seeds), but below the burst table (69.6).
- **The honest description of delay regions at k = 32 is a fine joint histogram of consecutive
  ISIs.** The bars add 4-14 points at k = 8 and almost nothing at k = 32. The exception is the
  124 s cortical blocks, where they add 2.5 points (and 5.6 in k-medoids).
- **Adding ISI bars helps on short retina samples** (+3 over regions alone, five seeds) **and
  hurts on long cortical ones** (-5).
  - ISI bars with the count left in are themselves weak (retina 40 s 43.6 vs ISI fractions
    64.3): long samples have many more ISIs, so the stable rank is dominated by the count.
  - The fix would be ISI fractions, which divide by the count, but that is the counting
    signature the user wanted to go beyond.
- **The order effect is the part that is not counting.** Shuffling ISIs costs every
  delay-cloud row 1-14 points and leaves ISI-only rows unchanged. The delay cloud does see
  serial structure, but its counting twin sees the same.
- **What failed:**
  - the bars as more than a fine histogram at k = 32 on short samples;
  - ISI bars without normalisation;
  - burst bars across stimuli (they are time-locked, as expected).

## Full tables (seed 0)

Each cell is 1-NN / k-medoids / k-means, %.

### Retina 10 s, frozen <-> unrepeated (piece 2 dropped)

60 neurons, 1701 + 1788 samples. Burst-bar rows: read the within-frozen column only.

| row | control | within frozen | within unrepeated | frozen -> unrepeated | unrepeated -> frozen |
|---|---|---|---|---|---|
| delay regions (k=32, H0) |  | 40.6 / 33.9 / 38.3 | 39.1 / 31.1 / 36.8 | 39.6 / 25.7 / 36.2 | 38.3 / 27.6 / 36.2 |
| delay regions (k=32, H0) | shuffled | 35.4 / 32.0 / 35.2 | 33.2 / 28.9 / 34.2 | 32.8 / 22.9 / 30.1 | 33.8 / 25.5 / 32.3 |
| twin: region counts |  | 41.4 / 33.3 / 39.5 | 38.6 / 29.4 / 35.8 | 39.5 / 27.9 / 33.8 | 38.9 / 26.7 / 34.7 |
| twin: region counts | shuffled | 31.7 / 28.0 / 34.5 | 32.4 / 27.5 / 32.5 | 33.2 / 23.7 / 29.8 | 30.7 / 26.8 / 29.1 |
| ISI bars |  | 41.4 / 28.8 / 28.4 | 37.0 / 24.9 / 25.7 | 40.5 / 20.4 / 20.4 | 37.4 / 23.1 / 24.5 |
| ISI bars | shuffled | 41.4 / 28.8 / 28.4 | 37.0 / 24.9 / 25.7 | 40.5 / 20.4 / 20.4 | 37.4 / 23.1 / 24.5 |
| twin: spike count |  | 11.0 / 19.5 / 19.3 | 8.9 / 17.7 / 17.7 | 11.7 / 12.5 / 13.6 | 10.3 / 11.5 / 13.3 |
| twin: spike count | shuffled | 11.0 / 19.5 / 19.3 | 8.9 / 17.7 / 17.7 | 11.7 / 12.5 / 13.6 | 10.3 / 11.5 / 13.3 |
| delay regions + ISI bars |  | 43.3 / 34.3 / 33.8 | 41.2 / 32.5 / 33.5 | 41.4 / 30.4 / 27.4 | 41.1 / 29.6 / 32.5 |
| delay regions + ISI bars | shuffled | 37.3 / 31.1 / 34.6 | 35.1 / 29.6 / 33.4 | 37.4 / 23.4 / 27.7 | 37.9 / 28.2 / 34.5 |
| twin: region counts + spike count |  | 39.7 / 29.9 / 32.2 | 38.3 / 30.1 / 30.7 | 38.9 / 22.7 / 23.3 | 37.4 / 28.6 / 30.5 |
| twin: region counts + spike count | shuffled | 34.2 / 29.6 / 31.3 | 31.4 / 28.5 / 30.3 | 34.9 / 22.5 / 23.2 | 31.7 / 25.3 / 29.9 |
| delay regions + ISI bars + burst bars |  | 82.3 / 46.6 / 47.3 | 37.2 / 28.6 / 32.4 | 38.9 / 26.5 / 29.1 | 36.2 / 24.6 / 32.4 |
| delay regions + ISI bars + burst bars | shuffled | 39.3 / 32.3 / 33.2 | 31.4 / 28.3 / 31.4 | 34.0 / 25.1 / 25.7 | 33.7 / 26.2 / 30.1 |
| twin: + burst count |  | 37.1 / 28.5 / 29.5 | 32.8 / 24.6 / 25.0 | 34.6 / 20.9 / 22.3 | 33.0 / 19.5 / 21.6 |
| twin: + burst count | shuffled | 31.0 / 27.5 / 28.8 | 29.4 / 23.9 / 24.8 | 30.5 / 20.1 / 20.5 | 30.0 / 18.2 / 20.7 |
| ref: ISI fractions |  | 50.9 / 37.6 / 40.4 | 43.5 / 33.6 / 37.0 | 49.0 / 32.7 / 32.8 | 46.2 / 30.4 / 35.2 |
| ref: ISI fractions | shuffled | 50.9 / 37.6 / 40.4 | 43.5 / 33.6 / 37.0 | 49.0 / 32.7 / 32.8 | 46.2 / 30.4 / 35.2 |
| ref: burst table |  | 52.6 / 34.7 / 40.8 | 45.6 / 34.6 / 39.3 | 47.4 / 30.4 / 35.3 | 46.1 / 32.9 / 37.9 |
| ref: burst table | shuffled | 44.3 / 37.1 / 37.4 | 43.3 / 33.8 / 35.6 | 44.8 / 29.9 / 32.3 | 41.6 / 31.3 / 35.4 |
| ref: multiscale sqrt time bars |  | 87.7 / 48.2 / 70.2 | 5.0 / 12.7 / 13.6 | 5.9 / 6.7 / 8.1 | 6.1 / 7.7 / 8.3 |
| ref: multiscale sqrt time bars | shuffled | 6.8 / 13.8 / 14.1 | 5.0 / 12.9 / 13.8 | 5.6 / 7.5 / 7.5 | 6.3 / 9.2 / 9.1 |

### Retina 40 s (clean), frozen x4 <-> unrepeated x4

60 neurons, 420 + 1795 samples. Burst-bar rows: read the within-frozen x4 column only.

| row | control | within frozen x4 | within unrepeated x4 | frozen x4 -> unrepeated x4 | unrepeated x4 -> frozen x4 |
|---|---|---|---|---|---|
| delay regions (k=32, H0) |  | 62.9 / 55.2 / 55.0 | 65.9 / 44.8 / 53.7 | 68.1 / 47.0 / 46.7 | 63.4 / 46.9 / 59.0 |
| delay regions (k=32, H0) | shuffled | 61.9 / 50.5 / 51.7 | 60.8 / 41.7 / 48.0 | 64.8 / 41.9 / 44.7 | 58.2 / 41.0 / 48.6 |
| twin: region counts |  | 63.1 / 55.0 / 54.3 | 63.9 / 42.7 / 49.0 | 67.9 / 46.7 / 46.2 | 61.4 / 43.1 / 52.4 |
| twin: region counts | shuffled | 54.8 / 48.1 / 49.0 | 57.9 / 43.0 / 46.7 | 63.8 / 39.1 / 42.7 | 55.9 / 45.0 / 47.9 |
| ISI bars |  | 43.6 / 38.8 / 40.0 | 57.4 / 28.6 / 30.8 | 64.0 / 26.8 / 26.9 | 46.0 / 25.0 / 25.7 |
| ISI bars | shuffled | 43.6 / 38.8 / 40.0 | 57.4 / 28.6 / 30.8 | 64.0 / 26.8 / 26.9 | 46.0 / 25.0 / 25.7 |
| twin: spike count |  | 15.2 / 31.4 / 31.4 | 13.1 / 21.4 / 21.5 | 14.8 / 15.3 / 13.3 | 14.9 / 16.4 / 15.0 |
| twin: spike count | shuffled | 15.2 / 31.4 / 31.4 | 13.1 / 21.4 / 21.5 | 14.8 / 15.3 / 13.3 | 14.9 / 16.4 / 15.0 |
| delay regions + ISI bars |  | 59.8 / 48.1 / 49.0 | 64.7 / 40.3 / 42.3 | 68.6 / 37.3 / 40.8 | 59.5 / 40.2 / 44.3 |
| delay regions + ISI bars | shuffled | 59.5 / 48.3 / 50.2 | 60.8 / 39.9 / 40.9 | 65.7 / 39.6 / 39.9 | 56.3 / 41.4 / 43.6 |
| twin: region counts + spike count |  | 56.2 / 48.1 / 47.9 | 61.1 / 38.3 / 38.9 | 65.0 / 36.2 / 36.4 | 53.9 / 37.4 / 36.7 |
| twin: region counts + spike count | shuffled | 48.6 / 46.0 / 46.0 | 55.1 / 36.1 / 40.4 | 60.7 / 34.5 / 33.6 | 50.4 / 36.7 / 40.7 |
| delay regions + ISI bars + burst bars |  | 82.4 / 58.3 / 58.8 | 61.4 / 40.8 / 39.8 | 64.5 / 40.9 / 41.7 | 57.0 / 41.9 / 40.7 |
| delay regions + ISI bars + burst bars | shuffled | 54.8 / 47.9 / 43.1 | 58.0 / 38.2 / 39.6 | 62.9 / 38.4 / 35.3 | 54.3 / 39.8 / 41.7 |
| twin: + burst count |  | 46.9 / 43.6 / 44.8 | 54.6 / 35.8 / 35.9 | 57.4 / 28.2 / 33.4 | 48.4 / 34.3 / 35.5 |
| twin: + burst count | shuffled | 42.4 / 42.6 / 45.2 | 50.0 / 35.2 / 35.3 | 54.5 / 30.1 / 31.8 | 44.3 / 34.3 / 33.1 |
| ref: ISI fractions |  | 64.3 / 48.8 / 48.1 | 70.0 / 44.3 / 49.9 | 75.0 / 39.4 / 39.2 | 64.7 / 45.2 / 51.0 |
| ref: ISI fractions | shuffled | 64.3 / 48.8 / 48.1 | 70.0 / 44.3 / 49.9 | 75.0 / 39.4 / 39.2 | 64.7 / 45.2 / 51.0 |
| ref: burst table |  | 68.3 / 53.3 / 49.0 | 70.8 / 46.8 / 49.7 | 76.0 / 41.4 / 41.3 | 66.6 / 49.5 / 50.5 |
| ref: burst table | shuffled | 65.7 / 49.8 / 49.8 | 66.8 / 44.8 / 45.6 | 71.9 / 38.2 / 41.8 | 62.2 / 44.0 / 45.2 |
| ref: multiscale sqrt time bars |  | 80.7 / 62.1 / 58.6 | 5.1 / 15.2 / 14.8 | 7.4 / 6.8 / 8.5 | 6.0 / 11.0 / 10.2 |
| ref: multiscale sqrt time bars | shuffled | 3.3 / 21.7 / 21.9 | 4.8 / 15.5 / 14.7 | 4.8 / 9.3 / 7.6 | 5.5 / 11.0 / 12.1 |

### OpenScope 10 s, gratings <-> grey

60 neurons, 1769 + 1490 samples. Burst-bar rows: read the within-gratings column only.

| row | control | within gratings | within grey | gratings -> grey | grey -> gratings |
|---|---|---|---|---|---|
| delay regions (k=32, H0) |  | 22.4 / 20.1 / 19.5 | 26.0 / 20.1 / 22.0 | 21.3 / 13.5 / 14.5 | 21.5 / 12.8 / 15.0 |
| delay regions (k=32, H0) | shuffled | 16.7 / 18.1 / 18.5 | 19.4 / 17.8 / 19.5 | 15.0 / 15.0 / 10.9 | 16.4 / 11.3 / 11.2 |
| twin: region counts |  | 23.9 / 20.7 / 21.9 | 28.1 / 22.2 / 21.0 | 20.7 / 14.1 / 15.0 | 23.6 / 13.4 / 14.2 |
| twin: region counts | shuffled | 17.6 / 20.5 / 21.0 | 20.5 / 18.0 / 19.9 | 15.7 / 12.8 / 13.6 | 16.6 / 10.3 / 11.6 |
| ISI bars |  | 18.8 / 16.4 / 15.8 | 22.3 / 15.9 / 16.0 | 14.9 / 8.7 / 8.2 | 17.5 / 8.5 / 8.2 |
| ISI bars | shuffled | 18.8 / 16.4 / 15.8 | 22.3 / 15.9 / 16.0 | 14.9 / 8.7 / 8.2 | 17.5 / 8.5 / 8.2 |
| twin: spike count |  | 4.4 / 12.6 / 10.6 | 4.6 / 11.8 / 11.5 | 4.1 / 5.8 / 4.2 | 4.2 / 4.1 / 3.9 |
| twin: spike count | shuffled | 4.4 / 12.6 / 10.6 | 4.6 / 11.8 / 11.5 | 4.1 / 5.8 / 4.2 | 4.2 / 4.1 / 3.9 |
| delay regions + ISI bars |  | 22.4 / 19.0 / 18.3 | 27.3 / 20.1 / 19.3 | 19.1 / 11.4 / 11.1 | 20.5 / 12.4 / 11.6 |
| delay regions + ISI bars | shuffled | 18.5 / 18.3 / 18.7 | 21.1 / 19.4 / 18.6 | 15.7 / 10.5 / 10.4 | 16.4 / 10.4 / 10.6 |
| twin: region counts + spike count |  | 23.3 / 18.5 / 19.3 | 26.5 / 19.1 / 18.3 | 20.0 / 11.1 / 10.5 | 21.3 / 10.6 / 10.7 |
| twin: region counts + spike count | shuffled | 17.1 / 17.9 / 17.9 | 19.7 / 18.0 / 17.7 | 14.1 / 9.8 / 9.6 | 14.7 / 8.5 / 9.9 |
| delay regions + ISI bars + burst bars |  | 23.7 / 18.0 / 17.9 | 27.2 / 19.8 / 20.0 | 18.0 / 8.9 / 10.7 | 19.5 / 10.9 / 11.9 |
| delay regions + ISI bars + burst bars | shuffled | 16.6 / 17.6 / 18.2 | 20.9 / 18.6 / 18.5 | 15.1 / 10.2 / 10.7 | 15.1 / 10.7 / 9.9 |
| twin: + burst count |  | 20.9 / 18.5 / 18.2 | 24.8 / 18.9 / 18.7 | 17.9 / 9.9 / 11.1 | 20.7 / 9.7 / 11.0 |
| twin: + burst count | shuffled | 16.8 / 18.5 / 18.0 | 20.9 / 17.9 / 18.3 | 15.3 / 10.3 / 10.7 | 16.4 / 9.3 / 10.5 |
| ref: ISI fractions |  | 26.1 / 21.9 / 22.4 | 29.4 / 22.4 / 22.3 | 21.0 / 14.5 / 15.8 | 24.0 / 12.3 / 12.4 |
| ref: ISI fractions | shuffled | 26.1 / 21.9 / 22.4 | 29.4 / 22.4 / 22.3 | 21.0 / 14.5 / 15.8 | 24.0 / 12.3 / 12.4 |
| ref: burst table |  | 31.4 / 22.4 / 23.1 | 34.8 / 23.6 / 23.2 | 25.4 / 14.8 / 16.6 | 28.1 / 13.1 / 15.1 |
| ref: burst table | shuffled | 22.6 / 21.8 / 22.2 | 27.7 / 21.4 / 21.3 | 20.1 / 15.0 / 15.3 | 23.1 / 14.7 / 15.0 |
| ref: multiscale sqrt time bars |  | 11.2 / 12.8 / 10.5 | 9.3 / 14.1 / 13.2 | 4.4 / 4.3 / 3.1 | 4.6 / 5.0 / 3.6 |
| ref: multiscale sqrt time bars | shuffled | 4.6 / 11.4 / 9.6 | 6.5 / 11.5 / 10.5 | 4.0 / 4.6 / 4.6 | 4.8 / 4.0 / 5.1 |

### OpenScope 124 s, gratings (within only)

60 neurons, 2310 + 0 samples. Burst-bar rows: read the within-gratings 124 s column only.

| row | control | within gratings 124 s |
|---|---|---|
| delay regions (k=32, H0) |  | 69.1 / 37.5 / 36.3 |
| delay regions (k=32, H0) | shuffled | 57.7 / 35.2 / 33.8 |
| twin: region counts |  | 66.6 / 31.9 / 32.2 |
| twin: region counts | shuffled | 52.6 / 30.6 / 29.7 |
| ISI bars |  | 42.4 / 20.3 / 19.0 |
| ISI bars | shuffled | 42.4 / 20.3 / 19.0 |
| twin: spike count |  | 5.0 / 13.9 / 13.5 |
| twin: spike count | shuffled | 5.0 / 13.9 / 13.5 |
| delay regions + ISI bars |  | 63.4 / 27.9 / 28.7 |
| delay regions + ISI bars | shuffled | 53.5 / 26.3 / 28.3 |
| twin: region counts + spike count |  | 59.0 / 25.8 / 26.3 |
| twin: region counts + spike count | shuffled | 48.0 / 24.1 / 24.9 |
| delay regions + ISI bars + burst bars |  | 66.3 / 27.6 / 29.2 |
| delay regions + ISI bars + burst bars | shuffled | 53.9 / 27.5 / 27.0 |
| twin: + burst count |  | 52.6 / 26.6 / 26.0 |
| twin: + burst count | shuffled | 45.8 / 26.0 / 26.1 |
| ref: ISI fractions |  | 64.2 / 31.5 / 32.4 |
| ref: ISI fractions | shuffled | 64.2 / 31.5 / 32.4 |
| ref: burst table |  | 71.8 / 35.2 / 35.8 |
| ref: burst table | shuffled | 60.8 / 33.4 / 30.5 |
| ref: multiscale sqrt time bars |  | 13.7 / 13.0 / 10.6 |
| ref: multiscale sqrt time bars | shuffled | 5.0 / 10.6 / 7.9 |

### Allen 30 s, movie <-> spontaneous (L17 pair)

60 neurons, 1783 + 1734 samples. Burst-bar rows: read the within-movie column only.

| row | control | within movie | within spontaneous | movie -> spontaneous | spontaneous -> movie |
|---|---|---|---|---|---|
| delay regions (k=32, H0) |  | 39.7 / 27.1 / 27.3 | 30.9 / 23.7 / 23.0 | 26.0 / 14.1 / 13.4 | 22.8 / 14.7 / 15.3 |
| delay regions (k=32, H0) | shuffled | 30.4 / 25.1 / 24.7 | 22.5 / 21.3 / 21.2 | 20.3 / 11.6 / 11.4 | 17.0 / 12.5 / 13.2 |
| twin: region counts |  | 39.3 / 27.0 / 27.0 | 30.9 / 22.8 / 25.7 | 25.9 / 11.1 / 12.7 | 22.4 / 13.4 / 15.1 |
| twin: region counts | shuffled | 28.2 / 23.9 / 23.3 | 23.2 / 23.0 / 22.7 | 17.3 / 10.2 / 11.5 | 15.3 / 13.4 / 11.4 |
| ISI bars |  | 28.2 / 17.3 / 17.6 | 22.5 / 17.6 / 17.9 | 15.4 / 7.6 / 7.9 | 15.6 / 8.8 / 8.3 |
| ISI bars | shuffled | 28.2 / 17.3 / 17.6 | 22.5 / 17.6 / 17.9 | 15.4 / 7.6 / 7.9 | 15.6 / 8.8 / 8.3 |
| twin: spike count |  | 8.0 / 13.6 / 13.5 | 4.1 / 12.7 / 12.3 | 4.2 / 3.6 / 3.0 | 3.2 / 4.8 / 4.8 |
| twin: spike count | shuffled | 8.0 / 13.6 / 13.5 | 4.1 / 12.7 / 12.3 | 4.2 / 3.6 / 3.0 | 3.2 / 4.8 / 4.8 |
| delay regions + ISI bars |  | 38.1 / 25.0 / 23.7 | 29.9 / 21.2 / 21.3 | 23.4 / 9.2 / 9.9 | 22.3 / 11.0 / 10.7 |
| delay regions + ISI bars | shuffled | 30.7 / 23.1 / 23.1 | 24.3 / 21.2 / 20.6 | 18.1 / 7.7 / 8.7 | 16.6 / 10.7 / 10.4 |
| twin: region counts + spike count |  | 35.8 / 24.1 / 24.2 | 29.6 / 20.0 / 20.3 | 22.1 / 9.5 / 9.6 | 20.7 / 10.9 / 10.6 |
| twin: region counts + spike count | shuffled | 27.4 / 22.8 / 22.5 | 21.2 / 19.8 / 20.0 | 15.5 / 7.8 / 8.1 | 14.2 / 10.0 / 10.1 |
| delay regions + ISI bars + burst bars |  | 44.4 / 25.9 / 23.6 | 27.6 / 21.3 / 19.8 | 22.9 / 11.0 / 9.5 | 21.9 / 10.9 / 10.3 |
| delay regions + ISI bars + burst bars | shuffled | 29.3 / 23.3 / 21.3 | 23.9 / 20.8 / 20.1 | 18.2 / 6.9 / 6.6 | 16.4 / 11.0 / 9.5 |
| twin: + burst count |  | 35.0 / 24.2 / 23.4 | 26.7 / 20.1 / 20.0 | 21.6 / 9.3 / 8.0 | 18.7 / 10.9 / 10.3 |
| twin: + burst count | shuffled | 28.2 / 23.2 / 22.3 | 20.8 / 19.6 / 20.1 | 14.7 / 7.4 / 7.0 | 13.7 / 11.4 / 10.3 |
| ref: ISI fractions |  | 44.8 / 27.2 / 28.0 | 35.8 / 26.0 / 25.8 | 25.4 / 12.5 / 11.8 | 24.5 / 14.8 / 13.8 |
| ref: ISI fractions | shuffled | 44.8 / 27.2 / 28.0 | 35.8 / 26.0 / 25.8 | 25.4 / 12.5 / 11.8 | 24.5 / 14.8 / 13.8 |
| ref: burst table |  | 47.7 / 28.2 / 29.4 | 37.9 / 28.8 / 28.7 | 30.6 / 13.5 / 14.0 | 27.9 / 14.3 / 15.0 |
| ref: burst table | shuffled | 39.6 / 24.6 / 25.1 | 31.5 / 24.7 / 25.1 | 22.8 / 12.9 / 11.4 | 22.4 / 12.2 / 14.5 |
| ref: multiscale sqrt time bars |  | 13.9 / 12.1 / 9.7 | 2.5 / 8.1 / 5.1 | 3.9 / 2.7 / 2.7 | 2.2 / 4.5 / 3.1 |
| ref: multiscale sqrt time bars | shuffled | 3.7 / 9.1 / 6.7 | 3.1 / 8.0 / 5.0 | 3.8 / 3.6 / 2.7 | 2.5 / 4.3 / 2.6 |

### Allen 30 s, movie block 1 <-> spontaneous before it (adjacent in time, L24)

55 neurons, 1551 + 781 samples. Burst-bar rows: read the within-movie 1 column only.

| row | control | within movie 1 | within spont early | movie 1 -> spont early | spont early -> movie 1 |
|---|---|---|---|---|---|
| delay regions (k=32, H0) |  | 45.5 / 31.5 / 33.7 | 42.8 / 32.8 / 32.9 | 37.4 / 19.6 / 21.9 | 29.3 / 24.2 / 23.8 |
| delay regions (k=32, H0) | shuffled | 38.0 / 29.9 / 32.0 | 35.2 / 30.2 / 30.3 | 27.4 / 15.7 / 17.7 | 25.2 / 19.4 / 21.9 |
| twin: region counts |  | 46.3 / 32.2 / 34.2 | 44.7 / 34.1 / 34.7 | 36.6 / 18.4 / 22.3 | 30.1 / 21.0 / 17.7 |
| twin: region counts | shuffled | 35.3 / 28.1 / 30.7 | 33.5 / 28.0 / 31.6 | 26.4 / 14.3 / 18.7 | 23.0 / 16.0 / 18.2 |
| ISI bars |  | 38.2 / 21.0 / 20.7 | 31.4 / 23.3 / 23.7 | 20.8 / 7.3 / 8.1 | 21.3 / 9.2 / 8.3 |
| ISI bars | shuffled | 38.2 / 21.0 / 20.7 | 31.4 / 23.3 / 23.7 | 20.8 / 7.3 / 8.1 | 21.3 / 9.2 / 8.3 |
| twin: spike count |  | 9.1 / 15.0 / 15.0 | 7.8 / 18.7 / 17.8 | 7.0 / 5.5 / 5.2 | 5.4 / 5.9 / 4.8 |
| twin: spike count | shuffled | 9.1 / 15.0 / 15.0 | 7.8 / 18.7 / 17.8 | 7.0 / 5.5 / 5.2 | 5.4 / 5.9 / 4.8 |
| delay regions + ISI bars |  | 46.2 / 28.1 / 29.0 | 41.7 / 27.5 / 29.7 | 31.4 / 13.7 / 17.2 | 28.7 / 12.8 / 12.3 |
| delay regions + ISI bars | shuffled | 40.7 / 25.1 / 26.9 | 32.8 / 30.2 / 29.1 | 25.8 / 13.4 / 13.2 | 24.6 / 15.3 / 12.8 |
| twin: region counts + spike count |  | 46.6 / 26.8 / 27.7 | 39.3 / 27.9 / 30.0 | 28.9 / 15.6 / 15.0 | 28.0 / 16.7 / 14.8 |
| twin: region counts + spike count | shuffled | 34.9 / 23.5 / 23.9 | 31.1 / 27.9 / 28.2 | 22.1 / 13.6 / 11.8 | 20.5 / 11.2 / 14.6 |
| delay regions + ISI bars + burst bars |  | 46.9 / 28.9 / 26.4 | 35.2 / 28.8 / 29.2 | 30.9 / 16.9 / 12.5 | 27.1 / 13.0 / 14.7 |
| delay regions + ISI bars + burst bars | shuffled | 38.9 / 25.8 / 25.0 | 31.0 / 29.2 / 26.0 | 24.6 / 15.4 / 14.3 | 22.7 / 10.9 / 11.7 |
| twin: + burst count |  | 43.8 / 26.4 / 26.2 | 36.9 / 28.6 / 28.4 | 27.5 / 13.6 / 10.6 | 25.5 / 12.0 / 14.4 |
| twin: + burst count | shuffled | 34.7 / 27.0 / 25.9 | 31.8 / 28.2 / 28.9 | 22.1 / 12.8 / 12.7 | 17.9 / 12.2 / 12.8 |
| ref: ISI fractions |  | 50.7 / 39.8 / 38.7 | 41.1 / 35.3 / 34.4 | 38.7 / 22.2 / 22.3 | 33.5 / 19.9 / 18.3 |
| ref: ISI fractions | shuffled | 50.7 / 39.8 / 38.7 | 41.1 / 35.3 / 34.4 | 38.7 / 22.2 / 22.3 | 33.5 / 19.9 / 18.3 |
| ref: burst table |  | 56.7 / 40.9 / 38.4 | 47.2 / 38.7 / 38.5 | 42.4 / 23.2 / 25.1 | 38.3 / 21.5 / 24.4 |
| ref: burst table | shuffled | 47.8 / 33.5 / 34.8 | 39.8 / 34.7 / 32.7 | 34.8 / 17.3 / 17.0 | 31.6 / 19.6 / 18.1 |
| ref: multiscale sqrt time bars |  | 6.6 / 9.1 / 7.3 | 1.7 / 11.9 / 8.3 | 3.3 / 4.5 / 3.6 | 3.2 / 4.7 / 3.5 |
| ref: multiscale sqrt time bars | shuffled | 3.0 / 8.6 / 5.9 | 3.1 / 12.8 / 9.5 | 3.1 / 3.2 / 4.1 | 2.0 / 4.4 / 4.0 |


## Seeds and sessions, full

### Seeds 0-4 (1-NN within A, mean [min-max]; k-medoids in brackets after)


Retina 10 s (5 seeds)

| row | control | nn_A | kmedoids_A | nn_Ato | nn_Bto |
|---|---|---|---|---|---|
| delay regions (k=32, H0) |  | 43.5 [39.7-47.6] | 35.3 [29.6-38.1] | 42.0 [36.2-46.9] | 40.6 [35.6-44.8] |
| delay regions (k=32, H0) | shuffled | 37.4 [30.6-43.0] | 33.2 [30.1-35.7] | 36.2 [30.7-42.0] | 36.1 [30.2-40.5] |
| twin: region counts |  | 43.2 [37.6-48.2] | 35.9 [31.9-39.2] | 42.0 [36.8-46.3] | 40.9 [35.9-44.6] |
| twin: region counts | shuffled | 34.9 [28.5-39.7] | 32.6 [28.0-36.6] | 35.2 [31.2-39.5] | 34.4 [30.6-38.7] |
| ISI bars |  | 44.7 [40.3-48.6] | 28.6 [27.2-29.5] | 42.3 [37.1-46.2] | 40.5 [37.4-43.5] |
| ISI bars | shuffled | 44.7 [40.3-48.6] | 28.6 [27.2-29.5] | 42.3 [37.1-46.2] | 40.5 [37.4-43.5] |
| twin: spike count |  | 11.0 [10.6-11.2] | 20.0 [19.5-20.7] | 10.7 [9.5-12.0] | 10.9 [10.1-11.9] |
| twin: spike count | shuffled | 11.0 [10.6-11.2] | 20.0 [19.5-20.7] | 10.7 [9.5-12.0] | 10.9 [10.1-11.9] |
| delay regions + ISI bars |  | 46.5 [41.7-50.3] | 34.2 [32.4-35.5] | 44.1 [38.4-47.6] | 42.3 [36.1-45.4] |
| delay regions + ISI bars | shuffled | 41.2 [34.5-46.5] | 33.7 [31.1-36.2] | 40.1 [35.7-43.9] | 39.1 [33.6-41.7] |
| twin: region counts + spike count |  | 42.9 [38.8-47.3] | 33.0 [29.9-36.3] | 41.1 [36.4-45.0] | 39.9 [34.2-43.9] |
| twin: region counts + spike count | shuffled | 36.1 [30.6-39.7] | 30.7 [29.0-33.6] | 35.2 [30.4-38.4] | 34.3 [30.7-37.9] |
| delay regions + ISI bars + burst bars |  | 83.2 [81.2-85.7] | 46.6 [39.4-51.4] | 41.2 [36.7-45.6] | 38.2 [32.4-42.6] |
| delay regions + ISI bars + burst bars | shuffled | 40.5 [33.7-45.3] | 32.4 [27.9-35.0] | 37.1 [31.7-40.5] | 35.3 [29.7-38.9] |
| twin: + burst count |  | 37.7 [34.8-39.9] | 28.7 [26.8-29.7] | 36.2 [31.1-39.5] | 34.9 [30.1-40.0] |
| twin: + burst count | shuffled | 33.8 [30.0-38.6] | 27.8 [25.6-29.6] | 32.4 [28.5-35.2] | 31.0 [26.5-34.4] |
| ref: ISI fractions |  | 52.5 [47.5-55.5] | 39.8 [35.3-43.1] | 51.6 [47.7-54.5] | 48.4 [43.8-51.3] |
| ref: ISI fractions | shuffled | 52.5 [47.5-55.5] | 39.8 [35.3-43.1] | 51.6 [47.7-54.5] | 48.4 [43.8-51.3] |
| ref: burst table |  | 54.5 [49.5-57.9] | 40.1 [34.7-45.5] | 50.8 [45.6-55.5] | 48.4 [42.4-52.2] |
| ref: burst table | shuffled | 48.4 [44.3-52.1] | 38.7 [34.5-42.3] | 47.1 [43.4-50.6] | 45.2 [40.9-49.0] |
| ref: multiscale sqrt time bars |  | 88.7 [87.7-89.7] | 52.2 [45.9-62.2] | 6.2 [5.0-7.2] | 6.5 [5.5-7.2] |
| ref: multiscale sqrt time bars | shuffled | 6.5 [5.5-7.2] | 13.9 [13.1-14.4] | 6.1 [5.4-7.6] | 6.1 [5.1-6.8] |

OpenScope 124 s (5 seeds)

| row | control | nn_A | kmedoids_A |
|---|---|---|---|
| delay regions (k=32, H0) |  | 67.1 [63.5-72.2] | 35.8 [33.8-38.1] |
| delay regions (k=32, H0) | shuffled | 55.8 [54.1-57.7] | 32.6 [31.0-35.2] |
| twin: region counts |  | 64.6 [60.9-68.7] | 31.8 [30.9-32.5] |
| twin: region counts | shuffled | 50.9 [49.9-52.6] | 30.1 [29.3-30.6] |
| ISI bars |  | 42.7 [40.2-44.8] | 19.0 [17.6-20.3] |
| ISI bars | shuffled | 42.7 [40.2-44.8] | 19.0 [17.6-20.3] |
| twin: spike count |  | 6.7 [5.0-7.8] | 13.6 [13.1-13.9] |
| twin: spike count | shuffled | 6.7 [5.0-7.8] | 13.6 [13.1-13.9] |
| delay regions + ISI bars |  | 62.4 [58.8-66.6] | 28.1 [27.2-29.2] |
| delay regions + ISI bars | shuffled | 52.9 [51.0-55.1] | 27.0 [26.3-27.4] |
| twin: region counts + spike count |  | 58.5 [55.8-61.8] | 25.7 [24.4-26.6] |
| twin: region counts + spike count | shuffled | 47.0 [45.2-48.0] | 24.8 [24.1-25.5] |
| delay regions + ISI bars + burst bars |  | 65.4 [62.6-68.8] | 28.2 [27.6-29.6] |
| delay regions + ISI bars + burst bars | shuffled | 51.9 [49.8-53.9] | 26.7 [25.5-27.5] |
| twin: + burst count |  | 53.7 [52.0-56.6] | 25.2 [23.7-26.6] |
| twin: + burst count | shuffled | 45.3 [43.8-45.9] | 25.3 [24.1-26.0] |
| ref: ISI fractions |  | 62.6 [59.2-64.2] | 32.3 [31.5-32.8] |
| ref: ISI fractions | shuffled | 62.6 [59.2-64.2] | 32.3 [31.5-32.8] |
| ref: burst table |  | 69.6 [67.2-72.5] | 34.4 [32.8-35.2] |
| ref: burst table | shuffled | 59.2 [57.1-61.6] | 31.6 [30.4-33.4] |
| ref: multiscale sqrt time bars |  | 10.1 [7.5-13.7] | 12.8 [11.5-13.8] |
| ref: multiscale sqrt time bars | shuffled | 3.7 [2.8-5.0] | 9.9 [9.0-10.6] |

### Nine retina sessions, within frozen trials (1-NN / k-medoids)

| row | control | 20171215_sr_le_sp | 20180116_sr_le_fp | 20180126_sr_le_sp | 20180222_fr_re | 20180222_sr_le_sp | 20180316_sr_le_fp | 20180316_sr_le_sp | 20180405_fr_re | 20180405_sr_le | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| delay regions (k=32, H0) |  | 24.3/22.0 | 31.0/26.9 | 40.7/33.7 | 28.8/26.4 | 26.7/21.5 | 21.2/24.0 | 18.4/20.2 | 28.7/24.7 | 30.7/28.6 | 27.8 |
| delay regions (k=32, H0) | shuffled | 22.2/21.4 | 27.4/27.2 | 36.7/30.9 | 24.3/26.0 | 21.5/22.7 | 19.7/23.1 | 15.5/18.8 | 24.7/24.8 | 26.8/25.2 | 24.3 |
| twin: region counts |  | 27.8/25.4 | 30.4/28.4 | 38.9/31.8 | 29.9/25.2 | 29.4/25.4 | 19.2/24.5 | 21.7/20.4 | 26.3/22.9 | 29.4/27.0 | 28.1 |
| twin: region counts | shuffled | 25.0/23.5 | 24.9/24.3 | 31.5/28.3 | 23.0/25.6 | 24.2/22.1 | 19.2/25.0 | 15.4/20.0 | 20.9/23.3 | 24.2/24.5 | 23.1 |
| ISI bars |  | 28.4/19.2 | 31.0/21.3 | 41.4/28.8 | 27.4/20.7 | 27.0/19.1 | 17.3/23.1 | 20.2/17.7 | 23.1/18.4 | 29.3/22.2 | 27.2 |
| ISI bars | shuffled | 28.4/19.2 | 31.0/21.3 | 41.4/28.8 | 27.4/20.7 | 27.0/19.1 | 17.3/23.1 | 20.2/17.7 | 23.1/18.4 | 29.3/22.2 | 27.2 |
| twin: spike count |  | 5.3/13.0 | 5.8/15.8 | 11.0/19.7 | 4.6/13.8 | 5.3/13.6 | 10.6/21.2 | 9.0/15.3 | 6.9/15.7 | 11.4/17.7 | 7.8 |
| twin: spike count | shuffled | 5.3/13.0 | 5.8/15.8 | 11.0/19.7 | 4.6/13.8 | 5.3/13.6 | 10.6/21.2 | 9.0/15.3 | 6.9/15.7 | 11.4/17.7 | 7.8 |
| delay regions + ISI bars |  | 28.5/23.2 | 32.7/25.1 | 42.9/32.3 | 29.2/24.8 | 30.0/24.6 | 21.6/23.6 | 22.6/20.4 | 25.8/23.1 | 30.5/25.3 | 29.3 |
| delay regions + ISI bars | shuffled | 24.7/21.5 | 28.8/24.9 | 41.3/33.2 | 24.6/22.7 | 25.8/23.0 | 17.3/22.6 | 19.1/19.9 | 25.4/23.3 | 28.3/24.5 | 26.2 |
| twin: region counts + spike count |  | 28.0/22.9 | 29.1/23.5 | 37.3/31.5 | 28.2/23.6 | 28.8/23.6 | 20.7/25.5 | 20.9/19.3 | 22.6/21.3 | 27.4/24.0 | 27.0 |
| twin: region counts + spike count | shuffled | 22.6/20.6 | 23.7/23.7 | 32.2/30.4 | 22.8/21.6 | 24.1/21.9 | 14.9/23.6 | 16.8/19.2 | 20.1/20.4 | 24.9/24.5 | 22.5 |
| delay regions + ISI bars + burst bars |  | 88.3/45.3 | 82.5/42.3 | 77.0/41.9 | 54.2/27.4 | 70.4/34.4 | 60.1/27.9 | 56.6/31.0 | 45.6/25.4 | 55.5/31.3 | 65.6 |
| delay regions + ISI bars + burst bars | shuffled | 26.4/21.4 | 28.4/24.5 | 39.5/33.8 | 24.6/23.5 | 27.1/21.8 | 21.2/23.6 | 18.6/20.2 | 23.3/22.5 | 28.8/24.0 | 26.4 |
| twin: + burst count |  | 25.6/23.2 | 29.7/25.9 | 38.6/31.5 | 27.6/23.2 | 27.7/23.6 | 19.7/25.0 | 18.7/19.8 | 24.5/20.4 | 28.1/25.9 | 26.7 |
| twin: + burst count | shuffled | 23.8/20.3 | 27.0/25.6 | 32.4/29.0 | 24.1/21.6 | 25.2/23.9 | 19.2/24.5 | 17.1/18.5 | 21.4/21.0 | 25.9/25.6 | 24.0 |
| ref: ISI fractions |  | 36.4/29.7 | 38.8/31.7 | 50.9/37.6 | 36.6/26.6 | 35.4/26.2 | 26.0/25.5 | 23.3/22.6 | 33.2/26.3 | 34.0/28.8 | 35.0 |
| ref: ISI fractions | shuffled | 36.4/29.7 | 38.8/31.7 | 50.9/37.6 | 36.6/26.6 | 35.4/26.2 | 26.0/25.5 | 23.3/22.6 | 33.2/26.3 | 34.0/28.8 | 35.0 |
| ref: burst table |  | 40.4/29.1 | 40.7/31.7 | 52.6/34.7 | 39.1/27.2 | 38.7/27.6 | 29.8/24.5 | 27.5/21.9 | 34.5/26.6 | 35.4/30.7 | 37.6 |
| ref: burst table | shuffled | 32.1/28.2 | 36.1/30.7 | 44.3/37.1 | 34.4/25.8 | 31.9/25.9 | 26.9/26.0 | 22.5/22.4 | 31.9/25.8 | 32.5/25.8 | 32.5 |
| ref: multiscale sqrt time bars |  | 90.9/61.1 | 88.6/58.1 | 87.8/48.3 | 52.6/28.7 | 75.5/36.9 | 71.2/42.8 | 54.9/30.1 | 57.0/32.6 | 53.4/33.8 | 70.2 |
| ref: multiscale sqrt time bars | shuffled | 5.6/10.7 | 5.8/12.2 | 6.9/13.7 | 4.5/11.9 | 6.2/11.3 | 8.7/19.2 | 5.9/11.4 | 6.3/12.6 | 4.6/13.3 | 6.1 |

20171215_sr_le_sp: 60 neurons, 1170 trials; 20180116_sr_le_fp: 59 neurons, 1547 trials; 20180126_sr_le_sp: 60 neurons, 1701 trials; 20180222_fr_re: 60 neurons, 1567 trials; 20180222_sr_le_sp: 51 neurons, 1332 trials; 20180316_sr_le_fp: 18 neurons, 208 trials; 20180316_sr_le_sp: 56 neurons, 1409 trials; 20180405_fr_re: 55 neurons, 1431 trials; 20180405_sr_le: 60 neurons, 1582 trials.
