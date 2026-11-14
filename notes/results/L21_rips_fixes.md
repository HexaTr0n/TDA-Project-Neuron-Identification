# L21 Three fixes for Rips: H1 on whole clouds, Betti curves for H1, a time-filtered complex

2026-11-13, local session. **Mostly failed; one small, consistent gain.**

- **H1 on the plain whole cloud, read as a Betti curve, adds to H0 on 4 of 5 datasets.**
  - Plain delay Rips, no regions; 1-NN within A (in brackets: ISIs shuffled).
  - H0 against H0 + H1 Betti curve: retina 10 s 31.0 -> 36.4 (31.5); retina 40 s 33.8 -> 47.1
    (44.0); OpenScope 124 s 34.0 -> 41.2 (36.1); OpenScope 10 s 12.8 -> 14.9 (14.3).
  - Allen is the exception: 20.7 -> 21.7, below its shuffled control (24.8).
  - The plain cloud is far below delay regions, so this does not change the headline.
- **H1 inside regions lowers regional H0 almost everywhere.**
  - k = 16: retina 10 s 43.3 -> 30.5-33.6; Allen 38.4 -> 27.7-29.1; OpenScope 124 s 63.5 ->
    58.5-60.4.
  - The one tie is OpenScope 124 s at k = 8 with the Betti curve (59.4 against 59.2).
- **H1 alone is above its shuffled control only on the long cortical samples.** OpenScope 124 s:
  regions k = 16 H1 stable rank 42.7 (36.8); plain cloud H1 Betti curve 34.1 (29.4). Elsewhere
  it is within 2 points of shuffled or below. By L23's own rule, L23 continues on OpenScope 124 s
  only.
- **Betti curve or stable rank for H1:** the Betti curve is better on the plain cloud (+1 to +12
  points) and worse inside regions (k = 16: -3 to -11). Inside a region the scale at which loops
  live is mostly the region's size.
- **The time-filtered complex fails.**
  - Best readout (Betti + births, r = 0.5): 8-22% 1-NN within A. That is 0-6 points above
    shuffled within a repeated stimulus, and below shuffled on retina 40 s (19.3 against 22.6).
  - At r = 1 and 2 it falls to 4-15%.
  - It is far below ISI fractions on the same samples (26-64%).

## Data

Same neurons and samples as L20 (L17 selection; L19 for OpenScope 124 s; seed 0), 60 neurons
each, chance 1.7%:

| dataset | samples | median spikes | time-filtered bars at r = 0.5 / 1 / 2 (median) |
|---|---|---|---|
| retina 10 s (piece 2 dropped), frozen <-> unrepeated | 1701 + 1788 | 166 | 102 / 30 / 6 |
| OpenScope 10 s, gratings <-> grey | 1769 + 1490 | 67 | 46 / 22 / 6 |
| Allen 30 s, movie <-> spontaneous | 1783 + 1734 | 101 | 76 / 34 / 10 |
| retina 40 s **clean** (`retina40clean`, L24), frozen x4 <-> unrepeated x4 | 420 + 1795 | 622 | 195 / 39 / 6 |
| OpenScope 124 s, gratings, within only | 2310 | 496 | 193 / 55 / 10 |

**Method** (`scripts/sb_rips_fixes.py`):
- **Whole clouds:** delay embedding m = 3 of log ISIs, every point, one barcode. Regions are
  k-means anchors (8, 16) on outside neurons. Rips H0 and H1, stable ranks and Betti curves are
  all in stablebear, with L1 distances.
- **Readouts:** H0 stable rank; H1 stable rank; H1 Betti curve; H0 + H1 stable rank; H0 stable
  rank + H1 Betti curve. Groups are median-scaled and added; regions are summed raw.
- **Time-filtered complex:**
  - Delay points (m = 3, raw log ISIs, as the web pilot exp32) enter at the time of their last
    spike. An edge joins two points closer than r from the moment both are in, with r = 0.5, 1, 2.
  - H0 bars are computed by union-find in numpy, because stablebear builds Rips only by distance,
    not by entry time. They are handed to stablebear as `Barcode`s.
  - Infinite bars end at the end of the sample (or the last spike, if a sample runs a few ms past
    its nominal length; this fixed the first Allen run).
  - Summaries in stablebear: the Betti curve over time; "births so far" (the Betti curve of bars
    [birth, end)); lifetimes (stable rank); Betti + births.
- **Control:** ISIs permuted within every sample.
- **Scores as L17:** 1-NN / k-medoids / k-means with `sb.mean`, within each stimulus and across.

## Commands

```sh
python scripts/sb_rips_fixes.py --dataset retina10 --drop-piece 2
python scripts/sb_rips_fixes.py --dataset openscope10
python scripts/sb_rips_fixes.py --dataset allen30
python scripts/sb_rips_fixes.py --dataset retina40clean
python scripts/sb_rips_fixes.py --dataset openscope124
```

OpenScope 124 s takes about 2 h and up to ~14 GB with swap (whole clouds up to 8667 points, with
H1); it was run alone.

## Summary: 1-NN within A / A -> B (in brackets: within A, ISIs shuffled)

On the same samples, ISI fractions give 50.9 / 26.1 / 44.8 / 64.3 / 64.2 within A, and the
whole-cloud delay regions with k = 32 (L20) give 43.7 / 21.5 / 39.0 / 67.4 / 67.2 (L20's retina
40 s is the old data).

| family | readout | retina 10 s | OpenScope 10 s | Allen 30 s | retina 40 s (clean) | OpenScope 124 s |
|---|---|---|---|---|---|---|
| whole cloud, plain | H0 | 31.0/28.0 (28.8) | 12.8/11.3 (13.5) | 20.7/12.1 (19.0) | 33.8/49.8 (36.0) | 34.0 (29.2) |
| whole cloud, plain | H1 stable rank | 16.8/17.4 (19.2) | 8.8/6.3 (8.4) | 11.7/7.8 (13.4) | 29.3/35.2 (35.0) | 22.5 (21.5) |
| whole cloud, plain | H1 Betti curve | 22.5/22.4 (22.8) | 9.8/7.9 (9.2) | 14.5/10.2 (16.8) | 38.1/36.4 (37.1) | 34.1 (29.4) |
| whole cloud, plain | H0 + H1 stable rank | 31.9/28.7 (29.1) | 13.0/10.6 (12.7) | 21.2/12.6 (20.1) | 39.0/52.4 (44.5) | 36.4 (33.7) |
| whole cloud, plain | H0 + H1 Betti curve | 36.4/32.3 (31.5) | 14.9/11.1 (14.3) | 21.7/14.7 (24.8) | 47.1/53.3 (44.0) | 41.2 (36.1) |
| whole cloud, regions k=8 | H0 | 42.4/41.3 (37.6) | 19.0/16.5 (16.4) | 34.4/22.7 (26.5) | 56.7/63.3 (56.0) | 59.2 (49.6) |
| whole cloud, regions k=8 | H1 stable rank | 21.6/21.0 (20.7) | 8.3/6.8 (8.4) | 15.1/11.9 (15.0) | 37.4/41.2 (37.4) | 40.3 (35.5) |
| whole cloud, regions k=8 | H1 Betti curve | 16.4/15.6 (16.8) | 5.9/4.6 (6.4) | 15.5/11.0 (13.2) | 40.5/42.4 (38.8) | 41.8 (37.4) |
| whole cloud, regions k=8 | H0 + H1 stable rank | 36.9/35.4 (34.2) | 13.4/13.2 (13.5) | 27.3/18.3 (23.3) | 49.0/57.9 (50.0) | 55.3 (45.5) |
| whole cloud, regions k=8 | H0 + H1 Betti curve | 37.2/33.8 (31.9) | 12.2/12.9 (12.7) | 29.7/19.3 (23.8) | 54.5/63.6 (51.2) | 59.4 (49.0) |
| whole cloud, regions k=16 | H0 | 43.3/40.7 (38.9) | 23.0/20.6 (17.2) | 38.4/25.4 (29.8) | 63.8/69.3 (60.5) | 63.5 (53.5) |
| whole cloud, regions k=16 | H1 stable rank | 19.9/19.6 (18.0) | 9.6/8.5 (8.4) | 17.4/13.5 (15.5) | 42.9/50.0 (41.7) | 42.7 (36.8) |
| whole cloud, regions k=16 | H1 Betti curve | 8.5/9.2 (10.6) | 6.8/5.5 (5.5) | 12.8/10.2 (11.9) | 36.2/40.2 (33.8) | 38.5 (36.7) |
| whole cloud, regions k=16 | H0 + H1 stable rank | 33.6/32.3 (32.2) | 17.5/15.5 (14.6) | 29.1/22.0 (24.1) | 58.1/63.1 (55.7) | 58.5 (49.1) |
| whole cloud, regions k=16 | H0 + H1 Betti curve | 30.5/30.4 (28.7) | 16.6/13.3 (13.7) | 27.7/21.3 (25.0) | 58.1/64.5 (56.2) | 60.4 (51.8) |
| time-filtered r=0.5 | Betti curve over time | 14.6/11.9 (10.6) | 6.7/6.1 (6.7) | 12.6/7.4 (9.4) | 14.0/19.5 (16.7) | 13.8 (14.2) |
| time-filtered r=0.5 | births so far | 16.0/11.5 (9.9) | 6.3/6.2 (7.0) | 13.0/6.2 (9.8) | 15.2/20.7 (17.9) | 16.0 (15.6) |
| time-filtered r=0.5 | lifetimes (stable rank) | 11.8/9.5 (9.3) | 4.7/4.7 (5.4) | 12.2/5.3 (6.2) | 11.9/18.3 (13.6) | 14.3 (12.6) |
| time-filtered r=0.5 | Betti + births | 17.2/15.5 (13.6) | 8.6/7.6 (8.5) | 17.8/8.0 (11.9) | 19.3/29.3 (22.6) | 22.4 (19.5) |
| time-filtered r=1 | Betti curve over time | 8.4/6.3 (6.8) | 5.8/5.5 (5.9) | 9.1/4.5 (6.6) | 9.0/8.6 (9.5) | 10.6 (8.4) |
| time-filtered r=1 | births so far | 9.0/7.9 (7.7) | 6.6/4.6 (4.7) | 8.3/5.6 (5.7) | 7.9/8.8 (10.7) | 10.1 (10.4) |
| time-filtered r=1 | lifetimes (stable rank) | 8.9/6.5 (7.9) | 5.6/4.6 (4.1) | 7.1/4.5 (4.8) | 7.4/7.9 (7.9) | 7.5 (10.1) |
| time-filtered r=1 | Betti + births | 9.9/9.9 (10.0) | 7.7/6.9 (6.3) | 11.3/6.0 (7.9) | 11.0/14.8 (15.2) | 14.9 (13.9) |
| time-filtered r=2 | Betti curve over time | 5.2/5.9 (5.1) | 4.4/4.2 (4.2) | 5.3/4.0 (4.7) | 7.1/8.1 (5.5) | 7.0 (6.4) |
| time-filtered r=2 | births so far | 4.3/4.2 (4.2) | 4.2/3.6 (5.3) | 5.6/3.9 (5.7) | 4.8/6.4 (7.1) | 7.4 (6.1) |
| time-filtered r=2 | lifetimes (stable rank) | 4.5/5.2 (5.0) | 4.4/4.5 (4.9) | 6.2/2.9 (5.6) | 4.3/6.2 (6.0) | 6.5 (7.0) |
| time-filtered r=2 | Betti + births | 6.6/5.9 (5.3) | 4.6/4.7 (5.7) | 7.0/4.3 (6.4) | 6.2/9.0 (4.5) | 7.9 (7.4) |

## What to take from it

- **H1 carries something on large clouds, but not what regions already give.**
  - On the plain whole cloud, H1 as a Betti curve adds 1-13 points to H0, and 3-5 points above
    its shuffled control on the retina and OpenScope 124 s. Loops in the delay cloud reflect
    ISI patterns that come back.
  - Fixed regions already capture where the cloud's mass sits. H1 inside a region is mostly
    noise about the region's own shape, and its weight in a plain sum drags H0 down.
- **The Betti curve is the right H1 summary for a whole cloud**, because it keeps the scale.
  Inside regions the stable rank does slightly better.
- **The time-filtered complex is not a useful identifier.**
  - Its bars count how often a new ISI pattern appears over time, a rate-like quantity. It keeps
    roughly the same score across stimuli as within (retina 10 s frozen 17.2 -> unrepeated 15.5;
    retina 40 s 19.3 -> 29.3), so it is not stimulus-locked.
  - It stays only 0-6 points above shuffled and far below ISI fractions.
  - The scale r = 0.5 is best. At larger r everything merges within a few spikes and the score
    falls to chance.
- **What failed:**
  - H1 as an addition to regional H0 (L23 now tests whether weighting or bar filtering fixes it,
    on OpenScope 124 s only);
  - H1 on short samples;
  - the time-filtered complex on every dataset (rate-like, not time-locked, below ISI fractions).

## Full tables

Each cell is 1-NN / k-medoids / k-means, %.

### retina 10 s

60 neurons, 1701 + 1788 samples.

| family | readout | control | within frozen | within unrepeated | frozen -> unrepeated | unrepeated -> frozen |
|---|---|---|---|---|---|---|
| whole cloud, plain | H0 |  | 31.0 / 28.7 / 30.9 | 25.9 / 27.3 / 29.5 | 28.0 / 21.7 / 24.2 | 27.1 / 25.7 / 25.2 |
| whole cloud, plain | H1 stable rank |  | 16.8 / 21.9 / 23.6 | 16.6 / 21.6 / 21.8 | 17.4 / 15.0 / 15.7 | 17.0 / 17.3 / 17.6 |
| whole cloud, plain | H1 Betti curve |  | 22.5 / 24.9 / 24.0 | 21.5 / 24.8 / 23.8 | 22.4 / 16.1 / 16.9 | 21.1 / 19.6 / 20.8 |
| whole cloud, plain | H0 + H1 stable rank |  | 31.9 / 27.7 / 29.3 | 29.1 / 27.7 / 27.6 | 28.7 / 22.8 / 24.3 | 28.7 / 25.7 / 23.1 |
| whole cloud, plain | H0 + H1 Betti curve |  | 36.4 / 31.7 / 32.2 | 31.9 / 31.0 / 31.4 | 32.3 / 25.3 / 26.1 | 31.4 / 26.6 / 26.6 |
| whole cloud, regions k=8 | H0 |  | 42.4 / 37.0 / 40.4 | 38.9 / 31.9 / 38.7 | 41.3 / 30.0 / 35.1 | 39.5 / 27.7 / 36.9 |
| whole cloud, regions k=8 | H1 stable rank |  | 21.6 / 23.1 / 23.2 | 22.0 / 21.1 / 24.6 | 21.0 / 17.3 / 18.0 | 22.5 / 14.2 / 18.0 |
| whole cloud, regions k=8 | H1 Betti curve |  | 16.4 / 18.3 / 19.2 | 16.8 / 16.6 / 18.6 | 15.6 / 14.1 / 14.7 | 16.5 / 12.7 / 15.0 |
| whole cloud, regions k=8 | H0 + H1 stable rank |  | 36.9 / 30.3 / 33.5 | 32.9 / 28.7 / 30.5 | 35.4 / 25.7 / 27.1 | 34.8 / 24.6 / 27.0 |
| whole cloud, regions k=8 | H0 + H1 Betti curve |  | 37.2 / 33.1 / 37.4 | 32.2 / 27.3 / 34.6 | 33.8 / 27.5 / 30.8 | 32.9 / 24.2 / 29.7 |
| whole cloud, regions k=16 | H0 |  | 43.3 / 32.7 / 38.2 | 39.5 / 31.2 / 38.0 | 40.7 / 27.0 / 33.2 | 39.7 / 26.5 / 34.6 |
| whole cloud, regions k=16 | H1 stable rank |  | 19.9 / 21.6 / 20.5 | 18.3 / 19.2 / 18.4 | 19.6 / 14.9 / 15.1 | 19.2 / 14.5 / 15.2 |
| whole cloud, regions k=16 | H1 Betti curve |  | 8.5 / 11.9 / 12.1 | 10.1 / 11.7 / 9.9 | 9.2 / 8.4 / 8.9 | 8.6 / 7.8 / 8.6 |
| whole cloud, regions k=16 | H0 + H1 stable rank |  | 33.6 / 31.5 / 32.4 | 33.6 / 27.0 / 28.8 | 32.3 / 24.0 / 27.0 | 33.4 / 22.6 / 24.8 |
| whole cloud, regions k=16 | H0 + H1 Betti curve |  | 30.5 / 27.2 / 28.0 | 29.0 / 24.9 / 23.0 | 30.4 / 20.7 / 22.3 | 30.1 / 21.6 / 20.2 |
| time-filtered r=0.5 (102 bars) | Betti curve over time |  | 14.6 / 18.6 / 20.2 | 11.1 / 18.4 / 19.6 | 11.9 / 12.8 / 11.6 | 11.8 / 14.1 / 14.3 |
| time-filtered r=0.5 (102 bars) | births so far |  | 16.0 / 18.6 / 19.8 | 13.7 / 18.8 / 19.6 | 11.5 / 10.9 / 11.7 | 11.0 / 13.1 / 14.2 |
| time-filtered r=0.5 (102 bars) | lifetimes (stable rank) |  | 11.8 / 17.5 / 17.2 | 10.5 / 15.4 / 16.2 | 9.5 / 10.3 / 9.3 | 9.5 / 10.2 / 10.1 |
| time-filtered r=0.5 (102 bars) | Betti + births |  | 17.2 / 21.7 / 22.5 | 14.9 / 20.1 / 21.6 | 15.5 / 13.6 / 13.6 | 14.4 / 15.7 / 17.6 |
| time-filtered r=1 (30 bars) | Betti curve over time |  | 8.4 / 15.4 / 14.9 | 7.7 / 14.4 / 14.7 | 6.3 / 8.7 / 8.1 | 7.4 / 7.8 / 8.3 |
| time-filtered r=1 (30 bars) | births so far |  | 9.0 / 14.4 / 14.3 | 7.4 / 14.2 / 14.5 | 7.9 / 6.9 / 8.1 | 8.2 / 7.6 / 7.9 |
| time-filtered r=1 (30 bars) | lifetimes (stable rank) |  | 8.9 / 14.6 / 15.9 | 7.3 / 14.1 / 14.3 | 6.5 / 6.6 / 7.9 | 8.2 / 8.0 / 7.0 |
| time-filtered r=1 (30 bars) | Betti + births |  | 9.9 / 17.2 / 16.2 | 11.0 / 16.3 / 16.4 | 9.9 / 11.0 / 9.2 | 9.4 / 11.1 / 10.6 |
| time-filtered r=2 (6 bars) | Betti curve over time |  | 5.2 / 12.6 / 11.8 | 4.8 / 12.7 / 10.9 | 5.9 / 6.5 / 5.6 | 5.8 / 6.1 / 5.6 |
| time-filtered r=2 (6 bars) | births so far |  | 4.3 / 12.6 / 10.5 | 5.2 / 12.4 / 10.7 | 4.2 / 4.6 / 3.9 | 4.6 / 4.5 / 5.3 |
| time-filtered r=2 (6 bars) | lifetimes (stable rank) |  | 4.5 / 12.0 / 10.9 | 5.0 / 12.7 / 11.6 | 5.2 / 4.7 / 4.9 | 4.7 / 6.5 / 4.9 |
| time-filtered r=2 (6 bars) | Betti + births |  | 6.6 / 13.5 / 11.8 | 5.7 / 12.8 / 10.7 | 5.9 / 6.4 / 5.8 | 6.3 / 5.7 / 5.6 |
| whole cloud, plain | H0 | shuffled | 28.8 / 28.0 / 29.0 | 24.8 / 26.0 / 27.6 | 28.2 / 19.9 / 23.4 | 26.1 / 24.6 / 24.3 |
| whole cloud, plain | H1 stable rank | shuffled | 19.2 / 22.8 / 22.6 | 15.6 / 21.9 / 21.1 | 17.9 / 15.5 / 14.5 | 16.3 / 17.6 / 16.8 |
| whole cloud, plain | H1 Betti curve | shuffled | 22.8 / 26.3 / 25.7 | 21.7 / 23.9 / 24.6 | 22.8 / 20.1 / 17.9 | 21.1 / 20.6 / 22.7 |
| whole cloud, plain | H0 + H1 stable rank | shuffled | 29.1 / 26.6 / 28.8 | 24.6 / 26.2 / 26.5 | 28.1 / 21.1 / 23.4 | 26.5 / 22.3 / 23.8 |
| whole cloud, plain | H0 + H1 Betti curve | shuffled | 31.5 / 31.2 / 32.5 | 28.3 / 29.5 / 29.5 | 31.9 / 25.1 / 26.0 | 30.4 / 26.5 / 29.0 |
| whole cloud, regions k=8 | H0 | shuffled | 37.6 / 34.3 / 36.6 | 35.3 / 28.5 / 34.8 | 36.2 / 28.1 / 30.4 | 34.1 / 22.7 / 32.7 |
| whole cloud, regions k=8 | H1 stable rank | shuffled | 20.7 / 21.9 / 22.0 | 19.3 / 20.4 / 22.0 | 19.4 / 17.6 / 16.6 | 18.8 / 16.8 / 19.0 |
| whole cloud, regions k=8 | H1 Betti curve | shuffled | 16.8 / 17.0 / 20.1 | 14.7 / 19.0 / 19.1 | 13.2 / 12.7 / 15.2 | 15.3 / 13.8 / 16.4 |
| whole cloud, regions k=8 | H0 + H1 stable rank | shuffled | 34.2 / 30.6 / 34.9 | 30.5 / 25.7 / 31.0 | 32.6 / 24.6 / 30.9 | 30.1 / 20.1 / 28.2 |
| whole cloud, regions k=8 | H0 + H1 Betti curve | shuffled | 31.9 / 29.5 / 32.0 | 29.6 / 27.7 / 30.1 | 31.9 / 25.9 / 26.2 | 31.0 / 22.6 / 27.7 |
| whole cloud, regions k=16 | H0 | shuffled | 38.9 / 33.2 / 40.0 | 31.9 / 30.0 / 33.3 | 35.4 / 25.3 / 31.7 | 35.0 / 30.6 / 33.6 |
| whole cloud, regions k=16 | H1 stable rank | shuffled | 18.0 / 19.3 / 20.5 | 15.0 / 19.9 / 18.2 | 16.9 / 12.9 / 15.0 | 18.1 / 14.6 / 14.6 |
| whole cloud, regions k=16 | H1 Betti curve | shuffled | 10.6 / 12.7 / 11.8 | 9.1 / 11.7 / 11.0 | 8.1 / 8.4 / 8.8 | 8.7 / 8.2 / 8.9 |
| whole cloud, regions k=16 | H0 + H1 stable rank | shuffled | 32.2 / 29.7 / 28.6 | 28.2 / 25.8 / 27.2 | 28.2 / 22.1 / 22.9 | 29.2 / 22.0 / 24.5 |
| whole cloud, regions k=16 | H0 + H1 Betti curve | shuffled | 28.7 / 27.4 / 28.6 | 26.2 / 26.2 / 23.3 | 27.9 / 21.3 / 24.2 | 27.5 / 22.2 / 23.0 |
| time-filtered r=0.5 (107 bars) | Betti curve over time | shuffled | 10.6 / 18.8 / 18.6 | 10.6 / 18.0 / 18.5 | 11.2 / 11.5 / 11.7 | 12.0 / 10.8 / 14.3 |
| time-filtered r=0.5 (107 bars) | births so far | shuffled | 9.9 / 18.6 / 18.1 | 8.8 / 17.2 / 18.3 | 9.9 / 12.4 / 11.5 | 10.0 / 12.2 / 11.6 |
| time-filtered r=0.5 (107 bars) | lifetimes (stable rank) | shuffled | 9.3 / 16.8 / 16.6 | 8.0 / 15.6 / 15.8 | 8.2 / 9.7 / 9.8 | 8.8 / 9.2 / 10.6 |
| time-filtered r=0.5 (107 bars) | Betti + births | shuffled | 13.6 / 20.7 / 21.6 | 13.5 / 20.1 / 20.4 | 14.2 / 15.0 / 14.1 | 13.9 / 15.1 / 15.8 |
| time-filtered r=1 (31 bars) | Betti curve over time | shuffled | 6.8 / 14.8 / 14.9 | 5.4 / 13.9 / 14.4 | 7.3 / 7.0 / 7.4 | 7.3 / 7.8 / 8.6 |
| time-filtered r=1 (31 bars) | births so far | shuffled | 7.7 / 14.4 / 14.9 | 5.8 / 14.9 / 14.6 | 6.7 / 7.2 / 8.1 | 7.3 / 8.1 / 8.1 |
| time-filtered r=1 (31 bars) | lifetimes (stable rank) | shuffled | 7.9 / 15.0 / 14.5 | 5.9 / 14.5 / 15.0 | 6.2 / 7.8 / 7.3 | 7.0 / 8.8 / 7.9 |
| time-filtered r=1 (31 bars) | Betti + births | shuffled | 10.0 / 17.2 / 16.3 | 8.4 / 16.4 / 17.2 | 10.6 / 10.0 / 9.6 | 10.3 / 9.8 / 10.8 |
| time-filtered r=2 (6 bars) | Betti curve over time | shuffled | 5.1 / 13.0 / 11.4 | 5.8 / 12.5 / 10.9 | 5.6 / 6.2 / 5.5 | 5.5 / 5.4 / 5.4 |
| time-filtered r=2 (6 bars) | births so far | shuffled | 4.2 / 12.1 / 10.6 | 5.0 / 11.9 / 10.5 | 4.5 / 5.5 / 4.4 | 5.3 / 4.2 / 4.4 |
| time-filtered r=2 (6 bars) | lifetimes (stable rank) | shuffled | 5.0 / 12.1 / 11.5 | 5.2 / 11.6 / 11.4 | 5.0 / 5.9 / 4.9 | 4.5 / 6.0 / 5.5 |
| time-filtered r=2 (6 bars) | Betti + births | shuffled | 5.3 / 13.1 / 11.4 | 6.5 / 12.5 / 10.3 | 5.7 / 6.1 / 5.3 | 5.5 / 6.2 / 5.1 |

### OpenScope 10 s

60 neurons, 1769 + 1490 samples.

| family | readout | control | within gratings | within grey | gratings -> grey | grey -> gratings |
|---|---|---|---|---|---|---|
| whole cloud, plain | H0 |  | 12.8 / 17.6 / 18.1 | 15.0 / 18.1 / 17.7 | 11.3 / 10.9 / 9.2 | 12.5 / 8.5 / 10.5 |
| whole cloud, plain | H1 stable rank |  | 8.8 / 15.5 / 15.3 | 9.5 / 14.8 / 15.9 | 6.3 / 8.0 / 8.2 | 7.5 / 7.1 / 7.8 |
| whole cloud, plain | H1 Betti curve |  | 9.8 / 14.9 / 15.3 | 9.0 / 15.0 / 15.0 | 7.9 / 7.3 / 7.1 | 8.0 / 7.4 / 8.6 |
| whole cloud, plain | H0 + H1 stable rank |  | 13.0 / 18.3 / 17.2 | 14.5 / 17.7 / 17.8 | 10.6 / 10.2 / 9.7 | 12.3 / 9.9 / 8.2 |
| whole cloud, plain | H0 + H1 Betti curve |  | 14.9 / 17.9 / 18.8 | 15.4 / 17.7 / 18.4 | 11.1 / 10.4 / 11.4 | 13.2 / 9.2 / 10.2 |
| whole cloud, regions k=8 | H0 |  | 19.0 / 19.7 / 20.1 | 22.8 / 20.5 / 20.6 | 16.5 / 11.2 / 13.2 | 18.6 / 12.5 / 12.4 |
| whole cloud, regions k=8 | H1 stable rank |  | 8.3 / 13.4 / 12.6 | 8.3 / 13.4 / 13.2 | 6.8 / 7.2 / 6.5 | 7.5 / 5.8 / 6.3 |
| whole cloud, regions k=8 | H1 Betti curve |  | 5.9 / 9.7 / 6.8 | 6.4 / 10.1 / 7.4 | 4.6 / 4.1 / 3.7 | 6.8 / 4.5 / 4.3 |
| whole cloud, regions k=8 | H0 + H1 stable rank |  | 13.4 / 16.5 / 16.3 | 16.6 / 16.8 / 16.8 | 13.2 / 11.1 / 10.0 | 13.5 / 9.4 / 8.8 |
| whole cloud, regions k=8 | H0 + H1 Betti curve |  | 12.2 / 15.6 / 14.0 | 16.2 / 16.4 / 15.1 | 12.9 / 10.7 / 7.6 | 12.7 / 10.6 / 7.4 |
| whole cloud, regions k=16 | H0 |  | 23.0 / 21.0 / 22.1 | 26.3 / 22.5 / 23.2 | 20.6 / 15.6 / 17.2 | 21.5 / 12.0 / 14.6 |
| whole cloud, regions k=16 | H1 stable rank |  | 9.6 / 12.7 / 10.6 | 12.7 / 15.0 / 13.6 | 8.5 / 8.9 / 6.8 | 10.4 / 7.1 / 6.2 |
| whole cloud, regions k=16 | H1 Betti curve |  | 6.8 / 9.8 / 6.1 | 7.9 / 11.1 / 7.7 | 5.5 / 5.4 / 4.0 | 6.5 / 3.6 / 3.8 |
| whole cloud, regions k=16 | H0 + H1 stable rank |  | 17.5 / 17.4 / 15.2 | 20.0 / 17.9 / 15.5 | 15.5 / 10.4 / 11.3 | 17.8 / 10.2 / 8.5 |
| whole cloud, regions k=16 | H0 + H1 Betti curve |  | 16.6 / 16.7 / 10.2 | 17.4 / 17.4 / 13.4 | 13.3 / 11.1 / 6.0 | 14.6 / 9.2 / 6.3 |
| time-filtered r=0.5 (46 bars) | Betti curve over time |  | 6.7 / 12.8 / 13.0 | 9.1 / 13.2 / 13.8 | 6.1 / 6.0 / 6.6 | 7.0 / 4.8 / 4.6 |
| time-filtered r=0.5 (46 bars) | births so far |  | 6.3 / 13.7 / 13.5 | 10.3 / 14.0 / 13.7 | 6.2 / 5.8 / 6.2 | 5.9 / 4.8 / 5.1 |
| time-filtered r=0.5 (46 bars) | lifetimes (stable rank) |  | 4.7 / 11.4 / 11.7 | 6.4 / 13.4 / 13.6 | 4.7 / 4.0 / 3.3 | 5.2 / 3.2 / 4.0 |
| time-filtered r=0.5 (46 bars) | Betti + births |  | 8.6 / 14.0 / 14.3 | 12.7 / 15.0 / 14.6 | 7.6 / 7.2 / 6.2 | 7.1 / 6.0 / 6.0 |
| time-filtered r=1 (22 bars) | Betti curve over time |  | 5.8 / 11.9 / 12.4 | 6.6 / 13.4 / 13.6 | 5.5 / 5.6 / 4.6 | 5.0 / 5.3 / 4.7 |
| time-filtered r=1 (22 bars) | births so far |  | 6.6 / 12.5 / 12.3 | 6.5 / 13.6 / 13.7 | 4.6 / 4.8 / 5.3 | 5.8 / 4.2 / 4.2 |
| time-filtered r=1 (22 bars) | lifetimes (stable rank) |  | 5.6 / 11.1 / 11.7 | 6.2 / 12.8 / 13.2 | 4.6 / 3.8 / 4.3 | 4.5 / 4.4 / 3.9 |
| time-filtered r=1 (22 bars) | Betti + births |  | 7.7 / 13.5 / 13.4 | 8.1 / 13.9 / 15.0 | 6.9 / 6.3 / 5.3 | 7.6 / 5.4 / 5.0 |
| time-filtered r=2 (6 bars) | Betti curve over time |  | 4.4 / 11.6 / 10.1 | 7.0 / 13.3 / 10.9 | 4.2 / 5.4 / 4.0 | 5.0 / 3.6 / 4.1 |
| time-filtered r=2 (6 bars) | births so far |  | 4.2 / 10.9 / 10.7 | 7.1 / 13.2 / 12.0 | 3.6 / 5.1 / 5.1 | 4.0 / 3.2 / 4.1 |
| time-filtered r=2 (6 bars) | lifetimes (stable rank) |  | 4.4 / 11.6 / 10.4 | 6.4 / 12.8 / 12.0 | 4.5 / 4.3 / 3.7 | 4.6 / 4.1 / 4.5 |
| time-filtered r=2 (6 bars) | Betti + births |  | 4.6 / 10.5 / 11.0 | 7.8 / 13.6 / 12.5 | 4.7 / 5.2 / 5.3 | 5.8 / 3.7 / 4.1 |
| whole cloud, plain | H0 | shuffled | 13.5 / 17.6 / 17.5 | 13.6 / 17.7 / 18.2 | 9.8 / 8.6 / 9.4 | 13.0 / 8.9 / 10.3 |
| whole cloud, plain | H1 stable rank | shuffled | 8.4 / 15.8 / 15.5 | 9.4 / 16.1 / 15.6 | 7.0 / 7.8 / 8.9 | 9.5 / 7.6 / 7.1 |
| whole cloud, plain | H1 Betti curve | shuffled | 9.2 / 15.4 / 14.3 | 9.3 / 15.9 / 15.5 | 8.0 / 7.5 / 7.0 | 9.1 / 8.0 / 6.7 |
| whole cloud, plain | H0 + H1 stable rank | shuffled | 12.7 / 17.5 / 17.4 | 13.6 / 18.5 / 18.6 | 11.0 / 6.9 / 10.5 | 13.0 / 9.0 / 9.0 |
| whole cloud, plain | H0 + H1 Betti curve | shuffled | 14.3 / 18.4 / 18.4 | 16.2 / 17.4 / 18.1 | 11.2 / 10.7 / 10.9 | 14.0 / 9.6 / 9.7 |
| whole cloud, regions k=8 | H0 | shuffled | 16.4 / 17.3 / 18.1 | 17.2 / 17.7 / 18.7 | 13.6 / 11.0 / 11.1 | 14.9 / 10.1 / 10.3 |
| whole cloud, regions k=8 | H1 stable rank | shuffled | 8.4 / 14.0 / 12.7 | 8.7 / 14.0 / 13.0 | 6.7 / 6.0 / 5.6 | 7.4 / 5.1 / 4.7 |
| whole cloud, regions k=8 | H1 Betti curve | shuffled | 6.4 / 10.5 / 6.4 | 8.1 / 11.6 / 8.9 | 6.3 / 5.4 / 4.6 | 6.4 / 6.0 / 5.4 |
| whole cloud, regions k=8 | H0 + H1 stable rank | shuffled | 13.5 / 16.9 / 17.0 | 14.6 / 16.1 / 17.6 | 11.4 / 9.1 / 9.1 | 11.2 / 8.0 / 10.6 |
| whole cloud, regions k=8 | H0 + H1 Betti curve | shuffled | 12.7 / 16.4 / 14.6 | 15.0 / 15.5 / 15.4 | 11.7 / 10.7 / 9.1 | 13.0 / 8.5 / 7.9 |
| whole cloud, regions k=16 | H0 | shuffled | 17.2 / 18.7 / 19.5 | 22.6 / 19.4 / 20.1 | 15.9 / 10.9 / 11.4 | 17.9 / 12.2 / 11.6 |
| whole cloud, regions k=16 | H1 stable rank | shuffled | 8.4 / 12.6 / 9.5 | 7.8 / 13.6 / 12.2 | 6.2 / 7.6 / 6.2 | 7.1 / 5.5 / 3.9 |
| whole cloud, regions k=16 | H1 Betti curve | shuffled | 5.5 / 10.4 / 5.6 | 7.4 / 9.8 / 6.9 | 5.1 / 5.6 / 3.6 | 6.3 / 5.3 / 3.8 |
| whole cloud, regions k=16 | H0 + H1 stable rank | shuffled | 14.6 / 16.2 / 14.6 | 16.5 / 18.3 / 15.5 | 11.8 / 9.2 / 10.7 | 13.4 / 10.2 / 9.0 |
| whole cloud, regions k=16 | H0 + H1 Betti curve | shuffled | 13.7 / 15.5 / 9.6 | 16.9 / 17.1 / 11.9 | 11.1 / 8.2 / 6.5 | 12.8 / 10.8 / 5.9 |
| time-filtered r=0.5 (47 bars) | Betti curve over time | shuffled | 6.7 / 13.8 / 13.2 | 6.2 / 13.3 / 13.8 | 6.1 / 6.0 / 4.7 | 7.3 / 6.0 / 5.2 |
| time-filtered r=0.5 (47 bars) | births so far | shuffled | 7.0 / 13.9 / 13.4 | 7.0 / 12.6 / 13.0 | 5.9 / 6.0 / 5.0 | 6.3 / 5.5 / 4.5 |
| time-filtered r=0.5 (47 bars) | lifetimes (stable rank) | shuffled | 5.4 / 11.9 / 12.0 | 6.0 / 12.2 / 12.6 | 5.1 / 4.6 / 3.8 | 5.5 / 3.7 / 4.3 |
| time-filtered r=0.5 (47 bars) | Betti + births | shuffled | 8.5 / 15.0 / 14.5 | 8.7 / 13.3 / 13.8 | 7.1 / 7.2 / 7.3 | 8.3 / 5.5 / 5.5 |
| time-filtered r=1 (23 bars) | Betti curve over time | shuffled | 5.9 / 12.7 / 12.5 | 7.2 / 13.0 / 13.0 | 4.7 / 6.7 / 5.1 | 5.4 / 5.8 / 4.8 |
| time-filtered r=1 (23 bars) | births so far | shuffled | 4.7 / 12.4 / 12.2 | 5.8 / 12.1 / 12.6 | 4.5 / 5.5 / 5.0 | 4.6 / 4.5 / 4.7 |
| time-filtered r=1 (23 bars) | lifetimes (stable rank) | shuffled | 4.1 / 11.2 / 11.4 | 5.2 / 12.3 / 13.2 | 3.2 / 4.5 / 4.7 | 4.0 / 4.2 / 4.7 |
| time-filtered r=1 (23 bars) | Betti + births | shuffled | 6.3 / 13.2 / 13.8 | 7.7 / 13.3 / 13.4 | 6.0 / 6.1 / 6.7 | 5.9 / 6.2 / 5.4 |
| time-filtered r=2 (6 bars) | Betti curve over time | shuffled | 4.2 / 11.3 / 10.1 | 5.8 / 12.3 / 11.1 | 3.1 / 4.6 / 3.8 | 4.1 / 4.4 / 4.0 |
| time-filtered r=2 (6 bars) | births so far | shuffled | 5.3 / 11.6 / 10.7 | 4.6 / 11.7 / 10.9 | 3.2 / 4.2 / 3.7 | 4.0 / 4.5 / 3.5 |
| time-filtered r=2 (6 bars) | lifetimes (stable rank) | shuffled | 4.9 / 11.6 / 11.0 | 4.3 / 12.5 / 11.9 | 4.4 / 4.6 / 4.3 | 3.9 / 4.8 / 5.3 |
| time-filtered r=2 (6 bars) | Betti + births | shuffled | 5.7 / 12.2 / 11.2 | 6.3 / 12.6 / 11.7 | 3.6 / 5.1 / 5.2 | 4.0 / 4.8 / 4.1 |

### Allen 30 s

60 neurons, 1783 + 1734 samples.

| family | readout | control | within movie | within spontaneous | movie -> spontaneous | spontaneous -> movie |
|---|---|---|---|---|---|---|
| whole cloud, plain | H0 |  | 20.7 / 19.0 / 19.9 | 17.4 / 18.6 / 18.9 | 12.1 / 6.5 / 8.1 | 11.2 / 9.8 / 9.1 |
| whole cloud, plain | H1 stable rank |  | 11.7 / 17.6 / 17.7 | 9.2 / 15.1 / 15.5 | 7.8 / 7.6 / 8.0 | 7.5 / 8.3 / 7.4 |
| whole cloud, plain | H1 Betti curve |  | 14.5 / 18.5 / 18.6 | 13.4 / 17.7 / 17.3 | 10.2 / 8.5 / 7.6 | 10.7 / 8.5 / 9.2 |
| whole cloud, plain | H0 + H1 stable rank |  | 21.2 / 19.9 / 20.0 | 17.2 / 18.3 / 18.3 | 12.6 / 9.1 / 8.8 | 11.8 / 8.2 / 10.3 |
| whole cloud, plain | H0 + H1 Betti curve |  | 21.7 / 21.9 / 21.4 | 19.7 / 19.6 / 20.6 | 14.7 / 8.5 / 9.0 | 14.7 / 9.6 / 12.7 |
| whole cloud, regions k=8 | H0 |  | 34.4 / 25.7 / 26.9 | 27.9 / 23.6 / 24.7 | 22.7 / 9.7 / 11.5 | 20.2 / 12.7 / 13.2 |
| whole cloud, regions k=8 | H1 stable rank |  | 15.1 / 18.7 / 16.6 | 13.1 / 16.7 / 15.3 | 11.9 / 8.7 / 6.1 | 8.5 / 8.0 / 8.8 |
| whole cloud, regions k=8 | H1 Betti curve |  | 15.5 / 13.9 / 14.2 | 9.9 / 12.5 / 11.9 | 11.0 / 5.8 / 7.3 | 7.9 / 7.2 / 7.2 |
| whole cloud, regions k=8 | H0 + H1 stable rank |  | 27.3 / 23.6 / 22.7 | 23.4 / 21.5 / 21.6 | 18.3 / 11.2 / 9.3 | 16.6 / 13.6 / 12.3 |
| whole cloud, regions k=8 | H0 + H1 Betti curve |  | 29.7 / 23.0 / 21.1 | 23.0 / 20.4 / 20.4 | 19.3 / 10.2 / 10.7 | 17.0 / 12.3 / 10.5 |
| whole cloud, regions k=16 | H0 |  | 38.4 / 27.3 / 28.2 | 30.9 / 24.0 / 24.7 | 25.4 / 12.6 / 12.3 | 22.4 / 13.2 / 16.4 |
| whole cloud, regions k=16 | H1 stable rank |  | 17.4 / 17.6 / 16.5 | 12.3 / 15.7 / 13.3 | 13.5 / 6.5 / 7.3 | 9.7 / 8.8 / 8.2 |
| whole cloud, regions k=16 | H1 Betti curve |  | 12.8 / 12.5 / 11.0 | 8.2 / 12.6 / 9.3 | 10.2 / 5.5 / 5.6 | 6.6 / 7.0 / 5.6 |
| whole cloud, regions k=16 | H0 + H1 stable rank |  | 29.1 / 22.5 / 21.4 | 24.6 / 22.4 / 18.6 | 22.0 / 10.3 / 8.5 | 18.2 / 11.9 / 10.3 |
| whole cloud, regions k=16 | H0 + H1 Betti curve |  | 27.7 / 21.9 / 18.4 | 23.5 / 20.2 / 14.2 | 21.3 / 11.2 / 8.8 | 16.4 / 12.6 / 9.5 |
| time-filtered r=0.5 (76 bars) | Betti curve over time |  | 12.6 / 15.6 / 15.3 | 7.3 / 13.0 / 12.9 | 7.4 / 4.4 / 4.1 | 5.1 / 4.8 / 4.4 |
| time-filtered r=0.5 (76 bars) | births so far |  | 13.0 / 15.5 / 15.4 | 6.7 / 13.7 / 13.1 | 6.2 / 5.3 / 3.8 | 5.1 / 4.9 / 4.5 |
| time-filtered r=0.5 (76 bars) | lifetimes (stable rank) |  | 12.2 / 14.7 / 14.9 | 6.5 / 12.2 / 12.5 | 5.3 / 4.4 / 4.5 | 5.1 / 4.4 / 3.6 |
| time-filtered r=0.5 (76 bars) | Betti + births |  | 17.8 / 16.2 / 16.9 | 8.8 / 13.4 / 13.8 | 8.0 / 5.4 / 4.2 | 6.0 / 6.2 / 5.3 |
| time-filtered r=1 (34 bars) | Betti curve over time |  | 9.1 / 13.9 / 14.2 | 5.5 / 13.3 / 13.0 | 4.5 / 5.5 / 4.2 | 4.7 / 5.0 / 5.1 |
| time-filtered r=1 (34 bars) | births so far |  | 8.3 / 13.7 / 13.7 | 4.8 / 12.7 / 12.6 | 5.6 / 4.3 / 4.8 | 4.7 / 4.4 / 6.1 |
| time-filtered r=1 (34 bars) | lifetimes (stable rank) |  | 7.1 / 13.5 / 13.9 | 5.2 / 12.1 / 12.2 | 4.5 / 3.7 / 4.6 | 4.4 / 4.5 / 3.5 |
| time-filtered r=1 (34 bars) | Betti + births |  | 11.3 / 15.0 / 15.3 | 7.6 / 13.5 / 13.8 | 6.0 / 5.5 / 5.1 | 6.7 / 6.1 / 4.9 |
| time-filtered r=2 (10 bars) | Betti curve over time |  | 5.3 / 12.1 / 10.6 | 5.6 / 12.1 / 11.2 | 4.0 / 3.5 / 4.0 | 3.6 / 4.6 / 3.5 |
| time-filtered r=2 (10 bars) | births so far |  | 5.6 / 11.6 / 11.2 | 4.3 / 11.5 / 11.3 | 3.9 / 4.3 / 4.2 | 3.6 / 5.6 / 3.4 |
| time-filtered r=2 (10 bars) | lifetimes (stable rank) |  | 6.2 / 11.9 / 11.8 | 4.4 / 11.3 / 11.6 | 2.9 / 4.7 / 4.3 | 3.3 / 3.9 / 3.0 |
| time-filtered r=2 (10 bars) | Betti + births |  | 7.0 / 12.5 / 12.3 | 6.1 / 12.0 / 12.4 | 4.3 / 3.8 / 5.2 | 4.2 / 4.3 / 4.3 |
| whole cloud, plain | H0 | shuffled | 19.0 / 19.2 / 19.0 | 15.4 / 18.1 / 18.9 | 11.9 / 7.7 / 7.3 | 10.3 / 8.4 / 10.6 |
| whole cloud, plain | H1 stable rank | shuffled | 13.4 / 18.0 / 17.3 | 9.5 / 15.8 / 16.5 | 8.9 / 6.9 / 6.7 | 6.2 / 8.8 / 9.1 |
| whole cloud, plain | H1 Betti curve | shuffled | 16.8 / 18.8 / 18.6 | 12.9 / 16.5 / 17.2 | 9.8 / 7.8 / 5.8 | 8.4 / 7.9 / 9.7 |
| whole cloud, plain | H0 + H1 stable rank | shuffled | 20.1 / 19.6 / 19.2 | 16.8 / 17.4 / 18.0 | 12.6 / 8.0 / 8.2 | 10.0 / 9.3 / 9.1 |
| whole cloud, plain | H0 + H1 Betti curve | shuffled | 24.8 / 20.9 / 21.5 | 19.4 / 20.1 / 20.5 | 15.1 / 8.7 / 9.2 | 12.4 / 10.3 / 10.9 |
| whole cloud, regions k=8 | H0 | shuffled | 26.5 / 22.2 / 22.0 | 20.8 / 20.9 / 22.5 | 15.4 / 9.9 / 9.3 | 13.7 / 10.9 / 12.9 |
| whole cloud, regions k=8 | H1 stable rank | shuffled | 15.0 / 16.7 / 16.4 | 10.7 / 17.0 / 14.0 | 9.7 / 5.7 / 6.6 | 7.1 / 7.9 / 6.6 |
| whole cloud, regions k=8 | H1 Betti curve | shuffled | 13.2 / 15.6 / 14.2 | 9.5 / 12.8 / 11.3 | 10.2 / 7.5 / 6.7 | 8.3 / 8.4 / 5.4 |
| whole cloud, regions k=8 | H0 + H1 stable rank | shuffled | 23.3 / 20.9 / 19.2 | 18.5 / 19.9 / 20.2 | 12.8 / 8.5 / 8.5 | 11.6 / 8.9 / 8.1 |
| whole cloud, regions k=8 | H0 + H1 Betti curve | shuffled | 23.8 / 20.8 / 19.3 | 18.6 / 20.9 / 19.0 | 16.0 / 8.6 / 7.7 | 13.4 / 9.3 / 10.5 |
| whole cloud, regions k=16 | H0 | shuffled | 29.8 / 25.0 / 24.7 | 22.4 / 21.5 / 22.3 | 17.9 / 12.5 / 11.0 | 16.6 / 15.0 / 12.7 |
| whole cloud, regions k=16 | H1 stable rank | shuffled | 15.5 / 16.5 / 15.3 | 12.3 / 15.1 / 13.6 | 10.0 / 5.8 / 6.5 | 8.8 / 6.6 / 7.1 |
| whole cloud, regions k=16 | H1 Betti curve | shuffled | 11.9 / 11.4 / 10.6 | 8.2 / 12.8 / 6.9 | 8.6 / 4.3 / 5.2 | 6.9 / 7.0 / 3.5 |
| whole cloud, regions k=16 | H0 + H1 stable rank | shuffled | 24.1 / 21.6 / 19.7 | 18.7 / 20.1 / 18.5 | 14.1 / 8.2 / 8.2 | 13.4 / 10.9 / 9.5 |
| whole cloud, regions k=16 | H0 + H1 Betti curve | shuffled | 25.0 / 21.1 / 16.9 | 18.3 / 18.3 / 14.0 | 15.5 / 7.7 / 8.5 | 14.1 / 9.8 / 7.1 |
| time-filtered r=0.5 (79 bars) | Betti curve over time | shuffled | 9.4 / 14.7 / 14.1 | 5.2 / 13.3 / 13.4 | 5.4 / 6.0 / 3.9 | 4.7 / 7.3 / 7.0 |
| time-filtered r=0.5 (79 bars) | births so far | shuffled | 9.8 / 14.2 / 14.9 | 5.4 / 13.9 / 14.0 | 5.6 / 4.7 / 5.1 | 4.7 / 5.8 / 6.2 |
| time-filtered r=0.5 (79 bars) | lifetimes (stable rank) | shuffled | 6.2 / 13.6 / 13.5 | 4.6 / 12.6 / 13.1 | 4.7 / 4.7 / 4.8 | 4.7 / 5.2 / 5.8 |
| time-filtered r=0.5 (79 bars) | Betti + births | shuffled | 11.9 / 15.0 / 16.0 | 6.1 / 14.3 / 13.8 | 6.2 / 4.6 / 5.7 | 4.6 / 7.2 / 7.2 |
| time-filtered r=1 (36 bars) | Betti curve over time | shuffled | 6.6 / 13.3 / 14.0 | 6.4 / 12.9 / 13.6 | 5.5 / 4.3 / 4.3 | 5.4 / 5.4 / 5.8 |
| time-filtered r=1 (36 bars) | births so far | shuffled | 5.7 / 14.0 / 13.9 | 6.0 / 13.2 / 13.6 | 6.0 / 4.4 / 5.8 | 5.2 / 4.8 / 6.3 |
| time-filtered r=1 (36 bars) | lifetimes (stable rank) | shuffled | 4.8 / 12.2 / 12.8 | 6.1 / 12.6 / 12.5 | 5.2 / 4.2 / 4.8 | 5.1 / 4.9 / 6.7 |
| time-filtered r=1 (36 bars) | Betti + births | shuffled | 7.9 / 15.0 / 14.8 | 8.2 / 14.6 / 15.1 | 6.6 / 6.5 / 6.2 | 6.2 / 6.4 / 4.9 |
| time-filtered r=2 (10 bars) | Betti curve over time | shuffled | 4.7 / 12.6 / 10.9 | 4.1 / 12.1 / 10.9 | 5.2 / 4.7 / 3.0 | 3.6 / 4.2 / 3.6 |
| time-filtered r=2 (10 bars) | births so far | shuffled | 5.7 / 12.7 / 12.3 | 3.9 / 12.1 / 11.9 | 4.9 / 4.3 / 4.4 | 4.3 / 4.9 / 4.0 |
| time-filtered r=2 (10 bars) | lifetimes (stable rank) | shuffled | 5.6 / 11.3 / 11.4 | 5.2 / 12.3 / 11.2 | 4.0 / 4.2 / 4.0 | 4.3 / 4.5 / 3.7 |
| time-filtered r=2 (10 bars) | Betti + births | shuffled | 6.4 / 12.6 / 12.5 | 4.9 / 12.3 / 12.9 | 4.9 / 3.6 / 4.4 | 4.3 / 4.6 / 4.8 |

### retina 40 s (clean)

60 neurons, 420 + 1795 samples.

| family | readout | control | within frozen x4 | within unrepeated x4 | frozen x4 -> unrepeated x4 | unrepeated x4 -> frozen x4 |
|---|---|---|---|---|---|---|
| whole cloud, plain | H0 |  | 33.8 / 40.0 / 40.0 | 43.1 / 32.3 / 35.7 | 49.8 / 27.4 / 27.8 | 38.8 / 32.4 / 32.6 |
| whole cloud, plain | H1 stable rank |  | 29.3 / 37.6 / 37.1 | 30.6 / 29.9 / 31.1 | 35.2 / 25.8 / 26.9 | 32.0 / 26.7 / 28.3 |
| whole cloud, plain | H1 Betti curve |  | 38.1 / 40.5 / 41.2 | 40.7 / 33.8 / 35.4 | 36.4 / 29.2 / 30.4 | 38.0 / 32.4 / 36.2 |
| whole cloud, plain | H0 + H1 stable rank |  | 39.0 / 41.2 / 42.4 | 47.0 / 31.9 / 33.6 | 52.4 / 29.9 / 29.4 | 43.2 / 31.4 / 33.1 |
| whole cloud, plain | H0 + H1 Betti curve |  | 47.1 / 46.2 / 43.3 | 51.1 / 36.0 / 38.7 | 53.3 / 36.4 / 32.9 | 47.6 / 37.9 / 38.8 |
| whole cloud, regions k=8 | H0 |  | 56.7 / 48.3 / 47.9 | 59.3 / 39.3 / 44.8 | 63.3 / 37.9 / 36.7 | 55.9 / 41.0 / 49.0 |
| whole cloud, regions k=8 | H1 stable rank |  | 37.4 / 40.5 / 43.8 | 39.2 / 32.5 / 39.2 | 41.2 / 28.6 / 30.0 | 38.9 / 28.3 / 36.0 |
| whole cloud, regions k=8 | H1 Betti curve |  | 40.5 / 40.2 / 47.1 | 41.5 / 33.2 / 40.9 | 42.4 / 30.1 / 36.2 | 42.2 / 30.7 / 39.8 |
| whole cloud, regions k=8 | H0 + H1 stable rank |  | 49.0 / 46.2 / 46.0 | 55.5 / 39.2 / 40.1 | 57.9 / 34.9 / 35.9 | 52.8 / 38.8 / 39.8 |
| whole cloud, regions k=8 | H0 + H1 Betti curve |  | 54.5 / 46.9 / 51.7 | 57.2 / 38.4 / 46.2 | 63.6 / 38.8 / 43.5 | 55.5 / 39.5 / 48.8 |
| whole cloud, regions k=16 | H0 |  | 63.8 / 52.4 / 53.6 | 64.4 / 46.5 / 48.9 | 69.3 / 42.8 / 42.9 | 61.7 / 46.2 / 47.1 |
| whole cloud, regions k=16 | H1 stable rank |  | 42.9 / 42.4 / 44.5 | 46.6 / 35.2 / 39.3 | 50.0 / 29.2 / 34.4 | 44.0 / 33.6 / 36.4 |
| whole cloud, regions k=16 | H1 Betti curve |  | 36.2 / 38.6 / 41.9 | 39.4 / 31.5 / 35.7 | 40.2 / 27.0 / 31.9 | 39.8 / 30.5 / 36.4 |
| whole cloud, regions k=16 | H0 + H1 stable rank |  | 58.1 / 51.9 / 52.4 | 61.6 / 40.6 / 47.1 | 63.1 / 40.9 / 42.9 | 57.6 / 40.2 / 49.0 |
| whole cloud, regions k=16 | H0 + H1 Betti curve |  | 58.1 / 51.0 / 54.0 | 62.6 / 40.3 / 49.4 | 64.5 / 42.5 / 45.3 | 60.3 / 42.6 / 48.6 |
| time-filtered r=0.5 (195 bars) | Betti curve over time |  | 14.0 / 30.0 / 29.8 | 16.7 / 22.7 / 22.5 | 19.5 / 15.2 / 12.5 | 18.1 / 19.0 / 18.3 |
| time-filtered r=0.5 (195 bars) | births so far |  | 15.2 / 32.9 / 32.9 | 19.2 / 22.9 / 22.8 | 20.7 / 16.3 / 16.8 | 19.7 / 17.1 / 19.0 |
| time-filtered r=0.5 (195 bars) | lifetimes (stable rank) |  | 11.9 / 27.9 / 30.7 | 17.8 / 21.7 / 21.6 | 18.3 / 14.3 / 15.2 | 15.5 / 18.8 / 19.8 |
| time-filtered r=0.5 (195 bars) | Betti + births |  | 19.3 / 34.3 / 34.5 | 24.2 / 26.6 / 27.0 | 29.3 / 20.5 / 20.9 | 24.0 / 23.8 / 21.9 |
| time-filtered r=1 (39 bars) | Betti curve over time |  | 9.0 / 24.8 / 25.0 | 9.2 / 15.3 / 16.9 | 8.6 / 6.4 / 6.0 | 9.2 / 9.0 / 12.6 |
| time-filtered r=1 (39 bars) | births so far |  | 7.9 / 24.5 / 25.2 | 7.4 / 14.9 / 14.9 | 8.8 / 8.2 / 8.0 | 8.0 / 9.0 / 9.0 |
| time-filtered r=1 (39 bars) | lifetimes (stable rank) |  | 7.4 / 24.5 / 25.5 | 8.0 / 16.3 / 15.5 | 7.9 / 5.7 / 7.7 | 8.9 / 8.6 / 9.3 |
| time-filtered r=1 (39 bars) | Betti + births |  | 11.0 / 28.3 / 27.1 | 11.7 / 17.6 / 18.1 | 14.8 / 12.1 / 9.6 | 11.3 / 12.9 / 12.6 |
| time-filtered r=2 (6 bars) | Betti curve over time |  | 7.1 / 22.1 / 18.6 | 7.2 / 13.3 / 9.7 | 8.1 / 5.4 / 5.0 | 5.8 / 5.7 / 7.1 |
| time-filtered r=2 (6 bars) | births so far |  | 4.8 / 21.9 / 19.8 | 6.6 / 12.4 / 9.7 | 6.4 / 4.1 / 4.5 | 5.2 / 5.2 / 3.8 |
| time-filtered r=2 (6 bars) | lifetimes (stable rank) |  | 4.3 / 22.4 / 22.9 | 6.5 / 12.7 / 10.7 | 6.2 / 4.3 / 4.2 | 5.6 / 6.0 / 8.1 |
| time-filtered r=2 (6 bars) | Betti + births |  | 6.2 / 23.1 / 21.2 | 6.5 / 13.1 / 10.0 | 9.0 / 6.4 / 4.7 | 5.7 / 5.5 / 5.0 |
| whole cloud, plain | H0 | shuffled | 36.0 / 41.4 / 41.9 | 41.6 / 33.1 / 32.9 | 45.5 / 28.6 / 28.7 | 38.7 / 30.5 / 27.6 |
| whole cloud, plain | H1 stable rank | shuffled | 35.0 / 38.8 / 39.3 | 30.3 / 28.6 / 30.6 | 38.8 / 25.2 / 24.4 | 33.0 / 27.1 / 30.0 |
| whole cloud, plain | H1 Betti curve | shuffled | 37.1 / 41.2 / 42.1 | 43.5 / 32.9 / 36.9 | 43.8 / 30.5 / 31.1 | 41.8 / 31.9 / 33.8 |
| whole cloud, plain | H0 + H1 stable rank | shuffled | 44.5 / 40.7 / 42.9 | 46.7 / 31.6 / 35.0 | 51.2 / 29.0 / 29.5 | 43.8 / 29.5 / 34.3 |
| whole cloud, plain | H0 + H1 Betti curve | shuffled | 44.0 / 46.0 / 46.7 | 52.4 / 36.6 / 40.6 | 54.8 / 35.3 / 36.7 | 47.6 / 37.6 / 39.0 |
| whole cloud, regions k=8 | H0 | shuffled | 56.0 / 49.5 / 49.3 | 56.2 / 40.7 / 44.7 | 59.5 / 41.3 / 38.9 | 51.3 / 43.8 / 45.5 |
| whole cloud, regions k=8 | H1 stable rank | shuffled | 37.4 / 41.0 / 42.4 | 36.8 / 34.5 / 37.0 | 42.9 / 28.0 / 28.6 | 37.2 / 33.6 / 36.4 |
| whole cloud, regions k=8 | H1 Betti curve | shuffled | 38.8 / 41.7 / 43.8 | 43.1 / 35.5 / 40.4 | 44.0 / 29.2 / 33.4 | 40.4 / 35.2 / 38.3 |
| whole cloud, regions k=8 | H0 + H1 stable rank | shuffled | 50.0 / 45.2 / 44.8 | 50.6 / 33.3 / 43.3 | 56.4 / 33.6 / 37.3 | 49.2 / 33.3 / 42.1 |
| whole cloud, regions k=8 | H0 + H1 Betti curve | shuffled | 51.2 / 46.9 / 48.6 | 56.2 / 38.3 / 45.9 | 56.7 / 37.3 / 39.1 | 51.5 / 38.8 / 45.5 |
| whole cloud, regions k=16 | H0 | shuffled | 60.5 / 50.7 / 52.9 | 61.1 / 45.3 / 47.9 | 64.3 / 41.7 / 43.6 | 56.9 / 46.9 / 49.5 |
| whole cloud, regions k=16 | H1 stable rank | shuffled | 41.7 / 43.6 / 43.8 | 41.7 / 34.8 / 39.6 | 44.0 / 31.0 / 32.9 | 43.3 / 34.5 / 35.5 |
| whole cloud, regions k=16 | H1 Betti curve | shuffled | 33.8 / 37.9 / 41.4 | 38.3 / 32.9 / 35.6 | 36.7 / 25.1 / 30.2 | 37.7 / 34.3 / 33.1 |
| whole cloud, regions k=16 | H0 + H1 stable rank | shuffled | 55.7 / 46.4 / 49.3 | 58.4 / 38.5 / 47.4 | 59.3 / 37.2 / 39.3 | 54.2 / 37.4 / 50.5 |
| whole cloud, regions k=16 | H0 + H1 Betti curve | shuffled | 56.2 / 52.6 / 51.7 | 59.8 / 42.5 / 48.6 | 59.5 / 43.5 / 42.6 | 56.2 / 43.6 / 47.1 |
| time-filtered r=0.5 (205 bars) | Betti curve over time | shuffled | 16.7 / 34.0 / 33.6 | 16.7 / 22.3 / 22.6 | 19.5 / 15.7 / 15.6 | 16.8 / 15.0 / 16.7 |
| time-filtered r=0.5 (205 bars) | births so far | shuffled | 17.9 / 32.4 / 32.6 | 17.4 / 24.0 / 24.1 | 22.6 / 15.8 / 16.3 | 18.7 / 19.8 / 21.4 |
| time-filtered r=0.5 (205 bars) | lifetimes (stable rank) | shuffled | 13.6 / 32.4 / 31.2 | 16.1 / 22.7 / 22.4 | 18.1 / 13.8 / 12.8 | 15.6 / 17.9 / 15.7 |
| time-filtered r=0.5 (205 bars) | Betti + births | shuffled | 22.6 / 36.4 / 36.0 | 24.3 / 27.5 / 28.7 | 27.1 / 19.4 / 21.3 | 23.0 / 21.9 / 21.0 |
| time-filtered r=1 (40 bars) | Betti curve over time | shuffled | 9.5 / 23.6 / 25.5 | 10.5 / 16.3 / 17.2 | 7.1 / 8.3 / 9.0 | 8.9 / 10.5 / 10.2 |
| time-filtered r=1 (40 bars) | births so far | shuffled | 10.7 / 26.4 / 25.5 | 8.0 / 15.5 / 15.8 | 11.0 / 7.2 / 8.5 | 8.8 / 9.5 / 9.5 |
| time-filtered r=1 (40 bars) | lifetimes (stable rank) | shuffled | 7.9 / 24.3 / 25.2 | 9.9 / 16.5 / 16.4 | 10.2 / 8.6 / 8.6 | 9.9 / 10.0 / 9.8 |
| time-filtered r=1 (40 bars) | Betti + births | shuffled | 15.2 / 26.2 / 27.6 | 12.8 / 19.2 / 18.1 | 14.3 / 11.1 / 11.9 | 12.2 / 13.3 / 13.6 |
| time-filtered r=2 (7 bars) | Betti curve over time | shuffled | 5.5 / 23.1 / 19.8 | 5.6 / 13.1 / 10.2 | 6.2 / 4.6 / 5.6 | 6.2 / 6.4 / 5.2 |
| time-filtered r=2 (7 bars) | births so far | shuffled | 7.1 / 21.2 / 19.0 | 5.2 / 11.4 / 9.7 | 3.6 / 5.4 / 4.0 | 6.2 / 5.2 / 4.8 |
| time-filtered r=2 (7 bars) | lifetimes (stable rank) | shuffled | 6.0 / 22.6 / 23.3 | 5.7 / 12.8 / 11.5 | 6.0 / 5.4 / 5.0 | 4.6 / 4.5 / 7.1 |
| time-filtered r=2 (7 bars) | Betti + births | shuffled | 4.5 / 21.9 / 18.1 | 7.1 / 12.8 / 10.2 | 6.2 / 5.8 / 5.6 | 7.3 / 5.5 / 5.7 |

### OpenScope 124 s

60 neurons, 2310 + 0 samples.

| family | readout | control | within gratings 124 s |
|---|---|---|---|
| whole cloud, plain | H0 |  | 34.0 / 23.0 / 22.9 |
| whole cloud, plain | H1 stable rank |  | 22.5 / 20.0 / 20.9 |
| whole cloud, plain | H1 Betti curve |  | 34.1 / 24.5 / 23.9 |
| whole cloud, plain | H0 + H1 stable rank |  | 36.4 / 22.3 / 23.2 |
| whole cloud, plain | H0 + H1 Betti curve |  | 41.2 / 25.8 / 26.2 |
| whole cloud, regions k=8 | H0 |  | 59.2 / 29.0 / 33.2 |
| whole cloud, regions k=8 | H1 stable rank |  | 40.3 / 26.2 / 26.6 |
| whole cloud, regions k=8 | H1 Betti curve |  | 41.8 / 26.0 / 27.0 |
| whole cloud, regions k=8 | H0 + H1 stable rank |  | 55.3 / 29.8 / 29.7 |
| whole cloud, regions k=8 | H0 + H1 Betti curve |  | 59.4 / 33.3 / 32.3 |
| whole cloud, regions k=16 | H0 |  | 63.5 / 31.8 / 35.6 |
| whole cloud, regions k=16 | H1 stable rank |  | 42.7 / 27.6 / 28.4 |
| whole cloud, regions k=16 | H1 Betti curve |  | 38.5 / 26.1 / 25.3 |
| whole cloud, regions k=16 | H0 + H1 stable rank |  | 58.5 / 31.0 / 31.7 |
| whole cloud, regions k=16 | H0 + H1 Betti curve |  | 60.4 / 29.4 / 33.4 |
| time-filtered r=0.5 (193 bars) | Betti curve over time |  | 13.8 / 16.6 / 15.9 |
| time-filtered r=0.5 (193 bars) | births so far |  | 16.0 / 16.4 / 16.8 |
| time-filtered r=0.5 (193 bars) | lifetimes (stable rank) |  | 14.3 / 16.2 / 15.6 |
| time-filtered r=0.5 (193 bars) | Betti + births |  | 22.4 / 18.8 / 19.7 |
| time-filtered r=1 (55 bars) | Betti curve over time |  | 10.6 / 13.6 / 13.9 |
| time-filtered r=1 (55 bars) | births so far |  | 10.1 / 14.6 / 14.4 |
| time-filtered r=1 (55 bars) | lifetimes (stable rank) |  | 7.5 / 13.9 / 13.3 |
| time-filtered r=1 (55 bars) | Betti + births |  | 14.9 / 16.7 / 17.4 |
| time-filtered r=2 (10 bars) | Betti curve over time |  | 7.0 / 12.6 / 9.6 |
| time-filtered r=2 (10 bars) | births so far |  | 7.4 / 11.3 / 11.3 |
| time-filtered r=2 (10 bars) | lifetimes (stable rank) |  | 6.5 / 11.9 / 10.9 |
| time-filtered r=2 (10 bars) | Betti + births |  | 7.9 / 12.0 / 11.4 |
| whole cloud, plain | H0 | shuffled | 29.2 / 21.6 / 21.5 |
| whole cloud, plain | H1 stable rank | shuffled | 21.5 / 21.3 / 21.1 |
| whole cloud, plain | H1 Betti curve | shuffled | 29.4 / 24.2 / 22.9 |
| whole cloud, plain | H0 + H1 stable rank | shuffled | 33.7 / 21.4 / 21.5 |
| whole cloud, plain | H0 + H1 Betti curve | shuffled | 36.1 / 24.5 / 25.1 |
| whole cloud, regions k=8 | H0 | shuffled | 49.6 / 28.9 / 29.4 |
| whole cloud, regions k=8 | H1 stable rank | shuffled | 35.5 / 25.5 / 26.8 |
| whole cloud, regions k=8 | H1 Betti curve | shuffled | 37.4 / 28.1 / 26.1 |
| whole cloud, regions k=8 | H0 + H1 stable rank | shuffled | 45.5 / 29.0 / 28.7 |
| whole cloud, regions k=8 | H0 + H1 Betti curve | shuffled | 49.0 / 29.2 / 29.4 |
| whole cloud, regions k=16 | H0 | shuffled | 53.5 / 32.0 / 30.0 |
| whole cloud, regions k=16 | H1 stable rank | shuffled | 36.8 / 26.4 / 27.6 |
| whole cloud, regions k=16 | H1 Betti curve | shuffled | 36.7 / 24.7 / 24.3 |
| whole cloud, regions k=16 | H0 + H1 stable rank | shuffled | 49.1 / 29.1 / 30.0 |
| whole cloud, regions k=16 | H0 + H1 Betti curve | shuffled | 51.8 / 31.4 / 29.0 |
| time-filtered r=0.5 (203 bars) | Betti curve over time | shuffled | 14.2 / 16.1 / 16.9 |
| time-filtered r=0.5 (203 bars) | births so far | shuffled | 15.6 / 18.3 / 17.5 |
| time-filtered r=0.5 (203 bars) | lifetimes (stable rank) | shuffled | 12.6 / 15.6 / 15.8 |
| time-filtered r=0.5 (203 bars) | Betti + births | shuffled | 19.5 / 20.0 / 19.4 |
| time-filtered r=1 (58 bars) | Betti curve over time | shuffled | 8.4 / 13.7 / 14.5 |
| time-filtered r=1 (58 bars) | births so far | shuffled | 10.4 / 15.0 / 14.8 |
| time-filtered r=1 (58 bars) | lifetimes (stable rank) | shuffled | 10.1 / 13.7 / 13.6 |
| time-filtered r=1 (58 bars) | Betti + births | shuffled | 13.9 / 17.1 / 17.7 |
| time-filtered r=2 (10 bars) | Betti curve over time | shuffled | 6.4 / 12.3 / 9.5 |
| time-filtered r=2 (10 bars) | births so far | shuffled | 6.1 / 12.4 / 11.9 |
| time-filtered r=2 (10 bars) | lifetimes (stable rank) | shuffled | 7.0 / 12.6 / 11.0 |
| time-filtered r=2 (10 bars) | Betti + births | shuffled | 7.4 / 13.1 / 12.7 |
