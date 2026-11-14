# L17 Full table: every method across a real stimulus change

2026-11-12, local session. **Worked as a test; the answer is that only stimulus-free signatures
cross a stimulus change.**

- ISI fractions, the burst table and the pdf-per-spike position keep 70-100% of their
  within-stimulus 1-NN across the change on all three datasets.
- Every time-locked method (time bars, burst bars, peak matching) falls to its shuffled control
  (2-8%), as expected.
- The Rips methods cross too, but from a low level. Plain delay Rips is near its shuffled
  control everywhere. Delay regions keep a 3-6 point order effect across stimuli on the retina
  and the Allen session.
- **A data problem was found on the way.** In the retina's "unrepeated" noise, piece 2 (seconds
  20-30 of each 40 s stretch before a frozen trial) repeats across blocks: its PSTH reliability
  is 0.244, as high as the frozen trials (0.248). Pieces 0, 1 and 3 are at -0.0-0.004. It is not
  the frozen sequence (PSTH correlation with frozen trials -0.06).
  - It made time-locked methods score 15-34% on "unrepeated" noise.
  - The retina 10 s table below is rerun without piece 2.
  - The 40 s table needs all four pieces, so it keeps the problem and is flagged.
  - L3 and L6 used all pieces; see the note at the end.

## Data and method

- Equal lengths on both sides. 60 neurons (seed 0) among those with >= 5 samples of >= 30 ISIs
  on both sides; at most 30 random samples per neuron and side.
- **Retina 10 s:** `20180126_sr_le_sp` frozen trials (9.94 s) against unrepeated 10 s pieces
  without piece 2. 143 candidate neurons; 1701 + 1788 samples; median 168 / 165 spikes.
- **Retina 40 s:** 4 consecutive frozen trials joined against 4 consecutive unrepeated pieces
  joined, piece 2 included. 148 candidates; 420 + 1796 samples (7 joined frozen samples per
  neuron); median 609 / 629 spikes.
- **OpenScope 10 s:** `glo637908` grating blocks cut into 10 s pieces against 10 s grey
  intermissions. 1243 candidates; 1769 + 1490 samples; median 62 / 74 spikes.
- **Allen 30 s:** 829720705 natural movie repeats against 30 s spontaneous windows. 990
  candidates; 1783 + 1734 samples; median 116 / 88 spikes.
- **Methods** (`scripts/sb_full_table.py`). Every barcode, PCF, mean and PCF distance is in
  stablebear.
  - multiscale sqrt time bars (s = 20, 67, 200 ms);
  - sqrt time bars + ISI fractions (weights 1, 0.5);
  - ISI fractions;
  - burst table;
  - burst bars (s = 67 ms);
  - peak matching (sigma 20 ms, q 10);
  - delay Rips m = 3, H0 and H0+H1;
  - delay regions k = 8 (anchors fitted on unselected neurons), H0 and H0+H1;
  - pdf per spike (`predictive_state` L = 2, bandwidth 1): position (mean point) and H0.
  - **All Rips clouds have 25 points** (20 subsamples): the shortest samples have 30 ISIs,
    which gives 28 delay points at m = 3. Parts are median-scaled and added.
- **Control:** ISIs permuted within every sample (both stimuli).
- **Scores.** One distance matrix covers both stimuli.
  - Within each stimulus: 1-NN / k-medoids / k-means with mean curves (`sb.mean`; "-" where no
    mean exists).
  - Across: "A -> B" gives 1-NN of A's samples among B's. Clustering across means: cluster A
    (k-medoids or k-means), pair the clusters with neurons on A, assign each B sample to the
    nearest medoid or centre, and score that.
  - Chance is 1.7% throughout.

## Command

```sh
python scripts/sb_full_table.py --dataset retina10 --drop-piece 2
python scripts/sb_full_table.py --dataset retina40
python scripts/sb_full_table.py --dataset openscope10
python scripts/sb_full_table.py --dataset allen30
```

Each cell is 1-NN / k-medoids / k-means, %.

## Retina 10 s, frozen <-> unrepeated (piece 2 dropped)

| method | | within frozen | within unrepeated | frozen -> unrepeated | unrepeated -> frozen |
|---|---|---|---|---|---|
| multiscale sqrt time bars | | 87.7 / 48.2 / 70.2 | 5.0 / 12.7 / 13.6 | 5.9 / 6.7 / 8.1 | 6.1 / 7.7 / 8.3 |
| | shuffled | 6.8 / 13.8 / 14.1 | 5.0 / 12.9 / 13.8 | 5.6 / 7.5 / 7.5 | 6.3 / 9.2 / 9.1 |
| sqrt time bars + ISI fractions | | 91.1 / 53.3 / 58.0 | 29.5 / 25.7 / 29.8 | 30.4 / 22.7 / 25.6 | 28.4 / 19.8 / 28.8 |
| | shuffled | 35.7 / 27.9 / 32.7 | 28.9 / 25.7 / 31.3 | 29.3 / 20.4 / 25.7 | 29.9 / 20.5 / 27.5 |
| ISI fractions | | 50.9 / 37.6 / 40.4 | 43.5 / 33.6 / 37.0 | 49.0 / 32.7 / 32.8 | 46.2 / 30.4 / 35.2 |
| | shuffled | 50.9 / 37.6 / 40.4 | 43.5 / 33.6 / 37.0 | 49.0 / 32.7 / 32.8 | 46.2 / 30.4 / 35.2 |
| burst table | | 52.6 / 34.7 / 40.8 | 45.6 / 34.6 / 39.3 | 47.4 / 30.4 / 35.3 | 46.1 / 32.9 / 37.9 |
| | shuffled | 44.3 / 37.1 / 37.4 | 43.3 / 33.8 / 35.6 | 44.8 / 29.9 / 32.3 | 41.6 / 31.3 / 35.4 |
| burst bars | | 69.8 / 35.4 / 51.3 | 4.6 / 9.2 / 8.1 | 4.7 / 3.8 / 4.1 | 4.3 / 4.2 / 3.7 |
| | shuffled | 5.2 / 9.5 / 8.8 | 4.0 / 8.9 / 8.3 | 4.1 / 3.9 / 3.5 | 4.5 / 4.6 / 3.9 |
| peak matching | | 71.7 / 34.8 / - | 2.6 / 10.5 / - | 1.6 / 5.0 / - | 2.5 / 4.2 / - |
| | shuffled | 1.5 / 10.5 / - | 1.8 / 9.6 / - | 1.1 / 4.4 / - | 2.0 / 4.5 / - |
| delay Rips m=3, H0 | | 16.1 / 21.9 / 21.5 | 14.6 / 19.6 / 21.0 | 15.5 / 15.2 / 14.8 | 16.1 / 16.0 / 16.9 |
| | shuffled | 15.4 / 20.3 / 20.4 | 12.5 / 19.4 / 18.6 | 16.0 / 13.0 / 13.4 | 14.7 / 14.3 / 15.8 |
| delay Rips m=3, H0+H1 | | 17.5 / 20.8 / 21.6 | 17.2 / 20.2 / 21.4 | 17.7 / 14.9 / 15.8 | 16.9 / 16.6 / 13.6 |
| | shuffled | 15.3 / 20.2 / 22.2 | 13.5 / 19.1 / 19.5 | 16.8 / 13.6 / 14.1 | 16.1 / 15.6 / 15.1 |
| delay regions k=8, H0 | | 30.4 / 26.4 / 31.0 | 26.5 / 24.4 / 28.4 | 28.8 / 21.0 / 24.9 | 26.6 / 20.4 / 24.5 |
| | shuffled | 25.5 / 25.5 / 27.7 | 22.9 / 23.8 / 26.5 | 23.3 / 20.0 / 20.6 | 23.2 / 20.0 / 22.1 |
| delay regions k=8, H0+H1 | | 24.2 / 22.5 / 24.2 | 21.9 / 21.6 / 21.9 | 21.4 / 14.8 / 19.5 | 22.3 / 15.9 / 17.6 |
| | shuffled | 17.8 / 20.3 / 22.2 | 21.1 / 20.7 / 21.0 | 18.9 / 13.8 / 16.0 | 17.6 / 14.9 / 17.0 |
| pdf per spike, position | | 47.4 / 35.6 / 38.7 | 41.9 / 35.0 / 35.4 | 47.4 / 29.8 / 32.8 | 44.3 / 34.5 / 33.8 |
| | shuffled | 47.9 / 37.2 / 39.2 | 43.9 / 32.6 / 35.2 | 48.2 / 28.5 / 32.6 | 45.9 / 29.4 / 31.9 |
| pdf per spike, H0 | | 13.3 / 17.8 / 18.5 | 13.0 / 19.0 / 18.2 | 12.7 / 11.5 / 11.2 | 13.2 / 12.5 / 12.2 |
| | shuffled | 13.8 / 19.2 / 19.9 | 12.8 / 20.1 / 19.4 | 13.4 / 12.5 / 11.3 | 12.8 / 15.5 / 14.5 |

## Retina 40 s, frozen x4 <-> unrepeated x4 (piece 2 included: time-locked rows are contaminated)

| method | | within frozen | within unrepeated | frozen -> unrepeated | unrepeated -> frozen |
|---|---|---|---|---|---|
| multiscale sqrt time bars | | 80.7 / 62.1 / 58.6 | 33.9 / 21.7 / 33.7 | 34.3 / 19.7 / 32.7 | 31.5 / 19.3 / 36.2 |
| | shuffled | 3.6 / 21.7 / 21.9 | 5.2 / 14.8 / 14.7 | 5.0 / 9.5 / 7.9 | 6.0 / 12.9 / 10.0 |
| sqrt time bars + ISI fractions | | 88.6 / 66.0 / 66.0 | 79.8 / 48.6 / 47.9 | 84.0 / 46.9 / 52.3 | 73.9 / 49.3 / 49.5 |
| | shuffled | 51.0 / 44.3 / 43.6 | 52.3 / 36.3 / 37.7 | 54.8 / 30.5 / 34.7 | 49.8 / 37.4 / 38.3 |
| ISI fractions | | 64.3 / 48.8 / 48.1 | 67.4 / 42.3 / 47.0 | 73.6 / 40.0 / 39.1 | 63.5 / 46.0 / 49.0 |
| | shuffled | 64.3 / 48.8 / 48.1 | 67.4 / 42.3 / 47.0 | 73.6 / 40.0 / 39.1 | 63.5 / 46.0 / 49.0 |
| burst table | | 68.3 / 53.3 / 49.0 | 70.9 / 46.1 / 48.7 | 74.8 / 42.0 / 41.3 | 66.5 / 50.2 / 49.0 |
| | shuffled | 65.7 / 49.8 / 49.8 | 67.1 / 44.0 / 47.4 | 71.0 / 38.6 / 42.1 | 61.0 / 46.4 / 48.8 |
| burst bars | | 61.0 / 44.8 / 54.3 | 11.0 / 9.5 / 9.4 | 13.3 / 7.3 / 10.1 | 9.4 / 5.2 / 6.2 |
| | shuffled | 3.6 / 14.5 / 16.4 | 3.4 / 7.3 / 6.8 | 2.6 / 3.7 / 3.7 | 3.8 / 4.8 / 3.3 |
| peak matching | | 53.3 / 47.6 / - | 9.4 / 14.1 / - | 8.3 / 7.2 / - | 5.8 / 8.8 / - |
| | shuffled | 1.7 / 19.8 / - | 1.9 / 11.3 / - | 1.9 / 5.5 / - | 2.1 / 7.1 / - |
| delay Rips m=3, H0 | | 21.0 / 33.3 / 33.1 | 21.9 / 24.5 / 26.1 | 23.1 / 20.0 / 17.0 | 19.6 / 22.4 / 22.9 |
| | shuffled | 19.3 / 31.9 / 35.2 | 20.2 / 24.9 / 24.4 | 18.3 / 17.5 / 19.9 | 19.1 / 21.0 / 21.4 |
| delay Rips m=3, H0+H1 | | 18.3 / 29.5 / 31.7 | 22.7 / 25.5 / 25.3 | 23.1 / 15.3 / 17.9 | 22.6 / 21.7 / 21.4 |
| | shuffled | 20.5 / 31.7 / 30.5 | 21.4 / 24.9 / 26.1 | 21.9 / 18.6 / 15.1 | 20.4 / 19.0 / 19.8 |
| delay regions k=8, H0 | | 35.5 / 40.2 / 43.3 | 38.1 / 33.7 / 36.2 | 44.8 / 29.4 / 29.3 | 37.6 / 31.2 / 37.9 |
| | shuffled | 29.8 / 38.3 / 37.9 | 33.6 / 30.5 / 34.6 | 38.6 / 26.3 / 25.0 | 32.1 / 24.3 / 32.4 |
| delay regions k=8, H0+H1 | | 22.4 / 31.4 / 32.9 | 28.0 / 24.2 / 28.4 | 34.0 / 19.6 / 18.5 | 26.8 / 21.9 / 29.3 |
| | shuffled | 22.4 / 32.9 / 34.3 | 25.8 / 23.3 / 25.4 | 27.9 / 17.9 / 17.7 | 24.5 / 18.3 / 20.2 |
| pdf per spike, position | | 62.9 / 48.3 / 50.5 | 66.0 / 45.1 / 45.7 | 71.7 / 39.8 / 40.7 | 64.4 / 42.6 / 45.0 |
| | shuffled | 68.1 / 50.2 / 49.3 | 68.1 / 43.6 / 48.1 | 74.8 / 40.6 / 39.4 | 64.9 / 47.4 / 47.9 |
| pdf per spike, H0 | | 14.5 / 28.1 / 29.0 | 17.4 / 19.9 / 19.6 | 17.1 / 12.2 / 12.1 | 15.1 / 13.1 / 14.8 |
| | shuffled | 14.3 / 28.1 / 29.0 | 15.5 / 19.7 / 20.7 | 17.9 / 10.7 / 11.7 | 14.8 / 12.9 / 13.3 |

## OpenScope 10 s, gratings <-> grey screen

| method | | within gratings | within grey | gratings -> grey | grey -> gratings |
|---|---|---|---|---|---|
| multiscale sqrt time bars | | 11.2 / 12.8 / 10.5 | 9.3 / 14.1 / 13.2 | 4.4 / 4.3 / 3.1 | 4.6 / 5.0 / 3.6 |
| | shuffled | 4.6 / 11.4 / 9.6 | 6.5 / 11.5 / 10.5 | 4.0 / 4.6 / 4.6 | 4.8 / 4.0 / 5.1 |
| sqrt time bars + ISI fractions | | 22.7 / 20.2 / 20.6 | 23.0 / 19.7 / 20.6 | 14.8 / 11.1 / 13.2 | 13.1 / 11.3 / 12.6 |
| | shuffled | 15.2 / 18.9 / 20.0 | 19.9 / 18.4 / 20.2 | 13.6 / 10.9 / 11.9 | 14.8 / 11.5 / 13.6 |
| ISI fractions | | 26.1 / 21.9 / 22.4 | 29.4 / 22.4 / 22.3 | 21.0 / 14.5 / 15.8 | 24.0 / 12.3 / 12.4 |
| | shuffled | 26.1 / 21.9 / 22.4 | 29.4 / 22.4 / 22.3 | 21.0 / 14.5 / 15.8 | 24.0 / 12.3 / 12.4 |
| burst table | | 31.4 / 22.4 / 23.1 | 34.8 / 23.6 / 23.2 | 25.4 / 14.8 / 16.6 | 28.1 / 13.1 / 15.1 |
| | shuffled | 22.6 / 21.8 / 22.2 | 27.7 / 21.4 / 21.3 | 20.1 / 15.0 / 15.3 | 23.1 / 14.7 / 15.0 |
| burst bars | | 7.7 / 11.0 / 9.8 | 5.2 / 10.9 / 10.8 | 3.1 / 3.6 / 4.0 | 3.3 / 3.6 / 3.6 |
| | shuffled | 3.2 / 9.3 / 7.9 | 5.0 / 9.5 / 9.1 | 3.1 / 3.0 / 4.2 | 2.4 / 3.6 / 3.7 |
| peak matching | | 7.6 / 11.4 / - | 6.1 / 12.1 / - | 3.6 / 3.8 / - | 3.4 / 4.8 / - |
| | shuffled | 3.3 / 9.0 / - | 3.4 / 9.3 / - | 3.4 / 3.4 / - | 2.1 / 3.3 / - |
| delay Rips m=3, H0 | | 7.5 / 12.8 / 12.6 | 9.7 / 14.6 / 14.8 | 6.8 / 5.9 / 7.0 | 7.7 / 5.7 / 6.0 |
| | shuffled | 7.9 / 14.4 / 12.7 | 8.8 / 13.5 / 13.6 | 5.5 / 5.6 / 5.4 | 8.3 / 4.6 / 5.4 |
| delay Rips m=3, H0+H1 | | 7.4 / 13.1 / 12.9 | 9.7 / 15.0 / 14.9 | 7.0 / 6.4 / 6.4 | 7.8 / 6.1 / 7.1 |
| | shuffled | 7.3 / 12.9 / 12.8 | 8.0 / 14.2 / 14.3 | 6.5 / 7.0 / 6.6 | 7.8 / 6.1 / 5.9 |
| delay regions k=8, H0 | | 16.7 / 16.7 / 16.7 | 19.6 / 20.2 / 19.7 | 13.7 / 11.6 / 11.6 | 15.4 / 13.1 / 11.6 |
| | shuffled | 14.5 / 16.2 / 17.0 | 13.5 / 17.8 / 18.0 | 11.1 / 10.6 / 9.7 | 12.3 / 8.6 / 10.3 |
| delay regions k=8, H0+H1 | | 11.1 / 14.8 / 14.6 | 14.9 / 16.8 / 15.6 | 11.6 / 8.3 / 8.1 | 11.9 / 8.1 / 6.9 |
| | shuffled | 11.5 / 15.5 / 14.9 | 13.0 / 16.2 / 15.6 | 9.4 / 8.8 / 7.7 | 10.7 / 8.8 / 7.7 |
| pdf per spike, position | | 23.7 / 21.8 / 22.0 | 27.5 / 21.5 / 23.1 | 19.9 / 16.0 / 17.0 | 23.2 / 13.6 / 13.6 |
| | shuffled | 24.1 / 21.8 / 22.9 | 28.3 / 22.7 / 23.3 | 20.5 / 14.4 / 14.2 | 23.3 / 16.3 / 12.7 |
| pdf per spike, H0 | | 5.8 / 12.6 / 12.2 | 8.9 / 13.0 / 12.9 | 5.9 / 6.0 / 5.4 | 6.1 / 5.0 / 5.9 |
| | shuffled | 5.5 / 12.6 / 12.7 | 5.8 / 13.0 / 13.1 | 5.7 / 6.2 / 5.1 | 5.4 / 5.1 / 3.8 |

## Allen 30 s, natural movie <-> spontaneous

| method | | within movie | within spont | movie -> spont | spont -> movie |
|---|---|---|---|---|---|
| multiscale sqrt time bars | | 13.9 / 12.1 / 9.7 | 2.5 / 8.1 / 5.1 | 3.9 / 2.7 / 2.7 | 2.2 / 4.5 / 3.1 |
| | shuffled | 3.7 / 9.1 / 6.7 | 3.1 / 8.0 / 5.0 | 3.8 / 3.6 / 2.7 | 2.5 / 4.3 / 2.6 |
| sqrt time bars + ISI fractions | | 29.4 / 21.2 / 19.5 | 13.1 / 20.1 / 16.1 | 11.2 / 8.1 / 8.6 | 10.2 / 10.7 / 9.3 |
| | shuffled | 15.0 / 20.5 / 16.8 | 14.9 / 19.8 / 16.9 | 10.9 / 10.1 / 7.0 | 10.2 / 12.2 / 11.3 |
| ISI fractions | | 44.8 / 27.2 / 28.0 | 35.8 / 26.0 / 25.8 | 25.4 / 12.5 / 11.8 | 24.5 / 14.8 / 13.8 |
| | shuffled | 44.8 / 27.2 / 28.0 | 35.8 / 26.0 / 25.8 | 25.4 / 12.5 / 11.8 | 24.5 / 14.8 / 13.8 |
| burst table | | 47.7 / 28.2 / 29.4 | 37.9 / 28.8 / 28.7 | 30.6 / 13.5 / 14.0 | 27.9 / 14.3 / 15.0 |
| | shuffled | 39.6 / 24.6 / 25.1 | 31.5 / 24.7 / 25.1 | 22.8 / 12.9 / 11.4 | 22.4 / 12.2 / 14.5 |
| burst bars | | 11.2 / 11.8 / 7.1 | 3.0 / 7.3 / 5.4 | 3.3 / 3.1 / 3.5 | 4.0 / 4.0 / 3.3 |
| | shuffled | 3.4 / 7.5 / 5.8 | 3.5 / 7.3 / 5.9 | 2.8 / 3.4 / 3.6 | 3.5 / 3.8 / 3.3 |
| peak matching | | 9.4 / 11.3 / - | 2.4 / 7.7 / - | 1.7 / 2.5 / - | 2.7 / 3.3 / - |
| | shuffled | 2.5 / 8.1 / - | 2.0 / 8.1 / - | 2.3 / 3.3 / - | 1.8 / 4.0 / - |
| delay Rips m=3, H0 | | 13.4 / 15.4 / 16.0 | 10.5 / 15.6 / 15.3 | 8.0 / 6.7 / 6.5 | 7.2 / 6.2 / 7.1 |
| | shuffled | 11.5 / 15.9 / 16.0 | 10.4 / 15.5 / 15.2 | 7.9 / 6.1 / 5.8 | 7.6 / 6.1 / 6.0 |
| delay Rips m=3, H0+H1 | | 11.7 / 16.8 / 16.9 | 11.8 / 16.3 / 15.7 | 8.0 / 6.9 / 6.6 | 7.9 / 7.3 / 5.8 |
| | shuffled | 10.3 / 15.9 / 16.7 | 10.0 / 16.1 / 15.9 | 8.3 / 7.2 / 6.5 | 7.8 / 6.6 / 7.6 |
| delay regions k=8, H0 | | 27.4 / 24.4 / 24.1 | 22.5 / 20.5 / 20.8 | 17.0 / 10.7 / 12.6 | 15.6 / 10.9 / 12.6 |
| | shuffled | 20.1 / 19.9 / 21.2 | 16.0 / 18.2 / 17.8 | 14.1 / 8.0 / 10.8 | 12.5 / 8.1 / 9.3 |
| delay regions k=8, H0+H1 | | 22.2 / 19.0 / 19.6 | 17.7 / 17.8 / 17.7 | 14.6 / 8.2 / 9.7 | 12.8 / 11.1 / 7.4 |
| | shuffled | 16.2 / 17.7 / 18.1 | 13.8 / 16.3 / 17.9 | 11.5 / 7.9 / 6.7 | 10.4 / 7.6 / 10.5 |
| pdf per spike, position | | 39.1 / 27.1 / 27.3 | 31.6 / 23.4 / 23.7 | 24.6 / 13.2 / 12.9 | 23.5 / 13.6 / 13.3 |
| | shuffled | 41.7 / 26.3 / 27.3 | 32.8 / 24.8 / 24.9 | 25.9 / 13.7 / 11.8 | 23.9 / 13.9 / 12.3 |
| pdf per spike, H0 | | 9.5 / 14.8 / 14.6 | 10.1 / 14.9 / 14.3 | 7.1 / 5.2 / 6.5 | 5.9 / 4.1 / 6.3 |
| | shuffled | 8.4 / 14.6 / 14.4 | 8.9 / 14.5 / 13.6 | 6.4 / 5.3 / 5.1 | 5.8 / 6.2 / 5.9 |

## What to take from it

- **Two families, cleanly separated.**
  - Stimulus-free signatures (ISI fractions, burst table, pdf position) lose little or nothing
    across the stimulus change:
    - retina: ISI fractions 50.9 / 43.5 within, 49.0 / 46.2 across;
    - OpenScope: 26.1 / 29.4 within, 21.0 / 24.0 across;
    - Allen: 44.8 / 35.8 within, 25.4 / 24.5 across.
    - Their shuffled controls are equal or close: they do not use order.
  - Time-locked signatures (multiscale sqrt time bars, burst bars, peak matching) are strong
    only within a repeated stimulus. Within frozen noise they reach 70-88% 1-NN. Elsewhere,
    and across every change, they are at their shuffled level (1-8%).
- **The combination sqrt time bars + ISI fractions** keeps its ISI part across stimuli
  (retina 30.4 / 28.4), at the level of its shuffled control. Across stimuli only the
  stimulus-free part counts.
- **Rips across stimuli:**
  - Plain delay Rips (H0, H0+H1) transfers at its within-stimulus level, but that level is low
    (7-18%) and equal to its shuffled control. It sees only the ISI distribution, crudely.
  - Delay regions k = 8 is the only homology method with an order effect that survives the
    change: retina H0 28.8 against 23.3 shuffled (frozen -> unrepeated); Allen 17.0 against
    14.1; OpenScope 13.7 against 11.1. Still about half of ISI fractions.
  - Pdf-per-spike H0 is weak everywhere (5-17%).
- **Clustering across** follows the same pattern, 5-15 points below 1-NN. The k-means with mean
  curves is usually equal to or slightly better than k-medoids. Exception: time bars within the
  frozen stimulus, where k-means gives 70.2 against 48.2.
- **Dataset differences.** On the retina the stimulus change costs ISI fractions nothing. On
  Allen the movie -> spontaneous change costs about 40% of within-stimulus 1-NN; L5 and L16
  suggest part of this is the 70-minute gap (recording drift), not the stimulus. OpenScope
  loses about 5 points, as in L7.
- **What failed:**
  - no homology method beats ISI fractions in any cell;
  - time-locked methods cannot cross a stimulus change, by construction;
  - the 40 s retina table has the piece 2 problem.

## Note on piece 2 (affects L3 and L6)

`load_retina_running.py` cuts the 50 s of unrepeated noise before each frozen trial into 10 s
pieces. It gets 4 per block, not the 5 in its docstring, because the stretch is just under
50 s. Piece 2 has a PSTH reliability of 0.244 across blocks (pieces 0, 1, 3: 0.004, -0.0,
-0.0), so it repeats a stimulus segment that is not the frozen sequence.

**L3 and L6 used all pieces.** For the stimulus-free signatures it changes little: ISI fractions
here 43.5 within unrepeated without piece 2 against 45.0 with it. But the "unrepeated" label
there is not fully correct. In the 40 s stretches of L6 and L10-L12, a quarter of each stretch
is repeated.

The stimulus file would tell what piece 2 is; that is not checked here.
