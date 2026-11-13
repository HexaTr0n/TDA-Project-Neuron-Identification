# L24 Three data checks

2026-11-13, local session. **Worked: three answers.**

1. **Retina piece 2 is a fixed 5 s chunk.** It sits at 24.85-29.9 s of every running stretch and
   is neither the frozen sequence nor a copy of another part. With it cut out:
   - the 40 s "unrepeated" time-locked scores fall from 34% to the shuffled level (5%);
   - ISI-based scores barely move (+1-3 points).
   - The 40 s retina tables need redoing only for time-locked rows; a clean loader now exists.
2. **On Allen, drift beats the stimulus change for single-neuron signatures.** The same movie
   70 min later is recognised worse than a different stimulus just before it: ISI fractions
   11.8% 1-NN against 38.7%.
3. **A second Allen session reproduces the first.**
   - ISI fractions 46.1 within / 28.0 across, against 44.8 / 25.4.
   - Burst table 54.2 / 31.6, against 47.7 / 30.6.
   - Whole-cloud delay regions (k = 16, H0) 43.6 / 29.1, against 38.4 / 25.4.
   - Time-locked square-root time bars are higher on the movie (24.5% against 13.9%).

## 1. Retina piece 2

**Data:** retina `20180126_sr_le_sp`, 156 cells, 30 blocks. Each running (unrepeated) stretch is
49.65 s (R = 1500 pulses at 33.1 ms) before a 9.93 s frozen trial. Raw spike times and frame
pulses come from `extracted/20180126_sr_le_sp`; stimulus parameters come from
`5_frozennoise8x8bw1blink3000run600freeze.mat` (RunningFrames 1500, FrozenFrames 300,
Nblinks 1, seed -1000, secondseed -10000).

**What it is** (script in the session scratchpad, not committed; method as in the board answer):
- PSTH reliability across the 30 blocks (50 ms bins, mean pairwise correlation per cell,
  averaged over cells), in windows over the whole stretch:
  - 1 s windows: 0.000 +- 0.004 everywhere, except 0.05 in the first second, and 0.43-0.49 from
    25 to 30 s;
  - 0.2 s windows: the edges are at 24.85 s (0.13) and 29.9 s (0.25), with 0.45-0.50 in
    between.
- So the repeat is 5 s = 150 pulses = 300 frames, the middle 300 of the stretch's 3000 frames,
  identical in all 30 blocks.
- It is not the frozen sequence: the PSTH correlation after removing each cell's mean, at any
  lag, is at most 0.018 (median -0.002).
- It is not another part of the stretch either: at most 0.037 outside its own position.
- The pulse intervals inside the stretches are regular (at most 33.3 ms).
- The manual (`data/manual.pdf`, section 2.1) says running noise resumes from the last seed after
  each frozen run, so a repeated 300-frame chunk is not the documented protocol. The files do not
  say why it happens. Regenerating the frames from the seeds with Gollisch lab's recreation code
  would settle it; not done.

**Fix:** `scripts/load_retina_running.py --cut 24.8 30.0` drops those seconds from every stretch
and joins the rest (44.45 s, i.e. 4 clean 10 s pieces). It writes
`data/retina_running_<session>_clean.parquet`, used as dataset `retina40clean` in
`sb_full_table.py`. Reliability of the 4 clean pieces: 0.005, 0.000, -0.000, 0.000.

```sh
python scripts/load_retina_running.py --extracted-dir extracted --session 20180126_sr_le_sp --cut 24.8 30.0
python scripts/sb_full_table.py --dataset retina40clean --skip-peaks --only "ISI fractions" "burst table" \
    "delay regions k=8, H0" "multiscale sqrt time bars" "burst bars"
```

**Retina 40 s, frozen x4 <-> unrepeated x4, before (L17) and after the cut.**
- Same neurons and selection as L17: 60 neurons, 420 + 1796 samples.
- Each cell is 1-NN / k-medoids / k-means within; 1-NN across; %.
- The frozen side is unchanged, so its scores are identical.

| method | | within unrepeated, L17 | within unrepeated, clean | frozen -> unrepeated, L17 / clean | unrepeated -> frozen, L17 / clean |
|---|---|---|---|---|---|
| multiscale sqrt time bars | real | 33.9 / 21.7 / 33.7 | **5.1** / 15.2 / 14.8 | 34.3 / **7.4** | 31.5 / **6.0** |
| | shuffled | 5.2 / 14.8 / 14.7 | 4.8 / 15.5 / 14.7 | 5.0 / 4.8 | 6.0 / 5.5 |
| burst bars | real | 11.0 / 9.5 / 9.4 | **2.7** / 7.2 / 7.0 | 13.3 / **4.5** | 9.4 / **3.2** |
| | shuffled | 3.4 / 7.3 / 6.8 | 3.1 / 7.5 / 6.8 | 2.6 / 4.5 | 3.8 / 3.3 |
| ISI fractions (shuffled identical) | real | 67.4 / 42.3 / 47.0 | 70.0 / 44.3 / 49.9 | 73.6 / 75.0 | 63.5 / 64.7 |
| burst table | real | 70.9 / 46.1 / 48.7 | 70.8 / 46.8 / 49.7 | 74.8 / 76.0 | 66.5 / 66.6 |
| | shuffled | 67.1 / 44.0 / 47.4 | 66.8 / 44.8 / 45.6 | 71.0 / 71.9 | 61.0 / 62.2 |
| delay regions k=8, H0 (25 points) | real | 38.1 / 33.7 / 36.2 | 38.1 / 31.8 / 35.8 | 44.8 / 47.6 | 37.6 / 38.2 |
| | shuffled | 33.6 / 30.5 / 34.6 | 36.5 / 30.0 / 32.6 | 38.6 / 38.3 | 32.1 / 31.6 |

- **Verdict:** time-locked rows on the retina's 40 s unrepeated noise must use the clean data;
  their old 31-34% was the repeated chunk.
- Stimulus-free rows change by at most 3 points. L6, L10-L12 and L17's 40 s conclusions stand.
  L19, L19x and L20 retina 40 s used the old data; their numbers are ISI-based and so close to
  right.
- L22 should use `retina40clean`.

## 2. Allen: stimulus change against drift

**Data:** Allen 829720705, the 60 neurons of L17 (`sb_full_table.choose`, seed 0). Per pair, the
neurons with >= 3 samples of >= 30 ISIs on both sides, and all their samples.

| group | time in session | windows |
|---|---|---|
| movie 1 | 38.2-53.2 min | 30 repeats |
| movie 2 | 123.2-138.3 min | 30 repeats |
| spontaneous early | 0.1-38.2 min (16-38 min, plus two at 0-1 min) | 17 |
| spontaneous late | 73.2-103.2 min | 60 |

**Signatures:**
- ISI fractions;
- burst table;
- delay regions k = 8 H0: the L17 version (25 random points, 20 draws), and the whole cloud
  (every point, as L20);
- ISIs shuffled as the control.

Scores as L17. Each cell is 1-NN within each side; 1-NN across in both directions; k-medoids
across (cluster one side, assign the other).

```sh
python scripts/sb_allen_drift.py
```

| pair | what changes | neurons, samples | ISI fractions within / across | burst table within / across (shuffled) | delay regions, whole cloud, within / across (shuffled) |
|---|---|---|---|---|---|
| movie 1 <-> spont early | stimulus; 0-38 min apart | 55; 1551 + 781 | 50.7, 41.1 / **38.7, 33.5** | 56.7, 47.2 / 42.4, 38.3 (34.8, 31.6) | 44.2, 38.4 / 32.4, 28.9 (23.1, 20.4) |
| movie 2 <-> spont late | stimulus; 20-65 min apart | 52; 1444 + 2669 | 55.4, 47.1 / 24.6, 17.7 | 58.0, 50.4 / 26.0, 18.5 (23.9, 16.0) | 47.3, 37.4 / 22.5, 14.3 (17.9, 12.6) |
| movie 1 <-> movie 2 | nothing but 70-100 min | 51; 1405 + 1421 | 50.0, 56.8 / **11.8, 12.0** | 54.2, 59.0 / 14.6, 17.0 (10.6, 12.9) | 43.8, 48.2 / 14.6, 15.3 (8.5, 8.9) |
| spont early <-> spont late | nothing but 35-100 min | 55; 781 + 2772 | 41.1, 45.4 / 23.2, 24.0 | 47.2, 48.4 / 28.3, 26.3 (23.4, 20.6) | 38.4, 34.7 / 19.0, 19.2 (16.0, 15.2) |

Clustering across (k-medoids, both directions):

| pair | ISI fractions | burst table | delay regions (25 points) | delay regions (whole cloud) |
|---|---|---|---|---|
| movie 1 <-> spont early | 22.2 / 19.9 | 23.2 / 21.5 | 14.2 / 15.7 | 14.3 / 15.2 |
| movie 2 <-> spont late | 13.0 / 15.3 | 12.7 / 14.8 | 8.4 / 11.6 | 10.7 / 7.9 |
| movie 1 <-> movie 2 | 6.3 / 10.1 | 5.9 / 10.2 | 10.1 / 10.0 | 10.5 / 7.8 |
| spont early <-> spont late | 11.9 / 11.3 | 13.4 / 14.7 | 11.1 / 12.4 | 10.0 / 11.8 |

- **Time apart, not the stimulus, sets the cross score.** Ordered by time apart, ISI fractions
  across go 38.7 (adjacent) > 24.6 / 23.2 (tens of minutes) > 11.8 (70-100 min, same movie).
  This is L16's population result again, now for single-neuron signatures.
- **The whole-cloud delay regions follow the same order.** They keep 6-9 points over their
  shuffled control in every pair, so ISI order survives drift too.
- **For the L17 Allen cross numbers** (25.4% for ISI fractions), part of the loss is drift: the
  movie and spontaneous blocks there are tens of minutes apart on average.
- Not done: a correction for drift (for example re-centring each unit's signature per block).

## 3. A second cortical session

**Data:** Allen Visual Coding functional connectivity, session 789848216 (DANDI 000022,
`sub-763808588_ses-789848216.nwb`, 1.9 GB). Same protocol as 829720705.
- Converted with `scripts/load_allen.py`: 1079 good units (CA1 279, grey 121, DG 100, VISam 94,
  VISl 90, CA3 57, LD 55, VISrl 54, LP 53, VISp 44, ...);
- 60 movie repeats and 77 spontaneous windows of 30.02 s;
- median 91 / 72 spikes.
- Dataset `allen30b` in `sb_full_table.py`, with the L17 selection: 858 candidates, 60 kept,
  1741 + 1650 samples, median 181 / 171 spikes.

```sh
dandi download "https://api.dandiarchive.org/api/dandisets/000022/versions/draft/assets/?path=sub-763808588/sub-763808588_ses-789848216.nwb"
python scripts/load_allen.py data/allen/sub-763808588_ses-789848216.nwb
python scripts/sb_full_table.py --dataset allen30b --skip-peaks --only "ISI fractions" "burst table" \
    "delay regions k=8, H0" "multiscale sqrt time bars"
python scripts/sb_cloud_sampling.py --dataset allen30b --modes whole
```

Each cell is 1-NN / k-medoids / k-means; %.

| method | | within movie | within spontaneous | movie -> spont | spont -> movie | session 1 (L17, L20): within movie / movie -> spont, 1-NN |
|---|---|---|---|---|---|---|
| ISI fractions (shuffled identical) | real | 46.1 / 28.0 / 29.3 | 45.4 / 32.4 / 30.5 | 28.0 / 16.3 / 16.5 | 29.9 / 16.0 / 13.6 | 44.8 / 25.4 |
| burst table | real | 54.2 / 29.6 / 30.6 | 51.8 / 33.5 / 34.7 | 31.6 / 17.5 / 18.4 | 33.5 / 17.2 / 16.8 | 47.7 / 30.6 |
| | shuffled | 44.0 / 29.9 / 29.8 | 42.6 / 30.8 / 32.0 | 26.8 / 15.3 / 17.8 | 28.2 / 16.9 / 15.0 | 39.6 / 22.8 |
| multiscale sqrt time bars | real | 24.5 / 18.5 / 16.3 | 5.4 / 12.3 / 11.2 | 6.8 / 7.0 / 6.4 | 6.1 / 7.1 / 5.5 | 13.9 / 3.9 |
| | shuffled | 6.2 / 11.6 / 10.6 | 5.2 / 12.2 / 10.4 | 5.9 / 8.2 / 5.5 | 5.6 / 6.7 / 4.9 | 3.7 / 3.8 |
| delay regions k=8, H0 (25 points) | real | 32.9 / 25.8 / 26.8 | 33.5 / 30.3 / 29.9 | 21.0 / 16.5 / 15.8 | 24.7 / 14.2 / 14.9 | 30.0 / 21.5 |
| | shuffled | 27.3 / 23.8 / 24.9 | 25.4 / 26.5 / 28.1 | 17.3 / 14.7 / 15.9 | 21.2 / 14.9 / 16.7 | |
| delay regions, whole cloud, k=8, H0 | real | 42.4 / 30.1 / 29.1 | 40.9 / 31.5 / 30.2 | 26.5 / 17.3 / 19.2 | 26.2 / 14.7 / 15.8 | 34.4 / 22.7 |
| | shuffled | 35.4 / 28.1 / 29.1 | 35.9 / 28.6 / 28.7 | 22.7 / 17.9 / 13.9 | 24.0 / 15.9 / 15.5 | 26.5 |
| delay regions, whole cloud, k=16, H0 | real | 43.6 / 30.3 / 29.8 | 42.3 / 28.4 / 29.8 | 29.1 / 14.5 / 17.5 | 29.3 / 17.1 / 16.8 | 38.4 / 25.4 |
| | shuffled | 36.2 / 28.4 / 29.2 | 36.0 / 29.3 / 29.6 | 24.1 / 15.3 / 14.8 | 23.9 / 15.9 / 16.9 | 29.8 |
| delay regions, whole cloud, k=32, H0 | real | 43.6 / 29.0 / 31.1 | 43.6 / 29.5 / 32.4 | 30.0 / 20.6 / 19.1 | 30.6 / 16.0 / 20.4 | 39.0 / 27.9 |
| | shuffled | 34.8 / 29.1 / 28.7 | 35.8 / 28.2 / 29.2 | 25.8 / 16.8 / 17.4 | 25.3 / 15.1 / 15.8 | 30.0 |
| delay Rips m=3, whole cloud, H0 | real | 27.6 / 25.5 / 26.1 | 25.5 / 25.3 / 26.0 | 15.7 / 11.5 / 12.9 | 17.5 / 12.7 / 12.1 | 20.7 / 12.1 |

- **The cortical conclusions hold on a second animal.**
  - The ISI signatures give 46-54% within and 28-34% across.
  - Whole-cloud delay regions (k = 16-32) are 2-11 points behind them, with 6-9 points of order
    over the shuffled control.
  - Plain delay Rips is far lower.
  - Time-locked signatures fall to their shuffled level across stimuli.
- **Session 2 is a few points easier everywhere** (+1 to +7). It has more spikes per sample
  (median 181 against 116). Time bars on the movie are higher (24.5 against 13.9), so this
  session has more stimulus-locked units.
- Not done: the spike distance, L19's cloud and the other L20 sampling modes on this session.
