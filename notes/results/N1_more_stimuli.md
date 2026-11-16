# N1 More stimuli for the report

2026-11-16, local session. **Neuropixels done (five stimuli in the pool). Retina converter
written and tested, then stopped at the user's request; the retina keeps its two noise stimuli.**

Work and results are on branch `night-run` (commit 350d3ce). This note is the report.

## Neuropixels (Allen Visual Coding, functional connectivity set, session 829720705)

**Converter.** I ran `python scripts/load_allen.py data/allen/sub-811322604_ses-829720705.nwb
--all-stimuli`.
- Every stimulus of the session appears, except receptive-field mapping (excluded on purpose)
  and invalid times.
- The window counts match the block lengths.
- The movie and grey-screen rows are identical to the old two-stimulus table: 184,128 rows,
  with the same starts and spike counts.
- No two windows overlap.
- The converter needed no change.
- The old table is kept as `data/allen/allen829720705_two_stimuli.parquet`. The new one
  replaces `allen829720705.parquet`. Data tables are not in git.

**Full stimulus list** (minutes from the start of the recording; windows of 30.02 s):

| stimulus | blocks (min) | windows | in the pool? |
|---|---|---|---|
| receptive-field mapping (Gabors) | 1.1-16.3 | - | no (left out by the converter) |
| grey screen (spontaneous) | 0.1-1.1, 16.3-38.2, 73.2-103.2 | 77 | yes, 10 per neuron |
| flashes | 21.2-26.1 | 9 | **no: fewer than 10 windows** |
| drifting gratings, contrast series | 27.2-36.2 | 17 | yes, 10 |
| natural movie, block 1 (reference) | 38.2-53.2 | 30 repeats | yes, 30 |
| natural movie, shuffled frames | 53.2-58.2, 118.2-123.2 | 20 | yes, 10 (first block, within 30 min) |
| drifting gratings (75 repeats) | 58.2-73.2, 103.2-118.2 | 58 | yes, 10 |
| natural movie, block 2 | 123.2-138.3 | 30 repeats | no (too far from block 1) |
| dot motion | 138.3-152.3 | 27 | no (too far from block 1) |

- **Picked by `04_extended.py`** (default `--max-gap 30`): natural movie (30), drifting
  gratings (10), contrast gratings (10), shuffled movie (10), grey screen (10). `--max-gap 45`
  gives the same list, so it was not needed.
- **Flashes are left out** because the step 4 rule needs at least 10 windows and their block
  holds 9. I did not change the rule.

**Across table** (40 neurons, at least 1 Hz, 5 draws, mean +- std; chance 2.5%):

| method | 1-NN, natural movie only | k-medoids, natural movie only | 1-NN, all stimuli mixed | k-medoids, all stimuli mixed | 1-NN, other stimuli only |
|---|---|---|---|---|---|
| B1 firing rate | 12.5 ± 2.7 | 17.8 ± 0.9 | 7.9 ± 2.1 | 13.1 ± 1.0 | 7.9 ± 1.1 |
| B2 ISI histogram | 44.0 ± 3.6 | 29.5 ± 1.8 | 32.1 ± 3.4 | 19.8 ± 1.1 | 30.8 ± 2.9 |
| B3 Victor-Purpura, q = 0.3/s | 27.3 ± 1.6 | 21.8 ± 0.6 | 11.5 ± 0.8 | 13.0 ± 0.8 | 10.5 ± 1.6 |
| B3 Victor-Purpura, q = 1/s | 28.2 ± 1.1 | 21.0 ± 1.5 | 12.0 ± 0.6 | 12.5 ± 1.0 | 10.2 ± 1.7 |
| B3 Victor-Purpura, q = 3/s | 23.6 ± 1.2 | 17.7 ± 1.7 | 10.2 ± 1.1 | 10.9 ± 0.8 | 8.6 ± 1.8 |
| B3 Victor-Purpura, q = 10/s | 9.8 ± 2.1 | 9.8 ± 1.2 | 6.0 ± 1.3 | 8.0 ± 0.7 | 5.3 ± 1.8 |
| B3 Victor-Purpura, q = 30/s | 3.9 ± 1.1 | 5.1 ± 0.9 | 3.1 ± 0.8 | 4.7 ± 0.7 | 3.2 ± 0.6 |
| B3 Victor-Purpura, q = 100/s | 2.5 ± 0.1 | 3.9 ± 0.4 | 2.6 ± 0.2 | 3.3 ± 0.1 | 2.6 ± 0.1 |
| T1 LINE H0 | 31.3 ± 3.9 | 27.3 ± 2.8 | 20.3 ± 3.1 | 18.8 ± 1.9 | 19.1 ± 2.7 |
| T1 LINE H0, log scale | 42.7 ± 2.2 | 22.8 ± 1.1 | 28.4 ± 3.0 | 15.3 ± 1.3 | 27.1 ± 2.4 |
| T1 LINE H0, log scale, share | 52.2 ± 3.5 | 36.1 ± 2.8 | 37.5 ± 2.9 | 24.9 ± 2.0 | 35.3 ± 2.5 |
| T2 DELAY H0 | 31.1 ± 2.2 | 28.8 ± 2.4 | 20.9 ± 2.9 | 20.4 ± 2.2 | 19.5 ± 2.3 |
| T2 DELAY H1 | 17.6 ± 2.8 | 21.6 ± 1.8 | 12.1 ± 1.9 | 16.7 ± 1.8 | 11.7 ± 2.1 |
| T2 DELAY H0+H1 | 29.3 ± 2.7 | 26.7 ± 2.0 | 19.3 ± 2.5 | 19.3 ± 1.9 | 18.7 ± 2.2 |
| C1 DELAY H0, ISIs shuffled | 29.7 ± 2.6 | 28.3 ± 2.3 | 20.2 ± 2.4 | 19.8 ± 2.0 | 19.4 ± 2.5 |
| C1 DELAY H0+H1, ISIs shuffled | 28.0 ± 2.7 | 27.4 ± 2.8 | 19.4 ± 1.5 | 19.2 ± 2.0 | 18.4 ± 2.2 |
| C2 LINE H0, random time cutoff | 24.0 ± 4.1 | 22.1 ± 2.3 | 16.1 ± 2.6 | 16.3 ± 1.5 | 15.6 ± 2.3 |
| C2 DELAY H0, random time cutoff | 21.8 ± 2.6 | 23.9 ± 2.2 | 16.9 ± 2.4 | 18.2 ± 1.4 | 15.7 ± 2.7 |
| C3 DELAY H0, share | 21.1 ± 2.0 | 20.1 ± 1.6 | 13.4 ± 2.0 | 14.4 ± 0.8 | 12.9 ± 1.6 |

What it shows:
- With five stimuli mixed, the ISI-distribution methods keep about 70% of their movie-only
  1-NN (LINE log-scale share: 52.2 -> 37.5; ISI histogram: 44.0 -> 32.1).
- Victor-Purpura loses more than half (28.2 -> 12.0), because it needs stimulus-locked timing.
- DELAY H0 stays within 1 point of its ISI-shuffled control.

Files on `night-run`: `results/extended_neuropixels_across.{md,csv}`, `..._runs.csv`, and
`figures/analysis/ext_neuropixels_across.png` and `ext_neuropixels_vp.png`. These replace the
two-stimulus version of the same files.

## Retina (recording 20180126_sr_le_sp)

Full stimulus list from `extracted/20180126_sr_le_sp/stimuli/*.mat`. Each stimulus is its own
recording with its own clock starting near 0; pulses are frame pulses.

| no. | stimpara.stimulus | original name | pulses | duration | 10 s windows | mean rate (first 40 cells) |
|---|---|---|---|---|---|---|
| 5 | frozencheckerflicker | frozennoise8x8bw1blink3000run600freeze | 54672 | 1809 s | 180 | 17.8 Hz |
| 6 | imagesequence (flashed natural images) | imgseq512x512_300im_10r_mixed | 3010 | 2988 s | 298 | 20.1 Hz |
| 7 | imagesequenceblur | imgseq512x512_40im_10r_scales6_15to180 | 2810 | 2789 s | 278 | 21.8 Hz |
| 9 | subunitflash | suf_noSpace_0.03_1_200ms_10contrasts_24angles_5r_14x14 | 1205 | 1195 s | 119 | 17.8 Hz |
| 10 | locallysparsesubunitflash | locallysparsesuf14x14con5ang8stixels2dur12 | 7654 | 1520 s | 151 | 18.3 Hz |
| 13 | reversinggratingswithvaryingspatialperiod | reversinggratingWVSP_30reversals_6widths_update | 869 | 890 s | 88 | 12.8 Hz |
| 14 | directiongratingsequence | directiongratingsequence_3_stimuli_contrast1 | 544 | 876 s | 87 | 11.2 Hz |
| 15 | saccadegrating | saccadegrating_16pixShift2_100ms_for60Hz | 2052 | 916 s | 91 | 14.3 Hz |

- **Converter:** `scripts/load_retina_stimuli.py` (in the working tree, not committed) cuts
  every stimulus except the frozen noise into 10 s windows. It uses the same columns and neuron
  ids as `load_retina_running.py`, keeps the first 30 windows per stimulus, and writes
  `data/retina_stimuli_20180126_sr_le_sp.parquet` (156 cells, 7 stimuli).
- **Tested:** I added the 7 stimuli to the retina `across` list at 15 samples each (the cap
  step 4 uses for extra Neuropixels stimuli). A 5-neuron test ran.
- **Stopped:** the user stopped the full run ("it was fine as it was"). The retina `across`
  list is back to the two noise stimuli; no retina change is committed.
- **Note:** my 5-neuron tests overwrote `results/extended_retina_across.*` and
  `figures/analysis/ext_retina_across.png` in the user's checkout. These are untracked, so the
  originals cannot be restored from git. The user said the retina results are done.

## What I had to change

- **Memory.** The first full retina run with 9 stimuli (6600 samples) crashed the machine.
  - `vector_matrix` (B1, B2) built an n x n x features array: 14 GB for 6600 samples x 40
    histogram bins. It now uses `scipy.spatial.distance.cdist(..., "cityblock")`, which gives
    the same numbers.
  - The across part kept all 19 distance matrices. It now scores each one as it is made and
    drops it; only H0 and H1 are kept until H0+H1 is formed.
  - Both changes give identical scores (checked: largest difference 0.0). They are committed
    to `night-run`.
- **Runs now go into capped user services:** `systemd-run --user -p MemoryMax=3G
  -p MemorySwapMax=10G`.
- **Not done:**
  - flashes in the Neuropixels pool (9 windows < 10);
  - the retina with extra stimuli (stopped by the user);
  - more Neuropixels sessions. The functional connectivity set (DANDI 000022) has 26 sessions
    with the same protocol, about 1.7-3 GB each. Two are on disk (829720705, 789848216).
    Only 9.8 GB is free here; the user will allocate more space.
