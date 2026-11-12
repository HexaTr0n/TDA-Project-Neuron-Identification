# L1 OpenScope, main table

2026-11-12, local session. **Partly worked:** the ISI signatures identify neurons (63-68% 1-NN),
but every time-locked signature is at chance level on OpenScope, and the port reproduces the
retina numbers.

## Data

- Dataset: OpenScope Global/Local Oddball, DANDI 000253, `sub-637908_ses-1213341633_ogen.nwb`
  (Pvalb-Cre;Ai32 mouse, Neuropixels, visual cortex, hippocampus and midbrain). 2031 units
  with `quality == "good"`, out of 2558.
- Table: `scripts/load_openscope.py <nwb> --grating-duration 124` writes
  `data/openscope/glo637908.parquet` (not committed, 800 MB).
- **How trials were joined.** One grating presentation is 0.5 s with a few spikes, so all 125
  presentations of a grating block (about 124.6 s, 0.5 s on and 0.5 s grey) form one sample, with
  times from block start. Blocks are cut to 124 s, which keeps 46 of the 49 blocks. Blocks
  overlapping receptive field mapping or optotagging are dropped. There is **no frozen repeat**:
  the 46 blocks carry 43 different orientation sequences (45 and 135 degrees, oddballs in
  different places).
- **Spontaneous samples** (session `glo637908_spont`) are the 53 grey-screen intermissions of
  10 s between blocks, one window each. The file's own `spontaneous_presentations` table is
  mostly 0.5 s gaps, with two rows where stop < start.
- Trial length `dur` = 124 s (grating), 10 s (spontaneous), in place of the retina's 9.94 s.
- Neurons: trials with >= 30 ISIs, then 60 random neurons with seed 0 (as `run_pipeline.py`).
  2397 trials, 46 per neuron (4 to 46). The 60 neurons: 40 wide-spike, 12 narrow-spike, 8
  PV-tagged; spread over CA1, SUB, POST, APN and visual areas VISp, VISrl, VISam, VISl, VISpm.
- Spikes per 124 s sample (these 60 neurons): median 484, quartiles 194 and 977. Spontaneous
  10 s windows (>= 10 ISIs, 57 of the 60 neurons, 2039 windows): median 51, quartiles 24 and 104.
- Grid: each curve has at most 15000 samples (RAM guard, `curves.MAX_POINTS`), so on 124 s the
  time-bar grid step is 8.3 ms instead of s/20 = 3.3 ms. Retina curves are below the cap.

## Commands

```sh
python scripts/curve_eval.py --data data/openscope/glo637908.parquet --session glo637908_grating
python scripts/curve_eval.py burst_bars burst_bars_shuffled --data data/openscope/glo637908.parquet --session glo637908_grating
python scripts/curve_eval.py isi_fractions cond_isi_fractions cond_isi_fractions_shuffled burst_table burst_table_shuffled \
    --data data/openscope/glo637908.parquet --session glo637908_grating --cross-session glo637908_spont
python scripts/curve_eval.py            # retina 20180126_sr_le_sp, same 60 neurons as run_pipeline
```

`scripts/curves.py` holds the signatures and `scripts/curve_eval.py` the scores, both new on this
branch. The burst-bars curve is the Betti curve of `barcodes.burst_bars` (1 inside a burst).

## Scores

1-NN / k-means with mean curves / k-medoids, then the cross test on half-trials (1-NN /
clustering, both directions averaged). Chance is about 1.7%. Retina = session
20180126_sr_le_sp, 60 neurons, run with the same code.

| Signature | OpenScope 1-NN | k-means | k-medoids | halves 1-NN | halves clust. | Retina 1-NN / k-means |
|---|---|---|---|---|---|---|
| Time bars | 6.8 | 6.9 | 9.3 | 3.7 | 5.3 | 75.6 / 58.5 |
| shuffled | 3.5 | 6.8 | 8.3 | 2.7 | 5.7 | |
| Square-root time bars | 7.2 | 8.4 | 10.9 | 4.8 | 6.4 | 84.9 / 68.7 |
| shuffled | 3.8 | 6.4 | 8.3 | 3.5 | 5.0 | 7.7 / 11.9 |
| Multi-scale square-root time bars | 7.6 | 8.0 | 9.9 | 4.7 | 5.5 | 87.5 / 68.8 |
| shuffled | 3.4 | 6.8 | 8.7 | 3.3 | 5.2 | 8.0 / 15.3 |
| Burst bars | 7.0 | 7.5 | 8.3 | 4.6 | 4.9 | 63.4 / 48.8 |
| shuffled | 3.4 | 5.3 | 6.6 | 3.7 | 3.9 | 4.7 / 8.4 |
| Square-root time bars + ISI fractions | 29.1 | 19.4 | 24.8 | 18.2 | 14.7 | 91.2 / 59.1 |
| shuffled | 19.0 | 19.0 | 20.9 | 13.5 | 14.1 | 36.5 / 32.2 |
| Current-ISI function | 13.1 | 18.5 | 15.9 | 8.2 | 9.4 | 77.0 / 52.3 |
| shuffled | 6.6 | 12.9 | 12.1 | 5.9 | 9.1 | 7.1 / 10.2 |
| Burst-size function | 7.3 | 6.2 | 9.1 | 4.7 | 5.5 | 77.6 / 55.9 |
| shuffled | 3.4 | 5.6 | 7.3 | 3.2 | 5.0 | 7.8 / 13.7 |
| Coincidence function | 6.5 | 7.8 | 10.1 | 4.6 | 5.5 | 84.3 / 67.6 |
| other neurons from next trial | 6.5 | 7.9 | 9.8 | 4.6 | 6.2 | 84.1 / 66.6 |
| **ISI fractions** | 62.6 | 31.6 | 28.9 | 51.1 | 26.4 | 54.3 / 42.4 (halves 39.2 / 30.3) |
| Conditional ISI fractions | 65.0 | 36.1 | 32.3 | 54.9 | 26.4 | 50.9 / 38.2 (halves 36.4 / 26.8) |
| shuffled | 58.4 | 33.3 | 28.5 | 50.7 | 28.8 | 48.6 / 39.2 (halves 32.6 / 22.5) |
| **Burst table** | 68.3 | 35.3 | 32.8 | 57.9 | 25.6 | 54.6 / 43.6 (halves 39.1 / 32.3) |
| shuffled | 58.9 | 30.9 | 27.9 | 52.8 | 28.9 | 49.0 / 41.3 (halves 33.8 / 28.4) |

### Real change of stimulus: grating blocks against spontaneous windows

Same 60 neurons. A spontaneous 10 s window looks for the nearest 124 s grating sample and the
reverse (1-NN), or the grating samples are clustered and the spontaneous windows assigned to the
nearest centre and the reverse (clustering); both directions averaged. Only stimulus-free
signatures can be compared across samples of different length.

| Signature | 1-NN | clustering |
|---|---|---|
| ISI fractions | 31.1 | 17.4 |
| Conditional ISI fractions | 31.7 | 17.7 |
| shuffled | 28.1 | 17.4 |
| Burst table | 33.0 | 17.8 |
| shuffled | 27.6 | 16.3 |

## What failed, plainly

- **Every time-locked signature is at chance level on OpenScope** (6.5-13% 1-NN against 3.5% for its
  shuffled control). There is no repeated stimulus: the blocks have different orientation
  sequences, and cortex and hippocampus do not fire at the same moments of a 124 s block. The
  retina results for time bars, square-root time bars, multi-scale, burst bars, burst size,
  current ISI and coincidences do not carry over. Current ISI is the only one a little above
  its control (13.1 against 6.6).
- **Coincidences carry no identity beyond the stimulus, even on the retina:** the next-trial
  control scores the same as the real thing (84.1 against 84.3; 6.5 against 6.5 here). So the
  retina's 84% is the neuron's own stimulus-locked timing seen through the population, not
  connectivity.
- **Adding time bars to ISI fractions hurts** here (29.1 against 62.6), because the time part is noise.
- **Clustering is much weaker than 1-NN** for the ISI signatures: 32-36% against 63-68%. k-means with
  mean curves beats k-medoids by 3-4 points for ISI signatures, not by 4-20 as on the retina.
- **Order effects are small:** conditional ISI fractions and the burst table beat their shuffled
  controls by 6-9 points in 1-NN but not in clustering (the shuffled burst table clusters
  across halves *better*, 28.9 against 25.6).
- **Across a real change of stimulus**, 1-NN drops from about 63-68% within grating to about 31-33%,
  and clustering to about 17%. The 10 s windows are short (median 51 spikes) compared with 124 s
  samples, so part of this drop is sample size, not stimulus; `cross_epoch.py` earlier found
  spontaneous-to-spontaneous 10 s windows at about the same level as across epochs.

## What worked

- The port reproduces the web chat's retina numbers: square-root time bars 84.9 / 68.7 (handoff
  85.0 / 66.8), plain time bars 75.6 / 58.5 (75.6 / 56.9), current ISI 77.0 (76.5), burst size
  77.6 / 55.9 (77.2 / 56.1), coincidence 84.3 / 67.6 (83.8 / 66.9), sqrt time bars + ISI fractions
  91.2 (90.8), burst bars 63.4 (about 64-70).
- On OpenScope the ISI signatures identify neurons better than on the retina (62-68% against
  51-55% 1-NN), helped by long samples (median 484 spikes). Across half-blocks they hold up
  (51-58%), better than retina halves (36-39%).
