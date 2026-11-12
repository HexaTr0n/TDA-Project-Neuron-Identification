# L5 A third dataset: Allen Visual Coding Neuropixels, functional connectivity set

2026-11-12, local session. **Partly worked:** ISI signatures identify neurons (49-53% 1-NN), and
with a repeated movie the time-locked signatures are above their controls but weak: 13% over
all areas, 18-22% in visual cortex, far from the retina's 75-88%.

## Data

- DANDI 000022, `sub-811322604/sub-811322604_ses-829720705.nwb` (1.7 GB, the smallest
  session). Pvalb-Cre;Ai32 mouse, Neuropixels in visual cortex (VISp, VISl, VISal, VISrl,
  VISam), hippocampus (CA1, CA3, DG) and superior colliculus. 1344 good units. No layer labels
  in this file.
- `scripts/load_allen.py <nwb>` writes `data/allen/allen829720705.parquet` (not committed):
  - `allen829720705_movie`: **one sample per repeat of natural movie one**, 60 repeats of
    30.02 s (frame 0 to frame 0 of the next repeat), times from repeat start. This is a
    frozen repeat like the retina's, so time-locked signatures get a fair test.
  - `allen829720705_spont`: the grey-screen spontaneous periods cut into 77 windows of
    30.02 s, the same length as a movie sample.
  - Optotagging (`opto_z`, `cell_type`) as in `load_openscope.py`: 145 PV-tagged, 206 narrow,
    993 wide.
- Neurons: movie samples with >= 30 ISIs, then 60 random neurons with seed 0: 2598 samples,
  spikes per sample median 127 (quartiles 62, 246). The 60 units: CA1 16, DG 9, VISl 10, VISal 8,
  VISrl 7, VISam 5, VISp 5. Second run with visual cortex only (`--area-prefix VIS`): 2829
  samples, median 155 spikes; VISl 15, VISam 14, VISal 14, VISrl 11, VISp 6.
- Spontaneous windows for the same 60 neurons (>= 10 ISIs): 4125 windows, median 68 spikes.

## Commands

```sh
python scripts/load_allen.py data/allen/sub-811322604_ses-829720705.nwb
python scripts/curve_eval.py --data data/allen/allen829720705.parquet --session allen829720705_movie
python scripts/curve_eval.py <methods> --data data/allen/allen829720705.parquet --session allen829720705_movie --area-prefix VIS
python scripts/sb_check.py --data data/allen/allen829720705.parquet --session allen829720705_movie --kmeans-seeds 3
python scripts/sb_cross.py --data data/allen/allen829720705.parquet --session allen829720705_movie --cross-session allen829720705_spont
python scripts/curve_eval.py isi_fractions cond_isi_fractions cond_isi_fractions_shuffled burst_table burst_table_shuffled \
    --data data/allen/allen829720705.parquet --session allen829720705_movie --cross-session allen829720705_spont
```

## Scores: movie repeats

1-NN / k-means with mean curves / k-medoids; cross test on half-repeats (15 s, 1-NN /
clustering). Chance 1.7%. With coincidences in the visual-cortex run, "other neurons" are the
visual-cortex units only.

| Signature | all areas 1-NN | k-means | k-medoids | halves 1-NN | halves clust. | visual cortex 1-NN / k-means |
|---|---|---|---|---|---|---|
| Time bars | 13.5 | 8.2 | 10.6 | 3.1 | 3.1 | 18.0 / 14.0 |
| shuffled | 2.9 | 5.5 | 6.5 | 3.3 | 4.2 | 3.3 / 6.1 |
| Square-root time bars | 12.9 | 9.9 | 13.4 | 3.5 | 4.6 | 17.9 / 12.5 |
| shuffled | 3.7 | 5.5 | 7.8 | 4.1 | 4.2 | 3.7 / 6.3 |
| Multi-scale square-root time bars | 13.1 | 7.6 | 11.8 | 3.3 | 3.3 | 22.5 / 15.5 |
| shuffled | 3.5 | 5.5 | 8.9 | 3.5 | 3.6 | 3.6 / 7.5 |
| Burst bars | 12.4 | 8.9 | 11.6 | 4.2 | 4.7 | |
| shuffled | 4.2 | 6.1 | 7.7 | 4.1 | 4.5 | |
| Current-ISI function | 25.8 | 22.6 | 19.3 | 6.0 | 4.8 | |
| shuffled | 7.7 | 11.2 | 12.5 | 5.9 | 6.0 | |
| Burst-size function | 13.8 | 8.4 | 12.0 | 3.3 | 3.1 | |
| shuffled | 3.7 | 5.4 | 6.7 | 3.4 | 3.3 | |
| Coincidence function | 12.1 | 9.4 | 11.3 | 3.3 | 4.4 | 14.2 / 13.2 |
| other neurons from next repeat | 12.1 | 8.3 | 11.2 | 3.3 | 3.7 | 14.2 / 11.3 |
| ISI fractions | 48.5 | 24.8 | 26.5 | 27.0 | 16.3 | 38.8 / 23.5 |
| Square-root time bars + ISI fractions | 29.4 | 22.6 | 22.2 | 9.7 | 8.5 | 43.9 / 22.5 |
| shuffled | 16.1 | 16.9 | 18.9 | 11.7 | 12.9 | 18.5 / 17.8 |
| Conditional ISI fractions | 46.3 | 27.9 | 26.8 | 26.5 | 16.2 | |
| shuffled | 41.7 | 23.5 | 24.5 | 29.5 | 18.1 | |
| Burst table | 52.8 | 28.1 | 27.1 | 29.5 | 17.4 | 45.3 / 25.4 |
| shuffled | 43.0 | 26.3 | 26.6 | 32.5 | 18.5 | 35.7 / 23.5 |

stablebear check (`sb_check.py`, square-root time bars, all areas): 1-NN 12.9% in both,
k-means 10.8% (stablebear) against 10.6% (numpy), distance matrices correlate at 0.999998.

## Scores: movie against spontaneous (equal 30 s samples)

ISI fractions in stablebear (`sb_cross.py`; clustering mean of 3 seeds):

| | ISI fractions | rate only |
|---|---|---|
| movie -> movie (1-NN) | 48.7 | 7.3 |
| spontaneous -> spontaneous | 32.8 | 6.5 |
| movie -> spontaneous | 28.0 | 5.5 |
| spontaneous -> movie | 23.7 | 4.6 |
| clustering within movie / within spontaneous | 25.3 / 23.3 | |
| cluster movie, assign spontaneous / the reverse | 13.2 / 17.0 | |

Numpy grids, both directions averaged (`curve_eval.py --cross-session`):

| Signature | cross 1-NN | cross clustering |
|---|---|---|
| ISI fractions | 26.0 | 15.7 |
| Conditional ISI fractions | 26.2 | 14.8 |
| shuffled | 23.0 | 12.5 |
| Burst table | 27.8 | 14.4 |
| shuffled | 23.6 | 13.8 |

## What failed, plainly

- **Time-locked signatures stay weak even with a frozen repeat.** Over all areas they give
  12-14% (controls 3-4%); in visual cortex alone 18-22%. On the retina the same code gives
  75-88%. Mouse cortex is far less reliable from repeat to repeat than retinal ganglion cells,
  and hippocampal units barely follow the movie. Current ISI is the best time-locked curve
  (25.8%).
- **Coincidences again equal their next-repeat control** (12.1 vs 12.1; 14.2 vs 14.2 in visual
  cortex): no identity beyond the stimulus.
- **Across a stimulus change ISI fractions lose about half**: 48.7% within movie, 28.0% movie to
  spontaneous. Unlike the retina (L3), the samples are equally long here, so this is a real
  change of firing statistics between movie and grey screen in cortex, not sample size.
  Spontaneous-to-spontaneous is also lower (32.8%) than movie-to-movie.
- **Shuffled controls beat the real ordering in the cross test** for conditional ISI fractions and
  the burst table (half-repeats 29.5 vs 26.5, 32.5 vs 29.5), so order effects do not transfer.
- Adding time bars to ISI fractions hurts over all areas (29.4 vs 48.5) but helps a little in
  visual cortex (43.9 vs 38.8), where time-locking exists.
- Half-repeat cross test for time-locked signatures is at chance (3-4%), as expected: the two
  halves of the movie are different stimuli.

## What worked

- ISI fractions and the burst table identify neurons at 49-53% 1-NN over all areas.
- stablebear and the numpy grids agree on a third dataset.
