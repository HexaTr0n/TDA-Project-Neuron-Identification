# L12 Rhythm signatures, hippocampus and cortex separately

2026-11-12, local session. **Weak.**

- Rhythm signatures identify neurons at 14-40% 1-NN, below ISI fractions (35-68%), on every
  dataset and in both regions.
- Hippocampus is not better than cortex, so no theta advantage showed up.
- The spectrum cloud's H0 is at chance level (2-7%).
- The best rhythm signal is OpenScope's long blocks (ACG curve 38%, spectrum position 40%,
  shuffled 25-29%).

## Data

**Units.**
- 60 per group (seed 0), among units with >= 10 samples of >= 100 ISIs.
- Hippocampus = CA1/CA3/DG/SUB/ProS/POST/HPF; cortex = VIS*.
- Retina: all cells.

| Dataset, group | candidate units | samples (spectrum) | median spikes | length |
|---|---|---|---|---|
| OpenScope gratings, hippocampus | 558 | 2400 (2287) | 465 | 124 s |
| OpenScope gratings, cortex | 629 | 2326 (2180) | 512 | 124 s |
| Allen movie, hippocampus | 266 | 2255 (2149) | 194 | 30 s |
| Allen movie, cortex | 406 | 2462 (2168) | 193 | 30 s |
| Allen spontaneous, hippocampus | 268 | 2325 (2173) | 185 | 30 s |
| Allen spontaneous, cortex | 371 | 3205 (3037) | 206 | 30 s |
| Retina unrepeated noise | 143 | 1702 (1702) | 680 | 40 s |

**Signatures** (`scripts/sb_rhythm.py`):

- **ACG:** lags 1-500 ms, 1 ms bins, 3 ms smoothing, 1 = Poisson. The lags run longer than
  exp21's 200 ms so that theta periods fit.
  - *curve:* L1 between ACGs.
  - *bars, lags dropped:* peak and dip prominences as bars [0, p], stable ranks in stablebear, L1.
  - *peaks matched in lag:* peaks with prominence >= 0.3, `matching_matrix`, q = 50.
- **Spectrum windows** (as `windowed_spectrum`):
  - Gaussian windows sigma 0.5 s every 0.5 s;
  - log power at 24 frequencies 1-100 Hz, relative to Poisson;
  - windows with effective count >= 4;
  - samples need >= 40 good windows.
  - *Position:* mean point.
  - *H0:* Rips in stablebear on 40 random windows, 10 subsamples, stable ranks averaged.
- **ACG bars + spectrum position:** both median-scaled and added.
- **Control:** ISIs permuted within each sample. ISI fractions are unchanged by this, as they
  should be.
- **Scores:** 1-NN / k-medoids, chance 1.7%.

## Command

```sh
python scripts/sb_rhythm.py --data data/openscope/glo637908.parquet --session glo637908_grating --group hippocampus
python scripts/sb_rhythm.py --data data/openscope/glo637908.parquet --session glo637908_grating --group cortex
python scripts/sb_rhythm.py --data data/allen/allen829720705.parquet --session allen829720705_spont --group hippocampus   # and cortex
python scripts/sb_rhythm.py --data data/allen/allen829720705.parquet --session allen829720705_movie --group hippocampus   # and cortex
python scripts/sb_rhythm.py --data data/retina_running_20180126_sr_le_sp.parquet --session 20180126_sr_le_sp_running --join-pieces --group all
```

## Scores

1-NN / k-medoids, %. "shuf" rows: ISIs permuted.

| signature | OS hippo | OS cortex | Allen movie hippo | Allen movie cortex | Allen spont hippo | Allen spont cortex | Retina |
|---|---|---|---|---|---|---|---|
| ISI fractions (reference) | 56.5/29.0 | 52.3/26.0 | 46.3/26.9 | 39.9/23.3 | 51.6/30.7 | 34.5/22.2 | 67.9/43.8 |
| ACG curve | 33.1/22.1 | 38.3/22.7 | 19.2/14.5 | 18.2/17.3 | 21.6/18.9 | 16.8/13.8 | 30.1/24.4 |
| shuf | 25.1/19.6 | 28.9/21.1 | 13.9/13.9 | 14.5/13.0 | 16.1/16.5 | 12.6/12.9 | 18.7/18.0 |
| ACG bars, lags dropped | 18.9/19.2 | 20.0/19.9 | 13.7/16.0 | 14.6/15.3 | 16.8/16.6 | 14.7/14.8 | 36.8/35.9 |
| shuf | 17.6/19.0 | 15.2/16.7 | 15.0/16.7 | 11.8/14.5 | 15.5/17.0 | 13.5/14.7 | 34.8/33.0 |
| ACG peaks matched in lag | 20.1/18.3 | 22.7/20.8 | 14.3/14.3 | 13.7/14.7 | 17.8/16.1 | 14.2/14.5 | 25.9/26.0 |
| shuf | 14.8/16.5 | 17.9/19.9 | 13.5/15.5 | 12.8/13.0 | 16.1/16.9 | 13.4/15.2 | 23.4/23.2 |
| spectrum position | 31.1/25.7 | 40.1/27.6 | 16.9/17.1 | 13.7/14.7 | 18.0/20.8 | 14.6/14.7 | 27.3/26.1 |
| shuf | 25.0/21.5 | 27.9/25.4 | 12.9/16.0 | 12.0/14.5 | 16.4/18.7 | 11.6/14.5 | 14.3/17.3 |
| spectrum H0 | 6.4/11.6 | 7.4/11.8 | 4.0/9.1 | 4.4/9.2 | 3.6/10.3 | 3.5/8.2 | 2.4/10.0 |
| shuf | 4.8/11.7 | 5.1/11.3 | 3.0/10.5 | 3.8/9.5 | 3.1/9.6 | 3.7/8.1 | 2.9/9.9 |
| ACG bars + spectrum position | 41.3/24.0 | 45.5/28.3 | 24.7/20.1 | 24.7/20.9 | 27.9/22.6 | 21.6/18.5 | 51.1/42.0 |
| shuf | 36.2/25.2 | 35.3/26.4 | 22.5/21.3 | 20.6/20.0 | 27.7/21.7 | 18.7/19.3 | 40.8/32.2 |

## What to take from it

- **Hippocampus does not stand out.**
  - On OpenScope, cortex is a little better than hippocampus with rhythm signatures (ACG curve
    38.3 vs 33.1, spectrum position 40.1 vs 31.1).
  - On Allen, hippocampus is a few points better (spontaneous ACG curve 21.6 vs 16.8).
  - Either way the differences are small, and ISI fractions carry the same edge (51.6 vs 34.5
    on Allen spontaneous), so it is the ISI distribution, not theta.
  - Whether these units are theta-modulated at all was not checked here (no LFP in the
    parquet).
- **Rhythm needs long samples.** With 124 s blocks, ACG curve and spectrum position reach
  31-40%, 6-12 points above shuffled. With 30 s they are at 14-22%, 2-6 points above shuffled.
- **Topology adds nothing here.**
  - ACG bars with lags dropped are within 0-5 points of their shuffle in cortex and hippocampus.
    The bar sizes mostly reflect burstiness, which survives shuffling.
  - The spectrum cloud's H0 is near chance everywhere: the spread of 40 window spectra is noise.
- **Combining helps but stays below ISI fractions**: ACG bars + spectrum position gives 22-51%,
  against 35-68%.
- **What failed:** every rhythm signature, as a neuron identifier, in every group; and the
  hoped-for theta advantage in hippocampus.
