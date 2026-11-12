# L4 OpenScope labels

2026-11-12, local session. **Weakly worked:** ISI-fraction profiles carry information about
layer, region and cell type well above shuffled labels, but clusters barely line up with
any of them (AMI 0.02-0.05).

## Data

- OpenScope oddball session `glo637908` (see L1), grating samples of 124 s, 46 blocks.
- Units: every good unit with >= 10 samples of >= 30 ISIs: **1744 units**, 71212 samples.
- Each unit's curve is `sb.mean` of the ISI-fraction PCFs of its samples (stablebear
  `barcode_to_stable_rank` / ISI count, as in `sb_cross.py`); unit-to-unit distances `sb.pdist`
  (p = 1); k-means with `sb.mean` centres, k = number of classes, mean of 3 seeds.
- Labels:
  - cell type: PV-tagged (light response z > 5; the mouse is Pvalb-Cre;Ai32) 222, narrow spike
    (< 0.4 ms) 351, wide 1171;
  - cortical layer, cortex only: L5 275, L2/3 201, L4 131, L6a 127 (734 units);
  - region: cortex 743, hippocampus 630, midbrain 203, thalamus 168;
  - PV-tagged against untagged narrow spikes, cortex only: 104 against 64.

## Command

```sh
python scripts/sb_labels.py --data data/openscope/glo637908.parquet --session glo637908_grating
```

## Scores

1-NN: each unit's label predicted from its nearest other unit, as balanced accuracy (mean recall
per class). Clustering: AMI between k-means clusters and labels. Shuffled: the same statistic
with labels permuted (200 permutations for 1-NN, 50 for AMI); p is one-sided and limited by the
number of permutations (smallest possible 0.005 and 0.020).

| Labels | units | 1-NN bal. acc. | shuffled (mean / 95%) | p | AMI | shuffled (mean / 95%) | p |
|---|---|---|---|---|---|---|---|
| Cell type (3 classes) | 1744 | **40.3** | 33.1 / 35.4 | 0.005 | 0.023 | 0.000 / 0.001 | 0.02 |
| Cortical layer (4 classes) | 734 | **41.9** | 25.1 / 28.6 | 0.005 | 0.037 | 0.000 / 0.003 | 0.02 |
| Region (4 classes) | 1744 | **45.5** | 25.1 / 27.3 | 0.005 | 0.022 | 0.000 / 0.001 | 0.02 |
| PV-tagged vs narrow, cortex | 168 | 60.5 | 49.6 / 57.9 | 0.02 | 0.051 | -0.002 / 0.005 | 0.02 |

## What to take from it

- **Nearest neighbours share labels well above chance.** A unit's nearest ISI profile is in the
  same layer 42% of the time (balanced; chance 25%) and in the same region 46% (chance 25%).
- **Clusters do not follow the labels.** AMI of 0.02-0.05 is above shuffled but tiny: the ISI
  profiles vary continuously, and the classes overlap. k-means with k = number of classes
  finds other structure (probably firing rate and burstiness).
- **Cell type is the weakest of the three**, apart from PV against narrow (60.5% against 49.6%,
  p = 0.02, only 168 units). The optotagging threshold (z > 5) was my choice and 222 tagged
  units looks high, so the PV label may be noisy.
- **Caveat, not tested:** layer and region also go with depth on the probe. Units recorded close
  together share drift and noise, so part of the layer and region signal may be
  recording position rather than biology. A control would be to predict layer only from units
  on another probe.
