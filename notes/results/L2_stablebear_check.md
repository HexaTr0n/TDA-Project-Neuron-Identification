# L2 Retina on real stablebear

2026-11-12, local session. **Worked:** stablebear agrees with the numpy version. 1-NN 85.0%
against 84.9%, k-means with `sb.mean` 66.0% against 65.8% (mean of 5 seeds).

## Data

- Retina frozen noise, session `20180126_sr_le_sp`, trial length 9.94 s.
- Trials with >= 30 ISIs, then 60 random neurons with seed 0 (same as `run_pipeline.py` and
  `curve_eval.py`): 1645 trials, about 27 per neuron.
- Shuffled control (from L1, same trials): square-root time bars with ISIs shuffled, 7.7% 1-NN,
  11.9% k-means.

## Command

```sh
python scripts/sb_check.py --session 20180126_sr_le_sp --neurons 60 --kmeans-seeds 5
```

`scripts/sb_check.py` (new): time bars `[t, t + 2/30)` -> `Barcode` -> `barcode_to_betti_curve`
(exact PCF) -> square root of its values -> `sb.Pcf`. Distances `sb.pdist(p=1)`; k-means
assigns by `sb.cdist(p=1)` and sets each centre to `sb.mean` of its members' PCFs (3 restarts,
25 iterations, k = 60, Hungarian pairing). The numpy column is `curves.sqrt_time_bars` on a
grid of step s/20 with the same k-means code.

## Scores

| | 1-NN | k-means mean | sd | min | max | seconds (5 seeds) |
|---|---|---|---|---|---|---|
| stablebear | 85.0 | 66.0 | 2.4 | 61.8 | 68.8 | 77 |
| numpy grid | 84.9 | 65.8 | 2.0 | 62.9 | 69.0 | 2 |
| web chat (numpy) | 85.0 | 66.8 | | | | |

Per seed, stablebear: 67.5, 61.8, 66.2, 68.8, 65.7; numpy: 65.1, 62.9, 66.4, 65.3, 69.0.

Distance matrices: correlation 0.999986, median relative difference 0.1%, largest 1.1%. The
two give the same nearest neighbour for 95.8% of trials (the rest are near ties).

## What to take from it

- The grid version is a faithful stand-in for stablebear here, and 35 times faster.
- **k-means scores move by about +-2.5 points between seeds** (61.8-68.8 on one session), so
  differences between methods of a few points in clustering are within noise. A single-seed
  number such as 66.8% should be reported as about 66 +- 2.5.
