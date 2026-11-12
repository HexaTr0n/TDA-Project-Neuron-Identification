# L3 Retina, real cross-stimulus

2026-11-12, local session. **Worked:** ISI fractions from a frozen trial find the same neuron's
unrepeated-noise pieces as well as they find its other frozen trials (57.6% 1-NN against 53.7%),
and the reverse holds too (49.0%). Clustering transfers with a loss of 3-6 points.

## Data

- Retina, session `20180126_sr_le_sp`, raw recording `20180126_sr_le_sp.rar` from G-Node GIN
  (the `.rar` files in `data/` are git-annex pointers; the real archive is 191 MB).
- Stimulus `5_frozennoise8x8bw1blink3000run600freeze`: R = 1500 pulses (49.7 s) of unrepeated
  noise, then F = 300 pulses (9.93 s) of the frozen movie, 30 times. Before frozen trial k, the
  49.7 s of unrepeated noise are cut into 4 pieces of 10 s (`block` = k, `piece` = 0-3; the last
  9.7 s are dropped).
- `scripts/load_retina_running.py` writes both into one table (not committed): session
  `20180126_sr_le_sp` (frozen trials, identical spike counts to `isi_frozennoise.parquet`) and
  `20180126_sr_le_sp_running` (unrepeated pieces). 156 units, 4680 frozen trials, 18720 pieces.
- Neurons: frozen trials with >= 30 ISIs, then 60 random neurons with seed 0 (the same 60 as in
  L1/L2). A: 1645 frozen trials, median 162 spikes. B: the same neurons' unrepeated pieces with
  >= 10 ISIs, 7056 pieces of 10 s, median 153 spikes.

## Commands

```sh
python scripts/load_retina_running.py --extracted-dir extracted --session 20180126_sr_le_sp
python scripts/sb_cross.py --data data/retina_running_20180126_sr_le_sp.parquet \
    --session 20180126_sr_le_sp --cross-session 20180126_sr_le_sp_running
python scripts/curve_eval.py isi_fractions cond_isi_fractions cond_isi_fractions_shuffled burst_table burst_table_shuffled \
    --data data/retina_running_20180126_sr_le_sp.parquet --session 20180126_sr_le_sp \
    --cross-session 20180126_sr_le_sp_running
```

`scripts/sb_cross.py` (new) does everything in stablebear: ISI bars `[0, log(ISI / 1e-4)]` ->
`barcode_to_stable_rank` -> divided by the ISI count -> `sb.Pcf`; `sb.pdist` / `sb.cdist`
(p = 1); k-means centres `sb.mean`. Clustering is the mean over 3 seeds.

## Scores

Chance is 1.7%. "far" never matches a piece from the same block, i.e. the noise recorded just
before that frozen trial, so closeness in time cannot explain the match.

| | ISI fractions (stablebear) | rate only (log spike count) |
|---|---|---|
| frozen -> frozen (1-NN) | 53.7 | 12.9 |
| unrepeated -> unrepeated | 52.1 | 11.9 |
| **frozen -> unrepeated** | **57.6** | 10.6 |
| **unrepeated -> frozen** | **49.0** | 10.4 |
| frozen -> unrepeated, far | 57.2 | 10.7 |
| unrepeated -> frozen, far | 48.4 | 10.1 |
| clustering within frozen | 41.1 (41.9, 39.4, 42.1) | |
| clustering within unrepeated | 37.7 (38.8, 36.3, 38.0) | |
| cluster frozen, assign unrepeated | 34.7 (35.4, 32.9, 35.9) | |
| cluster unrepeated, assign frozen | 37.9 (39.6, 36.1, 38.1) | |

Other stimulus-free signatures with the same trials (numpy grids, `curve_eval.py`; cross columns
average both directions):

| Signature | within 1-NN | k-means | cross 1-NN | cross clustering |
|---|---|---|---|---|
| ISI fractions | 54.3 | 42.4 | 53.3 | 36.3 |
| Conditional ISI fractions | 50.9 | 38.2 | 51.8 | 34.4 |
| shuffled | 48.6 | 39.2 | 48.9 | 32.4 |
| Burst table | 54.6 | 43.6 | 53.7 | 38.5 |
| shuffled | 49.0 | 41.3 | 50.2 | 36.6 |

## What to take from it

- **Changing the stimulus costs ISI fractions almost nothing on the retina**: 53.3% across against
  54.3% within (both directions averaged). Excluding the adjacent noise changes 0.4-0.6 points,
  so it is not drift or closeness in time.
- The earlier cross-stimulus number on the retina (half-trials, 39.2%) was low because a half is
  only 5 s. With 10 s on both sides there is no drop. The same holds for OpenScope (L1), where
  10 s windows are the limit, not the stimulus.
- The firing rate alone gives 10-13%, so the ISI shape, not the rate, carries the identity.
- **What did not work:** order effects are small. Conditional ISI fractions and the burst table
  beat their shuffled controls by 3-5 points in 1-NN and by 1-2 points in clustering, within
  the k-means seed noise (about +-2.5, see L2). Conditional ISI fractions are below plain ISI
  fractions.
- The direction matters a little: unrepeated -> frozen (49.0) is below frozen -> unrepeated (57.6),
  because there are 4 times more unrepeated pieces to be found.
