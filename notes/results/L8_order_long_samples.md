# L8 Order-only signature on long samples

2026-11-12, local session. **Worked:** with about 480 spikes per sample, ISI order alone identifies
OpenScope neurons at 49.1% 1-NN (shuffled 10.1%), against 27.6% on 10 s retina trials, and on
top of ISI fractions it adds 7 points in 1-NN and 4.4 in k-means. On the retina it adds nothing.

## Data

- OpenScope `glo637908_grating`: 124 s blocks, the 60 neurons of L1 (seed 0), 2397 samples,
  median 484 spikes. Half-blocks (62 s) for the cross-halves test; grey-screen 10 s windows
  (>= 10 ISIs, 2039 windows, median 51 spikes) for grating against grey.
- Retina `20180126_sr_le_sp`: frozen trials of 9.94 s, the same 60 neurons as L1-L3, 1645
  trials, median 162 spikes.
- Signatures (`scripts/curves.py`, numpy: stablebear has no two-parameter PCF):
  - `isi_joint`: share of consecutive ISI pairs with log ISI_k > a and log ISI_k+1 > b, on a
    24 x 24 grid from log 1 ms to log 3 s (as `exp25_isi_distances.py`);
  - `isi_order`: `isi_joint` minus the product of its two margins (zero when consecutive ISIs are
    independent);
  - `isi_fractions+isi_order`: each divided by its median pairwise L1 distance, order weighted 0.5.
  - "shuffled": ISIs permuted within each sample before the signature (both samples of a cross
    pair are shuffled).

## Commands

```sh
python scripts/curve_eval.py isi_order isi_order_shuffled isi_joint isi_joint_shuffled isi_fractions \
    --data data/openscope/glo637908.parquet --session glo637908_grating
python scripts/curve_eval.py isi_order isi_order_shuffled isi_joint isi_joint_shuffled isi_fractions \
    --data data/openscope/glo637908.parquet --session glo637908_grating --cross-session glo637908_spont
python scripts/curve_eval.py isi_fractions+isi_order isi_fractions+isi_order_shuffled \
    --data data/openscope/glo637908.parquet --session glo637908_grating [--cross-session glo637908_spont]
python scripts/curve_eval.py isi_order isi_order_shuffled isi_joint isi_joint_shuffled isi_fractions isi_fractions+isi_order isi_fractions+isi_order_shuffled
```

## Scores

1-NN / k-means with mean curves (one seed; seed noise about +-2.5) / cross 1-NN / cross
clustering. Chance 1.7%.

| Signature | OpenScope 124 s 1-NN | k-means | halves 1-NN | halves clust. | grating <-> grey 1-NN | clust. | Retina 10 s 1-NN | k-means | halves 1-NN |
|---|---|---|---|---|---|---|---|---|---|
| **Order only** | **49.1** | 25.2 | 38.7 | 18.5 | 21.3 | 11.8 | 27.6 | 23.8 | 17.2 |
| shuffled | 10.1 | 8.3 | 9.0 | 4.7 | 7.6 | 3.3 | 7.1 | 10.3 | 6.7 |
| Joint fractions | 61.3 | 31.5 | 51.9 | 24.1 | 29.7 | 15.9 | 52.9 | 37.4 | 37.3 |
| shuffled | 59.3 | 31.3 | 53.2 | 28.1 | 28.5 | 16.1 | 51.9 | 37.9 | 36.0 |
| ISI fractions (reference) | 62.6 | 31.6 | 51.1 | 26.4 | 31.1 | 17.4 | 54.3 | 42.4 | 39.2 |
| **ISI fractions + order** | **69.7** | **36.0** | 53.4 | 28.1 | 29.0 | 16.8 | 53.3 | 40.6 | 38.8 |
| shuffled | 51.0 | 28.7 | 40.0 | 24.8 | 22.5 | 14.6 | 46.1 | 36.4 | 27.2 |

## What to take from it

- **Order carries more identity with more spikes.** Order alone: 27.6% on 10 s retina trials
  (about 160 spikes), 49.1% on 124 s OpenScope blocks (about 480 spikes), 38.7% on 62 s half
  blocks. The shuffled control stays at 7-10%.
- **With long samples order is no longer redundant.** ISI fractions + order beats ISI fractions
  alone on OpenScope by 7.1 points in 1-NN (69.7 against 62.6) and 4.4 in k-means (36.0 against
  31.6), outside seed noise. On the retina it gives nothing (53.3 against 54.3), as the web chat
  found in W3.
- **Order transfers across stimuli only partly.** Grating blocks against grey windows: order alone
  21.3% (shuffled 7.6%), but adding it to ISI fractions does not help across stimuli (29.0
  against 31.1). The grey windows are only 10 s (about 50 spikes), so their order estimate is
  noisy; the equal-length version (as L7) is not done yet.
- **What failed:** joint fractions are no better than ISI fractions; the joint table is mostly
  the product of its margins. The shuffled combination is worse than ISI fractions alone (51.0
  against 62.6) because a shuffled order part is pure noise.
