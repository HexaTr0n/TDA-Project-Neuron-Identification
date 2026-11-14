# L25 Per-spike mixture pdf with the entire history (hidden Markov model)

2026-11-14, local session. **Small gain without homology; the homology readouts fail.**

- **The model finds memory where ISI order matters.** The mean probability of staying in a
  hidden state is the measure (no memory = 0.50 for 2 states, 0.33 for 3; ISIs shuffled:
  0.50 / 0.33 everywhere):

  | | 2 states | 3 states |
  |---|---|---|
  | OpenScope 124 s | 0.67 | 0.51 |
  | Allen 30 s | 0.60 | 0.42 |
  | retina 40 s | 0.52 | 0.40 |

- **The cloud without homology** (5 quantiles over spikes of the predicted pdfs) **is the best
  readout.** Added to ISI fractions it gives a small gain over ISI fractions on all three
  datasets with 3 states:
  - OpenScope 124 s: 68.3 vs 64.2 (shuffled 63.1);
  - retina 40 s: 66.7 vs 64.3 (64.0);
  - Allen: 46.0 vs 44.8 (41.2).
  - It stays below whole-cloud delay regions on OpenScope 124 s (69.4) and below the burst
    table (L22).
- **Rips H0 of the pdf cloud fails:** 7-21% whole and 19-36% in 8 regions. Adding it to ISI
  fractions lowers them by 7-17 points. The cloud's shape adds order (well above shuffled on
  OpenScope), but on its own it is far below every other signature.
- **What failed:** both homology readouts; the "spread" readout; the mean predicted pdf (below
  ISI fractions everywhere, as expected: with the history averaged out it is just the ISI
  distribution, smoothed).

## Data

L22's samples and loaders (`sb_headline.load`, seed 0), 60 neurons each. Chance 1.7%.

| dataset | samples | median spikes |
|---|---|---|
| OpenScope 124 s, gratings, within only | 2310 | 496 |
| retina 40 s clean (`retina40clean`), frozen x4 <-> unrepeated x4 | 420 + 1795 | 622 |
| Allen 30 s, movie <-> spontaneous (L17 pair) | 1783 + 1734 | 101 |

**Method** (`scripts/sb_hmm.py`, port of the web pilot `exp34_hmm_pdf_per_spike.py`, W12):
- **Model:** per sample, a hidden Markov model on log ISI with K = 2 or 3 states (one Gaussian
  each, a K x K transition table), fitted by EM (60 iterations, half a pseudo-count per
  transition).
- **Per-spike pdf:** after every ISI, the forward filter gives the weights of the next ISI's
  mixture. Each spike becomes that mixture's survival curve on 46 log-ISI points, so L1 between
  two curves is Wasserstein-1 between the pdfs.
- **Fitting:** samples are fitted in batches of 40, sorted by length, so the 124 s blocks fit
  in memory.
- **Readouts (as W12):**
  - mean predicted pdf;
  - cloud without homology (5 quantiles over spikes);
  - spread (sd over spikes);
  - Rips H0 of the whole cloud;
  - Rips H0 in 8 regions (k-means anchors on the curves of 200 outside-neuron samples).
  - Rips is in stablebear, Euclidean between curves, as stable ranks with L1.
  - Each readout is also added to ISI fractions (median-scaled).
- **References on the same samples:** ISI fractions; whole-cloud delay regions k = 32 H0 (L22's
  definition, anchors on outside neurons).
- **Control:** ISIs permuted before fitting.
- **Scores:** 1-NN / k-medoids, within each stimulus and across (as L17); no k-means.

## Commands

```sh
python scripts/sb_hmm.py --dataset openscope124
python scripts/sb_hmm.py --dataset retina40clean
python scripts/sb_hmm.py --dataset allen30
```

About 25 / 15 / 10 minutes.

## Summary: 1-NN within A / A -> B (in brackets: within A, ISIs shuffled)

| readout | OpenScope 124 s, K=2 / K=3 | retina 40 s clean, K=2 / K=3 | Allen 30 s, K=2 / K=3 |
|---|---|---|---|
| ref: ISI fractions | 64.2 | 64.3 / 75.0 | 44.8 / 25.4 |
| ref: delay regions k=32, whole cloud | 69.4 (57.5) | 62.9 / 68.8 (59.5) | 38.4 / 25.7 (30.3) |
| mean predicted pdf | 53.5 (55.2) / 59.4 (60.1) | 61.7 / 67.4 (60.2) ; 62.9 / 72.1 (63.8) | 37.1 / 21.9 (37.6) ; 40.7 / 24.2 (41.4) |
| cloud, no homology | 61.8 (55.8) / 65.4 (60.6) | 63.6 / 68.8 (61.4) ; 65.5 / 71.9 (61.9) | 40.3 / 24.2 (36.2) ; 42.0 / 26.4 (37.5) |
| **ISI fractions + cloud, no homology** | 66.5 (61.2) / **68.3** (63.1) | 66.9 / 71.7 (63.3) ; **66.7 / 74.3** (64.0) | 44.3 / 27.0 (41.4) ; **46.0 / 28.3** (41.2) |
| spread | 36.5 (26.8) / 33.7 (18.4) | 20.7 / 34.3 (22.6) ; 33.1 / 42.6 (10.0) | 24.2 / 12.8 (15.5) ; 21.6 / 15.4 (11.6) |
| cloud, Rips H0 (whole) | 18.0 (9.1) / 14.9 (9.4) | 10.7 / 15.2 (13.6) ; 20.7 / 23.3 (12.1) | 8.4 / 6.8 (6.7) ; 6.8 / 6.5 (5.5) |
| cloud, Rips H0 in 8 regions | 33.2 (17.1) / 34.4 (19.5) | 26.4 / 33.8 (26.0) ; 36.2 / 34.8 (21.7) | 18.7 / 12.8 (13.9) ; 18.8 / 11.3 (10.2) |
| ISI fractions + Rips H0 in 8 regions | 56.8 (45.6) / 55.4 (45.6) | 50.5 / 59.5 (45.2) ; 51.9 / 61.4 (45.7) | 35.3 / 22.5 (31.1) ; 34.2 / 21.0 (30.2) |

## What to take from it

- **History matters most in cortex**, as the stay probabilities show. The per-spike pdfs carry
  that order: the cloud without homology is 2-6 points above its shuffled control.
- **The useful part is the distribution of predictions, not its shape.** Five quantiles over
  spikes add 1-4 points to ISI fractions. The Rips barcode of the same cloud is far weaker,
  because the per-spike pdfs vary smoothly (they are mixtures of the same 2-3 Gaussians). The
  cloud is close to a curve, and its H0 mostly records spacing along it.
- **The retina has little memory** (stay 0.52 with 2 states), as W12 found. The 3-state model
  still gives a 2-point gain over ISI fractions there.
- **Compared with the other signatures on the same samples:** below the burst table (L22:
  OpenScope 124 s 71.8, retina 40 s 68.3, Allen 47.7). Below whole-cloud delay regions on
  OpenScope 124 s, above them on Allen.

## Full tables

Each cell is 1-NN / k-medoids, %.

### OpenScope 124 s, gratings (within only)

60 neurons. Mean probability of staying in a state: K=2 ISIs shuffled: 0.498; K=2 real: 0.667; K=3 ISIs shuffled: 0.331; K=3 real: 0.506.

| K | readout | control | within A |
|---|---|---|---|
| ref | ref: ISI fractions |  | 64.2 / 31.5 |
| ref | ref: ISI fractions | shuffled | 64.2 / 31.5 |
| ref | ref: delay regions k=32, H0 (whole cloud) |  | 69.4 / 33.9 |
| ref | ref: delay regions k=32, H0 (whole cloud) | shuffled | 57.5 / 33.2 |
| 2 | mean predicted pdf |  | 53.5 / 32.2 |
| 2 | mean predicted pdf | shuffled | 55.2 / 32.5 |
| 2 | ISI fractions + mean predicted pdf |  | 60.9 / 31.1 |
| 2 | ISI fractions + mean predicted pdf | shuffled | 60.3 / 31.8 |
| 2 | cloud, no homology (quantiles) |  | 61.8 / 32.8 |
| 2 | cloud, no homology (quantiles) | shuffled | 55.8 / 31.9 |
| 2 | ISI fractions + cloud, no homology (quantiles) |  | 66.5 / 33.0 |
| 2 | ISI fractions + cloud, no homology (quantiles) | shuffled | 61.2 / 32.6 |
| 2 | spread |  | 36.5 / 21.9 |
| 2 | spread | shuffled | 26.8 / 15.0 |
| 2 | ISI fractions + spread |  | 58.6 / 31.9 |
| 2 | ISI fractions + spread | shuffled | 49.4 / 22.8 |
| 2 | cloud, Rips H0 (whole) |  | 18.0 / 17.1 |
| 2 | cloud, Rips H0 (whole) | shuffled | 9.1 / 12.2 |
| 2 | ISI fractions + cloud, Rips H0 (whole) |  | 54.3 / 28.9 |
| 2 | ISI fractions + cloud, Rips H0 (whole) | shuffled | 43.1 / 23.3 |
| 2 | cloud, Rips H0 in 8 regions |  | 33.2 / 23.9 |
| 2 | cloud, Rips H0 in 8 regions | shuffled | 17.1 / 17.8 |
| 2 | ISI fractions + cloud, Rips H0 in 8 regions |  | 56.8 / 29.5 |
| 2 | ISI fractions + cloud, Rips H0 in 8 regions | shuffled | 45.6 / 24.4 |
| 3 | mean predicted pdf |  | 59.4 / 32.0 |
| 3 | mean predicted pdf | shuffled | 60.1 / 33.6 |
| 3 | ISI fractions + mean predicted pdf |  | 62.4 / 32.7 |
| 3 | ISI fractions + mean predicted pdf | shuffled | 62.8 / 31.2 |
| 3 | cloud, no homology (quantiles) |  | 65.4 / 32.9 |
| 3 | cloud, no homology (quantiles) | shuffled | 60.6 / 33.2 |
| 3 | ISI fractions + cloud, no homology (quantiles) |  | 68.3 / 34.0 |
| 3 | ISI fractions + cloud, no homology (quantiles) | shuffled | 63.1 / 33.4 |
| 3 | spread |  | 33.7 / 21.4 |
| 3 | spread | shuffled | 18.4 / 14.2 |
| 3 | ISI fractions + spread |  | 59.6 / 31.6 |
| 3 | ISI fractions + spread | shuffled | 42.9 / 24.7 |
| 3 | cloud, Rips H0 (whole) |  | 14.9 / 15.9 |
| 3 | cloud, Rips H0 (whole) | shuffled | 9.4 / 12.9 |
| 3 | ISI fractions + cloud, Rips H0 (whole) |  | 49.6 / 26.9 |
| 3 | ISI fractions + cloud, Rips H0 (whole) | shuffled | 43.4 / 24.6 |
| 3 | cloud, Rips H0 in 8 regions |  | 34.4 / 23.0 |
| 3 | cloud, Rips H0 in 8 regions | shuffled | 19.5 / 19.5 |
| 3 | ISI fractions + cloud, Rips H0 in 8 regions |  | 55.4 / 29.2 |
| 3 | ISI fractions + cloud, Rips H0 in 8 regions | shuffled | 45.6 / 27.7 |

### Retina 40 s clean, frozen x4 <-> unrepeated x4

60 neurons. Mean probability of staying in a state: K=2 ISIs shuffled: 0.499; K=2 real: 0.521; K=3 ISIs shuffled: 0.330; K=3 real: 0.395.

| K | readout | control | within A | within B | A -> B | B -> A |
|---|---|---|---|---|---|---|
| ref | ref: ISI fractions |  | 64.3 / 48.8 | 70.0 / 44.3 | 75.0 / 39.4 | 64.7 / 45.2 |
| ref | ref: ISI fractions | shuffled | 64.3 / 48.8 | 70.0 / 44.3 | 75.0 / 39.4 | 64.7 / 45.2 |
| ref | ref: delay regions k=32, H0 (whole cloud) |  | 62.9 / 53.6 | 66.6 / 45.9 | 68.8 / 46.6 | 63.5 / 52.6 |
| ref | ref: delay regions k=32, H0 (whole cloud) | shuffled | 59.5 / 52.4 | 59.8 / 45.3 | 62.9 / 43.5 | 59.1 / 46.2 |
| 2 | mean predicted pdf |  | 61.7 / 46.7 | 64.8 / 45.3 | 67.4 / 37.3 | 59.9 / 42.6 |
| 2 | mean predicted pdf | shuffled | 60.2 / 47.6 | 65.8 / 46.1 | 70.2 / 40.1 | 61.7 / 45.7 |
| 2 | ISI fractions + mean predicted pdf |  | 63.1 / 48.8 | 68.4 / 45.3 | 71.7 / 39.1 | 61.9 / 45.2 |
| 2 | ISI fractions + mean predicted pdf | shuffled | 63.1 / 46.7 | 68.8 / 46.6 | 71.7 / 39.2 | 63.3 / 45.7 |
| 2 | cloud, no homology (quantiles) |  | 63.6 / 46.9 | 66.2 / 45.1 | 68.8 / 38.3 | 60.8 / 44.5 |
| 2 | cloud, no homology (quantiles) | shuffled | 61.4 / 47.9 | 64.5 / 42.2 | 70.7 / 37.7 | 60.8 / 41.4 |
| 2 | ISI fractions + cloud, no homology (quantiles) |  | 66.9 / 47.1 | 69.1 / 47.6 | 71.7 / 40.2 | 63.4 / 46.2 |
| 2 | ISI fractions + cloud, no homology (quantiles) | shuffled | 63.3 / 48.3 | 68.2 / 45.2 | 73.1 / 39.1 | 63.1 / 43.1 |
| 2 | spread |  | 20.7 / 28.8 | 36.2 / 20.5 | 34.3 / 15.2 | 25.5 / 15.5 |
| 2 | spread | shuffled | 22.6 / 26.4 | 30.3 / 15.9 | 29.8 / 8.9 | 20.1 / 7.6 |
| 2 | ISI fractions + spread |  | 44.5 / 38.1 | 56.9 / 30.4 | 57.4 / 24.2 | 46.4 / 23.8 |
| 2 | ISI fractions + spread | shuffled | 40.5 / 33.1 | 53.5 / 21.7 | 54.5 / 15.8 | 39.3 / 15.7 |
| 2 | cloud, Rips H0 (whole) |  | 10.7 / 26.9 | 15.9 / 19.7 | 15.2 / 11.0 | 14.1 / 15.2 |
| 2 | cloud, Rips H0 (whole) | shuffled | 13.6 / 23.8 | 12.8 / 14.3 | 10.0 / 6.9 | 9.5 / 10.2 |
| 2 | ISI fractions + cloud, Rips H0 (whole) |  | 50.0 / 40.2 | 56.5 / 32.4 | 56.9 / 28.0 | 47.4 / 30.2 |
| 2 | ISI fractions + cloud, Rips H0 (whole) | shuffled | 41.7 / 35.7 | 51.8 / 25.7 | 55.7 / 20.4 | 41.4 / 20.7 |
| 2 | cloud, Rips H0 in 8 regions |  | 26.4 / 33.6 | 34.3 / 27.5 | 33.8 / 23.6 | 28.7 / 23.3 |
| 2 | cloud, Rips H0 in 8 regions | shuffled | 26.0 / 31.0 | 24.3 / 21.2 | 25.7 / 14.3 | 21.6 / 22.1 |
| 2 | ISI fractions + cloud, Rips H0 in 8 regions |  | 50.5 / 41.4 | 58.4 / 35.4 | 59.5 / 32.5 | 50.2 / 36.4 |
| 2 | ISI fractions + cloud, Rips H0 in 8 regions | shuffled | 45.2 / 40.7 | 54.2 / 28.6 | 58.6 / 24.1 | 46.2 / 24.5 |
| 3 | mean predicted pdf |  | 62.9 / 50.0 | 67.6 / 46.6 | 72.1 / 40.0 | 62.1 / 48.1 |
| 3 | mean predicted pdf | shuffled | 63.8 / 47.4 | 68.7 / 44.5 | 72.4 / 38.3 | 62.8 / 44.3 |
| 3 | ISI fractions + mean predicted pdf |  | 64.8 / 49.0 | 69.6 / 46.3 | 72.6 / 39.5 | 63.2 / 46.2 |
| 3 | ISI fractions + mean predicted pdf | shuffled | 63.8 / 46.7 | 69.7 / 45.6 | 72.9 / 36.8 | 63.6 / 46.4 |
| 3 | cloud, no homology (quantiles) |  | 65.5 / 51.7 | 66.4 / 45.2 | 71.9 / 42.2 | 62.1 / 44.3 |
| 3 | cloud, no homology (quantiles) | shuffled | 61.9 / 45.7 | 66.2 / 44.8 | 71.9 / 36.7 | 61.6 / 41.9 |
| 3 | ISI fractions + cloud, no homology (quantiles) |  | 66.7 / 51.4 | 70.5 / 44.1 | 74.3 / 42.5 | 63.8 / 41.4 |
| 3 | ISI fractions + cloud, no homology (quantiles) | shuffled | 64.0 / 48.3 | 68.6 / 45.6 | 73.8 / 38.2 | 63.0 / 43.1 |
| 3 | spread |  | 33.1 / 37.6 | 39.6 / 26.1 | 42.6 / 22.3 | 34.5 / 19.5 |
| 3 | spread | shuffled | 10.0 / 26.0 | 19.4 / 13.9 | 21.2 / 7.7 | 13.5 / 5.7 |
| 3 | ISI fractions + spread |  | 53.1 / 47.4 | 59.2 / 37.5 | 62.6 / 36.5 | 52.3 / 35.2 |
| 3 | ISI fractions + spread | shuffled | 35.5 / 35.2 | 48.4 / 25.9 | 47.9 / 19.7 | 39.0 / 21.7 |
| 3 | cloud, Rips H0 (whole) |  | 20.7 / 32.9 | 22.3 / 22.7 | 23.3 / 14.4 | 20.4 / 15.5 |
| 3 | cloud, Rips H0 (whole) | shuffled | 12.1 / 27.1 | 13.6 / 15.9 | 16.0 / 7.7 | 12.5 / 8.3 |
| 3 | ISI fractions + cloud, Rips H0 (whole) |  | 47.4 / 44.3 | 53.4 / 33.1 | 59.3 / 28.7 | 46.1 / 29.0 |
| 3 | ISI fractions + cloud, Rips H0 (whole) | shuffled | 45.5 / 37.4 | 50.1 / 28.6 | 52.9 / 19.9 | 41.3 / 26.2 |
| 3 | cloud, Rips H0 in 8 regions |  | 36.2 / 36.4 | 37.9 / 29.6 | 34.8 / 24.0 | 32.3 / 23.1 |
| 3 | cloud, Rips H0 in 8 regions | shuffled | 21.7 / 31.7 | 22.6 / 21.8 | 24.3 / 16.2 | 22.7 / 15.7 |
| 3 | ISI fractions + cloud, Rips H0 in 8 regions |  | 51.9 / 45.0 | 61.8 / 36.9 | 61.4 / 33.8 | 52.3 / 33.1 |
| 3 | ISI fractions + cloud, Rips H0 in 8 regions | shuffled | 45.7 / 38.8 | 54.0 / 34.1 | 56.2 / 29.6 | 46.9 / 29.3 |

### Allen 30 s, movie <-> spontaneous

60 neurons. Mean probability of staying in a state: K=2 ISIs shuffled: 0.495; K=2 real: 0.600; K=3 ISIs shuffled: 0.324; K=3 real: 0.423.

| K | readout | control | within A | within B | A -> B | B -> A |
|---|---|---|---|---|---|---|
| ref | ref: ISI fractions |  | 44.8 / 27.2 | 35.8 / 26.0 | 25.4 / 12.5 | 24.5 / 14.8 |
| ref | ref: ISI fractions | shuffled | 44.8 / 27.2 | 35.8 / 26.0 | 25.4 / 12.5 | 24.5 / 14.8 |
| ref | ref: delay regions k=32, H0 (whole cloud) |  | 38.4 / 27.0 | 31.0 / 23.0 | 25.7 / 15.2 | 22.6 / 13.6 |
| ref | ref: delay regions k=32, H0 (whole cloud) | shuffled | 30.3 / 23.8 | 23.0 / 20.1 | 19.7 / 11.5 | 16.1 / 10.3 |
| 2 | mean predicted pdf |  | 37.1 / 25.1 | 30.6 / 25.5 | 21.9 / 13.2 | 21.7 / 13.7 |
| 2 | mean predicted pdf | shuffled | 37.6 / 26.8 | 30.6 / 25.2 | 22.0 / 14.4 | 21.3 / 13.7 |
| 2 | ISI fractions + mean predicted pdf |  | 41.7 / 25.7 | 33.6 / 25.0 | 24.3 / 12.4 | 23.8 / 15.3 |
| 2 | ISI fractions + mean predicted pdf | shuffled | 42.5 / 27.2 | 33.7 / 25.4 | 25.5 / 12.4 | 23.5 / 12.7 |
| 2 | cloud, no homology (quantiles) |  | 40.3 / 27.4 | 33.6 / 25.1 | 24.2 / 14.1 | 23.0 / 12.2 |
| 2 | cloud, no homology (quantiles) | shuffled | 36.2 / 25.5 | 29.1 / 24.4 | 22.2 / 11.6 | 20.1 / 11.9 |
| 2 | ISI fractions + cloud, no homology (quantiles) |  | 44.3 / 25.8 | 35.9 / 25.8 | 27.0 / 13.0 | 25.1 / 12.8 |
| 2 | ISI fractions + cloud, no homology (quantiles) | shuffled | 41.4 / 26.5 | 32.2 / 25.3 | 23.9 / 14.6 | 22.5 / 13.2 |
| 2 | spread |  | 24.2 / 17.3 | 19.7 / 15.8 | 12.8 / 6.3 | 14.7 / 6.0 |
| 2 | spread | shuffled | 15.5 / 13.4 | 13.1 / 12.9 | 11.7 / 5.0 | 10.7 / 4.6 |
| 2 | ISI fractions + spread |  | 37.4 / 24.7 | 30.2 / 21.2 | 23.4 / 13.0 | 22.3 / 10.0 |
| 2 | ISI fractions + spread | shuffled | 32.2 / 19.6 | 26.4 / 18.2 | 20.3 / 8.9 | 19.1 / 9.3 |
| 2 | cloud, Rips H0 (whole) |  | 8.4 / 13.8 | 8.0 / 14.2 | 6.8 / 5.6 | 6.0 / 5.4 |
| 2 | cloud, Rips H0 (whole) | shuffled | 6.7 / 12.3 | 5.5 / 11.8 | 5.5 / 4.5 | 4.5 / 5.1 |
| 2 | ISI fractions + cloud, Rips H0 (whole) |  | 34.7 / 24.3 | 30.2 / 22.6 | 21.1 / 12.0 | 20.9 / 12.2 |
| 2 | ISI fractions + cloud, Rips H0 (whole) | shuffled | 29.6 / 19.6 | 25.1 / 19.5 | 17.7 / 9.9 | 17.5 / 10.8 |
| 2 | cloud, Rips H0 in 8 regions |  | 18.7 / 18.6 | 15.6 / 18.7 | 12.8 / 10.8 | 11.0 / 8.5 |
| 2 | cloud, Rips H0 in 8 regions | shuffled | 13.9 / 16.9 | 9.3 / 15.2 | 8.4 / 8.7 | 8.4 / 6.7 |
| 2 | ISI fractions + cloud, Rips H0 in 8 regions |  | 35.3 / 26.0 | 31.5 / 24.1 | 22.5 / 11.9 | 20.7 / 15.1 |
| 2 | ISI fractions + cloud, Rips H0 in 8 regions | shuffled | 31.1 / 20.3 | 25.1 / 19.7 | 18.3 / 9.9 | 18.8 / 9.7 |
| 3 | mean predicted pdf |  | 40.7 / 25.9 | 31.9 / 24.2 | 24.2 / 11.3 | 23.0 / 12.8 |
| 3 | mean predicted pdf | shuffled | 41.4 / 26.1 | 32.7 / 24.7 | 23.4 / 13.2 | 22.5 / 12.3 |
| 3 | ISI fractions + mean predicted pdf |  | 43.9 / 27.0 | 34.4 / 25.3 | 24.7 / 12.9 | 23.9 / 12.7 |
| 3 | ISI fractions + mean predicted pdf | shuffled | 43.1 / 26.3 | 34.7 / 25.4 | 24.7 / 11.5 | 23.6 / 13.1 |
| 3 | cloud, no homology (quantiles) |  | 42.0 / 28.2 | 34.7 / 27.5 | 26.4 / 14.9 | 24.3 / 16.2 |
| 3 | cloud, no homology (quantiles) | shuffled | 37.5 / 26.9 | 30.8 / 24.5 | 23.1 / 12.6 | 22.1 / 12.6 |
| 3 | ISI fractions + cloud, no homology (quantiles) |  | 46.0 / 28.7 | 36.1 / 27.4 | 28.3 / 14.1 | 25.4 / 14.3 |
| 3 | ISI fractions + cloud, no homology (quantiles) | shuffled | 41.2 / 26.3 | 34.0 / 23.4 | 25.0 / 12.2 | 23.9 / 12.8 |
| 3 | spread |  | 21.6 / 19.8 | 17.5 / 16.4 | 15.4 / 7.7 | 15.7 / 9.5 |
| 3 | spread | shuffled | 11.6 / 13.6 | 12.5 / 12.7 | 8.4 / 4.4 | 9.1 / 4.3 |
| 3 | ISI fractions + spread |  | 38.4 / 25.5 | 34.3 / 24.7 | 27.1 / 14.8 | 25.8 / 15.2 |
| 3 | ISI fractions + spread | shuffled | 30.1 / 20.5 | 26.5 / 19.3 | 18.7 / 8.0 | 19.4 / 9.4 |
| 3 | cloud, Rips H0 (whole) |  | 6.8 / 15.1 | 7.0 / 13.0 | 6.5 / 5.3 | 5.7 / 6.1 |
| 3 | cloud, Rips H0 (whole) | shuffled | 5.5 / 12.9 | 3.6 / 11.9 | 4.4 / 3.2 | 3.4 / 5.0 |
| 3 | ISI fractions + cloud, Rips H0 (whole) |  | 31.8 / 21.9 | 26.0 / 20.1 | 20.8 / 11.2 | 20.0 / 11.8 |
| 3 | ISI fractions + cloud, Rips H0 (whole) | shuffled | 27.4 / 21.0 | 23.6 / 19.3 | 16.8 / 7.4 | 16.1 / 7.5 |
| 3 | cloud, Rips H0 in 8 regions |  | 18.8 / 18.6 | 13.7 / 17.7 | 11.3 / 8.7 | 12.3 / 9.4 |
| 3 | cloud, Rips H0 in 8 regions | shuffled | 10.2 / 17.6 | 8.9 / 15.8 | 7.7 / 6.5 | 8.7 / 6.3 |
| 3 | ISI fractions + cloud, Rips H0 in 8 regions |  | 34.2 / 26.6 | 28.2 / 22.9 | 21.0 / 11.7 | 22.0 / 12.5 |
| 3 | ISI fractions + cloud, Rips H0 in 8 regions | shuffled | 30.2 / 21.5 | 23.9 / 21.9 | 17.8 / 10.4 | 18.1 / 11.5 |
