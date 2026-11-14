# The analysis, explained

This file explains what the code in `analysis/` does, what every figure shows, what the
numbers mean, and which questions the professor is likely to ask. Read it before writing the
Methods and Results sections of the report.

## 1. How to run it

From the repo root, with the virtual environment active:

```bash
python analysis/01_explore.py        # data figures                     (~5 s)
python analysis/02_stable_ranks.py   # barcodes and stable ranks         (~5 s)
python analysis/03_identify.py       # main experiment, 5 neuron draws   (~4 min)
```

Figures go to `figures/analysis/`, tables to `results/` (`identification.md` is the main one).
All scripts take `--n-neurons`, `--seed`/`--seeds` and `--min-rate`; run with `--help`.

## 2. The question

> Given **one** 10 s trial of spikes, can we tell **which neuron** it came from?

Every trial shows the same frozen noise movie, so a neuron's trials are repeats under identical
conditions. If each neuron has its own "signature", trials of the same neuron should be closer
to each other than to trials of other neurons.

**Setup:** 40 neurons drawn at random from the 9 sessions (only neurons firing ≥ 1 Hz on
average), 30 trials each, so 1200 trials. Guessing at random gives **2.5 %**. Everything is
repeated for 5 different random draws of 40 neurons, and we report mean ± std.

## 3. The pipeline, step by step

```
trial (spike times) → point cloud → Vietoris–Rips → barcodes H0, H1 → stable ranks
                    → L1 distance between stable ranks → which neuron is closest?
```

This is the same recipe as the earlier student projects in `prev-projects/` (Parkinson,
music, epilepsy).

### 3.1 Two point clouds per trial (`tda_utils.py: cloud_line, cloud_delay`)

**LINE cloud (method T1).** The spike times as points on the real line ℝ.
In the Rips filtration, two neighbouring spikes get connected exactly when the scale t
reaches the gap between them, i.e. their **interspike interval (ISI)**. So:

- every H0 bar is `[0, ISI)`: there is **one bar per ISI** (checked in the code with an `assert`);
- the H0 stable rank at t = **number of ISIs longer than t**.

This is the report's "first procedure" (partition the data at scale t, count the
classes). It's the simplest possible TDA representation of a spike train. Good point for the
oral: *on a line, H0 persistence gives back exactly the ISIs.*

**DELAY cloud (method T2).** One point in ℝ² per pair of consecutive ISIs:
`(log10 ISI_i, log10 ISI_{i+1})`. This is a delay (Takens-style) embedding. It shows **which ISI
follows which**: a burst (short, short) sits bottom-left, a pause after a burst
(short, long) sits bottom-right, and so on.

- We use **log** ISIs because ISIs go from ~1 ms (inside bursts) to seconds (pauses). On a linear
  scale all the short ISIs would be squashed into one point near 0.
- H0 of this cloud: how the points cluster. H1: loops (holes) in the cloud.
- Trials with fewer than 2 ISIs (46 of 1200, from 4 neurons that are almost silent in some
  trials) get a single point. That has no finite bars, so its stable rank is 0.

### 3.2 Barcodes and stable ranks (stablebear)

`stablebear.persistence.compute_persistent_homology` builds the Vietoris–Rips filtration
(using Ripser) and returns the barcodes. We use `reduced=True`, which drops the single infinite
H0 bar, so every function goes to 0.

`barcode_to_stable_rank` turns a barcode into the function

```
stable rank(t) = number of bars longer than t
```

This is a **piecewise constant function** (PCF). Two reasons to use it instead of the raw
barcode:
1. PCFs live in a vector space: we can add them, average them, and measure distances
   (`∫ |f − g| dt`).
2. It is **stable**: moving the points a little changes the function a little (the theorem in
   Section 1 of the report).

### 3.3 Distances and scores (`tda_utils.py: pcf_distances, score_*`)

The distance between two trials is the **L1 distance between their stable ranks**,
`d(f, g) = ∫₀^∞ |f(t) − g(t)| dt`, computed by `stablebear.pdist(p=1)`.

For H0+H1, the H0 and H1 distance matrices are each divided by their median before being
added. Otherwise H0 (many more bars) would completely dominate H1.

Two ways to decide which neuron a trial belongs to. Both are evaluated with **leave-one-out
cross-validation**: a train/test split where the test set is one trial. Each trial in turn is
the test trial, and the other 1199 trials are the training set.

- **1-NN:** find the training trial closest to the test trial. Correct if it comes from the
  same neuron.
- **Nearest mean:** average the stable ranks of each neuron's training trials (the "average
  stable rank" of the earlier projects) and assign the test trial to the closest average.

The test trial's neuron is never used to classify it, only to check the answer.

**Implementation shortcut (a likely exam question).** Removing trial i from the data changes
only one average: the average of the neuron it comes from (the others never contained it).
So the code computes the averages once with all trials, and then corrects that single
distance with `‖mean without i − f_i‖ = n/(n−1) · ‖mean − f_i‖`. The code uses the label of
i to find which entry to correct, but the result is exactly the same as recomputing all 40
averages without trial i. We checked this: the full recomputation gives the same prediction
on 150 of 150 test trials. It's only a speed-up, not a use of the answer.

**Why not split into two halves (15 train / 15 test trials)?** It would also be correct, but
the averages would use 15 trials instead of 29, so the scores would be lower and noisier.
Leave-one-out uses the data better and tests every trial once.

### 3.4 Baselines and the control

To know whether topology adds anything, we compare with methods that use no topology:

- **B1 firing rate:** only the number of spikes. This tells us how much of the identification is
  "just the rate".
- **B2 ISI histogram:** histogram of log10 ISI (40 bins, 1 ms to 10 s), L1 distance. It's the
  non-topological twin of T1: it uses the same information (the set of ISIs).

The **control C** shuffles the ISIs inside every trial before building the delay cloud. The ISI
distribution stays identical but the **order** is destroyed. If T2 and C score the same, the
delay cloud only sees the ISI distribution, not the patterns of consecutive ISIs.

The code also runs a sanity check: with randomly permuted neuron labels the 1-NN score drops to
2.6 %, which is chance. This shows the pipeline doesn't cheat.

## 4. The figures

| File | What it shows | What to say |
|---|---|---|
| `explore_rasters.png` | Spikes of 3 neurons, all 30 trials | Vertical stripes: the response is locked to the movie. The neurons fire at very different rates. The top neuron is almost silent in its first ~14 trials (the dataset's slow rate drift) |
| `explore_isi_hist.png` | log ISI distributions of the same 3 neurons | Different shapes: some neurons fire in bursts (peak at a few ms), others more regularly |
| `explore_delay_clouds.png` | Delay cloud of a typical trial and of all trials, per neuron | The clouds look different (clusters, density, position). This is what TDA sees |
| `sr_line_example.png` | One trial: spike times → H0 barcode (one bar per ISI) → stable rank | The cleanest illustration of the pipeline |
| `sr_delay_example.png` | One trial: delay cloud → H0 and H1 barcodes → stable ranks | Same pipeline in 2-D, with H1 loops (short bars = small holes) |
| `sr_mean_per_neuron.png` | Average stable rank ± std over trials, 5 neurons | The classic figure of earlier projects. The bands are separated for some neurons and overlap for others, which explains why the scores are well above chance but far from 100 % |
| `id_distance_matrix.png` | Distances between the trials of 10 neurons, sorted by neuron | Light squares on the diagonal = trials of the same neuron are close. Visible for all methods |
| `id_mds.png` | Trials placed in 2-D (classical MDS) so that distances are preserved, 8 neurons | Some neurons form their own group (pink, brown), others overlap |
| `id_scores.png` | Accuracy of every method, with the chance line | The main result |
| `id_sample_length.png` | Accuracy for 10, 20 and 30 s samples | More data → better identification |

## 5. Results

Accuracy (%), 40 neurons, 10 s trials, mean ± std over 5 neuron draws, chance = 2.5 %
(from `results/identification.md`):

| Method | 1-NN | Nearest mean |
|---|---|---|
| B1 firing rate | 13.2 ± 2.5 | 16.0 ± 3.0 |
| B2 ISI histogram | 38.1 ± 7.4 | 42.1 ± 5.3 |
| T1 LINE H0 | 32.1 ± 5.3 | 31.2 ± 3.6 |
| T2 DELAY H0 | 27.9 ± 6.2 | 29.0 ± 4.6 |
| T2 DELAY H1 | 13.7 ± 3.6 | 18.6 ± 3.5 |
| T2 DELAY H0+H1 | 26.0 ± 5.8 | 27.3 ± 4.9 |
| C DELAY H0+H1 shuffled | 24.6 ± 5.9 | 27.6 ± 5.0 |

Nearest-mean accuracy against sample length (`results/sample_length.csv`):

| Method | 10 s | 20 s | 30 s |
|---|---|---|---|
| B1 firing rate | 16.0 | 17.8 | 17.3 |
| B2 ISI histogram | 42.1 | 49.3 | 51.2 |
| T1 LINE H0 | 31.2 | 35.5 | 37.7 |
| T2 DELAY H0+H1 | 27.3 | 32.1 | 35.0 |
| C DELAY H0+H1 shuffled | 27.6 | 30.4 | 33.8 |

### What the results mean

1. **Stable ranks identify neurons far above chance.** From one 10 s trial, the LINE H0 stable
   rank finds the right neuron among 40 about 31 % of the time, **12 times chance**.
   Each neuron does have a signature.
2. **It isn't just the firing rate.** Rate alone gives 13–16 %. The stable ranks roughly double
   that, so they capture *how* the neuron fires, not only *how much*.
3. **The delay cloud does not see more than the ISI distribution.** T2 (27.3 %) and its
   shuffled control C (27.6 %) score the same. The topology of the delay cloud is decided by
   *which* ISIs occur, not by their *order*. That's an honest negative result, and an
   interesting one: the H0 of the delay cloud mostly measures how spread out the ISIs are.
4. **H1 is weak** (14–19 %) and adding it to H0 makes H0 slightly worse (29.0 → 27.3 %). Short
   trials give clouds of only ~10–100 points, too few to have meaningful loops. The H1 bars are
   short, which is mostly noise.
5. **The simple ISI histogram is the best method** (42 %). Our check: a histogram on a
   **linear** ISI scale scores 24–32 % (3 draws), about the same as T1. So the log scale, which
   makes short ISIs (bursts) count as much as long ones, is what gives the histogram its edge.
   The L1 distance between LINE stable ranks is an integral over linear time, so it's
   dominated by the long ISIs. (Feeding log ISIs into the LINE cloud didn't help: 23–31 %.)
6. **More data helps every method except rate** (+6–9 points from 10 s to 30 s). The
   signatures are real but noisy in short samples.

**One-sentence conclusion:** topological summaries (stable ranks) of single spike trains
identify neurons well above chance, but in this setting the information they use is the ISI
distribution, which a well-chosen simple histogram captures even better.

## 6. Questions you might get at the oral

- **Why is H0 of the LINE cloud equal to the ISIs?** On a line, at scale t, two neighbouring
  points are connected iff their gap ≤ t. A component dies when it merges with its neighbour,
  i.e. at t = the gap. So the n spikes give n−1 finite bars of lengths = the n−1 ISIs.
- **Why stable rank and not the barcode?** It's a function, so we can average and compute
  distances. And it's stable to noise.
- **Why L1?** It's the standard distance between stable ranks (used in the course and earlier
  projects), and stablebear computes it exactly for PCFs.
- **Why log ISIs in the delay cloud?** ISIs span three orders of magnitude (ms to s). Without
  the log, all burst ISIs collapse to one point.
- **Why divide by the median when adding H0 and H1?** The two distances have different scales.
  Without it, H0+H1 ≈ H0.
- **How do you know it isn't luck?** 5 random neuron draws (the std is in the table), a
  label-shuffle check that gives chance (2.6 %), and leave-one-out scoring.
- **Why is the delay cloud no better than its shuffle?** Shuffling keeps the set of ISIs, so it
  keeps the set of coordinates of the points. Only the pairing changes. Rips H0/H1 of these
  small, noisy clouds are mostly decided by the spread of the coordinates.
- **Why not 100 %?** Only 10 s per trial (~10–150 spikes), neurons of the same type behave
  similarly, and the firing rate drifts over the recording.

## 7. Limitations

- Every trial shows the **same movie**, so a signature could partly be "the response to this
  movie" rather than the neuron itself.
- The firing rate **drifts** over the recording (more spikes in later trials; some neurons are
  almost silent at the start). This adds trial-to-trial variation.
- 10 s samples are short; scores grow with longer samples.
- Only neurons firing ≥ 1 Hz are used.

## 8. Possible extensions (only if time allows)

- Use spike **timing** in the movie (the vertical stripes of the rasters). The earlier analysis
  in git history found that timing-based methods identify neurons much better (> 80 %), but
  they are harder to present as topology.
- Compare stable ranks on a log scale of t, to give bursts more weight.
