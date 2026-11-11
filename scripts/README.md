# Analysis scripts

Can we tell which neuron a single trial came from? Every method here turns trials into a
distance matrix. A trial is then scored by whether its nearest other trial (1-NN) comes from
the same neuron.

## Running it

From the repo root:

```sh
pip install stablebear scikit-learn scipy pandas pyarrow
python scripts/test_methods.py                  # checks the maths, and the stablebear calls
python scripts/run_pipeline.py --list           # every config
python scripts/run_pipeline.py --neurons 60 --jobs 4
python scripts/compare.py --neurons 60          # the summary table
python scripts/cross_segment.py --neurons 60    # does it survive a change of stimulus?
```

Add `--seed 1` to any of them for another random set of neurons, `--session NAME` for
another recording, and leave out `--neurons` to use every neuron of the session.

To run part of it, pass names or patterns, such as
`python scripts/run_pipeline.py "time_bars_*" isi_bars_log isi_histogram`.

Results go to `results/`. Finished configs are skipped, so use `--force` after changing code.
`compare.py` picks a family's settings on half the neurons and scores them on the other half,
so families with many settings get no unfair advantage.

## Files

| File | Contents |
|---|---|
| `embeddings.py` | Spike train to point cloud: the ISI delay embedding |
| `barcodes.py` | Spike train to bars: time bars, ISI bars, peaks of the smoothed rate |
| `spike_distance.py` | Matching distances: spike distance and peak matching |
| `run_pipeline.py` | All configs, stablebear calls, baselines, 1-NN and AMI scoring |
| `compare.py` | Held-out comparison of the saved distance matrices |
| `cross_segment.py` | Half-trials matched across different parts of the movie |
| `test_methods.py` | Checks that each claim below holds |

## Methods

| Configs | Method | Homology | Uses stablebear |
|---|---|---|---|
| `m*_all`, `m*_fixed` | ISI delay embedding, Rips | Real: H0, H1 | Yes |
| `regions_*` | Delay cloud split around fixed anchors, one barcode per region | Real: H0, H1 | Yes |
| `isi_bars_raw` | Stable rank of ISIs. Same bars as H0 Rips of the spike times | Real but trivial | Yes |
| `isi_bars_log` | The same with log ISIs | Rescaled by hand | Yes |
| `time_bars_*` | One bar `[t, t+s)` per spike, Betti curve | Written by hand | Yes |
| `burst_bars_*` | One bar per burst: H0 of the train thickened by `s`, Betti curve | Real: H0 | Yes |
| `peak_s*_q*` | H0 of the smoothed rate, matched in time | Real: H0 | No, scipy and numpy |
| `window_count_*` | Spikes in a sliding window, L1 | None | No |
| `spike_distance_q*` | Victor-Purpura | None | No |
| `van_rossum_*` | L2 between smoothed trains | None | No |
| `isi_histogram`, `spike_count` | Baselines | None | No |

**Why Rips struggles here.** Rips barcodes do not change when a cloud moves, and they do not
record when a feature happened. Identity sits largely in when a neuron fires during the
movie, so the delay embedding mostly sees the ISI distribution. Its shuffled control scores
about the same.

**Time bars.** With movie time as the filtration parameter, time survives. A spike at `t`
is a component born at `t` that disappears at `t + s`, so the Betti curve at `t` is the number
of spikes in the last `s` seconds. With `s = 2/q`, shifting one spike by `d` costs `min(qd, 2)`
and adding one costs 1, as in the spike distance. Edits close together can cancel, so
this is a lower bound on the spike distance. It is much faster. Nothing merges, so
this is homology only in a formal sense.

**Burst bars.** Thicken each spike `t` to `[t, t+s)` and take the union. Spikes closer than
`s` fuse into one piece, a burst, and each burst is one bar from its first spike to its last
spike plus `s`. The Betti curve is 1 during a burst and 0 outside, so the spike count inside a
burst is ignored. The number of bursts at scale `s` is the number of ISIs above `s`, plus 1,
so ISI bars count the same pieces without their times.

**ISI bars.** The H0 bars of spike times on a line are the ISIs. The stable rank counts how
many ISIs exceed each length. It complements the time bars: one says when, the other how.

**Peak matching.** Smooth the train into a rate `r(t)` with a Gaussian of width `sigma`. Sweep a
level down from the top. Each peak of `r` starts a component of `{r > level}`, which ends when
it merges with a taller peak. Each peak's bar length is its prominence, and
`prominence * sigma * sqrt(2 pi)` is its size in spikes. Trials become time-ordered lists of
peaks, compared with the same dynamic program as the spike distance, with masses.

## Scores

- `acc_1nn`: each trial takes the label of its nearest other trial. This needs labelled
  trials of every neuron.
- `cluster_acc`: k-medoids with k = number of neurons, which uses no labels to form the
  groups. Each cluster is then paired with one neuron, and the score is the share of trials
  in the right cluster. `cluster_ami` and `ami` (average linkage) measure the same agreement
  without the pairing.
- `cross_segment.py`: methods locked to movie time must drop to chance on `cross`. What
  survives there does not depend on the stimulus.

## Caveats

- Every trial shows the same movie, so features locked to movie time may describe the response
  to this movie, not the neuron.
- `compare.py` scores 50 random splits of one session.
- `time_bars` and `window_count` are the same function. `window_count` uses a grid with step
  `s/20`, while `time_bars` integrates exactly.
- Regions with under 2 points are stored as two equal points, because empty clouds are
  undocumented in stablebear.
- Changing `embeddings.py` or `barcodes.py` needs `--force`, because saved results do not
  record the code version.
