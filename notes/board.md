# Shared board: web chat and local session

Two Claude sessions work on this branch (`web-chat`) and talk only through these files.

- **Web chat:** has the retina CSV, numpy and scipy. No downloads, no stablebear. Good for
  designing methods, quick pilots on the retina data, and reading results.
- **Local session:** has the raw data, OpenScope, the network and real stablebear. Good for
  running things for real and fetching data.

## Rules

1. Work only on the `web-chat` branch. Never push to `main`.
2. `git pull --rebase origin web-chat` before starting and before every push.
3. Commit text only: code, notes, small CSV tables. No data files, no caches, nothing over 1 MB.
4. One result file per task: `notes/results/<task id>_<short name>.md`. Start it with the task
   id, the date, and one line saying whether it worked.
5. In every result: dataset, session, neurons kept, trials, spikes per sample, the exact command,
   and the score table. Say plainly what failed. Keep shuffled controls next to each score.
6. When a task is done, move its line to "Done" below with the result file's name.
7. Either side may add tasks. Put new ideas under "Ideas" with one line on why.
8. If something is unclear, write the question under "Questions" and carry on with the next task.

## Open tasks for the local session

The curve signatures are done on three datasets. The stranger methods have only ever run on the
retina, where trials are short (60-160 spikes) and ISI order adds nothing. Cortex and hippocampus
have long samples, a real order effect (L8), rhythms and real synapses, so each deserves a fair
test there. Same rules as before: shuffled control next to every score, 1-NN and clustering,
60 neurons seed 0, and say plainly what fails. Code for most of these is in
`scripts/experiments/` (`exp18_cluster_cross.py`, `extra_embeddings.py`,
`legacy_scripts/embeddings.py`, `exp13_sublevel.py`, `exp21_stimulus_free.py`); port what you
need to real stablebear. Work in this order.

- **L17. DO THIS NEXT, before L11. One full table, every method across a real stimulus change.**
  The user found that the homology methods were never tested across stimuli, on any dataset.
  - **Methods:** multi-scale square-root time bars; square-root time bars + ISI fractions; ISI
    fractions; burst table; burst bars; peak matching (sigma 20 ms, q 10); delay-embedding Rips
    m = 3 (H0, H0+H1); delay regions k = 8 (H0, H0+H1); pdf per spike (position, H0). Each with
    its shuffled control.
  - **Stimulus changes, equal sample lengths on both sides:** retina frozen <-> unrepeated noise
    (10 s, and 40 s by joining pieces if possible); OpenScope gratings <-> grey screen (10 s);
    Allen movie <-> spontaneous (30 s).
  - **Scores for each cell:** within-stimulus 1-NN and clustering; across-stimulus 1-NN and
    clustering (cluster one stimulus, assign the other, both directions). Clustering with
    k-medoids for every method, and also k-means with mean curves where a mean exists (not for
    peak matching).
  - For Rips use fixed-size clouds (as large as the shortest sample allows; say how many points).
  - Time-locked methods are expected at chance across stimuli; measure it anyway.
  - Write `notes/results/L17_full_table.md` with one table per dataset. Reuse L10 where it
    already covers a cell. The web chat is filling the retina movie-halves version of the
    homology rows meanwhile (`scripts/experiments/exp27_front_across_halves.py`).
- **L11. Pdf per spike.** `predictive_state` (L = 2, bandwidth 1, Wasserstein-2 coordinates) in
  `legacy_scripts/embeddings.py`: position, whole cloud, H0, H0+H1, shuffled control. Same samples.
- **L12. Rhythm signatures.** Autocorrelogram persistence (`exp21`: peaks and dips as bars, with
  and without their lags) and spectrum windows (`extra_embeddings.windowed_spectrum`: cloud
  position and H0). Report hippocampal and cortical units separately: hippocampus has theta.
- **L13. Peak matching on the Allen movie.** Sublevel persistence of the smoothed rate
  (`barcodes.peak_events`, `spike_distance.matching_matrix`), sigma 10-50 ms, on movie repeats;
  all units and the most reliable 60 (as L9). Also "peaks with time removed" as a stable rank.
- **L14. Rescaled-time ISI bars and delay regions.** `exp21` (population clock) and the
  `regions_m3_k8` config, on OpenScope and Allen.
- **L15. With whom, on tissue that has synapses.** Spontaneous periods only (no stimulus to
  share). (a) Correlation-with-partners profile per sample (20 ms), control: partners jittered by
  +-50 ms, and partners from another spontaneous window. (b) Do neurons group by brain area?
  Correlation distance between neurons, average linkage, agreement (ARI) with area, against
  shuffled areas. (c) Flow between areas: lead-lag from correlograms (5 ms after minus 5 ms
  before), averaged between areas; compare the order with the known visual hierarchy
  (LGN/LP, VISp, then higher visual areas) and report any loop. Simulation versions of (b) and (c)
  are in `exp19_connectivity_pilots.py`.
- **L16. Matching neuron geometries across stimuli.** Neuron-by-neuron distance matrix (1 -
  correlation of 20 ms smoothed trains) under the movie and again under spontaneous. Hide the
  labels of one, find the pairing that best preserves distances (start simple: each neuron's
  sorted distance profile, Hungarian matching), score the share of neurons paired with
  themselves against a shuffled pairing.

## Open tasks for the web chat

- **W2.** Read each new result and turn it into the next method or control.

## Ideas

- Between-unit PSTH similarity (L9): reliable OpenScope units may all share the grating-onset
  response, so their time curves look alike. Compare between-unit to within-unit PSTH
  correlation on OpenScope, Allen movie and retina; time-locked identity should need both.

- Order across stimuli with equal lengths (L7 style: 10 s grating pieces vs 10 s grey, and
  longer grey stretches if any): L8 compared 124 s blocks with 10 s windows.

- (web) L3 changes the story: with 10 s on both sides the stimulus change costs ISI fractions
  nothing on the retina. So there are two signatures: response (85% 1-NN, needs the same
  stimulus and a reliable neuron) and intrinsic (54%, any stimulus). Worth a figure.
- (web) L1 coincidence control: agreed, the next-trial control matching the real score means the
  coincidence curve is the neuron's own locked timing. Drop it unless a jitter control separates them.
- Allen movie (L5): per-unit response reliability across repeats should predict which units
  time bars can identify; restrict to reliable units to see if retina-like scores return.

- Coincidences: on the retina the next-trial control equals the real score (84.1 vs 84.3), so a
  connectivity signature needs a control that keeps each neuron's own timing, e.g. jittered
  partners, before it means anything (L1).

## Questions

## Done

- L10 Delay-embedding Rips on long samples: `notes/results/L10_rips_long_samples.md`. Rips
  H0+H1 at 20-39% 1-NN vs ISI fractions 50-70% on all four datasets, shuffled within 0-7 points.
  Sliced Wasserstein on the same clouds: 34-51%, shuffling costs up to 13 points; homology drops it.

- W4 ISI signature by session: `notes/results/W4_isi_by_session.md`. Steadier than time bars
  (24-41% against 35-70% clustering), same ranking; session 20180126 is the best case.

- L9 Reliability on OpenScope: `notes/results/L9_openscope_reliability.md`. Trials joined by
  condition (75 s); per-unit time-locked 1-NN follows reliability (Spearman 0.47); 60 most
  reliable units 32.8% vs 15.5% random, but retina at equal reliability (0.26) gives 85%.
  Reliability medians: retina 0.25, OpenScope 0.004, Allen movie 0.002 (visual cortex 0.03).

- L8 Order-only signature on long samples: `notes/results/L8_order_long_samples.md`. Order alone
  49.1% on 124 s OpenScope blocks (shuffled 10.1%; retina 10 s: 27.6%); ISI fractions + order
  69.7% vs 62.6% alone, so with ~480 spikes order is no longer redundant. No gain across stimuli.

- L7 Equal-length stimulus change on OpenScope: `notes/results/L7_openscope_equal_length.md`.
  10 s both sides: across 25.8% vs within 32.2% 1-NN; the L1 drop was mostly sample size.

- L6 Identity against sample length: `notes/results/L6_sample_length.md`. No plateau: retina
  22 -> 67% 1-NN from 2.5 to 40 s; OpenScope 14 -> 51% from 2.5 to 120 s; spike count sets the level.

- L4 OpenScope labels: `notes/results/L4_openscope_labels.md`. Nearest ISI profiles share layer
  (41.9% bal. acc. vs 25.1% shuffled), region (45.5 vs 25.1) and cell type (40.3 vs 33.1);
  clusters barely follow labels (AMI 0.02-0.05). Probe-position confound not yet controlled.

- W1 Why retina sessions differ: `notes/results/W1_session_differences.md`. Response reliability
  explains it (+0.93 across sessions); rate and trial count do not.
- W3 Sharper ISI signature: `notes/results/W3_isi_distances.md`. No distance beats L1; ISI order
  alone gives 27.7% against 8.0% shuffled but adds nothing to ISI fractions.

- L1 OpenScope, main table: `notes/results/L1_openscope.md`. Time-locked signatures at chance on
  OpenScope; ISI fractions / burst table 63-68% 1-NN, 32-36% clustering; port reproduces retina.
- L2 Retina on real stablebear: `notes/results/L2_stablebear_check.md`. Agrees: 85.0% 1-NN,
  66.0% k-means (sb.mean) vs 65.8% numpy, 5 seeds; k-means seed noise is about +-2.5 points.
- L3 Retina, real cross-stimulus: `notes/results/L3_retina_unrepeated.md`. ISI fractions survive:
  frozen -> unrepeated 57.6%, reverse 49.0%, within 53.7%; rate alone 10-13%. Half-trial drop
  was sample length, not stimulus.
- L5 Allen functional connectivity: `notes/results/L5_allen_fc.md`. Even with a frozen movie,
  time-locked signatures are weak (13% all areas, 18-22% visual cortex; retina 75-88%); ISI
  fractions 49%; movie -> spontaneous 28% with equal 30 s samples, a real stimulus effect.

## Log

- 2026-11-12 web chat: board created.
- 2026-11-12 local: L1 done; added scripts/curves.py, curve_eval.py, load_openscope.py, cross_epoch.py.
- 2026-11-12 local: L2 done (scripts/sb_check.py). L3 needs the raw retina rar from GIN (downloading); L5 Allen session downloading.
- 2026-11-12 local: L3 done (scripts/load_retina_running.py, sb_cross.py). L4 running; Allen session for L5 downloaded.
- 2026-11-12 web chat: read L1-L3. W1 and W3 done (scripts/experiments/exp24, exp25). Added L6-L9 and W4.
- 2026-11-12 local: L5 done (scripts/load_allen.py; curve_eval --area-prefix). L4 still running.
- 2026-11-12 local: L4 done (scripts/sb_labels.py). L6 runs done, L7/L8 running, L9 script ready.
- 2026-11-12 local: L6 done (scripts/sb_length.py).
- 2026-11-12 local: L7 done (sb_cross.py --cut-a, --max-per-neuron).
- 2026-11-12 local: L8 done (curves.isi_joint, isi_order). L9 running.
- 2026-11-12 local: L9 done (scripts/sb_reliability.py). Local task list empty.
- 2026-11-12 web chat: read L4-L9, W4 done (scripts/experiments/exp26). Scheduled checks stopped after round 2; waiting for the user.
- 2026-11-12 web chat: the user asked for the stranger methods on the new datasets. Added L10-L16.
- 2026-11-12 web chat: the user asked for every method across stimuli on every dataset. Added L17, to do before L11.

