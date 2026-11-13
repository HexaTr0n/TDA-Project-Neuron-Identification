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

- **L21. Three fixes for what Rips cannot do (from the user).** On real stablebear, on the
  whole-cloud versions of L20, all datasets, within and across, 1-NN and clustering, each with
  ISIs shuffled.
  - **H1 on the whole cloud.** H0+H1 for plain delay Rips and for delay regions (8 and 16
    anchors). H1 was only ever tested on 25-point clouds, where it hurt.
  - **Betti curves instead of stable ranks for H1.** A stable rank keeps only bar lengths; a
    Betti curve also keeps the scale at which each loop lives. `barcode_to_betti_curve`, same
    L1 distance. Report H1 alone and H0 (stable rank) + H1 (Betti curve).
  - **Time-filtered complex** (the proposal slides). Fix a Rips scale r in ISI-pattern space
    (delay points, m = 3, log ISIs; r = 0.5, 1, 2) and let the points enter in spike order: a
    point enters at the time of its last spike, and an edge exists between two points closer
    than r from the moment both are in. H0 bars then live on the time axis: a component is born
    when a pattern unlike every earlier one appears and dies when a later point joins it to an
    older component. Summaries: the Betti curve over time; the number of births so far over
    time; the stable rank of lifetimes. stablebear only builds Rips from points or a distance
    matrix, so this may need the bars computed outside it (union-find, as in
    `scripts/experiments/exp32_rips_fixes.py`) and handed over as `Barcode`s; say which you did.
    Expected to be locked to the stimulus, so report within a repeated stimulus and across.
  - The web chat is piloting the time-filtered complex and H1 inside regions on the retina
    (`exp32_rips_fixes.py`; its own Rips code is too slow for H1 on whole clouds without regions).
  - Write `notes/results/L21_rips_fixes.md`.

- **L22. Headline homology numbers (from the user: "good homology numbers, even if we rely on
  counting spikes").** After L20 and L21. One table per dataset in the L17 layout (retina 10 s
  with piece 2 dropped, retina 40 s, OpenScope 10 s and 124 s, Allen 30 s; within each stimulus
  and across; 1-NN / k-medoids / k-means with mean curves), real stablebear, every row built
  from barcodes only:
  - **Delay regions on the whole cloud**, H0, point count left in, with the number of anchors
    that L20 found best (fix one number for all datasets; do not pick per dataset).
  - **The same plus ISI bars** (stable rank of the log ISI bars, not divided by the count):
    the sum of the two distance matrices, each divided by its median over pairs of outside
    neurons. Both are H0, one of the delay cloud and one of the spike times on a line.
  - **Within a repeated stimulus only: the same plus burst bars** (H0 that keeps time).
  - Next to every row: ISIs shuffled, and the twin without bars (region counts; spike count for
    the ISI bars), so it is visible what the bars add beyond counting.
  - Reference rows from L17 on the same samples: ISI fractions, burst table, multi-scale
    square-root time bars.
  - **How stable is it:** repeat the retina 10 s and OpenScope 124 s tables for neuron seeds
    0-4 and give mean and range; and give the retina within-frozen row for all nine sessions.
  - Write `notes/results/L22_headline_homology.md`. Start with a four-line summary: best
    barcode-only score within and across on retina and on cortex, each with its control.

  - **Two corrections from L24 (web, 2026-11-13):** use `retina40clean` for the 40 s retina
    table; and for Allen report "across" on the pair that is adjacent in time (movie 1 <->
    spontaneous early) next to the L17 pair, since L24 shows that time apart costs more than
    the stimulus change.

- **L23. H0+H1: make it work or close it.** H1 lowered H0 in every test so far (L19 on 25 and
  100 points; W8 inside regions on the retina). Likely reasons: at these cloud sizes H1 bars are
  sampling noise, and a plain sum gives that noise the same weight as H0. After L21:
  - If H1 on the whole cloud (L21) is nowhere above its shuffled control, write one line under
    "Done" and stop.
  - Otherwise try, on the dataset where it was above: (a) divide each degree's distance by its
    median over outside-neuron pairs and weight H1 by 0.25, 0.5, 1; (b) keep only H1 bars
    longer than the 95th percentile of H1 bar lengths in ISI-shuffled clouds of the same size;
    (c) clouds of at least 200 points only.
  - Write `notes/results/L23_h1.md`.

- **L25. Per-spike mixture pdf with the entire history (the user's idea; small).** After L22.
  A hidden Markov model on log ISI, fitted per sample: K states, one Gaussian each, a
  transition table. After every ISI the filter gives the weights of the next ISI's mixture, so
  each spike becomes a mixture-of-Gaussians pdf that depends on the whole past. Code and the
  retina result (negative: almost no memory there) are in
  `scripts/experiments/exp34_hmm_pdf_per_spike.py` and `notes/results/W12_hmm_pdf_per_spike.md`.
  - Run it where ISI order matters: OpenScope 124 s and `retina40clean` (and Allen 30 s if
    cheap). K = 2 and 3; ISIs shuffled as the control.
  - Readouts as in W12: mean predicted pdf, the cloud without homology, the spread, Rips H0 of
    the cloud (whole, and in 8 regions with anchors from outside neurons) in stablebear, and
    each added to ISI fractions. Report the mean probability of staying in a state.
  - Compare with ISI fractions and whole-cloud delay regions on the same samples.
  - Write `notes/results/L25_hmm_pdf.md`.

- **Freedom to explore (from the user).** If other ideas come up while implementing L18 or
  L19, try them as well, as long as the original task is also done as specified. Give each
  extra its own short section or result file, with its control, and add one line about it
  under "Ideas". The three narrow designs under "Ideas" are candidates.

- **Status (local, 2026-11-13 ~10:15).** L24 done.
  - Only L20's last dataset runs now: OpenScope 124 s, with whole clouds up to 8667 points.
    It takes about 14 GB with swap, so it runs alone.
  - L21: retina 10 s and OpenScope 10 s done. The Allen 30 s run crashed on bars of negative
    length (fixed in `c7331ba`).
  - Queued after L20 (`results/run_L21c.sh`): Allen 30 s, `retina40clean`, OpenScope 124 s.
  - L21 retina 10 s interim, 1-NN within frozen trials (in brackets: ISIs shuffled):
    - H1 helps only on the plain whole cloud as a Betti curve: H0 31.0 (28.8) -> H0 + H1
      Betti 36.4 (31.5).
    - Inside regions H1 still lowers H0: k=8 42.4 -> 36.9-37.2.
    - Time-filtered complex at most 17.2 (13.6).
  - L20 interim (4 of 5 datasets), delay regions H0, 1-NN within A / A -> B (in brackets: ISIs
    shuffled within A):
    - Whole cloud beats every other sampling: retina 10 s k=8 42.4/41.3 (37.6) against 25
      random points 27.9/28.4.
    - Time crop 70% is in between; 50% and consecutive stretches are lower.
    - Anchors k = 16-32 are best. Retina 40 s whole cloud k=32: 67.4/70.7 (60.5).
    - Dividing by the point count costs 0-4 points.
  - The previous L24 edit had removed this section's header and W2 by mistake; restored here.

## Open tasks for the web chat

- **W2.** Read each new result and turn it into the next method or control.
- **W9.** Check this branch every hour (from 2026-11-13 about 01:30, up to 12 rounds): read new
  result files, answer anything under "Questions", add follow-up tasks.
- **W10.** When L20-L22 are in, bring `notes/state.md` and the write-up up to date. L19x already
  changes one claim: at equal data, regional Rips is level with or ahead of the ISI
  distribution on long samples.

## Ideas

- (web, recommended next, from the L19 status at 21:00; **done as L19x**) **Was Rips held back by cloud size?**
  With 100-point clouds on retina 40 s, delay regions (m = 2, K = 16) H0 ties ISI fractions
  within a stimulus (64/63 against 64/67), where 25-point clouds gave about half. ISI fractions
  always used every ISI. A fair test: the same neurons and samples, cloud sizes 25, 50, 100,
  200 and 400 points (long samples: retina 40 s, OpenScope 124 s), delay regions m = 2 K = 16
  H0 and its counts twin, against ISI fractions computed from the SAME number of randomly
  chosen ISIs. If the curves coincide at equal data, the earlier gap was the subsampling, and
  the statement becomes "regional Rips equals the ISI distribution at equal data, and adds
  0-3 points of order".

- (web) Three narrower designs, replaced by L19 but still open as extras. General rules for them: The user
  feels the Markov/pdf, spectral and cross-correlation families were not thought through. Each
  design below follows what we learned: (1) Rips cannot see where a cloud sits, so identity must
  be turned into shape or pinned by fixed regions; (2) a regional stable rank starts at the
  number of points in the region, so regions carry a histogram and the bars add local spread;
  (3) each point needs enough data; (4) the summary must be a PCF so that k-means with
  `sb.mean` works. **For every design report:** 1-NN, k-medoids and k-means; within each
  stimulus and across (as in L17, same selection code); its control; and its "no homology"
  twin, so we see what the barcode adds. Datasets: retina 10 s (piece 2 dropped) and 40 s,
  OpenScope 10 s and 124 s blocks, Allen 30 s, unless a design says otherwise.

- **Extra A. Markov-state regions (Markov / pdf family).**
  - **Idea.** A Markov chain on ISI states is a table of transitions. Make each transition a
    region of the delay cloud and take Rips inside it. The stable rank of region (i, j) starts
    at the number of i -> j transitions, so the barcode contains the transition matrix; its bar
    lengths say how tightly that transition's ISI pairs concentrate. Reversing a train
    transposes the regions, so this is orientation-aware (the slide's fix).
  - **Cloud.** Points (log I_k, log I_(k+1)), m = 2.
  - **States.** 4 states from fixed log-ISI edges: the 25, 50, 75% quantiles of ISIs pooled over
    neurons NOT in the comparison, one set of edges per dataset. This gives 16 regions.
  - **Barcode.** Subsample the cloud to a fixed size (25 points on short samples, 100 on long),
    20 subsamples; Rips H0 inside every region (a region with fewer than 2 points has an empty
    barcode); stable ranks averaged over subsamples and concatenated over the 16 regions. Also
    report H0+H1.
  - **Control.** ISIs shuffled.
  - **No-homology twin.** The 4 x 4 transition counts of the same subsamples (the stable ranks'
    values at zero), L1.
  - **Also report** m = 3 with 3 states (27 regions) on the long samples only.
  - **Compare with** delay regions k = 8 and ISI fractions from L17.

- **Extra B. Persistence spectrum (spectral family).**
  - **Idea.** In a sliding-window embedding a rhythm becomes a loop, and the loop is roundest
    when the window matches the period. So sweep the window length and record H1: a
    topological periodogram, with the frequency kept as the coordinate instead of being lost
    as a position.
  - **Signal.** Spike train smoothed with a Gaussian of width w / 8 for window length w.
  - **Embedding.** For each w in 12 log-spaced values from 20 ms to 2 s: points
    (f(t), f(t + w/8), ..., f(t + 7w/8)) for t on a grid of step w / 8; drop windows with no
    spike; subtract each window's mean and divide by its norm (as SW1PerS does); keep 100
    points by farthest-point sampling (fewer if the sample has fewer non-empty windows; say so).
  - **Barcode.** Rips H1 per w. Two summaries: (a) the longest H1 bar as a function of w, a PCF
    with 12 steps; (b) the H1 stable ranks of all 12 w concatenated.
  - **Control.** ISIs shuffled, and a second control with each spike jittered by +-w/2 (computed
    per w), which removes rhythm at that scale but keeps slower structure.
  - **No-homology twin.** Power of the smoothed train at the 12 frequencies 1/w (Welch or direct
    Fourier sum), L1 after dividing by total power.
  - **Samples.** Long ones only: retina 40 s, OpenScope 124 s, Allen 30 s movie and spontaneous.
    Report hippocampal and cortical units separately as in L12.

- **Extra C. Lead / lag persistence diagram (cross-correlation family).**
  - **Idea.** Earlier versions either used partner names (a lookup table) or threw them away and
    the lags with them. Keep the lag as the meaningful coordinate: for every partner take the
    persistence of its cross-correlogram with the neuron, and sort the resulting bars into
    "partner leads", "synchronous" and "partner follows". The three stable ranks say how many
    strong partners fire before, with and after the neuron: a per-neuron in/out profile, which
    is as close to connectivity as spike trains get.
  - **Samples.** Long ones, because a pair's correlogram needs many spikes: Allen spontaneous in
    300 s stretches and the movie blocks in 300 s stretches (time-matched pair for "across", as
    in L16); OpenScope grey screen and grating blocks in the longest equal stretches available.
    Retina 40 s for completeness (no synapses between these cells; expect nothing).
  - **Partners.** A fixed random set of 200 units per dataset, the same for every sample, each
    with >= 1 Hz; never the neuron itself.
  - **Per partner.** Cross-correlogram at 1 ms bins for lags within +-25 ms; subtract a slow
    baseline (the correlogram smoothed with a 10 ms boxcar) and divide by the square root of
    the baseline (noise units); find peaks and their prominences (sublevel persistence of the
    negated curve); keep the most prominent peak and its lag.
  - **Barcode.** Three sets of bars [0, prominence]: lag in [-15, -1.5) ms (partner leads),
    [-1.5, 1.5] (synchronous), (1.5, 15] (partner follows). Stable rank of each, concatenated.
  - **Second variant where areas exist (Allen, OpenScope).** Split partners by brain area as
    well (hippocampus, thalamus, primary visual, higher visual, other), so the signature says
    which areas lead and follow the neuron.
  - **Control.** The neuron's own spikes jittered by +-25 ms (removes millisecond structure,
    keeps rates and slow co-fluctuation); and partners taken from another stretch.
  - **No-homology twin.** The summed correlogram over the 200 partners as a function of lag
    (population coupling curve, 51 bins), L1.


- (web, from the user's question on Rips) Three ways to get identity into a Rips barcode, since
  Rips cannot see where a cloud sits. Not yet tasks; waiting for the user's go-ahead.
  1. **Conditional-pdf cloud with references.** Split ISIs into about 8 groups by the previous
     ISI; one next-ISI pdf per group (quantile coordinates, Wasserstein). Add fixed reference
     points to every cloud: the sample's overall ISI pdf, a Poisson pdf at the same rate, a few
     fixed pdfs at set time scales. No memory = the 8 points coincide; shuffled collapses by
     construction. Needs about 60 ISIs per group, so long samples only.
  2. **Persistence spectrum.** Sliding-window embedding (f(t), f(t+tau), ...) of the smoothed
     train for a range of window lengths; record the longest H1 bar at each length. A PCF over
     window length that peaks at the neuron's rhythms. Hippocampal units first.
  3. **Lag diagram.** For each partner, persistence of the cross-correlogram (bars with lag and
     prominence); pool over partners into one diagram per neuron; compare by matching in lag.
     Spontaneous periods only.
- (web, from L19's own conclusion) **Scale the joint cloud's coordinates on outside neurons.**
  With equal weights, seven coordinates dilute the two good ones. Fit one weight per coordinate
  on neurons outside the comparison (for example by maximising 1-NN on those), then rerun the
  joint cloud's regional H0 and sliced W1. Not a task yet; only worth it if L22 leaves a gap.
- (web) L16 is the strongest "with whom" result so far: 99.5-100% of 851 units re-identified
  from correlation distances alone within a condition. Worth a drift-free stimulus test.

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

- (local) Retina "unrepeated" piece 2 (seconds 20-30 of the 50 s before each frozen trial)
  repeats across blocks: PSTH reliability 0.244, the others ~0. Not the frozen sequence. What is
  it in the stimulus protocol? L3 and L6 (and the 40 s stretches of L10-L12, L17) include it;
  L17's 10 s retina table is rerun without it. Should load_retina_running.py drop it?
  - **Answer (local, 2026-11-13, part of L24).** The repeat is exactly 5 s: seconds 24.85-29.9 of
    each 49.65 s running stretch. That is pulses ~750-900 of R = 1500, the middle 300 of its
    3000 frames.
    - Reliability in 0.2 s windows is 0.45-0.50 inside it (all 30 blocks) and 0.00 +- 0.004
      everywhere else.
    - It is not the frozen sequence: the rate-corrected PSTH match at any lag is at most 0.018.
      It is not a copy of any other part of the stretch either (at most 0.037).
    - The pulse intervals are regular (at most 33.3 ms), so no timing glitch.
    - The manual says running noise resumes from the last seed after each frozen run. A fixed
      300-frame chunk in the middle of every run is not in the documented protocol, so I cannot
      say from the files *why* it repeats. Recreating the frames from the seeds would settle it;
      not done.
    - Consequences:
      - 10 s piece 2 (20-30 s) holds all of it, so dropping piece 2 removes it fully.
      - The 40 s joined stretches (0-40 s) hold it, so 12.5% of every unrepeated 40 s sample
        repeats. Time-locked rows there must be redone (L17 time bars "within unrepeated x4"
        33.9% are mostly this). ISI rows are barely affected (10 s ISI fractions 45.0 with vs
        43.5 without).
      - Fix: build the 40 s stretches from 0-24.8 s plus 30.0-49.65 s, which gives 44.45 s, and
        keep the first 40 s. I will add this as an option to `load_retina_running.py` with L24.

## Done

- L20 Cloud sampling: `notes/results/L20_cloud_sampling.md`.
  - The whole cloud (every point, one barcode) beats crops, random draws and consecutive
    stretches on all five datasets.
  - With 32 regions (fixed for L22), whole-cloud delay regions H0 are level with or above ISI
    fractions on long samples: OpenScope 124 s 67.2 vs 64.2 (shuffled 56.4); retina 40 s 67.4
    vs 64.3.
  - On short samples they are 5-7 points behind: Allen 39.0 vs 44.8, retina 10 s 43.7 vs 50.9.
  - Dividing by the point count changes at most 2 points. H0+H1 is 6-12 points below H0.

- L24 Data checks: `notes/results/L24_data_checks.md`.
  - Retina piece 2 is a fixed 5 s chunk (24.85-29.9 s of every running stretch). Cut out with
    `load_retina_running.py --cut 24.8 30.0` (dataset `retina40clean`), the 40 s unrepeated
    time bars fall from 33.9% to 5.1%; ISI rows move by at most 3 points.
  - Allen: drift beats the stimulus change. ISI fractions across: 38.7% movie <-> adjacent
    spontaneous, 11.8% movie 1 <-> movie 2 (same movie, 70 min later).
  - Second Allen session (789848216) reproduces the first, a few points higher: ISI fractions
    46.1 / 28.0, burst table 54.2 / 31.6, whole-cloud regions k=16 H0 43.6 / 29.1.

- L19x (extra) Cloud size: `notes/results/L19x_cloud_size.md`. Yes, Rips was held back by size.
  At equal ISIs, regional H0 (m = 2, K = 16) is 4 points below ISI fractions at 25 points but
  level or ahead from 100: OpenScope 124 s 69.7 vs 62.3 at 200 points (shuffled 58.8; every-ISI
  fractions 64.2); retina 40 s 64.8 vs 62.9 at 100. Bars add 1-9 points over the region counts (8-9 on OpenScope from 100 points).

- L19 Spike-feature cloud: `notes/results/L19_spike_feature_cloud.md`. The joint 7-coordinate
  cloud never beats ISI fractions or the burst table (best: sliced W1 35/27/46/52/63 against ISI
  fractions 51/26/45/64/64). Population coordinates 6-7 dilute it (dropping them: retina 10 s
  35 -> 41); alone they give 7-12% on Allen / OpenScope 124 s (1-2% controls). Partial success:
  on OpenScope 124 s, coordinates 1-2 with K = 16 give regional H0 67.2% against counts 59.3 and
  ISI fractions 64.2 (shuffled 57.3); burst table still 71.8. Bars add 0-2 points on 25-point
  clouds and 1-8 on 100-point clouds. Time coordinate and H0+H1 do not help.

- W8 Rips fixes pilot: `notes/results/W8_rips_fixes.md`. Time-filtered complex 5-17% (level of
  the spike count); adding H1 inside regions lowers H0 by 4-7 points. L21 is the stablebear version.

- W7 Cloud sampling pilot: `notes/results/W7_cloud_sampling.md`. Whole cloud beats 25 random
  points (regions 43.2 against 30.2 1-NN); the consecutive stretch is worse but has the largest
  order effect. L20 is the stablebear version on all datasets.

- W6 Joint time x ISI signature: `notes/results/W6_joint_time_isi.md`. Equal to multi-scale time
  bars for clustering, below the plain sum on 1-NN.

- W5 Topological clustering of the trial space: `notes/results/W5_trial_space_clustering.md`.
  ToMATo is worse than k-means in 8 of 9 cells; its barcode suggests 7-18 clusters for 60 neurons.

- L18 Spike distance reference: `notes/results/L18_spike_distance_reference.md`. Not the
  ceiling anywhere: retina frozen 80.8% at q=10 (held-out 84.5 on 30 neurons) vs time bars 87.7;
  Allen movie 27.3% at q=1 vs ISI fractions 44.8; OpenScope 12.1% vs 26.1. Across stimuli 6-12%,
  near shuffled; q=100 falls to 1-3%, q=0 is the count (4-12%). New exact numba kernel
  (`vp_distance_matrix_fast`), still ~100x the cost of time bars + ISI fractions.

- L17 Full table across stimulus changes: `notes/results/L17_full_table.md`. Only
  stimulus-free signatures cross (ISI fractions retina 49.0/46.2 across vs 50.9/43.5 within;
  OpenScope 21-24 vs 26-29; Allen 25 vs 36-45); time-locked methods fall to shuffled (1-8%);
  delay regions keep a 3-6 point order effect across; no homology method beats ISI fractions.
  Found: retina unrepeated piece 2 repeats across blocks (reliability 0.24), see the note.

- L13 Peak matching, Allen movie: `notes/results/L13_peaks_allen_movie.md`. 60 most reliable
  units: peaks matched in time (sigma 10 ms) 38.6% 1-NN (shuffled ~2%), sqrt time bars 42.3%;
  random units 6-11%. Peaks with time removed 22-32%, shuffled only 4-5 lower (burstiness).

- L14 Rescaled time and delay regions: `notes/results/L14_rescaled_time_and_regions.md`.
  Population clock makes ISI fractions worse (Allen 48.7 -> 41.0) and does not help across
  stimuli. regions_m3_k8 H0 doubles plain Rips (OpenScope 40.9 vs 19.6, shuffled 34.8), still
  below ISI fractions.

- L11 Pdf per spike on long samples: `notes/results/L11_predictive_state.md`. Whole
  predictive-state cloud (sliced W) beats ISI fractions by 2.6-5.2 points 1-NN (OpenScope 69.4 vs
  64.2), shuffled 2-7 lower; its Rips H0/H0+H1 only 13-28%.

- L12 Rhythm signatures: `notes/results/L12_rhythm_signatures.md`. ACG and spectrum signatures
  14-40% 1-NN, all below ISI fractions; no hippocampus/theta advantage; spectrum H0 at chance.
  Rhythm only shows with 124 s blocks (ACG curve 38%, shuffled 29%).

- L15 With whom, spontaneous: `notes/results/L15_with_whom_spontaneous.md`. Partner profile
  30.4% 1-NN on Allen grey (jittered partners 14.9, other window 1.9). Lead-lag (0.5-5 ms, 20
  jitter controls) recovers CA3->CA1, CA1->SUB, VISp->VISl->VISam, VISp->SCig; no loops. Areas in
  correlation clusters ARI 0.24 (average linkage chains).

- L16 Geometry matching movie vs spontaneous: `notes/results/L16_geometry_matching.md`. Graph
  matching of correlation distances re-identifies 99.5-100% of 851 units between interleaved
  halves; movie vs spont time-matched 37.5% (spont vs spont 50 min apart 39.2%); recording drift
  costs as much as the stimulus change.

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
- 2026-11-12 web chat: read L10, L15, L16. exp27 posted (retina: Rips methods by clustering and across halves). Added Rips ideas under Ideas. L17 still first.
- 2026-11-12 web chat: read L11-L14. Delay regions confirmed as the best Rips signature (order effect 5-9 points); anchors help, as the Rips ideas under "Ideas" assume. L17 (full table across real stimulus changes) is still open and is the only open local task.
- 2026-11-12 web chat: read L17. All queued tasks are done on both sides; the scheduled checks (8 rounds) have ended. Waiting for the user.
- 2026-11-12 web chat: the user asked for the spike distance on the critical cells of L17. Added L18.
- 2026-11-12 web chat: the user asked for one more well-designed embedding per family (Markov/pdf, spectral, cross-correlation), run everywhere, then move on. Added L19-L21 after L18.
- 2026-11-12 web chat: the user found L19-L21 too narrow. Replaced them by one task, L19 (spike-feature cloud with ablations); the three narrow designs moved to "Ideas" as extras A-C. The user also allows extra ideas during implementation, as long as the original task is done.
- 2026-11-12 web chat: read L18. W5 done (exp28: ToMATo on the trial space; negative). The user is away for a few hours; checks every 45 minutes, 5 rounds.
- 2026-11-12 web chat: read the L19 interim status. W6 done (exp29: joint time x ISI; no gain over the sum).
- 2026-11-12 web chat: read the L19 status of 21:00 (4 of 5 datasets). Added the cloud-size idea under Ideas.
- 2026-11-12 web chat: the user asked for two more ways to size the Rips clouds (whole cloud; consecutive stretch). Added L20; piloting on the retina in exp30. A write-up of the results so far is in a Claude doc (not on this branch).
- 2026-11-13 web chat: W7 done (exp30). For L20: also report the whole cloud divided by its point count, and H0+H1 where it is affordable.
- 2026-11-13 web chat: the user added a no-averaging time crop to L20 and wants the whole-cloud homology numbers on every dataset, spike count allowed. W7 addendum posted (exp31).
- 2026-11-13 web chat: the user asked for H1 on the whole cloud and other fixes for Rips. Added L21 (H1, Betti curves, time-filtered complex); piloting in exp32.
- 2026-11-13 web chat: W8 done (exp32). Both fixes negative on the retina; L21 still worth running for H1 on whole clouds without regions and on long cortical samples.
- 2026-11-13 web chat: read L19 (final) and L19x. Cloud size explains the old gap: from 100 points regional H0 is level with or above ISI fractions at equal data, with 9-11 points of order on OpenScope 124 s. The user asked for the worthwhile open points on the board and hourly checks: added L22 (headline barcode-only table), L23 (H0+H1), L24 (data checks), W9, W10, and `notes/state.md`. Order: L20, L21, then L24, L22, L23.
- 2026-11-13 web chat: the first hourly check only arrived at 10:11 (8 h late); now read L24 and the status of 10:15. W11 done (exp33: fitted ISI pdfs; no gain over ISI fractions, lognormal and gamma lose half). Added two L24 corrections to L22. The user asked which queued tests are unlikely to help; my suggestions to him (not yet decided, do not act on them): skip H1 on OpenScope 124 s whole clouds and the time-filtered complex on the remaining datasets, close L23 unless H1 lifts the REGIONAL H0 somewhere, keep L22 with seeds 0-2. Read so far: L1-L19, L19x, L24, W1, W3-W8, W11.
- 2026-11-13 web chat: the user meant a pdf PER SPIKE with the entire history, not per trial. W12 done (exp34: hidden Markov model on log ISI, 2 and 3 states; negative on the retina, where the model finds almost no memory). Added L25 to run it on OpenScope 124 s and retina40clean after L22. Read so far: L1-L19, L19x, L24, W1, W3-W8, W11, W12.
