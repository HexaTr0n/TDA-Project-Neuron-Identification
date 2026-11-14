# Methods: keep for main and the report, or cut

2026-11-14, web chat. A proposal for the user to decide on; nothing has been moved to `main`.

Scores are for the retina, 10 s samples, 60 neurons (chance 1.7%), written
**1-NN / k-medoids**, within one stimulus and across stimuli (frozen noise -> unrepeated
noise). In brackets: 1-NN within, ISIs shuffled. Sources: L17, L18, L20, L21, L22 on real
stablebear unless a row says otherwise.

## Keep

| Method | What it is | Within | Across | Why keep |
|---|---|---|---|---|
| Spike distance (Victor-Purpura) | Edit distance between two trains: 1 to add or delete a spike, `q` per second to shift one | 80.8 / 44.3 | 6-12 | The reference named in the proposal; shows the costly standard is neither best nor stimulus-free |
| Square-root time bars, three window sizes | One bar `[t, t+s)` per spike; the Betti curve counts spikes in the last `s` seconds (20, 67, 200 ms); square root of the curve | 87.7 / 48.2 (6.8) | 5.9 / 6.7 | Best single signature within a stimulus; lower bound of the spike distance at 1% of the cost |
| ISI fractions | One bar per ISI of length log ISI; stable rank divided by the count = share of ISIs longer than each value. L1 = Wasserstein-1 between ISI distributions | 50.9 / 37.6 | 49.0 / 32.7 | Best signature across stimuli; it is H0 of the spike times on a line |
| Burst table | For 20 time scales, the number of bursts with at least 1, 2, 3, ... spikes | 52.6 / 34.7 (44.3) | 47.4 / 30.4 | Best conventional signature on cortex and on long samples; the non-topological reference |
| Time bars + ISI fractions | Sum of the two distances | 91.1 / 53.3 (35.7) | 30.4 / 22.7 | Best score within a stimulus; shows the two signatures add within and hurt across |
| Burst bars | Thicken each spike by `s`; each connected piece is one bar over time | 69.8 / 35.4 (5.2) | 4.7 / 3.8 | Real H0 that keeps time; carries the barcode-only headline within a stimulus |
| Peak matching | H0 of the smoothed rate: each peak is a bar with a time and a prominence; trials compared by matching peaks in time order | 71.7 / 34.8 (1.5) | 1.6 / 5.0 | Sublevel persistence of a function, a different kind of barcode; one row is enough (not stablebear, no mean curve) |
| Plain delay (Takens) Rips, H0 and H0+H1 | Points are windows of 3 consecutive log ISIs; Rips barcode of the whole cloud | 31.0 / 28.7 (28.8) | 28.0 / 21.7 | The proposal's own method; kept to show it equals its shuffled control, and why |
| Delay regions, whole cloud | The same cloud split around fixed anchors (from other neurons), Rips H0 inside each region, stable ranks summed | 8 regions: 42.4 / 37.0 (37.6); 32 regions: 40.6 / 33.9 (35.4) | 41.3 / 30.0; 39.6 / 25.7 | The main Rips result: level with ISI fractions on long samples (OpenScope 124 s: 67.1 against 62.6 over five seeds) and sees ISI order |
| Delay regions, random time cut | As above, after cutting a random amount of time off the start and end so that 70% (or 50%) of the spikes remain; one barcode, no averaging | 70%: 34.3 (29.7); 50%: 31.2 (23.2), 8 regions | 33.8; 28.3 | Shows the signature survives when samples do not share the same cut and spike count; requested by the user. Only run with 8 regions so far |
| Region counts (twin) | Points per region, no bars | 32 regions: 41.4 / 33.3 (31.7) | 39.5 / 27.9 | The control that says how much is homology: bars add 4-14 points at 8 regions and about 0 at 32 |
| Barcode-only combination | Delay regions + ISI bars + burst bars, distances summed after scaling | 82.3 / 46.6 (39.3) | regions + ISI bars: 41.4 / 30.4 (37.4) | The headline "homology only" numbers |
| Pdf family (report section; code after a stablebear rerun) | Per-trial fits (lognormal, gamma, mixtures); per spike from the last two ISIs (kernel); per spike from the last 16-64 ISIs (mixture); per spike from the whole past (state model, running mixture) | best cloud without homology 52-54; Rips H0 13-23 (= shuffled) | best 35-47 | The user wants the family shown. One result worth a line: two parameters keep 27% of 54% |
| Controls and scores | ISIs shuffled; counting twins; 1-NN and k-medoids (k-means with mean curves as a third column) | | | Every claim above depends on them |
| What drives the scores | Sample length (22% at 2.5 s to 67% at 40 s); response reliability across nine retinas; recording drift on Allen (38.7% next to each other, 11.8% 70-100 min apart) | | | Not methods, but they explain most of the variation and belong in the report |

## Cut

| Method | What it is | Score | Why cut |
|---|---|---|---|
| Random draws of 25 points, 20 draws averaged | The first way clouds were cut to equal size | 27.9 (25.2), 8 regions | The whole cloud is better everywhere; one sentence in the report as the handicap we removed |
| Consecutive stretch of 25 points | 25 consecutive delay points from a random start, averaged | 27.5 (19.6) | Lowest of all cuts |
| H1 added to H0 inside regions | Stable ranks of both degrees summed | 32.0 against 43.7 for H0 (32 regions) | Lowers the score on every dataset; one row in a table of negatives |
| H1 on its own | Loops of the delay cloud | above shuffled only on OpenScope 124 s (42.7 against 36.8) | Too weak; the plain-cloud Betti curve (31.0 -> 36.4) can be one sentence |
| Time-filtered complex | Points enter in spike order at a fixed Rips scale | 17.2 (13.6) | Fails on all five datasets (8-22%) |
| Pdf per spike read by Rips | H0 of the cloud of predicted pdfs | 13.3 (13.8) | Equals shuffled; covered inside the pdf section |
| Joint spike-feature cloud | One point per spike with 7 coordinates (ISIs before and after, burst context, population counts) | never above ISI fractions (L19) | The extra coordinates dilute the two good ones |
| Joint time x ISI signature | One object holding spike time and ISI | no gain over the sum (W6, not stablebear) | Adding the two distances does the same |
| Population-rescaled time | Clock that runs with population activity | lower than plain time (L14) | Hurts |
| Rhythm: autocorrelogram bars, spectrum | Peaks of the autocorrelogram as bars; averaged spectrum | 14-40% on cortex, below ISI fractions (L12) | Weak; no advantage for hippocampal units |
| Short-time spectrum cloud | One point per 0.5 s window: log power at 24 frequencies | 17.8 by position, 1.8 by H0 (not stablebear) | Each window is noise of fixed shape; Rips sees the same blob for every neuron |
| Anonymous cross-correlogram cloud | One point per partner: lag, height, width of the correlogram peak | 3.6 by H0 (not stablebear) | A single-trial correlogram is noise |
| Coincidences with other neurons during a stimulus | Coincidence counts as a function of window width | 84.3 against 84.1 for its control | Nothing beyond the response timing |
| Partner profile, geometry matching, lead-lag flow | Correlations with named partners; matching the correlation geometry between conditions; millisecond lead-lag between areas | 30.4 (14.9 jittered); 99.5-100% within a condition, 37.5% across | Strong, but a different question (population, not one train). At most one outlook slide; the user's call |
| Topological clustering of trials (ToMATo) | Density peaks merged by persistence, instead of k-medoids | worse than k-means in 8 of 9 cases (W5, not stablebear) | No gain |
| Window count, van Rossum | Sliding spike count; L2 between smoothed trains | same function as time bars | Duplicates of time bars |
| ISI bars with the count left in, alone | Stable rank of ISI bars, not divided | 41.4 / 28.8 | Weaker than ISI fractions; kept only inside the barcode-only combination |
| Log-normal and gamma windows | Mean and sd of the last 20 log ISIs, one point per spike | 24.9 by position, 4.0 by H0 (not stablebear) | Superseded by the sliding-window mixture in the pdf section |
| Simulated neurons and networks | Izhikevich neurons; synthetic connectivity pilots | | Off the question |
