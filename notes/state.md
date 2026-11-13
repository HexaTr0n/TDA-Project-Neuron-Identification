# Where things stand

2026-11-13, web chat. A short summary for both sessions and the user; `notes/results/` has
the tables. All scores: 60 neurons, chance 1.7%, real stablebear unless a result file says
otherwise.

## The question

Can one short sample of spikes tell which neuron produced it, within one stimulus and across
a change of stimulus, scored by clustering as well as 1-NN?

## What holds so far

- **Two signatures.** When the neuron fires during a repeated stimulus (time bars, burst bars,
  peak matching), and how its ISIs are distributed (ISI fractions, burst table).
  - Retina 10 s, one stimulus: square-root time bars + ISI fractions 91.1% 1-NN; multi-scale
    square-root time bars 70.2% by k-means clustering (L17).
  - Across stimuli the timing signatures fall to their shuffled level; ISI fractions keep
    49.0% 1-NN (L17).
  - Cortex (OpenScope 10 s, Allen 30 s): timing is near chance for typical units; ISI
    fractions and the burst table give 26-48% within and 21-30% across (L17).
- **Sample length is the main lever** for the ISI signature: 22% at 2.5 s to 67% at 40 s on the
  retina (L6).
- **Spike distance** (Victor-Purpura) is below time bars on the retina (80.8% against 87.7%),
  at chance across stimuli, at about 100 times the cost (L18).
- **Population:** correlation geometry re-identifies 99.5-100% of 851 units within a condition
  and 37.5% across; lead-lag flow follows anatomy (L15, L16).

## Rips

- Plain delay-embedding Rips equals its shuffled control everywhere (L10, L17).
- **Delay regions** (Rips inside the regions of fixed anchors) are the best Rips signature.
- **Cloud size was the handicap (L19x).** With 25-point clouds regional H0 is 4 points below
  ISI fractions from the same ISIs. From 100 points it is level or ahead: OpenScope 124 s at
  200 points 69.7% against 62.3% 1-NN (ISIs shuffled: 58.8%); retina 40 s at 100 points 64.8%
  against 62.9%.
- The bars add 4-9 points over the region counts on long samples, and ISI order adds 9-11
  points on OpenScope and 2-6 on the retina (L19x).
- **The whole cloud is the simplest fix (L20).** Every point, one barcode, 32 regions, 1-NN
  within / across (ISIs shuffled, within):
  - long samples: OpenScope 124 s 67.2 (56.4) against ISI fractions 64.2; retina 40 s
    67.4 / 70.7 (60.5) against 64.3 / 73.6 (old retina data with the repeating chunk);
  - short samples: retina 10 s 43.7 / 40.0 (35.3) against 50.9 / 49.0; Allen 30 s 39.0 / 27.9
    (30.0) against 44.8 / 25.4; OpenScope 10 s 21.5 / 19.3 (17.5) against 26.1 / 21.0.
  - Dividing by the point count changes scores by at most 2 points. Clustering lags: k-means
    36.9 on OpenScope 124 s where 1-NN is 67.2.
- The burst table is still the best single signature on every dataset (71.8% on OpenScope
  124 s, L19).
- **Did not work:** the joint 7-coordinate spike-feature cloud (L19); H0+H1 (below H0 in every
  test); the time-filtered complex on the retina (W8); pdf-per-spike H0 (L11, L17); ToMATo
  clustering of trials (W5); a joint time x ISI signature (W6).

## Data checks (L24) and fitted pdfs (W11)

- The repeating retina piece is a fixed 5 s chunk (24.85-29.9 s of every running stretch).
  Cut out, the time-locked 40 s "unrepeated" scores fall from 34% to the shuffled level (5%);
  ISI-based scores move by 1-3 points. Use `retina40clean`.
- **Drift costs more than a stimulus change on Allen.** ISI fractions across: 38.7% for a
  movie block against the spontaneous block next to it, 11.8% for the same movie 70-100 min
  later. Whole-cloud delay regions keep 6-9 points over shuffled in every pair.
- A second Allen session reproduces the first (ISI fractions 46.1 within / 28.0 across).
- Per-spike pdfs with the whole history (a state model, W12; a running mixture of sqrt(n)
  Gaussians, W13) do not beat ISI fractions on retina 10 s; their Rips H0 is near shuffled.
- Fitting a pdf to the ISIs does not beat the empirical distribution. A lognormal or gamma
  (2 parameters) keeps 27% of 54%; a mixture of 2 Gaussians on log ISI recovers 50.6% (W11,
  retina, not stablebear).

## Open

- L21: H1 on whole clouds, Betti curves for H1, time-filtered complex (two datasets done).
- L22: one headline table of barcode-only scores with their counting twins.
- L23: H0+H1, fix or close. L25: the state model (per-spike mixture pdf) on long samples.

## Limits

- Retina numbers come from the best of nine sessions; the average over nine is about 50%
  clustering for time bars and 30% for ISI fractions (W1, W4).
- One session per cortical dataset.
- Clustering scores move by about 2.5 points between random starts.
