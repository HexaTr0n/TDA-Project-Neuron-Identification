# L16 Matching neuron geometries across stimuli

2026-11-12, local session. **Worked, and drift matters more than the stimulus.**

- Within one condition, interleaved halves of the recording re-identify all 851 units from
  distances alone (99.5-100%).
- Movie against spontaneous gives 15.6% overall, and 37.5% when the two are recorded close in
  time. Spontaneous against spontaneous recorded far apart gives 39.2%. Chance is 0.12%.

## Data

- Allen FC session 829720705, natural movie repeats (60 x 30 s, two blocks 70 min apart:
  2290-3161 s and 7395-8265 s) and spontaneous grey-screen windows (77 x 30 s, 8-6162 s).
- **851 quality-good units with >= 1 Hz in both conditions.**

**Distance matrix.**
- D = 1 - correlation of the trains, binned at 10 ms and smoothed with a Gaussian of 20 ms.
- Computed over all samples of a condition laid end to end.

**Pairing** (labels of B hidden):
- *sorted profile*: each unit's sorted row of distances, L1, Hungarian matching.
- *refined*: from that start, 20 graph-matching rounds. The cost of pairing i with j is
  mean_k |D_A[i, k] - D_B[j, pi(k)]| under the current pairing pi.
- *rate only*: Hungarian matching on |log rate difference|.
- *+ rate*: sorted-profile cost and rate cost, each median-scaled, added.

**Score:** the share of units paired with themselves. Random: 1/851 = 0.12% (95% of 200 random
pairings: 0.35%).

## Command

```sh
python scripts/sb_geometry.py --data data/allen/allen829720705.parquet \
    --a allen829720705_movie --b allen829720705_spont
```

## Scores

% of 851 units paired with themselves. "r(D)" is the correlation of the two distance matrices;
"r(rate)" is the correlation of the log rates.

| A vs B | time between | r(D) | r(rate) | sorted profile | refined | rate only | + rate | + rate, refined |
|---|---|---|---|---|---|---|---|---|
| movie, even vs odd repeats | interleaved | 0.972 | 0.996 | 56.3 | 99.5 | 4.0 | 79.3 | **100.0** |
| spont, even vs odd windows | interleaved | 0.971 | 0.995 | 42.7 | 99.8 | 2.8 | 66.5 | **99.8** |
| spont, 1st vs 2nd half | ~50 min | 0.822 | 0.649 | 9.6 | 34.7 | 1.1 | 11.8 | 39.2 |
| movie, block 1 vs block 2 | 70 min | 0.535 | 0.394 | 0.8 | 1.3 | 0.5 | 0.7 | 1.2 |
| **movie block 1 vs spont within 1500 s** (stimulus change, time-matched; 931 s vs 776 s) | close | 0.838 | 0.816 | 10.6 | 29.7 | 0.7 | 15.2 | **37.5** |
| movie (all) vs spont (all) (stimulus change) | mixed | 0.711 | 0.836 | 3.4 | 11.9 | 0.2 | 4.8 | 15.6 |

## What to take from it

- **The geometry of a population is a near-perfect fingerprint within a condition.** From
  correlation distances alone, without spike shapes or positions, graph matching recovers every
  unit between interleaved halves (99.5-100%). The sorted profile alone gets 43-56%, so a unit's
  distance distribution is already distinctive; refining with the partners' identities does the
  rest.
- **Across stimuli, about 30-40% survive when recorded close in time.** Movie against
  spontaneous, time-matched: 37.5%. That is no worse than spontaneous against spontaneous ~50
  minutes apart (39.2%). At this resolution the stimulus change costs about as much as
  recording drift.
- **Drift dominates.** The two movie blocks, 70 minutes apart, are not even matched by rate
  (rate correlation 0.39, pairing 1.2%). The recording changed between them (units lost or
  drifting), so "movie vs spontaneous" on all data (15.6%) is mostly a drift effect.
- **Rate alone does not identify** (0.2-4%), and adding it helps a little (+3-8 points).
- **What failed:**
  - matching across the 70-minute gap;
  - the simple start (sorted profile) is weak across conditions (3-11%) and needs the refinement
    to be useful.
- **Next:** a drift-free test of the stimulus, with spontaneous windows interleaved with movie
  repeats. This session has none within the movie blocks. Alternatively, a session where
  stimuli alternate.
