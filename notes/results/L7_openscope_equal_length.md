# L7 Equal-length stimulus change on OpenScope

2026-11-12, local session. **Worked, with a modest stimulus effect:** with 10 s on both sides,
ISI fractions cross from gratings to grey screen at 23.5-28.0% 1-NN, against 27.7-36.6% within
each condition. The L1 drop (63% -> 31%) was mostly sample size; about 6 points are stimulus.

## Data

- OpenScope oddball session `glo637908`, the same 60 neurons as L1 (seed 0).
- A: the 124 s grating blocks cut into 10 s pieces (12 per block, `block` = original block),
  pieces with >= 10 ISIs, at most 40 random pieces per neuron: **2279 pieces, median 39 spikes**.
- B: the 10 s grey-screen intermissions, >= 10 ISIs, at most 40 per neuron: **1773 windows of
  57 neurons, median 48 spikes**.
- ISI fractions in stablebear (stable rank / ISI count), `sb.pdist` / `sb.cdist`, k-means
  `sb.mean`, 3 seeds.

## Command

```sh
python scripts/sb_cross.py --data data/openscope/glo637908.parquet --session glo637908_grating \
    --cross-session glo637908_spont --cut-a 10 --max-per-neuron 40
```

## Scores

Chance 1.7%. The "far" rows exclude nothing here (grey-screen windows have no grating block), so
they equal the plain rows.

| | ISI fractions (stablebear) | rate only |
|---|---|---|
| grating -> grating (1-NN) | 27.7 | 5.5 |
| grey -> grey | 36.6 | 5.8 |
| **grating -> grey** | **23.5** | 3.9 |
| **grey -> grating** | **28.0** | 5.1 |
| clustering within grating / within grey | 21.0 / 23.9 | |
| cluster grating, assign grey / the reverse | 15.8 / 12.9 | |

## What to take from it

- **Across-stimulus 1-NN (mean 25.8%) is about 6 points below within-stimulus (mean 32.2%)**, so
  gratings and grey screen change cortical/hippocampal ISI statistics a little. In L1 the drop
  looked like 63% -> 31% because 124 s blocks were compared with 10 s windows.
- **The three datasets differ:** retina noise (L3) loses nothing across stimuli at 10 s;
  OpenScope loses about 6 points at 10 s; the Allen session (L5) loses about half at 30 s
  (48.7/32.8 within against 28.0/23.7 across). Grey screen against gratings is a smaller change
  than grey screen against a natural movie.
- Grey windows identify neurons better than grating pieces (36.6 against 27.7) although they have
  only a few more spikes (48 against 39). Activity without a stimulus may be more characteristic
  of the neuron.
- **What failed:** clustering transfers poorly: 13-16% across against 21-24% within.
- Rate alone gives 4-6%.
