# Step 4: the extended experiment

`04_extended.py` keeps the protocol of step 3 (40 random neurons that fire at least 1 Hz, 30
samples per neuron, 5 random draws, mean ± std) and adds the points below. Nothing in steps 1-3
was changed; `tda_utils.py` only gained new functions and three optional arguments of
`load_trials` whose defaults give the old behaviour.

## What is new

| | Step 3 | Step 4 |
|---|---|---|
| Data | retina, frozen noise | retina, and Neuropixels recordings from mouse visual cortex |
| Question | within one stimulus | within one stimulus, and across two stimuli |
| Scores | 1-NN, nearest mean | 1-NN, k-medoids (nearest mean is kept for the old methods) |
| References without topology | firing rate, ISI histogram | the same, and the Victor-Purpura distance |
| Controls for the delay cloud | shuffled ISIs | shuffled ISIs, random time cutoff, share of bars |

**Across two stimuli.** The neurons are known from stimulus A; samples recorded under stimulus B
must be given to the right neuron.

- 1-NN: every B sample takes the neuron of its closest A sample.
- k-medoids: the A samples are clustered without labels, each cluster is named after a neuron
  with the A labels, and every B sample goes to the cluster of its closest medoid.

| Dataset | Stimulus A | Stimulus B |
|---|---|---|
| retina | frozen noise, 30 repeats of 9.94 s | the first 10 s of the unrepeated noise shown before each frozen trial (30 pieces) |
| neuropixels | natural movie, first block, 30 repeats of 30 s | grey screen (spontaneous activity) recorded just before that block, 15 windows of 30 s |

The retina's unrepeated noise only exists for the recording `20180126_sr_le_sp`, so the retina
"across" part draws its 40 neurons from that recording. The retina "within" part uses all
recordings, exactly like step 3.

**k-medoids** is k-means for a distance matrix: every centre is one of the samples. It uses no
labels. The score is the share of samples that sit in the cluster paired with their neuron
(clusters and neurons are paired with the Hungarian method). It replaces nearest mean as the
second score because it works for every distance, also for the Victor-Purpura distance, where
no "mean spike train" exists.

**Victor-Purpura distance.** The cheapest way to turn one spike train into the other, where
deleting or adding a spike costs 1 and shifting a spike by `dt` seconds costs `q * dt`. The
parameter `q` is the time precision: `q = 0` compares spike counts, large `q` only accepts
spikes at the same time. One row per `q` in 0.3, 1, 3, 10, 30, 100 per second, and a figure of
the score against `q`. The rows are a sweep: if the report quotes the best `q`, it should say
that it was picked on the scored data.

**Random time cutoff.** From every sample, a random part of the start and a random part of the
end are removed (each between 0 and 25 % of the sample, independently). The neuron's ISI
distribution is unchanged, but two samples of one neuron no longer have comparable spike counts
and no longer cover the same part of the stimulus. What a method loses under the cutoff is what
it owed to those two things. The shuffled-ISI control answers the other question: what a method
owes to the order of the ISIs.

**Share of bars.** The stable rank divided by its value at 0. The number of bars (the spike
count) then no longer enters the distance, and no data is removed.

**LINE on a log scale** (two extra rows, optional for the report). The LINE barcode with bar
lengths `log10(ISI / 0.1 ms)`, and the same divided by the number of bars. The second one is
the share of ISIs longer than each value, so its L1 distance is the Wasserstein-1 distance
between the log-ISI distributions of two samples.

## Data files

| File | Used by | In git? |
|---|---|---|
| `data/isi_frozennoise.parquet` | retina, within | yes |
| `data/retina_running_20180126_sr_le_sp_clean.parquet` | retina, across | no |
| `data/allen/allen829720705.parquet` | neuropixels, both parts | no |

The two files that are not in git were made earlier on Maxim's machine (branch `main`,
`scripts/README.md`, steps A and C explain how). They stay in place when the branch is switched,
because git does not touch untracked files.

## How to run

```bash
pip install -r scripts/requirements.txt        # numba makes the Victor-Purpura distance fast
python analysis/04_extended.py --dataset retina --part within --seeds 1 --n-neurons 10   # quick test
python analysis/04_extended.py --dataset retina
python analysis/04_extended.py --dataset neuropixels
```

Tables are printed and written to `results/extended_<dataset>_within.md` and
`..._across.md` (plus `.csv`, and `..._runs.csv` with one row per draw). Figures go to
`figures/analysis/ext_<dataset>_*.png`.

Expected time: the retina should take a few minutes more than step 3. Neuropixels samples are
three times longer, so H1 of the delay cloud and the Victor-Purpura distance take longer there;
if it is too slow, `--seeds 2` gives a first answer.

## Checks to make on the first run

1. **The old rows must reproduce step 3.** In `extended_retina_within.md`, the 1-NN and
   nearest-mean columns of B1, B2, T1 LINE H0, T2 DELAY H0 / H1 / H0+H1 and
   "C1 DELAY H0+H1, ISIs shuffled" must equal `results/identification.md` (same data, same
   draws, same code).
2. **Victor-Purpura at small `q`** must be close to the firing-rate row, since it then compares
   spike counts.
3. **Every score must be above chance** (2.5 %) except where a method has nothing to use.

## Status of the code

Written on 2026-11-16 without access to stablebear, the Neuropixels table or the unrepeated
retina table. It was run end to end on the retina frozen-noise data with a small stand-in for
the stablebear calls and with synthetic second-stimulus tables, so the logic, the scores and
the outputs are tested. Not tested: the two new stablebear lines in `normalise_pcfs`
(`sb.Pcf(array)` and `sb.PcfTensor(list)`, taken from scripts that ran on real stablebear), the
numba version of the Victor-Purpura distance, and the real second-stimulus tables. If the first
run fails, it will most likely be in one of those places.
