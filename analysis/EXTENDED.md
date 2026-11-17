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
| Scores | 1-NN, nearest mean | 1-NN, k-medoids (nearest mean is dropped: it needs labelled neuron means and has no mean for the Victor-Purpura distance) |
| References without topology | firing rate, ISI histogram | the same, and the Victor-Purpura distance |
| Controls for the delay cloud | shuffled ISIs | shuffled ISIs, random time cutoff, share of bars |
| Delay cloud | 2 consecutive log ISIs per point | 2 and 3 consecutive log ISIs per point |

**Across stimuli.** The same neurons are recorded under several stimuli. The across table has
three groups of scores:

- **First stimulus only:** 1-NN and k-medoids inside the first stimulus, as in the within part,
  on the neurons of this part.
- **All stimuli mixed** (the main one): every neuron gives the same number of samples from every
  stimulus. The samples go into one pool, and the stimulus of a sample is not used. 1-NN asks
  whether the nearest other sample of the pool is from the same neuron. k-medoids sorts the
  unlabelled pool into as many buckets as there are neurons; buckets are then paired with
  neurons, and the score is the share of samples that sit in their neuron's bucket.
- **Other stimuli only:** a stricter 1-NN, where the nearest sample is searched only among the
  samples of the other stimuli. In the mixed pool a sample can find its neighbour inside its
  own stimulus; here it cannot, so this column shows what carries over between stimuli.

| Dataset | Stimuli in the pool |
|---|---|
| retina | frozen noise (30 repeats of 9.94 s); the first 10 s of the unrepeated noise shown before each frozen trial (30 pieces) |
| neuropixels | natural movie, first block (30 repeats of 30 s); every other stimulus in the table with at least 10 windows of 30 s recorded within 30 min of that block (`--max-gap`): the grey screen, and after `load_allen.py --all-stimuli` also gratings, flashes, dot motion and so on |

**More stimuli.** On Neuropixels, `python scripts/load_allen.py <nwb> --all-stimuli` adds every
stimulus of the session to the table (receptive field mapping and invalid times are left out),
each block cut into windows of 30 s. `04_extended.py` then picks them up by itself and prints
the list it uses. On the retina only the two noise stimuli are in a table; the recordings also
contain flashed natural images, which would need their own converter.

The retina's unrepeated noise only exists for the recording `20180126_sr_le_sp`, so the retina
"across" part draws its 40 neurons from that recording. The retina "within" part uses all
recordings, exactly like step 3.

On Neuropixels the stimuli must be close in time: for one neuron, the same movie 70-100 minutes
later is harder to recognise than a different stimulus recorded right next to it (recording
drift). That is why only the first movie block is used, together with stimuli recorded within
30 minutes of it. All stimuli except the movie get the same number of samples per neuron: the
smallest number available, at most 15.

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

**Delay cloud in 3D.** Each point is (log10 ISI_i, log10 ISI_i+1, log10 ISI_i+2) instead of
two consecutive ISIs. Rows `T2 DELAY 3D H0 / H1 / H0+H1`, with the shuffled and cutoff controls
(`C1 DELAY 3D ...`, `C2 DELAY 3D ...`). The 2D rows are unchanged (`cloud_delay(isis)` still
gives the 2D cloud; `cloud_delay(isis, 3)` the 3D one).

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

The two files that are not in git are made from the raw data with the scripts below (also
on this branch). If they already exist in another copy of the repository, copying them into
`data/` is enough.

```bash
# retina, unrepeated noise: needs the raw session in extracted/20180126_sr_le_sp
python scripts/extract.py --data-dir data --out-dir extracted --download      # only if extracted/ is missing
python scripts/load_retina_running.py --extracted-dir extracted --session 20180126_sr_le_sp --cut 24.8 30.0

# Neuropixels: needs the NWB file of each session from DANDI 000022 (functional connectivity set)
python scripts/load_allen.py data/allen/sub-811322604_ses-829720705.nwb --all-stimuli
for f in data/allen/*.nwb; do python scripts/load_allen.py "$f" --all-stimuli; done   # every downloaded session
```

`--cut 24.8 30.0` removes a 5 s chunk that repeats inside every "unrepeated" stretch.

## How to run

```bash
pip install -r scripts/requirements.txt        # numba makes the Victor-Purpura distance fast
python analysis/04_extended.py --dataset retina --part within --seeds 1 --n-neurons 10   # quick test
python analysis/04_extended.py --dataset retina
python analysis/04_extended.py --dataset neuropixels                          # session 829720705
python analysis/04_extended.py --dataset neuropixels --npx-session all        # every data/allen/allen<id>.parquet
python analysis/04_extended.py --dataset neuropixels --npx-session all --part length
```

**Several Neuropixels sessions.** Each session is run on its own, with its own neurons and
stimuli, and its outputs are named `neuropixels_<id>` (for example
`results/extended_neuropixels_829720705_within.md`, `figures/analysis/ext_neuropixels_829720705_across.png`).
The first block of movie repeats is found from the gaps between repeats (more than 2 minutes);
the script prints it. A session whose first block has fewer than 30 repeats, or that has fewer
than 40 neurons with enough samples, is skipped with a message. With more than one session,
`results/extended_neuropixels_sessions.md` gives the mean ± std over sessions of the within and
the all-stimuli-mixed scores. If a run is stopped early, `--npx-session all --replot` redraws the figures
and writes this summary from the sessions that are finished. A session that fails for any
reason is skipped with its error message, and the next one is run.

**Order of the work and stopping early.** Sessions are run one after the other, and inside a
session draw by draw: draw 0 within and across, then draw 1, and so on. The tables
(`..._runs.csv`, `.csv`, `.md`) are rewritten after every draw, so stopping the run (Ctrl+C)
keeps every finished draw; the figures are drawn when a session is done, or later with
`--replot`. `--resume` keeps the draws already in the `_runs.csv` files and only computes the
missing ones, so a stopped run can be continued with the same command plus `--resume`. Many
draws on a few mice: list the mice, e.g.
`--npx-session 829720705 839557629 821695405 835479236 --seeds 40`. Sessions are not pooled: different sessions are different mice
and may show the other stimuli at different times.

Tables are printed and written to `results/extended_<dataset>_within.md` and
`..._across.md` (plus `.csv`, and `..._runs.csv` with one row per draw). Figures go to
`figures/analysis/ext_<dataset>_*.png`. Every figure shows 1-NN and k-medoids side by side;
the within and the across part have one figure each.

**k-medoids chance.** k-medoids has a higher chance level than 1-NN (about 10 % with 30 samples
per neuron, instead of 2.5 %), because pairing clusters with neurons picks the best pairing. The
columns `kmedoids_chance` (within) and `kmed_within_chance`, `kmed_pooled_chance` (across) give
it: the same clustering scored against shuffled labels. It is only in the tables, not in the
figures.

| Report figure | Command | File |
|---|---|---|
| distance matrices (as Fig. 7), one per stimulus | `--part across` (or `--part across --figure-only`, draw 0 only) | `ext_<dataset>_matrix_<stimulus>`, e.g. `ext_retina_matrix_frozen_noise`, `ext_retina_matrix_unrepeated_noise` |
| scores (as Fig. 8), within | `--part within` | `ext_<dataset>_within`, `ext_<dataset>_vp` |
| scores (as Fig. 8), across (all stimuli mixed) | `--part across` | `ext_<dataset>_across` |
| sample length (as Fig. 9), within and across | `--part length` | `ext_<dataset>_length_within`, `ext_<dataset>_length_across`, `results/extended_<dataset>_length.md` |
| redraw the bar and q-sweep figures from saved tables | `--replot` | the three bar/q figures |

Every figure is written as `.png` and `.svg`.

The distance-matrix figures show the same 10 neurons of draw 0 for four methods, one per
family: ISI histogram, the best Victor-Purpura `q`, LINE H0 log share, DELAY H0+H1. One figure
per stimulus, its samples against each other, with the same colour scale per method in every
figure. For unrepeated noise every sample is a different piece of stimulus, so a light diagonal
block there cannot come from a shared stimulus. The length figures show the same four methods.
`--part length` glues 1, 2 or 3
consecutive samples into one (10, 20, 30 s on the retina, as in step 3) and scores both the
within part and the mixed pool (one figure each); with 30 samples per neuron and stimulus
there are 30, 15 and 10 glued samples per neuron and stimulus.

**Methods in the bar figures.** A fixed set, chosen by role and not by score (`BAR_GROUPS` in
the script): firing rate, ISI histogram, the Victor-Purpura `q` with the best 1-NN score (the
whole sweep is in `ext_<dataset>_vp`); LINE H0, LINE H0 log share and the LINE cutoff control;
DELAY H0+H1 and DELAY 3D H0+H1, each with its shuffled and cutoff controls (figures never show
H1 alone). Controls are hatched. `BAR_GROUPS = None` plots every row. The tables have every row.
Every figure is written as `.png` and `.svg`.

Expected time: the retina should take a few minutes more than step 3. Neuropixels samples are
three times longer, so H1 of the delay cloud and the Victor-Purpura distance take longer there;
if it is too slow, `--seeds 2` gives a first answer.

## Checks to make on the first run

1. **The old rows must reproduce step 3.** In `extended_retina_within.md`, the 1-NN column of
   B1, B2, T1 LINE H0, T2 DELAY H0 / H1 / H0+H1 and "C1 DELAY H0+H1, ISIs shuffled" must equal
   `results/identification.md` (same data, same draws, same code). This was checked on the
   first real run, 2026-11-16.
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
