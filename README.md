# TDA Project – Neuron Identification

Interspike intervals (ISIs) of mouse retinal ganglion cells responding to a repeated
("frozen") white-noise movie, extracted from the public dataset of
Karamanlis & Gollisch (2021), *J Neurosci* 41:3479-3498
([GIN repository](https://gin.g-node.org/gollischlab/Karamanlis_Gollisch_2021_RGC_spiketrains_natural_image_encoding),
CC BY-SA 4.0).

**What's in the dataset:** 850 neurons from 9 recording sessions. Each neuron has 30–34 trials,
and each trial is one ~9.94 s presentation of the same frozen noise movie. There are no
cell-type labels, only neuron ids, spike times and ISIs.

## Repository layout

```
data/
  isi_frozennoise.parquet          the dataset (recommended format)
  isi_frozennoise.csv              same data as CSV
  frozennoise_neuron_summary.csv   one row per neuron: rate, number of trials, reliability
  manual.pdf, README.md, ...       documentation of the original dataset
  *.rar                            original archives (12 are only git-annex pointers, see below)
scripts/
  load_dataset.py                  load the dataset in Python  <- start here
  extract.py                       download + extract the raw archives
  build_isi_dataset.py             rebuild the ISI dataset from the raw data
  README.md                        the analysis: methods, and how to replicate every result
  sb_*.py, run_pipeline.py, ...    the analysis scripts (see scripts/README.md)
  pilots/                          small pdf-fitting experiments (not stablebear)
  requirements.txt
notes/
  summary.md                       the results in one page
  results/                         one file per experiment, with full tables and commands
  keep_or_cut.md                   every method tried, and why it was kept or dropped
figures/                           example raster plots
```

**Analysis.** The project asks whether one short sample of spikes identifies its neuron, with
topological signatures computed in stablebear. Start with `notes/summary.md` for the results
and `scripts/README.md` for how to rerun them. Two more datasets (OpenScope and Allen
Neuropixels) are used there; they are too large for this repository and
`scripts/README.md` says how to download them.

## Quick start (use the ready-made dataset)

You don't need the raw data for this.

```bash
git clone https://github.com/HexaTr0n/TDA-Project-Neuron-Identification.git
cd TDA-Project-Neuron-Identification
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r scripts/requirements.txt

python scripts/load_dataset.py     # prints a summary to check that everything works
```

Then in Python (run from the repo root):

```python
import sys; sys.path.insert(0, "scripts")
from load_dataset import load_dataset, isis_by_neuron, spike_trains, neuron_summary

df = load_dataset()                       # data/isi_frozennoise.parquet by default
df = load_dataset(min_rate=0.5)           # optional: drop neurons firing < 0.5 Hz
isis = isis_by_neuron(df)                 # {neuron_id: [ISI array of trial 0, trial 1, ...]}
pooled = isis_by_neuron(df, pooled=True)  # {neuron_id: all ISIs in one array}
trains = spike_trains(df, "20180222_sr_le_sp_C65_1")   # spike times, one array per trial
summary = neuron_summary(df)              # per neuron: session, n_trials, n_spikes, rate_hz
```

### Columns (one row per neuron × trial)

| column | content |
|---|---|
| `neuron_id` | `<session>_C<channel>_<cluster>`, unique across sessions |
| `session` | recording session (date, `fr`/`sr` = setup, `le`/`re` = left/right eye) |
| `trial` | 0..n-1, in time order |
| `isis` | interspike intervals in seconds (numpy array) |
| `spike_times` | spike times since trial start, in seconds (numpy array) |

In the raw CSV, `isis` and `spike_times` are semicolon-separated strings. `load_dataset`
converts them to arrays for you. A trial with fewer than 2 spikes has an **empty** `isis`
array. These trials are kept on purpose (1,658 of 25,873).

### Things to know

- **Repeats:** all trials of a neuron show the *same* stimulus. Responses are locked to it,
  so the rasters show vertical stripes (see `figures/`).
- **Rate drift:** firing rate increases over the recording (about 93 spikes/trial on
  average in the first trials vs. about 115 in the last ones).
- **Trial counts:** two sessions have 31 and 34 trials instead of 30.
- **Low rates:** 42 neurons fire below 0.5 Hz, which gives few ISIs per trial.

## Rebuilding the dataset from raw data (optional)

Only needed if you want to change how the dataset is built. Requirements: ~4 GB of free
disk space and a RAR extractor (`unrar`, `unar` or `7z`):

- Arch: `sudo pacman -S unrar`
- Ubuntu/Debian: `sudo apt install unrar`
- macOS: `brew install unar`
- Windows: install 7-Zip and add it to PATH

```bash
# 1. Download the archives (~870 MB) and extract all 13 sessions to extracted/
#    The .rar files in data/ are git-annex pointers (~64 bytes); --download fetches
#    the real files from GIN into data/rar_full/ and checks their MD5.
python scripts/extract.py --data-dir data --out-dir extracted --download

# 2. Build the dataset and run the sanity checks (rates, raster plots)
python scripts/build_isi_dataset.py --extracted-dir extracted \
    --out data/isi_frozennoise.csv --fig-dir figures
```

`build_isi_dataset.py`:
- keeps the 9 sessions that have a frozen-noise stimulus and skips the other 4;
- cuts each frozen segment (1500 running + 300 frozen frame pulses at 30 Hz) into a trial;
- writes the `.csv`, the `.parquet` and the per-neuron summary.

It stops if a session's timing looks inconsistent. Run either script with `--help` to see
all options.

`data/rar_full/`, `extracted/` and `.venv/` are git-ignored. Never commit them: some
archives are over GitHub's 100 MB file limit.

## Citation

If you use this data, cite: Karamanlis D, Gollisch T (2021). *Nonlinear spatial integration
underlies the diversity of retinal ganglion cell responses to natural images.*
J Neurosci 41:3479-3498. doi:10.1523/JNEUROSCI.3075-20.2021
