# Analysis scripts: how to replicate the results

**Question.** Can one short sample of spikes tell which neuron produced it? Every method turns a
sample into a signature and compares signatures by a distance. Two scores are reported
everywhere: **1-NN** (a sample takes the label of its nearest other sample) and **k-medoids**
(samples are grouped without labels into as many clusters as neurons; the score is the share of
samples in the right cluster). Most tables also give k-means with mean curves as cluster centres.
Chance is 1.7% for 60 neurons.

- `notes/summary.md`: the results in one page.
- `notes/results/`: one file per experiment, with the full tables and the exact commands.
- `notes/keep_or_cut.md`: every method that was tried, what it is, and why it is kept or dropped.
- Everything not kept here (exploratory code, the other result files, the working log) is on
  the branch `web-chat`.

## 1. Setup

```sh
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r scripts/requirements.txt
python scripts/test_methods.py       # checks the maths of each method and the stablebear calls
```

All commands are run from the repo root. Outputs go to `results/`, which git ignores.

## 2. Data

| Dataset | Used as | In the repo? | How to get it |
|---|---|---|---|
| Retina, frozen noise ([Karamanlis and Gollisch 2021](https://gin.g-node.org/gollischlab/Karamanlis_Gollisch_2021_RGC_spiketrains_natural_image_encoding)) | 10 s trials of one repeated movie, 9 sessions, 850 cells | Yes: `data/isi_frozennoise.parquet` | Nothing to do |
| Retina, unrepeated noise between the frozen trials (same recordings) | The second stimulus for the across-stimuli tests | No | Step A |
| OpenScope Global/Local Oddball, DANDI 000253, `sub-637908_ses-1213341633_ogen.nwb` | Mouse visual cortex and hippocampus; grating blocks and grey screen | No (large) | Step B |
| Allen Visual Coding Neuropixels, DANDI 000022, sessions 829720705 and 789848216 | Mouse visual cortex, thalamus, hippocampus; natural movie and spontaneous activity | No (large) | Step C |

**Step A, retina unrepeated noise.** Needs `unrar`, `unar` or `7z`.

```sh
python scripts/extract.py --data-dir data --out-dir extracted --download
python scripts/load_retina_running.py --extracted-dir extracted --session 20180126_sr_le_sp
python scripts/load_retina_running.py --extracted-dir extracted --session 20180126_sr_le_sp --cut 24.8 30.0
```

This writes `data/retina_running_20180126_sr_le_sp.parquet` and `..._clean.parquet`. The second
removes a 5 s chunk that repeats in every "unrepeated" stretch (`notes/results/L24_data_checks.md`).

**Step B, OpenScope.** Download the NWB file named above from
[DANDI 000253](https://dandiarchive.org/dandiset/000253) into `data/openscope/`, then:

```sh
python scripts/load_openscope.py data/openscope/sub-637908_ses-1213341633_ogen.nwb
```

This writes `data/openscope/glo637908.parquet`.

**Step C, Allen.** Download into `data/allen/`, then convert:

```sh
dandi download "https://api.dandiarchive.org/api/dandisets/000022/versions/draft/assets/?path=sub-763808588/sub-763808588_ses-789848216.nwb"
python scripts/load_allen.py data/allen/sub-811322604_ses-829720705.nwb
python scripts/load_allen.py data/allen/sub-763808588_ses-789848216.nwb
```

The `dandi download` line is the one recorded for the second session; the first session
(`sub-811322604/sub-811322604_ses-829720705.nwb`) is fetched the same way. The loaders write
`data/allen/allen829720705.parquet` and `data/allen/allen789848216.parquet`.

## 3. Results and the commands that produce them

The commands are the ones recorded in each result file. They were run on Linux with real
stablebear.

| Result | File in `notes/results/` | Data needed | Commands |
|---|---|---|---|
| stablebear agrees with a plain numpy version of the same curves | `L2_stablebear_check.md` | retina frozen | `python scripts/sb_check.py --session 20180126_sr_le_sp --neurons 60 --kmeans-seeds 5` |
| Every method within one stimulus and across a stimulus change | `L17_full_table.md` | A, B, C | `python scripts/sb_full_table.py --dataset retina10 --drop-piece 2`; the same with `--dataset retina40`, `openscope10`, `allen30` |
| Spike distance (Victor-Purpura) as the reference | `L18_spike_distance_reference.md` | A, B, C | `NUMBA_NUM_THREADS=8 python scripts/sb_spike_distance.py --dataset retina10 --drop-piece 2 --jobs 8`; the same with `openscope10`, `allen30` (no `--drop-piece`) |
| Score against sample length | `L6_sample_length.md` | A, B | `python scripts/sb_length.py --data data/retina_running_20180126_sr_le_sp.parquet --select-session 20180126_sr_le_sp --session 20180126_sr_le_sp_running --join-pieces`; `python scripts/sb_length.py --data data/openscope/glo637908.parquet --select-session glo637908_grating --session glo637908_grating --lengths 2.5 5 10 20 40 120` |
| Rips against the ISI distribution at equal cloud size | `L19x_cloud_size.md` | A, B | `python scripts/sb_cloud_size.py --dataset retina40`; `--dataset openscope124` |
| Whole cloud, random time cut, random draws, consecutive stretch | `L20_cloud_sampling.md` | A, B, C | `python scripts/sb_cloud_sampling.py --dataset retina10 --drop-piece 2`; the same with `openscope10`, `allen30`, `retina40`, `openscope124` |
| H1 on whole clouds, Betti curves for H1, time-filtered complex | `L21_rips_fixes.md` | A, B, C | `python scripts/sb_rips_fixes.py --dataset retina10 --drop-piece 2`; the same with `openscope10`, `allen30`, `retina40clean`, `openscope124` (this last one took about 2 h and up to 14 GB of memory) |
| Headline table from barcodes only, with counting twins, five seeds, nine retina sessions | `L22_headline_homology.md` | retina frozen for the nine sessions; A, B, C for the rest | `python scripts/sb_headline.py --dataset retina10 --drop-piece 2` (also `retina40clean`, `openscope10`, `allen30`, `allen30adj`, `openscope124`); seeds: add `--seed 1 --no-kmeans` (1 to 4); sessions: `python scripts/sb_headline.py --dataset retina:20171215_sr_le_sp --no-kmeans` for each session; twins: `python scripts/sb_twins.py --dataset retina10 --drop-piece 2` (also the other datasets) |
| Data checks: the repeating retina chunk, drift on Allen, a second Allen session | `L24_data_checks.md` | A, C | `python scripts/sb_allen_drift.py`; the other commands are in the file |
| Fitted pdfs (pilots, retina only, **not stablebear**) | `W11` to `W14` | retina frozen | section 5 |

The first pipeline (`run_pipeline.py`, retina frozen noise only) still works on its own:

```sh
python scripts/run_pipeline.py --list            # every config
python scripts/run_pipeline.py --neurons 60 --jobs 4
python scripts/compare.py --neurons 60           # held-out comparison of the saved distance matrices
python scripts/cross_segment.py --neurons 60     # halves of the movie against each other
```

## 4. Methods

| Method | What it is | Where |
|---|---|---|
| Time bars | One bar `[t, t+s)` per spike. The Betti curve at `t` is the number of spikes in the last `s` seconds. Used with a square root and three window sizes (20, 67, 200 ms). With `s = 2/q` its L1 distance never exceeds the spike distance | `barcodes.py`, `curves.py` |
| ISI fractions | One bar per inter-spike interval, of length log ISI. The stable rank divided by the number of bars is the share of ISIs longer than each value, so L1 is the Wasserstein-1 distance between ISI distributions. These bars are H0 of the spike times on a line | `curves.py` |
| Burst bars | Thicken each spike `t` to `[t, t+s)`. Spikes closer than `s` fuse into one piece, a burst; each burst is one bar over time | `barcodes.py` |
| Burst table | For 20 time scales, the number of bursts with at least 1, 2, 3, ... spikes. No homology | `curves.py` |
| Peak matching | Smooth the train into a rate. Each peak is an H0 bar of the superlevel sets, with a time and a prominence. Trials are compared by matching peaks in time order | `barcodes.py`, `spike_distance.py` |
| Spike distance | Victor-Purpura edit distance: 1 to add or delete a spike, `q` per second to shift one | `spike_distance.py`, `curve_eval.py` |
| Delay-embedding Rips | Points are windows of consecutive log ISIs `(I_k, ..., I_(k+m-1))`. Rips H0 and H1, stable ranks | `embeddings.py`, `sb_full_table.py` |
| Delay regions | The same cloud split around fixed anchor points (k-means on other neurons). Rips inside each region; stable ranks summed. The twin without bars counts points per region | `sb_feature_cloud.py`, `sb_headline.py`, `sb_twins.py` |
| Pdf per spike | For each spike, the estimated distribution of the next ISI given the last two, by kernel weighting over the same sample | `predictive_state.py` |

**Controls.** ISIs shuffled within each sample (same distribution, no order, no locking to the
stimulus); a counting twin for every barcode (what do the bars add beyond counting); settings
chosen on other neurons than the ones scored.

## 5. Pdf pilots (`scripts/pilots/`)

Four small experiments on fitting a pdf to the ISIs. They use numpy, scipy and scikit-learn,
**not stablebear**, and only the retina frozen-noise data in the repo.

```sh
python scripts/pilots/isi_pdf_fit.py          # W11: one pdf per trial (lognormal, gamma, mixtures), about 5 min
python scripts/pilots/hmm_pdf_per_spike.py    # W12: per spike, whole history, hidden Markov model, about 10 min
python scripts/pilots/running_mixture.py      # W13: per spike, sqrt(n) Gaussians on the first n ISIs, about 30 min
gcc -O3 -march=native -fopenmp -shared -fPIC scripts/pilots/rips.c -o scripts/pilots/librips.so
python scripts/pilots/window_mixture.py       # W14: per spike, mixture on the last 16, 32, 64 ISIs, about 30 min
```

Times are for 2 cores. `window_mixture.py` needs the small Rips library built by the `gcc` line.

## 6. Caveats

- Retina numbers come mostly from one session (`20180126_sr_le_sp`), the best of nine for
  ISI-based methods. `L22_headline_homology.md` has all nine.
- One OpenScope session and two Allen sessions.
- Clustering scores move by about 2.5 points between random starts, and the choice of the 60
  neurons moves every row by about 4 points.
- Saved results do not record the code version: rerun after changing a method.
