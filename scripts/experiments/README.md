# Experiments

Everything here compares embeddings on the same trials with the same scores (see `lab.py`).
Results land in `results/experiments/results.csv`, unless `LAB_OUT` says otherwise.

`rips.c` is a small H0 and H1 Rips routine, checked against brute force in `test_tda.py`.
It lets these scripts run without stablebear. It is a test tool, not a replacement: use
stablebear through `run_pipeline.py` for the real analysis.

| Script | Question |
|---|---|
| `exp0_baselines.py` | Baselines with all three scores |
| `exp1_memory.py` | Does history predict the next ISI within a trial? |
| `exp2_fits.py` | Which pdf model predicts the next ISI best? |
| `exp3_impact.py` | Effect of bandwidth, cloud size, and global-information fixes |
| `exp4_ccg_spectrum.py` | Spectral and cross-correlation embeddings |
| `exp5_fitted.py` | Topology on clouds of fitted pdfs, delay clouds, time-tagged clouds |
| `exp6_shape.py` | How many peaks does one trial's ISI distribution need? |
| `exp7_controls.py` | Shuffled controls for the fixes, and held-out weighting |
| `exp8_robustness.py` | Is topology's small gain real? Other seeds, all 146 neurons |
| `exp9_families.py` | Topology on neuron space: families or one blob? |
| `exp10_stacked_ccg.py` | Stacked cross-correlation as a matrix, an image, and a 2D cloud |
| `exp11_without_rate.py` | The family test again with the firing-rate axis removed |
| `exp12_check_signals.py` | Are the two signals from exp11 real? |
| `exp13_sublevel.py` | Sublevel-set persistence of the smoothed rate, peak matching in time |
| `exp14_izhikevich.py` | Single Izhikevich neurons: does H0 count the ISI period, and how drive changes memory |
| `exp15_exact_correlation.py` | Pair correlations from exact spike-pair sums against 2 ms binning |
| `exp16_window_count.py` | Sliding-window count: L1 embedding that matches the spike distance for single edits |
| `exp17_heldout.py` | Settings chosen on half the neurons, scored on the other half (needs exp13) |
| `exp18_cluster_cross.py` | The exploratory embeddings scored by clustering and across movie halves |
| `netsim.py` | Simulated spiking networks with known wiring |
| `exp19_connectivity_pilots.py` | Pilots: single connections, groups, flow between groups, ring wiring |
| `exp20_cluster_anatomy.py` | Fewer or more clusters, and which neurons and trials get misclustered (needs the pipeline's saved distances) |
| `exp21_stimulus_free.py` | Rescaled-time ISI bars, burst tables and autocorrelogram persistence, across movie halves |
| `exp22_pcf_ideas.py` | Seven more piecewise constant signatures, and k-means with mean curves |
| `exp23_other_sessions.py` | The best signatures on all nine sessions, other neuron sets, and without the first trials |
| `exp24_session_differences.py` | Why retina sessions differ in clustering: reliability, rate, margin |
| `exp25_isi_distances.py` | Other distances for ISI fractions, and a two-parameter order-only signature |

`exp5` needs `hmms.pkl` from `exp2`. `exp3` to `exp12` need `baselines.npz` from `exp0`.
`exp11` and `exp12` need `neuron_spaces.npz` from `exp9`.
`run_all.sh` runs them in order. Linux or macOS with gcc, 2 cores took about an hour.
