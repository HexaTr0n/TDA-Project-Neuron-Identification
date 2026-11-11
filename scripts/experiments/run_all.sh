#!/bin/sh
# Reproduces every experiment. Run from this folder after:
#   gcc -O3 -march=native -fopenmp -shared -fPIC rips.c -o librips.so
#   python ../spike_distance.py --neurons 60 --q 1 10 100 1000 --jobs 4   (from the repo root)
set -e
export PYTHONDONTWRITEBYTECODE=1
python test_tda.py
python exp0_baselines.py
python exp1_memory.py
python exp2_fits.py
python exp3_impact.py bandwidth
python exp5_fitted.py
python exp3_impact.py size
python exp3_impact.py global
python exp4_ccg_spectrum.py spectrum
python exp4_ccg_spectrum.py ccg
python exp6_shape.py
python exp7_controls.py
python exp8_robustness.py
python exp10_stacked_ccg.py
python exp9_families.py
python exp11_without_rate.py
python exp12_check_signals.py
python exp13_sublevel.py
python exp14_izhikevich.py
python exp15_exact_correlation.py
python exp16_window_count.py
python exp17_heldout.py
python exp18_cluster_cross.py
python exp19_connectivity_pilots.py
