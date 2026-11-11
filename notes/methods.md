# Notes from the web chat

This branch carries the exploratory work done in the web chat, so that a session with the data
and real stablebear can rerun it. Everything under `scripts/experiments/` ran on a stand-in for
stablebear or on plain numpy, except where a log says otherwise.

Take only what you need, for example:

    git checkout web-chat -- scripts/experiments notes

The files in `scripts/` on this branch are the web chat's copies of the pipeline. If the local
ones are newer, keep the local ones.

## What we learned (retina, frozen noise, 60 neurons unless stated)

- A barcode worked when each bar stayed attached to a meaningful coordinate (movie time or ISI
  length). It failed when a small cloud was filtered by distance and only bar lengths were kept.
- Rips on ISI delay embeddings equals its shuffled control in every setting. H1 never helped.
- For richer clouds (pdf per spike, fitted models, spectrum, correlograms) the cloud's position
  always beat its barcode, for example 40.9% against 16.2% clustering for pdf per spike.
- One stimulus: time-locked curves win. Across stimuli they fall to chance, and only ISI
  features survive. One signature cannot serve both tasks.
- 1-NN flatters: about 90% where clustering gives 40-67%.
- Clustering improved most from two changes: the square root of the count curve, and k-means
  with mean curves as centres instead of k-medoids.
- Across the nine retina sessions, clustering with square-root time bars and mean curves is
  51% on average: 67-72% in three sessions, 36-58% in six. Report the average.
- Trial 0 is the worst clustered trial. Dropping the first five trials changes little.

## Signatures

Every signature is a curve sampled on a grid and compared by L1, so each is a PCF. `dur` is the
trial length (9.94 s on the retina data; replace it for other data). Scores are 1-NN /
clustering with mean curves, session 20180126_sr_le_sp, unless stated.

| Signature | Definition | Control | Retina |
|---|---|---|---|
| Time bars | spikes in `(t - s, t]`, `s = 2/30` s, grid step `s/20`; Betti curve of bars `[t_k, t_k + s)` | ISIs shuffled | 75.6% / 56.9% |
| Square-root time bars | square root of the above | ISIs shuffled | 85.0% / 66.8% |
| Multi-scale square-root time bars | the square-root curve at `s` = 0.02, 2/30, 0.2 s; each divided by its median pairwise L1 distance; concatenated | ISIs shuffled | 88.1% / 66.7% |
| ISI fractions | stable rank of bars `[0, log(max(ISI, 1e-4))]` divided by the number of ISIs; 230 grid points from `log(1e-4)` to `log(10)` | none (order-free) | across halves: 41.5% 1-NN, 28-30% clustering |
| Square-root time bars + ISI fractions | both, each divided by its median distance, ISI part weighted 0.5 | ISIs shuffled | 90.8% / 67.1% (63.6% with k-medoids) |
| Current-ISI function | `log` of the gap containing `t`, gaps between consecutive points of `[0, spikes, dur]`; step 5 ms | ISIs shuffled | 76.5% / 54.5% |
| Burst bars | one bar `[first spike, last spike + s)` per run of spikes with gaps `<= s`; Betti curve | ISIs shuffled | about 64-70% 1-NN |
| Burst-size function | square root of the size of the burst alive at `t`, `s = 2/30`; step 5 ms | ISIs shuffled | 77.2% / 56.1% |
| Coincidence function over time | per own spike, the number of other neurons' spikes within 10 ms; summed over own spikes in `(t - 2/30, t]`; square root | other neurons from the next trial | 83.8% / 66.9% |
| Conditional ISI fractions | ISI fractions separately for ISIs that follow a short or a long ISI (split at the trial median) | ISIs shuffled first | across halves 37.9% against 34.0% |
| Burst table | for 20 scales from 2 ms to 1 s and m in 1,2,3,4,6,8,12,16: bursts with at least m spikes, divided by the spike count | ISIs shuffled | across halves 40.7% against 37.0% |
| Peak matching | H0 of the smoothed rate; peaks with time and mass, matched in time order | ISIs shuffled | 75% held-out 1-NN |

Tested without gain: rescaled-time ISI bars, autocorrelogram persistence, ISI fractions split
by busy and quiet moments of the population, coincidences as a function of the radius, Rips on
any delay embedding, persistence of the cross-correlogram peak for detecting connections,
ring-shaped wiring from correlations in simulation.

## Scores

- **Clustering with mean curves:** assign each trial to the nearest centre by L1, centre = mean
  curve of its members, k = number of neurons, 3 random restarts, 25 iterations. Score = share of
  trials in the right cluster after pairing clusters with neurons (Hungarian method).
- **Across stimuli:** cut each trial in two. 1-NN: nearest half-trial among the other half of the
  other trials. Clustering: cluster one half without labels, name clusters by neuron, then assign
  the other half's trials to the nearest centre.
- **Held-out settings:** choose settings on half the neurons, score on the other half.

## Where the code is

| Script | Contents |
|---|---|
| `exp18_cluster_cross.py` | Older embeddings scored by clustering and across halves |
| `exp19_connectivity_pilots.py`, `netsim.py` | Simulated networks: connections, groups, flow between groups |
| `exp20_cluster_anatomy.py` | Which neurons and trials are misclustered |
| `exp21_stimulus_free.py` | ISI fractions, rescaled time, burst table, autocorrelogram |
| `exp22_pcf_ideas.py` | The seven curve signatures and k-means with mean curves |
| `exp23_other_sessions.py` | The best signatures on all nine sessions |

`scripts/experiments/README.md` lists the rest. The older scripts import `lab.py`, which needs
`librips.so`: build it with `gcc -O3 -march=native -fopenmp -shared -fPIC rips.c -o librips.so`.

## Asked of the local session

1. Run `exp22_pcf_ideas.py`-style signatures on the OpenScope table, with the trial length changed.
2. Write results to `notes/results_openscope.md`: trial length, neurons kept, spikes per sample,
   how trials were joined, and the score table.
3. Commit to this branch, so the web chat can read them.
