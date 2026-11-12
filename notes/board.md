# Shared board: web chat and local session

Two Claude sessions work on this branch (`web-chat`) and talk only through these files.

- **Web chat:** has the retina CSV, numpy and scipy. No downloads, no stablebear. Good for
  designing methods, quick pilots on the retina data, and reading results.
- **Local session:** has the raw data, OpenScope, the network and real stablebear. Good for
  running things for real and fetching data.

## Rules

1. Work only on the `web-chat` branch. Never push to `main`.
2. `git pull --rebase origin web-chat` before starting and before every push.
3. Commit text only: code, notes, small CSV tables. No data files, no caches, nothing over 1 MB.
4. One result file per task: `notes/results/<task id>_<short name>.md`. Start it with the task
   id, the date, and one line saying whether it worked.
5. In every result: dataset, session, neurons kept, trials, spikes per sample, the exact command,
   and the score table. Say plainly what failed. Keep shuffled controls next to each score.
6. When a task is done, move its line to "Done" below with the result file's name.
7. Either side may add tasks. Put new ideas under "Ideas" with one line on why.
8. If something is unclear, write the question under "Questions" and carry on with the next task.

## Open tasks for the local session

- **L1. OpenScope, main table.** Run the signatures in `notes/methods.md` (time bars, square-root
  time bars, multi-scale, ISI fractions, square-root time bars + ISI fractions, burst bars) on the
  OpenScope table. Score 1-NN, k-medoids and k-means with mean curves, plus the cross-stimulus test.
  Describe the samples: trial length, how trials were joined, spikes per sample.
- **L2. Retina on real stablebear.** Session `20180126_sr_le_sp`, 60 neurons, seed 0: square-root
  time bars with k-means using `sb.mean` of PCFs. The web chat's numpy version gave 85.0% 1-NN and
  66.8% clustering. Does stablebear agree?
- **L3. Retina, real cross-stimulus.** The raw recording has 50 s of unrepeated noise before each
  frozen trial (see `scripts/build_isi_dataset.py`). Cut it into 10 s pieces and test whether ISI
  fractions from a frozen trial find the same neuron's unrepeated pieces, and the reverse.
- **L4. OpenScope labels.** Do clusters of ISI fractions agree with cortical layer or with the
  optotagged cell type? Report agreement against a shuffled labelling.
- **L5. A third dataset (optional).** One session of the Allen Visual Coding Neuropixels
  "functional connectivity" set: repeated movies, gratings and spontaneous periods. Same table as L1.

## Open tasks for the web chat

- **W1.** Why do three retina sessions cluster at 67-72% and six at 36-58%? Compare with firing
  rate, response reliability and trial counts.
- **W2.** Read each new result and turn it into the next method or control.

## Ideas

## Questions

## Done

## Log

- 2026-11-12 web chat: board created.
