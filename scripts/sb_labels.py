"""Task L4: do ISI-fraction profiles agree with anatomy or cell type? All in stablebear.

Each unit gets one curve: sb.mean of the ISI-fraction PCFs of its samples (sb_cross.py).
For each labelling (cell type, cortical layer, brain region) the script asks two things,
and compares each with the same statistic under shuffled labels (--perms permutations):

    clustering  k-means with mean curves (sb.mean / sb.cdist), k = number of classes;
                adjusted mutual information (AMI) between clusters and labels
    1-NN        each unit's label predicted from its nearest other unit (sb.pdist);
                balanced accuracy (mean recall per class), so a big class cannot win alone

p is the share of shuffled labellings scoring at least as high (one-sided).

Run from the repo root:
    python scripts/sb_labels.py --data data/openscope/glo637908.parquet --session glo637908_grating
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curve_eval import read
from sb_check import sb_kmeans
from sb_cross import isi_fraction_pcfs

REGIONS = {  # coarse grouping of the Allen CCF acronyms in this file
    "cortex": ("VIS",),
    "hippocampus": ("CA1", "CA2", "CA3", "DG", "SUB", "ProS", "POST", "PRE", "HPF"),
    "thalamus": ("LP", "LGd", "LGv", "MGd", "MGm", "MGv", "PoT", "POL", "TH", "SGN", "Eth", "PO"),
    "midbrain": ("APN", "MB", "MRN", "SCig", "SCiw", "NB", "PPT", "NOT", "OP"),
}


def region(area):
    for name, prefixes in REGIONS.items():
        if area.startswith(prefixes):
            return name
    return "other"


def main():
    import stablebear as sb

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--min-isis", type=int, default=30, help="per sample")
    ap.add_argument("--min-samples", type=int, default=10, help="usable samples per unit")
    ap.add_argument("--perms", type=int, default=200, help="shuffled labellings for 1-NN")
    ap.add_argument("--cluster-perms", type=int, default=50, help="shuffled labellings for AMI")
    ap.add_argument("--kmeans-seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = read(args.data, [args.session])
    df = df[df["isis"].map(len) >= args.min_isis]
    counts = df["neuron_id"].value_counts()
    df = df[df["neuron_id"].isin(counts.index[counts >= args.min_samples])]
    units = df.drop_duplicates("neuron_id").set_index("neuron_id").sort_index()
    print(f"{args.session}: {len(units)} units with >= {args.min_samples} samples of >= "
          f"{args.min_isis} ISIs; {len(df)} samples", flush=True)

    t0 = time.time()
    curves = []
    for nid in units.index:
        p = isi_fraction_pcfs(list(df.loc[df["neuron_id"] == nid, "spike_times"]))
        curves.append(sb.mean(sb.PcfTensor(p)))
    curves = [c[0] for c in curves]
    D = np.asarray(sb.pdist(sb.PcfTensor(curves), p=1).to_dense(), dtype=float)
    np.fill_diagonal(D, np.inf)
    print(f"curves and distances: {time.time() - t0:.0f} s", flush=True)

    labellings = {"cell type (PV_tagged / narrow / wide)": units["cell_type"].astype(str)}
    lay = units["layer"].astype(str)
    labellings["cortical layer (cortex only)"] = lay[lay.isin(["2/3", "4", "5", "6a"])]
    labellings["region (cortex / hippocampus / thalamus / midbrain)"] = \
        units["area"].astype(str).map(region).loc[lambda r: r != "other"]
    labellings["PV_tagged vs narrow (cortex only)"] = units.loc[
        units["area"].astype(str).str.startswith("VIS") & units["cell_type"].isin(["PV_tagged", "narrow"]),
        "cell_type"].astype(str)

    rng = np.random.default_rng(args.seed)
    rows = []
    for name, lab in labellings.items():
        idx = np.searchsorted(units.index, lab.index)
        y = np.unique(lab.to_numpy(), return_inverse=True)[1]
        k = y.max() + 1
        Dsub = D[np.ix_(idx, idx)]
        sub = [curves[i] for i in idx]
        nn = Dsub.argmin(1)
        bacc = balanced_accuracy_score(y, y[nn])
        null_b = np.array([balanced_accuracy_score(p, p[nn]) for p in
                           (rng.permutation(y) for _ in range(args.perms))])
        lab_k = [sb_kmeans(sub, k, np.random.default_rng(s)) for s in range(args.kmeans_seeds)]
        ami = np.mean([adjusted_mutual_info_score(y, lk) for lk in lab_k])
        null_a = np.array([np.mean([adjusted_mutual_info_score(p, lk) for lk in lab_k]) for p in
                           (rng.permutation(y) for _ in range(args.cluster_perms))])
        rows.append(dict(labels=name, units=len(y),
                         classes=" / ".join(f"{c}:{n}" for c, n in lab.value_counts().items()),
                         nn_bal_acc=bacc, nn_shuffled=null_b.mean(), nn_shuffled_95=np.quantile(null_b, 0.95),
                         nn_p=(np.sum(null_b >= bacc) + 1) / (len(null_b) + 1),
                         ami=ami, ami_shuffled=null_a.mean(), ami_shuffled_95=np.quantile(null_a, 0.95),
                         ami_p=(np.sum(null_a >= ami) + 1) / (len(null_a) + 1)))
        print(pd.Series(rows[-1]).to_string(), "\n", flush=True)
    out = pd.DataFrame(rows)
    print(out.drop(columns="classes").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
