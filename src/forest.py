"""Model 4: the paper's random forest on its 10 acoustic features, on our splits.

Same features and log transforms as the paper (cause~ac_RF.R). Each feature is
standardised per baby, as in the paper, but from the baby's other episodes only
(the same no-peeking rule as our per-baby CNN). Classes are balanced per tree,
like the paper's equal stratified sampling. Cries are scored, then averaged per
episode, as for the CNN.

Run:  python src/forest.py --split B
      python src/forest.py --split A
Writes runs/forest_split<X>/fold<k>_seed<s>.csv, readable by evaluate.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
META = ROOT / "data/raw/enes/meta/dataset_44605_short.csv"
CLASSES = ["hunger", "discomfort", "loneliness"]
FEATURES = sorted(["duration_segment_P", "voiced_S", "pitch_median_S", "pitch_iqr_S",
                   "specCentroidVoiced_median_S", "entropyVoiced_median_S", "HNRVoiced_median_S",
                   "roughnessVoiced_median_S", "jitter_P", "shimmer_P"])
LOGGED = ["duration_segment_P", "pitch_median_S", "pitch_iqr_S", "specCentroidVoiced_median_S"]


def features():
    df = pd.read_csv(META)
    df = df[df.cause_stop_engl.isin(CLASSES)].rename(columns={"file_seq_S": "episode", "cause_stop_engl": "cause"})
    for v in LOGGED:
        df[v] = np.log(df[v] + 1e-3)
    # standardise per baby, leaving out the episode being scored
    x = df[FEATURES].to_numpy(float)
    out = np.empty_like(x)
    for _, gb in df.groupby("baby"):
        for _, ge in gb.groupby("episode"):
            rest = gb.index.difference(ge.index)
            ref = df.loc[rest, FEATURES].to_numpy(float)
            mean, std = ref.mean(0), ref.std(0) + 1e-9
            out[df.index.get_indexer(ge.index)] = (df.loc[ge.index, FEATURES].to_numpy(float) - mean) / std
    df[FEATURES] = out
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["A", "B"], required=True)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    df = features()
    splits = json.loads((DATA / "splits.json").read_text())["A_by_episode" if args.split == "A" else "B_by_baby"]
    unit = "episode" if args.split == "A" else "baby"
    out_dir = ROOT / "runs" / f"forest_split{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(args.seeds):
        for fold, s in enumerate(splits):
            tr = df[df[unit].isin(s["train"] + s["val"])]  # a forest needs no early stopping
            te = df[df[unit].isin(s["test"])]
            rf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                        random_state=1000 * seed + fold, n_jobs=1)
            rf.fit(tr[FEATURES], tr.cause)
            p = pd.DataFrame(rf.predict_proba(te[FEATURES]), columns=rf.classes_, index=te.index)[CLASSES]
            p[["episode", "baby", "cause"]] = te[["episode", "baby", "cause"]]
            ep = p.groupby("episode").agg({**{c: "mean" for c in CLASSES}, "baby": "first", "cause": "first"})
            ep["pred"] = ep[CLASSES].idxmax(axis=1)
            ep["seed"], ep["fold"] = seed, fold
            ep.reset_index().to_csv(out_dir / f"fold{fold}_seed{seed}.csv", index=False)
            (out_dir / f"fold{fold}_seed{seed}.json").write_text(json.dumps({"split": args.split, "fold": fold, "seed": seed}))
            acc = np.mean([(ep[ep.cause == c].pred == c).mean() for c in CLASSES if (ep.cause == c).any()])
            print(f"seed {seed} fold {fold}: balanced accuracy {acc:.3f} on {len(ep)} episodes", flush=True)


if __name__ == "__main__":
    main()
