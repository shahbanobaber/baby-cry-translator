"""Model 4: the paper's random forest on its 10 acoustic features, on our splits.

Same features and log transforms as the paper (cause~ac_RF.R). Each feature is
standardised per baby, as in the paper, but from the baby's other episodes only
(the same no-peeking rule as our per-baby CNN). Classes are balanced per tree,
like the paper's equal stratified sampling. Cries are scored, then averaged per
episode, as for the CNN.

Run:  python src/forest.py --split B
      python src/forest.py --split A
      python src/forest.py --split C --personal   (one forest per baby, trained on that baby only)
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
SPLIT_KEY = {"A": "A_by_episode", "B": "B_by_baby", "C": "C_by_time"}
FEATURES = sorted(["duration_segment_P", "voiced_S", "pitch_median_S", "pitch_iqr_S",
                   "specCentroidVoiced_median_S", "entropyVoiced_median_S", "HNRVoiced_median_S",
                   "roughnessVoiced_median_S", "jitter_P", "shimmer_P"])
LOGGED = ["duration_segment_P", "pitch_median_S", "pitch_iqr_S", "specCentroidVoiced_median_S"]


def features(reference=None):
    """reference: episodes allowed as scaling reference (split C: training period only,
    so the future is never used). The episode being scaled is always left out."""
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
            if reference is not None:
                rest = rest[gb.loc[rest, "episode"].isin(reference)]
            ref = df.loc[rest, FEATURES].to_numpy(float)
            mean, std = ref.mean(0), ref.std(0) + 1e-9
            out[df.index.get_indexer(ge.index)] = (df.loc[ge.index, FEATURES].to_numpy(float) - mean) / std
    df[FEATURES] = out
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["A", "B", "C"], required=True)
    ap.add_argument("--personal", action="store_true", help="one forest per baby (split C)")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    splits = json.loads((DATA / "splits.json").read_text())[SPLIT_KEY[args.split]]
    df = features(set(splits[0]["train"] + splits[0]["val"]) if args.split == "C" else None)
    unit = "baby" if args.split == "B" else "episode"
    if args.personal and args.split != "C":
        ap.error("--personal needs split C")
    out_dir = ROOT / "runs" / (f"forest_personal_split{args.split}" if args.personal else f"forest_split{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(args.seeds):
        for fold, s in enumerate(splits):
            tr = df[df[unit].isin(s["train"] + s["val"])]  # a forest needs no early stopping
            te = df[df[unit].isin(s["test"])]
            groups = [(tr[tr.baby == b], te[te.baby == b]) for b in te.baby.unique()] if args.personal else [(tr, te)]
            parts = []
            for tr_g, te_g in groups:
                rf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                            random_state=1000 * seed + fold, n_jobs=1)
                rf.fit(tr_g[FEATURES], tr_g.cause)
                pg = pd.DataFrame(0.0, columns=CLASSES, index=te_g.index)
                pg[list(rf.classes_)] = rf.predict_proba(te_g[FEATURES])  # a cause absent from training gets 0
                parts.append(pg)
            p = pd.concat(parts)
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
