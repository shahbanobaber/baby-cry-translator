"""Model 7: AST (Audio Spectrogram Transformer, pretrained on AudioSet), frozen,
with a logistic regression on top.

AST has heard thousands of baby cries in AudioSet. If our CNN fails because 612
episodes are too few to learn from, a model that arrives already trained should do better.

Step 1 (GPU):  python src/ast_probe.py extract
    One 768-number embedding per cry, saved to data/processed/ast_emb.npy.
    AST uses its own front end (128 mel bands, 16 kHz); each cry is padded to its
    10.24 s input window.
Step 2 (CPU):  python src/ast_probe.py fit --split B
    Logistic regression on the embeddings of training babies. Each episode gets the
    same total weight, classes are balanced, and cries are averaged per episode.
    Writes runs/ast_split<X>/fold<k>_seed0.csv, readable by evaluate.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
AUDIO = ROOT / "data/raw/enes/audio/00_pooled_separate"
CLASSES = ["hunger", "discomfort", "loneliness"]
CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 16_000


def extract():
    import librosa
    import torch
    from transformers import ASTFeatureExtractor, ASTModel

    idx = pd.read_csv(DATA / "index.csv")
    fe = ASTFeatureExtractor.from_pretrained(CHECKPOINT)
    model = ASTModel.from_pretrained(CHECKPOINT).cuda().eval().half()
    out = np.zeros((len(idx), model.config.hidden_size), np.float16)
    batch = 32
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            waves = [librosa.load(AUDIO / f, sr=SR, mono=True)[0] for f in idx.file[i:i + batch]]
            x = fe(waves, sampling_rate=SR, return_tensors="pt")["input_values"].cuda().half()
            out[i:i + len(waves)] = model(input_values=x).pooler_output.float().cpu().numpy()
            if (i // batch) % 100 == 0:
                print(f"{i + len(waves)}/{len(idx)} cries", flush=True)
    np.save(DATA / "ast_emb.npy", out)
    print("saved", DATA / "ast_emb.npy", out.shape)


def fit(split):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    idx = pd.read_csv(DATA / "index.csv")
    emb = np.load(DATA / "ast_emb.npy").astype(np.float32)
    splits = json.loads((DATA / "splits.json").read_text())["A_by_episode" if split == "A" else "B_by_baby"]
    unit = "episode" if split == "A" else "baby"
    out_dir = ROOT / "runs" / f"ast_split{split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold, s in enumerate(splits):
        tr = idx[idx[unit].isin(s["train"] + s["val"])]
        te = idx[idx[unit].isin(s["test"])]
        # each episode weighs the same in total; each class weighs the same in total
        w = 1.0 / tr.groupby("episode").episode.transform("size")
        w = w / w.groupby(tr.cause).transform("sum")
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
        clf.fit(emb[tr.index], tr.cause, logisticregression__sample_weight=w.to_numpy())
        p = pd.DataFrame(clf.predict_proba(emb[te.index]), columns=clf.classes_, index=te.index)[CLASSES]
        p[["episode", "baby", "cause"]] = te[["episode", "baby", "cause"]]
        ep = p.groupby("episode").agg({**{c: "mean" for c in CLASSES}, "baby": "first", "cause": "first"})
        ep["pred"] = ep[CLASSES].idxmax(axis=1)
        ep["seed"], ep["fold"] = 0, fold
        ep.reset_index().to_csv(out_dir / f"fold{fold}_seed0.csv", index=False)
        (out_dir / f"fold{fold}_seed0.json").write_text(json.dumps({"split": split, "fold": fold, "seed": 0}))
        acc = np.mean([(ep[ep.cause == c].pred == c).mean() for c in CLASSES if (ep.cause == c).any()])
        print(f"fold {fold}: balanced accuracy {acc:.3f} on {len(ep)} episodes", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["extract", "fit"])
    ap.add_argument("--split", choices=["A", "B"], default="B")
    args = ap.parse_args()
    extract() if args.step == "extract" else fit(args.split)


if __name__ == "__main__":
    main()
