"""Train the CNN on one split and save episode-level test predictions.

Run:  python src/train.py --split B --fold 0
      python src/train.py --split B --fold all
      python src/train.py --split A --fold all
      python src/train.py --split B --fold all --per-baby-norm      (experiment 2)
      python src/train.py --split B --fold 0 --epochs 2 --smoke     (quick check)

Writes runs/<name>/fold<k>.csv (one row per test episode) and fold<k>.json (summary).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
CLASSES = ["hunger", "discomfort", "loneliness"]

FRAMES = 151           # 1.5 s at a 10 ms hop
SILENCE = -80.0        # dB floor, used for padding
CRIES_PER_EPISODE = 32
BATCH = 64
LR, WEIGHT_DECAY = 1e-3, 1e-4
FREQ_MASKS, FREQ_MASK_MAX = 2, 8
TIME_MASKS, TIME_MASK_MAX = 2, 20


# ---------- data ----------

class Cries:
    def __init__(self, per_baby_norm):
        self.specs = np.load(DATA / "specs.npy")  # (total_frames, 64) float16
        self.idx = pd.read_csv(DATA / "index.csv")
        self.idx["y"] = self.idx.cause.map(CLASSES.index)
        self.per_baby_norm = per_baby_norm
        self.splits = json.loads((DATA / "splits.json").read_text())

    def rows(self, split, fold, part):
        s = self.splits["A_by_episode" if split == "A" else "B_by_baby"][fold][part]
        key = "episode" if split == "A" else "baby"
        return self.idx[self.idx[key].isin(s)]

    def fit_norm(self, train_rows):
        """Mean and spread per mel band. Global: from training cries only.
        Per baby: from each baby's own cries (no labels needed; an app would record a few first)."""
        def stats(rows):
            fr = np.concatenate([self.specs[o:o + n] for o, n in zip(rows.offset, rows.n_frames)]).astype(np.float32)
            return fr.mean(0), fr.std(0) + 1e-5
        if self.per_baby_norm:
            self.norm = {b: stats(g) for b, g in self.idx.groupby("baby")}
        else:
            self.norm = {"all": stats(train_rows)}

    def crop(self, r, train, rng):
        x = self.specs[r.offset:r.offset + r.n_frames].astype(np.float32)
        n = len(x)
        if n >= FRAMES:
            start = rng.integers(0, n - FRAMES + 1) if train else (n - FRAMES) // 2
            x = x[start:start + FRAMES]
        else:
            out = np.full((FRAMES, x.shape[1]), SILENCE, np.float32)
            start = rng.integers(0, FRAMES - n + 1) if train else (FRAMES - n) // 2
            out[start:start + n] = x
            x = out
        mean, std = self.norm[r.baby if self.per_baby_norm else "all"]
        return ((x - mean) / std).T  # (64, 151)

    def batch(self, rows, train, rng):
        x = np.stack([self.crop(r, train, rng) for r in rows.itertuples()])
        if train:
            x = spec_augment(x, rng)
        return torch.from_numpy(x[:, None]), torch.from_numpy(rows.y.to_numpy(copy=True))


def spec_augment(x, rng):
    for i in range(len(x)):
        for _ in range(FREQ_MASKS):
            w = rng.integers(0, FREQ_MASK_MAX + 1)
            f0 = rng.integers(0, x.shape[1] - w + 1)
            x[i, f0:f0 + w, :] = 0.0
        for _ in range(TIME_MASKS):
            w = rng.integers(0, TIME_MASK_MAX + 1)
            t0 = rng.integers(0, x.shape[2] - w + 1)
            x[i, :, t0:t0 + w] = 0.0
    return x


def epoch_rows(rows, rng):
    """Up to 32 random cries from each episode, so every episode counts about equally."""
    parts = []
    for _, g in rows.groupby("episode"):
        k = min(CRIES_PER_EPISODE, len(g))
        parts.append(g.iloc[rng.choice(len(g), k, replace=False)])
    out = pd.concat(parts)
    return out.iloc[rng.permutation(len(out))]


# ---------- model ----------

def block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2))


class CryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, len(CLASSES)))

    def forward(self, x):
        return self.head(self.features(x))


# ---------- evaluation ----------

@torch.no_grad()
def predict_episodes(model, data, rows, device, rng):
    """Score every cry (centre crop), then average the probabilities over each episode."""
    model.eval()
    probs = []
    for i in range(0, len(rows), 256):
        x, _ = data.batch(rows.iloc[i:i + 256], train=False, rng=rng)
        probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
    p = pd.DataFrame(np.concatenate(probs), columns=CLASSES, index=rows.index)
    p["episode"], p["baby"], p["cause"] = rows.episode, rows.baby, rows.cause
    ep = p.groupby("episode").agg({**{c: "mean" for c in CLASSES}, "baby": "first", "cause": "first"})
    ep["n_cries"] = p.groupby("episode").size()
    ep["pred"] = ep[CLASSES].idxmax(axis=1)
    return ep.reset_index()


def balanced_accuracy(ep):
    return float(np.mean([(ep[ep.cause == c].pred == c).mean() for c in CLASSES if (ep.cause == c).any()]))


# ---------- training ----------

def run_fold(data, split, fold, args, device, out_dir):
    rng = np.random.default_rng(args.seed + fold)
    torch.manual_seed(args.seed + fold)
    train, val, test = (data.rows(split, fold, p) for p in ("train", "val", "test"))
    if args.smoke:  # tiny run to check the code end to end
        train, val, test = (r[r.episode.isin(r.episode.drop_duplicates().sample(12, random_state=0))] for r in (train, val, test))
    data.fit_norm(train)

    n_ep = train.drop_duplicates("episode").cause.value_counts()
    # class weight = total / (3 x episodes of that class): rarer causes count more
    weights = torch.tensor([n_ep.sum() / (len(CLASSES) * n_ep.get(c, 1)) for c in CLASSES], dtype=torch.float32)

    model = CryCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    best, best_state, history = -1.0, None, []
    for epoch in range(args.epochs):
        model.train()
        t0, losses = time.time(), []
        rows = epoch_rows(train, rng)
        for i in range(0, len(rows), BATCH):
            x, y = data.batch(rows.iloc[i:i + BATCH], train=True, rng=rng)
            loss = loss_fn(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        val_bacc = balanced_accuracy(predict_episodes(model, data, val, device, rng))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_bacc": val_bacc})
        print(f"  fold {fold} epoch {epoch + 1:2d}  loss {np.mean(losses):.3f}  val {val_bacc:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if val_bacc > best:
            best, best_state = val_bacc, {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ep = predict_episodes(model, data, test, device, rng)
    ep.to_csv(out_dir / f"fold{fold}.csv", index=False)
    summary = {"split": split, "fold": fold, "per_baby_norm": args.per_baby_norm,
               "test_bacc": balanced_accuracy(ep), "val_bacc_best": best,
               "test_episodes": len(ep), "test_babies": sorted(ep.baby.unique()), "history": history}
    (out_dir / f"fold{fold}.json").write_text(json.dumps(summary, indent=1))
    print(f"fold {fold}: test balanced accuracy {summary['test_bacc']:.3f} on {len(ep)} episodes", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["A", "B"], required=True)
    ap.add_argument("--fold", default="all", help="fold number or 'all'")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--per-baby-norm", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="12 episodes per part, to test the code")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="run folder name")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(1)
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name()})" if device == "cuda" else ""))

    data = Cries(args.per_baby_norm)
    n_folds = len(data.splits["A_by_episode" if args.split == "A" else "B_by_baby"])
    folds = range(n_folds) if args.fold == "all" else [int(args.fold)]
    name = args.name or f"split{args.split}" + ("_perbaby" if args.per_baby_norm else "") + ("_smoke" if args.smoke else "")
    out_dir = ROOT / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in folds:
        run_fold(data, args.split, f, args, device, out_dir)


if __name__ == "__main__":
    main()
