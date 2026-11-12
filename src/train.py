"""Train the CNN on one split and save episode-level test predictions.

Run:  python src/train.py --split B --fold 0
      python src/train.py --split B --fold all
      python src/train.py --split A --fold all
      python src/train.py --split B --fold all --per-baby-norm      (experiment 2)
      python src/train.py --split B --fold 0 --epochs 2 --smoke     (quick check)
      python src/train.py --task identity                           (model 8: name the baby, split A)

Writes runs/<name>/fold<k>_seed<s>.csv (one row per test episode) and .json (summary).
Each fold is trained with 3 seeds by default.
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
TARGET, LABELS = "cause", CLASSES   # the identity task switches these to "baby" and the 24 babies

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
        self.idx["y"] = self.idx[TARGET].map(LABELS.index)
        self.per_baby_norm = per_baby_norm
        self.splits = json.loads((DATA / "splits.json").read_text())

    def rows(self, split, fold, part):
        s = self.splits["A_by_episode" if split == "A" else "B_by_baby"][fold][part]
        key = "episode" if split == "A" else "baby"
        return self.idx[self.idx[key].isin(s)]

    def fit_norm(self, train_rows):
        """Mean and spread per mel band.
        Global: from training babies only.
        Per baby: from the same baby's other episodes, never the episode being scored.
        No labels are used; an app would calibrate on a baby's earlier cries the same way."""
        if not self.per_baby_norm:
            fr = np.concatenate([self.specs[o:o + n] for o, n in zip(train_rows.offset, train_rows.n_frames)]).astype(np.float32)
            self.norm = {"all": (fr.mean(0), fr.std(0) + 1e-5)}
            return
        sums = {}  # episode -> (sum, sum of squares, frame count)
        for e, g in self.idx.groupby("episode"):
            fr = np.concatenate([self.specs[o:o + n] for o, n in zip(g.offset, g.n_frames)]).astype(np.float64)
            sums[e] = (fr.sum(0), (fr ** 2).sum(0), len(fr), g.baby.iloc[0])
        baby = {}
        for s1, s2, n, b in sums.values():
            t = baby.get(b, (0.0, 0.0, 0))
            baby[b] = (t[0] + s1, t[1] + s2, t[2] + n)
        self.norm = {}
        for e, (s1, s2, n, b) in sums.items():
            t1, t2, tn = baby[b][0] - s1, baby[b][1] - s2, baby[b][2] - n
            mean = t1 / tn
            self.norm[e] = (mean.astype(np.float32), (np.sqrt(np.maximum(t2 / tn - mean ** 2, 0)) + 1e-5).astype(np.float32))

    def window(self, r, start, rng, train):
        """One 1.5 s window of a cry. Short cries are padded with silence."""
        x = self.specs[r.offset:r.offset + r.n_frames].astype(np.float32)
        n = len(x)
        if n >= FRAMES:
            if start is None:
                start = rng.integers(0, n - FRAMES + 1)
            x = x[start:start + FRAMES]
        else:
            out = np.full((FRAMES, x.shape[1]), SILENCE, np.float32)
            pad = rng.integers(0, FRAMES - n + 1) if train else (FRAMES - n) // 2
            out[pad:pad + n] = x
            x = out
        mean, std = self.norm[r.episode if self.per_baby_norm else "all"]
        return ((x - mean) / std).T  # (64, 151)

    def windows_for_test(self, rows):
        """Sliding 1.5 s windows (half overlap) so long cries are scored in full."""
        out = []
        for r in rows.itertuples():
            if r.n_frames <= FRAMES:
                out.append((r, 0))
            else:
                starts = list(range(0, r.n_frames - FRAMES + 1, FRAMES // 2))
                if starts[-1] != r.n_frames - FRAMES:
                    starts.append(r.n_frames - FRAMES)
                out += [(r, st) for st in starts]
        return out

    def batch(self, rows, rng):
        """Training batch: a random window of each cry, then SpecAugment."""
        x = np.stack([self.window(r, None, rng, train=True) for r in rows.itertuples()])
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
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, len(LABELS)))

    def forward(self, x):
        return self.head(self.features(x))


# ---------- evaluation ----------

@torch.no_grad()
def predict_episodes(model, data, rows, device, rng):
    """Score every window of every cry, average windows into cries,
    then average cries into one answer per episode."""
    model.eval()
    wins = data.windows_for_test(rows)
    probs = []
    for i in range(0, len(wins), 512):
        chunk = wins[i:i + 512]
        x = np.stack([data.window(r, st, rng, train=False) for r, st in chunk])
        probs.append(torch.softmax(model(torch.from_numpy(x[:, None]).to(device)), 1).cpu().numpy())
    p = pd.DataFrame(np.concatenate(probs), columns=LABELS)
    p["cry"] = [r.Index for r, _ in wins]
    cry = p.groupby("cry")[LABELS].mean()
    cry = cry.join(rows[["episode", "baby", "cause"]])
    ep = cry.groupby("episode").agg({**{c: "mean" for c in LABELS}, "baby": "first", "cause": "first"})
    ep["n_cries"] = cry.groupby("episode").size()
    ep["pred"] = ep[LABELS].idxmax(axis=1)
    return ep.reset_index()


def balanced_accuracy(ep):
    t = ep[TARGET]
    return float(np.mean([(ep[t == c].pred == c).mean() for c in LABELS if (t == c).any()]))


# ---------- training ----------

def run_fold(data, split, fold, seed, args, device, out_dir):
    rng = np.random.default_rng(1000 * seed + fold)
    torch.manual_seed(1000 * seed + fold)
    train, val, test = (data.rows(split, fold, p) for p in ("train", "val", "test"))
    if args.smoke:  # tiny run to check the code end to end
        train, val, test = (r[r.episode.isin(r.episode.drop_duplicates().sample(12, random_state=0))] for r in (train, val, test))
    data.fit_norm(train)

    n_ep = train.drop_duplicates("episode")[TARGET].value_counts()
    # class weight = total / (classes x episodes of that class): rarer classes count more
    weights = torch.tensor([n_ep.sum() / (len(LABELS) * n_ep.get(c, 1)) for c in LABELS], dtype=torch.float32)

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
            x, y = data.batch(rows.iloc[i:i + BATCH], rng)
            loss = loss_fn(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        val_bacc = balanced_accuracy(predict_episodes(model, data, val, device, rng))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_bacc": val_bacc})
        print(f"  seed {seed} fold {fold} epoch {epoch + 1:2d}  loss {np.mean(losses):.3f}  val {val_bacc:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if val_bacc > best:
            best, best_state = val_bacc, {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ep = predict_episodes(model, data, test, device, rng)
    ep["seed"], ep["fold"] = seed, fold
    ep.to_csv(out_dir / f"fold{fold}_seed{seed}.csv", index=False)
    summary = {"split": split, "task": args.task, "fold": fold, "seed": seed, "per_baby_norm": args.per_baby_norm,
               "test_bacc": balanced_accuracy(ep), "val_bacc_best": best,
               "test_episodes": len(ep), "test_babies": sorted(ep.baby.unique()), "history": history}
    (out_dir / f"fold{fold}_seed{seed}.json").write_text(json.dumps(summary, indent=1))
    print(f"seed {seed} fold {fold}: test balanced accuracy {summary['test_bacc']:.3f} on {len(ep)} episodes", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["A", "B"], default="A")
    ap.add_argument("--task", choices=["cause", "identity"], default="cause",
                    help="identity = model 8, the positive control: which of the 24 babies is this?")
    ap.add_argument("--fold", default="all", help="fold number or 'all'")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--per-baby-norm", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="12 episodes per part, to test the code")
    ap.add_argument("--seeds", type=int, default=3, help="train each fold this many times with different seeds")
    ap.add_argument("--name", default=None, help="run folder name")
    args = ap.parse_args()
    if args.task == "identity":
        # every baby must be on both sides, so only split A works; per-baby scaling would erase the answer
        if args.split != "A" or args.per_baby_norm:
            ap.error("--task identity needs split A and no --per-baby-norm")
        global TARGET, LABELS
        TARGET, LABELS = "baby", sorted(pd.read_csv(DATA / "index.csv").baby.unique())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(1)
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name()})" if device == "cuda" else ""))

    data = Cries(args.per_baby_norm)
    n_folds = len(data.splits["A_by_episode" if args.split == "A" else "B_by_baby"])
    folds = range(n_folds) if args.fold == "all" else [int(args.fold)]
    name = args.name or ("identity_" if args.task == "identity" else "") + f"split{args.split}" + ("_perbaby" if args.per_baby_norm else "") + ("_smoke" if args.smoke else "")
    out_dir = ROOT / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seeds):
        for f in folds:
            run_fold(data, args.split, f, seed, args, device, out_dir)


if __name__ == "__main__":
    main()
