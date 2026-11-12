"""Turn every labelled cry into a log-mel spectrogram and write the splits.

Output (data/processed/):
  specs.npy    all spectrogram frames stacked, shape (total_frames, 64), float16
  index.csv    one row per cry: where its frames sit in specs.npy, plus baby, episode, cause
  splits.json  split A (by episode, as in the paper), split B (by baby),
               split C (by time: each baby's earlier episodes train, later ones test)

Cries are stored at full length. Cropping to 1.5 s happens at training time,
so long cries can be cropped at a random place each epoch.

Decibels are measured against the loudest point of the whole crying episode.
That keeps how loud each cry is within its episode, and removes the distance
to the microphone, which changes between episodes.

Run:  python src/prep.py            (all 39,201 cries)
      python src/prep.py --limit 200 (quick check)
      python src/prep.py --splits-only (rewrite splits.json from index.csv)
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/enes"
OUT = ROOT / "data/processed"

CLASSES = ["hunger", "discomfort", "loneliness"]
SR = 16_000        # resample 22,050 Hz to 16 kHz
N_FFT = 512        # 32 ms Hann window
HOP = 160          # 10 ms hop
N_MELS = 64
FMIN, FMAX = 100, 8_000
TOP_DB = 80        # floor, relative to the episode's loudest point

SEED = 0
N_FOLDS = 6        # split B: 24 babies -> 6 groups of 4
N_REPEATS_A = 5    # split A: repeats of the paper's 66/34 episode split
TRAIN_SHARE_A = 0.66
VAL_SHARE = 0.15   # share of training episodes held out for early stopping
TRAIN_SHARE_C = 2 / 3  # split C: each baby's first two thirds of episodes, in time order


def log_mel(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    m = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
        window="hann", center=True, power=2.0,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    db = librosa.power_to_db(m, ref=1.0, amin=1e-10, top_db=None)  # absolute; made relative per episode in main()
    return db.T.astype(np.float32)  # (frames, 64)


def load_labels():
    df = pd.read_csv(RAW / "meta/dataset_44605_short.csv")
    df = df[df.cause_stop_engl.isin(CLASSES)]
    df = df.rename(columns={"file_seq_S": "episode", "cause_stop_engl": "cause"})
    return df[["file", "episode", "baby", "sex", "age_month", "cause"]].reset_index(drop=True)


def split_by_baby(eps, rng):
    """Split B. Babies are dealt into 6 groups of similar size.
    Fold i: test = group i, validation = group i+1, train = the other 16 babies."""
    per_baby = eps.groupby("baby").size().sort_values(ascending=False)
    babies = list(per_baby.index)
    groups = [[] for _ in range(N_FOLDS)]
    # deal in snake order (0..5, 5..0, ...) so each group gets a mix of big and small babies
    for i in range(0, len(babies), N_FOLDS):
        chunk = babies[i:i + N_FOLDS]
        rng.shuffle(chunk)
        order = range(N_FOLDS) if (i // N_FOLDS) % 2 == 0 else reversed(range(N_FOLDS))
        for g, b in zip(order, chunk):
            groups[g].append(b)
    folds = []
    for i in range(N_FOLDS):
        test, val = groups[i], groups[(i + 1) % N_FOLDS]
        train = [b for b in babies if b not in test and b not in val]
        folds.append({"train": train, "val": val, "test": test})
    return folds


def split_by_episode(eps, rng):
    """Split A, as in the paper: 66% of episodes per cause for training, the rest for testing.
    The same baby appears on both sides. 15% of training episodes go to validation."""
    repeats = []
    for _ in range(N_REPEATS_A):
        train, test = [], []
        for _, g in eps.groupby("cause"):
            ids = list(g.episode)
            rng.shuffle(ids)
            k = int(TRAIN_SHARE_A * len(ids))
            train += ids[:k]
            test += ids[k:]
        rng.shuffle(train)
        k = int(VAL_SHARE * len(train))
        repeats.append({"train": sorted(train[k:]), "val": sorted(train[:k]), "test": sorted(test)})
    return repeats


def episode_time(episode):
    """Recording date and time from the episode name, e.g. BM29_M_2_F_F_10102019_2058.wav."""
    p = episode.replace(".wav", "").split("_")
    return pd.to_datetime(p[5] + p[6], format="%d%m%Y%H%M")


def split_by_time(eps):
    """Split C, the personal translator. Each baby's episodes in time order:
    the first two thirds train (the last 15% of those validate), the rest test.
    Every baby is on both sides, and the model always predicts the future."""
    train, val, test = [], [], []
    eps = eps.assign(t=eps.episode.map(episode_time)).sort_values(["baby", "t"])
    for _, g in eps.groupby("baby"):
        ids = list(g.episode)
        k = int(np.floor(TRAIN_SHARE_C * len(ids)))
        v = int(np.ceil(VAL_SHARE * k))
        train += ids[:k - v]
        val += ids[k - v:k]
        test += ids[k:]
    return [{"train": sorted(train), "val": sorted(val), "test": sorted(test)}]


def write_splits(df):
    eps = df.drop_duplicates("episode")[["episode", "baby", "cause"]]
    rng = np.random.default_rng(SEED)
    splits = {"A_by_episode": split_by_episode(eps, rng), "B_by_baby": split_by_baby(eps, rng),
              "C_by_time": split_by_time(eps)}
    (OUT / "splits.json").write_text(json.dumps(splits, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only process the first N cries")
    ap.add_argument("--splits-only", action="store_true", help="rewrite splits.json from index.csv")
    args = ap.parse_args()
    if args.splits_only:
        write_splits(pd.read_csv(OUT / "index.csv"))
        print("wrote", OUT / "splits.json")
        return

    df = load_labels()
    if args.limit:
        df = df.iloc[:args.limit]
    OUT.mkdir(parents=True, exist_ok=True)

    specs, offsets, lengths = [], [], []
    pos = 0
    audio = RAW / "audio/00_pooled_separate"
    for i, f in enumerate(df.file):
        s = log_mel(audio / f)
        specs.append(s)
        offsets.append(pos)
        lengths.append(len(s))
        pos += len(s)
        if (i + 1) % 2000 == 0:
            print(f"{i + 1}/{len(df)} cries", flush=True)

    # loudest point of each episode = 0 dB; floor at -80 dB
    df["offset"], df["n_frames"] = offsets, lengths
    peak = {e: max(specs[i].max() for i in g.index) for e, g in df.groupby("episode")}
    for i, e in enumerate(df.episode):
        specs[i] = np.maximum(specs[i] - peak[e], -TOP_DB).astype(np.float16)
    np.save(OUT / "specs.npy", np.concatenate(specs))
    df.to_csv(OUT / "index.csv", index=False)

    write_splits(df)

    print(f"done: {len(df)} cries, {df.episode.nunique()} episodes, {df.baby.nunique()} babies, "
          f"{pos} frames, median {int(np.median(lengths))} frames per cry")


if __name__ == "__main__":
    main()
