"""Score finished runs against the baselines, all on the same splits.

Run:  python src/evaluate.py runs/splitB runs/splitB_perbaby
      python src/evaluate.py runs/splitA

Everything is scored per episode with balanced accuracy (chance = 33%).
Test predictions from all folds are pooled first; seeds are averaged.
95% intervals and paired differences come from resampling babies (2,000 times),
because babies are the independent units.
Also reports accuracy within each age group and for day vs night recordings,
to check that the model is not just hearing age or time of day.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)  # empty classes in a resample give NaN, skipped

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
CLASSES = ["hunger", "discomfort", "loneliness"]
N_BOOT = 2000
PASS_MARK = 1 / 3  # the CNN beats chance only if its whole 95% interval is above this


def episode_info():
    idx = pd.read_csv(DATA / "index.csv")
    ep = idx.drop_duplicates("episode")[["episode", "baby", "cause", "age_month"]].copy()
    hour = ep.episode.str.split("_").str[6].str[:2].astype(int)
    ep["time"] = np.where((hour >= 22) | (hour < 7), "night", "day")
    return ep.set_index("episode")


# ---------- baselines: counting rules, no training ----------

def baseline(split, rule, info):
    """Majority: the most common cause in training.
    Age-only: the most common cause for that age in training.
    Baby-only: the most common cause for that baby in training (split A only)."""
    splits = json.loads((DATA / "splits.json").read_text())
    key = "A_by_episode" if split == "A" else "B_by_baby"
    rows = []
    for fold, s in enumerate(splits[key]):
        unit = "episode" if split == "A" else "baby"
        tr = info.reset_index()
        tr = tr[tr[unit].isin(s["train"] + s["val"])]
        te = info.reset_index()
        te = te[te[unit].isin(s["test"])]
        overall = tr.cause.mode()[0]
        if rule == "majority":
            pred = [overall] * len(te)
        else:
            col = {"age-only": "age_month", "baby-only": "baby"}[rule]
            by = tr.groupby(col).cause.agg(lambda c: c.mode()[0])
            pred = [by.get(v, overall) for v in te[col]]
        rows.append(pd.DataFrame({"episode": te.episode, "baby": te.baby, "cause": te.cause,
                                  "pred": pred, "seed": 0, "fold": fold}))
    return pd.concat(rows)


# ---------- scoring ----------

def counts(pred, babies):
    """Confusion counts per seed and baby: shape (seeds, babies, 3, 3)."""
    seeds = sorted(pred.seed.unique())
    c = np.zeros((len(seeds), len(babies), 3, 3))
    b_i = {b: i for i, b in enumerate(babies)}
    for si, s in enumerate(seeds):
        p = pred[pred.seed == s]
        np.add.at(c[si], (p.baby.map(b_i).values, p.cause.map(CLASSES.index).values, p.pred.map(CLASSES.index).values), 1)
    return c


def bacc(c):
    """Balanced accuracy from confusion counts (seeds, 3, 3), averaged over seeds."""
    tot = c.sum(-1)
    with np.errstate(invalid="ignore"):
        rec = np.diagonal(c, axis1=-2, axis2=-1) / tot
    return float(np.nanmean(np.nanmean(rec, -1)))


def boot(cs, rng):
    """Resample babies; return the balanced accuracy of each count array per resample."""
    n = cs[0].shape[1]
    out = np.empty((N_BOOT, len(cs)))
    for i in range(N_BOOT):
        pick = rng.integers(0, n, n)
        out[i] = [bacc(c[:, pick].sum(1)) for c in cs]
    return out


def ci(x):
    return np.nanpercentile(x, [2.5, 97.5])


def load_run(path):
    files = sorted(Path(path).glob("fold*_seed*.csv"))
    if not files:
        sys.exit(f"no results in {path}")
    split = json.loads(files[0].with_suffix(".json").read_text())["split"]
    return split, pd.concat(pd.read_csv(f) for f in files)


def main():
    runs = sys.argv[1:]
    if not runs:
        sys.exit(__doc__)
    info = episode_info()
    rng = np.random.default_rng(0)

    loaded = {Path(r).name: load_run(r) for r in runs}
    splits = {s for s, _ in loaded.values()}
    if len(splits) > 1:
        sys.exit("compare runs from one split at a time")
    split = splits.pop()

    models = {name: p for name, (_, p) in loaded.items()}
    for rule in ["majority", "age-only"] + (["baby-only"] if split == "A" else []):
        models[rule] = baseline(split, rule, info)

    babies = sorted(info.baby.unique())
    cs = {m: counts(p, babies) for m, p in models.items()}
    b = boot(list(cs.values()), rng)
    names = list(cs)

    print(f"\nSplit {split}: balanced accuracy per episode, chance 33%, 95% interval over babies\n")
    summary = {}
    for j, m in enumerate(names):
        c = cs[m].sum(1)
        lo, hi = ci(b[:, j])
        rec = (np.diagonal(c, axis1=-2, axis2=-1) / c.sum(-1)).mean(0)
        summary[m] = {"bacc": bacc(c), "ci": [lo, hi], "recall": dict(zip(CLASSES, rec.round(3).tolist()))}
        flag = "  above chance" if lo > PASS_MARK else ""
        print(f"  {m:22s} {bacc(c):.1%}  [{lo:.1%}, {hi:.1%}]   "
              + "  ".join(f"{k} {v:.0%}" for k, v in summary[m]["recall"].items()) + flag)

    trained = [m for m in names if m in loaded]
    print("\nPaired differences (model minus baseline), 95% interval over babies")
    for m in trained:
        for other in names:
            if other == m or (other in trained and names.index(other) < names.index(m)):
                continue
            d = b[:, names.index(m)] - b[:, names.index(other)]
            lo, hi = ci(d)
            diff = bacc(cs[m].sum(1)) - bacc(cs[other].sum(1))
            summary[m][f"minus_{other}"] = {"diff": diff, "ci": [lo, hi]}
            print(f"  {m} - {other}: {diff:+.1%}  [{lo:+.1%}, {hi:+.1%}]")

    print("\nWithin groups (balanced accuracy), to check age and time-of-day shortcuts")
    for col in ["age_month", "time"]:
        for m in trained:
            p = models[m].join(info[[col]], on="episode")
            row = {}
            for g, pg in p.groupby(col):
                row[str(g)] = bacc(counts(pg, babies).sum(1))
            summary[m][f"by_{col}"] = row
            print(f"  {m:22s} by {col:9s} " + "  ".join(f"{k}: {v:.0%}" for k, v in row.items()))

    out = Path(runs[0]) / "summary.json"
    out.write_text(json.dumps(summary, indent=1, default=float))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
