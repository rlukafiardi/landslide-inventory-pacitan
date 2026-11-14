"""Phase 5: results table and paired comparisons from the evaluated out-of-fold rasters (docs/validation-design.md).

modeling_results.csv  per config at each recall target: precision at the operating point (mean over repeats, 95% interval over block
                      resamples of the repeat-averaged precision), flagged area share, recall by size class, weighted pixel AP.
modeling_pairs.csv    each config against the baselines at recall 0.8: mean difference, 95% paired interval, repeats won (of the repeats run).
A difference counts only if the interval excludes zero and the config wins at least 4 of 5 repeats (the rule fixed before the models).
Usage: python -m src.modeling.summarize
"""

import numpy as np
import pandas as pd

from src.features.build import INTERIM
from src.modeling.prep import CONFIGS

OOF = f"{INTERIM}/oof"
raw = pd.read_csv("reports/tables/modeling_results_raw.csv")
BASELINES = {
    "naive_dB": "naive -dB",
    "naive_dBmin5": "naive -dB_min5",
    "floor_logreg": "site factors only",
    "simple_logreg": "best simple (dNDVI + factors, logistic)",
}
configs = list(dict.fromkeys(raw.config))


def boots(cfg, tg):
    reps = sorted(raw[(raw.config == cfg)].repeat.unique())
    names = [cfg if r == -1 else f"{cfg}_r{r}" for r in reps]
    return np.stack([np.load(f"{OOF}/boot_{n}_{tg}.npy") for n in names])  # (repeats, B)


rows = []
for cfg in configs:
    for tg in (0.7, 0.8, 0.9):
        d = raw[(raw.config == cfg) & (raw.recall_target == tg)]
        b = boots(cfg, tg)
        mb = np.nanmean(b, axis=0) if not np.isnan(b).all() else np.full(b.shape[1], np.nan)
        row = {
            "config": cfg,
            "features": CONFIGS.get(cfg, ("rule", "-dB" if cfg == "naive_dB" else "-dB_min5"))[1],
            "split": "block CV 3 km, 5 repeats",
            "recall_target": tg,
            "repeats": len(d),
            "reached": int(d.reached.sum()),
            "precision": d.precision_at_target.mean(),
            "precision_sd_over_repeats": d.precision_at_target.std(),
            "ci_low": np.nanpercentile(mb, 2.5),
            "ci_high": np.nanpercentile(mb, 97.5),
            "flagged_share": d.flagged_share.mean(),
            "recall_ge5": d.recall_ge5.mean(),
            "recall_<5 px": d["recall_<5 px"].mean(),
            "recall_5-20 px": d["recall_5-20 px"].mean(),
            "recall_>20 px": d["recall_>20 px"].mean(),
            "weighted_pixel_AP": d.weighted_pixel_AP.mean(),
            "date": d.date.iloc[0],
        }
        rows.append(row)
res = pd.DataFrame(rows)
res.to_csv("reports/tables/modeling_results.csv", index=False)

pairs = []
tg = 0.8


def per_repeat(cfg):
    """precision at the operating point by repeat; the naive rules have one raster, which serves every repeat."""
    s = raw[(raw.config == cfg) & (raw.recall_target == tg)].set_index("repeat").precision_at_target
    return (lambda r: s.loc[-1]) if -1 in s.index else (lambda r: s.loc[r])


for cfg in configs:
    for base in BASELINES:
        if cfg == base:
            continue
        diff = np.nanmean(boots(cfg, tg), axis=0) - np.nanmean(boots(base, tg), axis=0)
        diff = diff[~np.isnan(diff)]
        get_c, get_b = per_repeat(cfg), per_repeat(base)
        reps = sorted(r for r in raw.repeat.unique() if r >= 0)
        pc = np.array([get_c(r) for r in reps], dtype=float)
        pb = np.array([get_b(r) for r in reps], dtype=float)
        won = int(np.sum(~np.isnan(pc) & (np.nan_to_num(pc, nan=-1) > np.nan_to_num(pb, nan=-1))))
        lo, hi = np.percentile(diff, [2.5, 97.5]) if len(diff) else (np.nan, np.nan)
        pairs.append(
            {
                "config": cfg,
                "baseline": base,
                "mean_diff_precision": np.nanmean(pc) - np.nanmean(pb),
                "ci_low": lo,
                "ci_high": hi,
                "repeats_won": f"{won} of {len(reps)}",
                "counts": bool(lo > 0 and won >= 4),
            }
        )
pairs = pd.DataFrame(pairs)
pairs.to_csv("reports/tables/modeling_pairs.csv", index=False)
show = res[res.recall_target == 0.8][
    [
        "config",
        "features",
        "reached",
        "precision",
        "ci_low",
        "ci_high",
        "flagged_share",
        "recall_ge5",
        "recall_<5 px",
        "recall_5-20 px",
        "recall_>20 px",
        "weighted_pixel_AP",
    ]
]
print("Precision at 80% object recall (polygons >= 5 px), block CV, mean over repeats, 95% block-bootstrap interval:")
print(show.round(3).to_string(index=False))
print(
    "\nPaired differences in precision at 80% recall (a difference counts only if the interval excludes 0 and the config wins >= 4 of 5 repeats):"
)
print(pairs[pairs.baseline.isin(["naive_dBmin5", "simple_logreg"])].round(3).to_string(index=False))
