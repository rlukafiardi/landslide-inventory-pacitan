"""Do the logistic baselines change when the exact differences dA and dB are dropped (dA = postA - pre, dB = postB - pre)?

Reads reports/tables/collinear_logreg_raw.csv (src.modeling.evaluate_oof --out ...) and the paired bootstrap files, and writes
reports/tables/collinear_logreg.csv: per base config and repeat, object precision at 80% recall with and without dA/dB, the difference and
its 95% interval over the same block resamples, and recall at that point. Same tuned C, same folds; training blocks only.
Prepare with: python -m src.modeling.ablate <seed> simple_logreg_no_diff all_logreg_no_diff (seeds 0-4), then evaluate_oof for the pairs.
Usage: python -m src.modeling.collinear_check
"""

import numpy as np
import pandas as pd

from src.features.build import INTERIM

OOF = f"{INTERIM}/oof"
raw = pd.read_csv("reports/tables/collinear_logreg_raw.csv")
raw = raw[raw.recall_target == 0.8].set_index(["config", "repeat"])
rows = []
for base in ("simple_logreg", "all_logreg"):
    var = f"abl_{base}_no_diff"
    for seed in range(5):
        d = np.load(f"{OOF}/boot_{var}_r{seed}_0.8.npy") - np.load(f"{OOF}/boot_{base}_r{seed}_0.8.npy")
        d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        a, b = raw.loc[(base, seed)], raw.loc[(var, seed)]
        rows.append(
            {
                "base": base,
                "repeat": seed,
                "precision_with_dAdB": a.precision_at_target,
                "precision_without": b.precision_at_target,
                "diff": b.precision_at_target - a.precision_at_target,
                "ci_low": lo,
                "ci_high": hi,
                "interval_excludes_zero": bool(lo > 0 or hi < 0),
                "recall_with": a.recall_ge5,
                "recall_without": b.recall_ge5,
            }
        )
out = pd.DataFrame(rows)
out.round(4).to_csv("reports/tables/collinear_logreg.csv", index=False)
print(out.round(4).to_string(index=False))
print(out.groupby("base")[["precision_with_dAdB", "precision_without", "diff"]].mean().round(4).to_string())
print("repeats with interval excluding zero:", out.groupby("base").interval_excludes_zero.sum().to_dict())
