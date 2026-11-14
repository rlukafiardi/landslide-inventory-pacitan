"""Phase 5: ablation table. Each variant of c_hgb (one group or feature removed) against the full c_hgb, repeat <seed>, at 80% object recall.

Reads reports/tables/ablation_raw.csv (written by evaluate_oof --out) and the per-resample bootstrap files. The 95% interval is of the
difference in operating precision (variant minus full model) over the same block resamples. A drop counts only when the interval
excludes zero; with one repeat there is no repeats-won count.
Usage: python -m src.modeling.ablation_report <seed>
"""

import sys

import numpy as np
import pandas as pd

from src.features.build import INTERIM

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OOF = f"{INTERIM}/oof"
raw = pd.read_csv("reports/tables/ablation_raw.csv")
raw = raw[(raw.recall_target == 0.8) & (raw.repeat == seed)].set_index("config")
full = np.load(f"{OOF}/boot_c_hgb_r{seed}_0.8.npy")
rows = []
for cfg in raw.index:
    r = raw.loc[cfg]
    b = np.load(f"{OOF}/boot_{cfg}_r{seed}_0.8.npy")
    d = b - full
    d = d[~np.isnan(d)]
    lo, hi = np.percentile(d, [2.5, 97.5]) if len(d) else (np.nan, np.nan)
    rows.append(
        {
            "variant": cfg,
            "precision_at_80_recall": r.precision_at_target,
            "diff_vs_full": r.precision_at_target - raw.loc["c_hgb", "precision_at_target"],
            "ci_low": lo,
            "ci_high": hi,
            "flagged_share": r.flagged_share,
            "weighted_pixel_AP": r.weighted_pixel_AP,
            "interval_excludes_zero": bool(lo > 0 or hi < 0),
        }
    )
out = pd.DataFrame(rows).round(3)
out.to_csv("reports/tables/ablation.csv", index=False)
print(out.to_string(index=False))
