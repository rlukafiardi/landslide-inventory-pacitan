"""Phase 4: how much do random pixel folds flatter a model compared with block folds? One regularised logistic regression,
the same table, three ways of grouping pixels into folds. Weighted average precision (negatives carry their sampling weight)
approximates precision-recall at true prevalence. Usage: python -m src.modeling.leak_check
"""

import json

import numpy as np
import pandas as pd

from src.cv_utils import SEED, oof
from src.embedding_utils import auc
from src.features.build import INTERIM, apply_fills
from src.features.registry import GROUPS, NAMES
from src.modeling.metrics import pixel_ap

t = pd.read_parquet(f"{INTERIM}/features_train.parquet")
fills = json.load(open(f"{INTERIM}/fill_values.json"))
rng = np.random.default_rng(SEED)
sub = pd.concat([t[t.y == 1], t[t.y == 0].sample(100_000, random_state=SEED)]).reset_index(drop=True)
sub["w"] = np.where(sub.y == 1, 1.0, sub.w * 4)  # 4x fewer negatives than the table, so 4x the weight
sub = apply_fills(sub, fills)
y = sub.y.to_numpy() == 1
cat = ["geologi", "landuse"]
sets = {
    "all 95 numeric features": [n for n in NAMES if n not in cat],
    "dNDVI + terrain (no embedding)": [n for n in GROUPS["ndvi_pixel"] + GROUPS["ndvi_nbr"] + GROUPS["terrain"]],
}
groupings = {
    "random pixels": rng.permutation(len(sub)),
    "3 km blocks": sub.block.to_numpy(),
    "6 km blocks": (sub.row // 600).to_numpy() * 1000 + (sub.col // 600).to_numpy(),
}
rows = []
for sname, cols in sets.items():
    for gname, g in groupings.items():
        p = oof(sub, cols, "logreg", g, y)
        rows.append((sname, gname, auc(p[y], p[~y]), pixel_ap(y, p, sub.w.to_numpy())))
out = pd.DataFrame(rows, columns=["features", "folds", "AUC", "weighted_AP"])
out.to_csv("reports/tables/leak_check.csv", index=False)
print(out.round(3).to_string(index=False))
for sname in sets:
    s = out[out.features == sname].set_index("folds")
    print(
        f"{sname}: random folds flatter weighted AP by {s.loc['random pixels', 'weighted_AP'] - s.loc['3 km blocks', 'weighted_AP']:+.3f} "
        f"(AUC {s.loc['random pixels', 'AUC'] - s.loc['3 km blocks', 'AUC']:+.3f}) against 3 km blocks"
    )
