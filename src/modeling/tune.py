"""Phase 5: small hyperparameter search on the STRICTER scheme (6 km block folds, seed 0), scored by weighted average precision of
the pooled out-of-fold predictions on the training table. Logistic regression: grid over C. Boosting: 8 random configurations
per feature set. Writes data/interim/tuned_params.json and reports/tables/tuning.csv. The later re-score with 3 km blocks and 5
repeats shares seed 0 with this search, so its gain is slightly optimistic; only the test blocks give the honest number.
Usage: python -m src.modeling.tune
"""
import json
import time

import numpy as np
import pandas as pd

from src.embedding_utils import auc
from src.features.build import INTERIM
from src.modeling.folds import block_folds, fold_of
from src.modeling.metrics import pixel_ap
from src.modeling.prep import CONFIGS, SETS, default_params, design, make_model

t = pd.read_parquet(f"{INTERIM}/features_train.parquet")
y = t.y.to_numpy() == 1
w = t.w.to_numpy()
folds = fold_of(t.row.to_numpy(), t.col.to_numpy(), block_folds(0, size_px=600), 600)
assert (folds >= 0).all()
rng = np.random.default_rng(0)
GRID = {"max_depth": [3, 4, 6], "learning_rate": [0.03, 0.06, 0.1], "max_iter": [200, 300, 400],
        "min_samples_leaf": [50, 200, 500], "l2_regularization": [0.0, 1.0, 10.0]}


def cv_score(kind, fset, params):
    cols = SETS[fset]
    X = design(t, cols, kind)
    p = np.zeros(len(t))
    for k in range(5):
        tr, te = folds != k, folds == k
        p[te] = make_model(kind, params, cols).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return pixel_ap(y, p, w), auc(p[y], p[~y])


rows, best = [], {}
t0 = time.time()
for name, (kind, fset) in CONFIGS.items():
    cands = ([{"C": c} for c in (0.003, 0.01, 0.03, 0.1, 1.0)] if kind == "logreg"
             else [{k: int(rng.choice(v)) if k != "learning_rate" and k != "l2_regularization" else float(rng.choice(v)) for k, v in GRID.items()} for _ in range(8)])
    scored = []
    for params in cands:
        ap, a = cv_score(kind, fset, params)
        scored.append((ap, a, params))
        rows.append({"config": name, "kind": kind, "set": fset, "params": json.dumps(params), "weighted_AP": ap, "AUC": a})
        print(f"{name:<14} {json.dumps(params):<110} AP {ap:.3f} AUC {a:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    best[name] = max(scored, key=lambda s: s[0])[2]
json.dump(best, open(f"{INTERIM}/tuned_params.json", "w"), indent=1)
pd.DataFrame(rows).to_csv("reports/tables/tuning.csv", index=False)
print("tuned:", json.dumps(best))
