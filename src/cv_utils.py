"""Spatial block cross-validation helpers shared by the supervised checks."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.embedding_utils import auc

SEED = 42
MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
    "rf": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
}


def blocks(x, y_, size_m):
    """Block ids of a size_m grid over projected coordinates x, y_ (pandas Series)."""
    return (
        np.floor(x / size_m).astype(int).astype(str) + "_" + np.floor(y_ / size_m).astype(int).astype(str)
    ).to_numpy()


def oof(df, cols, model, groups, labels):
    """Out-of-fold predicted probability with 5-fold GroupKFold."""
    pred = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=5).split(df, labels, groups):
        m = MODELS[model]().fit(df.iloc[tr][cols], labels[tr])
        pred[te] = m.predict_proba(df.iloc[te][cols])[:, 1]
    return pred


def pooled_auc(p, labels):
    return auc(p[labels], p[~labels])


def block_boot(preds, groups, y, fn, n=1000):
    """95% CI of fn(preds, labels) over bootstrap resamples of whole blocks."""
    rng = np.random.default_rng(SEED)
    ub = pd.unique(groups)
    idx_by = {b: np.flatnonzero(groups == b) for b in ub}
    out = []
    for _ in range(n):
        idx = np.concatenate([idx_by[b] for b in rng.choice(ub, len(ub))])
        if y[idx].all() or not y[idx].any():
            continue
        out.append(fn({k: v[idx] for k, v in preds.items()}, y[idx]))
    return np.percentile(out, [2.5, 97.5])
