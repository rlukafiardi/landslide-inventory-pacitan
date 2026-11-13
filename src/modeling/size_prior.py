"""Phase 5 extra: does a size-dependent rule raise object precision at 80% object recall? (idea "object-size prior", 2026-11-13)

Rule: an object of the c_hgb out-of-fold raster is kept if size >= a_min and (size >= S or its peak probability >= P), so a small object needs a
confident peak. S = 0 is no rule (the stage-1 baseline, same code path). Per (flagged-share threshold, S, P, a_min): per-block counts of
polygons >= 5 px detected, objects and true objects, as in evaluate_oof. Dropping objects only removes pixels, it cannot change the geometry of the
kept ones, so the geometry guard of docs/validation-design.md is met by construction.
Reading: per rule and repeat, the best precision over (threshold, a_min) with recall >= 80%; the rule with the best mean precision over the repeats is then
fixed (so only (threshold, a_min) are searched, as for the baseline) and compared with the baseline per repeat and with the block-resample bootstrap of
evaluate_oof (same resamples, so paired). Choosing the rule from 12 candidates on the same data is optimistic; the interval does not cover that selection.
Usage: python -m src.modeling.size_prior <seed> [seed ...]   (counts)   |   python -m src.modeling.size_prior --analyse
"""
import sys

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from src.modeling import evaluate_oof as ev
from src.modeling.metrics import EIGHT

RULES = [(0, 0.0)] + [(s, p) for s in (6, 10, 15) for p in (0.3, 0.5, 0.7, 0.85)]
OUT = ev.OOF / "size_prior"


def counts(prob):
    """cnt[t, rule, a, block, k]: k = 0 polygons >= 5 px detected, 1 objects, 2 true objects."""
    vals = prob.ravel()[ev.train_flat]
    thr = np.quantile(vals[~np.isnan(vals)], 1 - np.array(ev.SHARES))
    out = np.zeros((len(thr), len(RULES), len(ev.A_MINS), ev.NB, 3), dtype=np.int32)
    for ti, th in enumerate(thr):
        pm = (prob >= th) & ev.train
        lab, n = ndi.label(pm, structure=EIGHT)
        if n == 0:
            continue
        idx = np.flatnonzero(pm.ravel())
        labs = lab.ravel()[idx]
        sizes = np.bincount(labs, minlength=n + 1)
        pmax = ndi.maximum(prob, lab, index=np.arange(n + 1))
        uniq, first = np.unique(labs, return_index=True)
        obj_block = ev.bi_of_flat[idx[first]]
        tp_lab = np.zeros(n + 1, bool)
        tp_lab[np.unique(labs[(ev.poly_flat[idx] > 0) | ev.ring_flat[idx]])] = True
        pid = ev.poly_flat[idx]
        for ri, (s, p) in enumerate(RULES):
            ok = (sizes >= s) | (pmax >= np.float64(p))
            for ai, a in enumerate(ev.A_MINS):
                keep_lab = (sizes >= a) & ok
                keep_lab[0] = False
                ko = keep_lab[uniq]
                out[ti, ri, ai, :, 1] = np.bincount(obj_block[ko], minlength=ev.NB)
                out[ti, ri, ai, :, 2] = np.bincount(obj_block[ko & tp_lab[uniq]], minlength=ev.NB)
                hit = np.unique(pid[keep_lab[labs] & (pid > 0)])
                rows = ev.poly_to_row[hit]
                det = np.zeros(len(ev.pt), bool)
                det[rows[rows >= 0]] = True
                out[ti, ri, ai, :, 0] = np.bincount(ev.pt.bi[det & ev.big], minlength=ev.NB)
    return thr, out


def best(c, tot, target=0.8):
    """c (..., T, A, 3) summed over blocks, tot = polygons >= 5 px -> best precision with recall >= target (NaN if never reached)."""
    prec = np.where(c[..., 1] > 0, c[..., 2] / np.maximum(c[..., 1], 1), -1.0)
    prec = np.where(c[..., 0] / tot >= target, prec, -1.0)
    m = prec.reshape(*prec.shape[:-2], -1).max(-1)
    return np.where(m >= 0, m, np.nan)


def analyse():
    seeds = sorted(int(p.stem.split("_r")[1]) for p in OUT.glob("c_hgb_r*.npz"))
    W = np.random.default_rng(ev.SEED).multinomial(ev.NB, np.full(ev.NB, 1 / ev.NB), size=ev.N_BOOT).astype("float64")
    tot_all = ev.n_poly_big.sum()
    tabs = {s: np.load(OUT / f"c_hgb_r{s}.npz")["cnt"] for s in seeds}  # (T, R, A, B, 3)
    # grid: any rule, per repeat; precision per rule (rules on the second axis), best over (T, A)
    rows, win = [], []
    for s in seeds:
        c = tabs[s].sum(3)  # (T, R, A, 3)
        per_rule = np.array([best(c[:, r], tot_all) for r in range(len(RULES))])
        win.append(int(np.nanargmax(per_rule)))
        rows.append(per_rule)
    per_rule = np.array(rows)  # (seeds, rules)
    print("precision at 80% recall per repeat (OOF), rows = rule (S, P):")
    for r, (sz, p) in enumerate(RULES):
        print(f"  S={sz:2d} P={p:.2f}: " + " ".join(f"{x:.3f}" for x in per_rule[:, r]) + f"   mean {np.nanmean(per_rule[:, r]):.3f}")
    fixed = max(range(1, len(RULES)), key=lambda r: np.nanmean(per_rule[:, r]))
    print(f"\nbest fixed rule by mean over repeats: S={RULES[fixed][0]}, P={RULES[fixed][1]} (wins per repeat: {[RULES[w] for w in win]})")
    d_rep = per_rule[:, fixed] - per_rule[:, 0]
    wins = int((d_rep > 0).sum())
    tot_b = (W @ ev.n_poly_big.astype(float))[:, None, None, None]  # polygons >= 5 px in each resample
    boot = []
    for s_ in seeds:
        cb = np.einsum("bn,ntrak->brtak", W, tabs[s_].transpose(3, 0, 1, 2, 4)[:, :, [0, fixed]].astype("float64"))  # (resample, rule, T, A, 3)
        boot.append(best(cb, tot_b))  # (resample, 2): baseline, fixed rule
    boot = np.array(boot)  # (seeds, resamples, 2)
    diff = (boot[:, :, 1] - boot[:, :, 0]).mean(0)
    lo, hi = np.nanpercentile(diff, [2.5, 97.5])
    print(f"\nfixed rule minus baseline: mean over repeats {d_rep.mean():+.4f} (per repeat {np.round(d_rep, 4).tolist()}), repeats won {wins}/{len(seeds)}, "
          f"bootstrap 95% interval of the 5-repeat mean difference [{lo:+.4f}, {hi:+.4f}]")
    print("rule of docs/validation-design.md: interval excludes 0 and wins >= 4 of 5 repeats ->", "PASS" if (lo > 0 and wins >= 4) else "NOT met")
    pd.DataFrame({"rule_S": [r[0] for r in RULES], "rule_P": [r[1] for r in RULES], **{f"repeat{s}": per_rule[i] for i, s in enumerate(seeds)}}).to_csv("reports/tables/size_prior_results.csv", index=False)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    if sys.argv[1:] == ["--analyse"]:
        analyse()
    else:
        for seed in sys.argv[1:]:
            thr, cnt = counts(np.asarray(np.load(ev.OOF / f"c_hgb_r{seed}.npy", mmap_mode="r")))
            np.savez_compressed(OUT / f"c_hgb_r{seed}.npz", thr=thr, cnt=cnt)
            base = cnt[:, 0].sum(2)  # (T, A, 3)
            print(f"seed {seed}: baseline precision at 80% recall {best(base, ev.n_poly_big.sum()):.4f}", flush=True)
