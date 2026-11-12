"""Metric-optimal selection of candidate instances.

For pixel-disjoint predictions S, the pooled PQ over (annotator, image) pairs is

    PQ(S) = sum_{s in S} R_s * u_s / (0.5 * sum_{s in S} R_s + 0.5 * G),

with R_s the number of annotators of the image of s, u_s the annotator-averaged
matched IoU (IoU * [IoU > 0.5]) and G the total number of GT instances. This is a
ratio of additive set functions, so (Dinkelbach) the optimum S* maximises
sum_s R_s * (u_s - tau) with tau = PQ(S*) / 2. Inside one image R is constant,
so each image reduces to a maximum-weight antichain in its component tree,
solved exactly by a bottom-up DP. The estimated u_s comes from the scorer.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def tree_dp(df: pd.DataFrame, gain: np.ndarray) -> np.ndarray:
    """Max-weight antichain per image. df needs columns stem, node, parent, level. Returns bool mask."""
    gain = np.asarray(gain, np.float64)
    sel = np.zeros(len(df), bool)
    pos_all = np.arange(len(df))
    for _, g in df.groupby("stem", sort=False):
        rows = pos_all[df.index.get_indexer(g.index)]
        node, parent, level = g["node"].to_numpy(), g["parent"].to_numpy(), g["level"].to_numpy()
        gn = gain[rows]
        pos = {n: i for i, n in enumerate(node)}
        n = len(node)
        best, csum = np.zeros(n), np.zeros(n)
        children = defaultdict(list)
        for i in np.argsort(-level, kind="stable"):
            best[i] = max(gn[i], csum[i], 0.0)
            p = parent[i]
            if p >= 0 and p in pos:
                csum[pos[p]] += best[i]
                children[pos[p]].append(i)
        stack = [i for i in range(n) if parent[i] < 0 or parent[i] not in pos]
        while stack:
            i = stack.pop()
            if gn[i] > 0 and gn[i] >= csum[i]:
                sel[rows[i]] = True
            else:
                stack.extend(children[i])
    return sel


def pq_of_selection(df: pd.DataFrame, sel: np.ndarray, G_total: int) -> dict:
    s = df.loc[sel]
    num = float(s["u_sum"].sum())
    den = 0.5 * float(s["R"].sum()) + 0.5 * G_total
    tp = int(s["hits"].sum())
    return dict(pq=num / den if den else 0.0, n_pred=int(sel.sum()), tp_pairs=tp,
                sq=num / tp if tp else 0.0)


def dinkelbach(df: pd.DataFrame, u_hat: np.ndarray, G_total: int, tau0: float = 0.2, iters: int = 20,
               tol: float = 1e-5) -> tuple[np.ndarray, float, list]:
    """Iterate tau <- PQ(S(tau))/2 using estimated u_hat for selection and true labels for PQ.

    Note: with estimated u_hat the fixed point is a heuristic; use select_tau_cv for an honest tau.
    """
    tau, hist = tau0, []
    for _ in range(iters):
        sel = tree_dp(df, np.asarray(u_hat) - tau)
        res = pq_of_selection(df, sel, G_total)
        hist.append((tau, res["pq"], res["n_pred"]))
        new = res["pq"] / 2
        if abs(new - tau) < tol:
            break
        tau = new
    return sel, tau, hist


def sweep_tau(df: pd.DataFrame, u_hat: np.ndarray, G_total: int, taus) -> pd.DataFrame:
    out = []
    for t in taus:
        sel = tree_dp(df, np.asarray(u_hat) - t)
        out.append(dict(tau=t, **pq_of_selection(df, sel, G_total)))
    return pd.DataFrame(out)


def single_level_baseline(df: pd.DataFrame, G_total: int, min_areas=(16, 50, 100, 200, 400)) -> pd.DataFrame:
    """Connected components at one fixed threshold, filtered by area (the usual post-processing)."""
    out = []
    for lvl, g in df.groupby("level"):
        for a in min_areas:
            sel = (df["level"].to_numpy() == lvl) & (df["area"].to_numpy() >= a)
            out.append(dict(level=lvl, thr=float(g["thr"].iloc[0]), min_area=a, **pq_of_selection(df, sel, G_total)))
    return pd.DataFrame(out).sort_values("pq", ascending=False)
