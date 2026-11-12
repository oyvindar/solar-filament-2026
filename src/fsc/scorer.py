"""Expected matched-IoU scorer for candidate nodes (LightGBM).

Target per node: u = (1/R) * sum_a IoU_a * [IoU_a > 0.5], the annotator-averaged
contribution of the node to the PQ numerator; rows are weighted by R so that
images count as often as they do in the pooled metric. Two estimators:
  * "xent":   one cross-entropy regressor on u in [0, 1];
  * "hurdle": P(at least one annotator matched) x E[u | matched].
Scores must be cross-fitted (train on other folds) before they are used to
choose tau, otherwise the selection overfits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURES = ["thr", "level", "area", "bbox_h", "bbox_w", "fill", "p_mean", "p_max", "p_q25", "p_q75", "p_mass",
                 "major", "minor", "elong", "thick", "orient", "r_norm", "p_ring", "ff_in", "ff_ring", "ff_contrast",
                 "n_child", "parent_area_ratio", "child_area_ratio", "level_count", "persist"]
EXTRA_FEATURES = ["log_area", "edge", "p_over_thr", "site", "n_roots",
                  "depth_left", "n_leaves", "maxchild_ratio", "parent_p_mean", "n_sib"]
FEATURES = BASE_FEATURES + EXTRA_FEATURES
SITES = ["Bh", "Ch", "Lh", "Mh", "Th", "Uh"]
PARAMS = dict(objective="cross_entropy", learning_rate=0.05, num_leaves=63, min_data_in_leaf=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=6)


def _tree_context(g: pd.DataFrame) -> pd.DataFrame:
    """Per image: subtree depth, number of leaves, largest child, parent probability, sibling count."""
    g = g.copy()
    pos = {n: i for i, n in enumerate(g["node"].to_numpy())}
    par, lvl, area = g["parent"].to_numpy(), g["level"].to_numpy(), g["area"].to_numpy()
    maxlvl = lvl.copy()
    leaves = np.zeros(len(g))
    maxchild = np.zeros(len(g))
    for i in np.argsort(-lvl, kind="stable"):  # children (higher level) first
        if leaves[i] == 0:
            leaves[i] = 1  # a node without processed children is a leaf
        p = par[i]
        if p >= 0 and p in pos:
            j = pos[p]
            maxlvl[j] = max(maxlvl[j], maxlvl[i])
            leaves[j] += leaves[i]
            maxchild[j] = max(maxchild[j], area[i])
    g["depth_left"] = maxlvl - lvl
    g["n_leaves"] = leaves
    g["maxchild_ratio"] = maxchild / area
    p_of = dict(zip(g["node"], g["p_mean"]))
    g["parent_p_mean"] = g["parent"].map(p_of).fillna(0.0)
    sib = g.groupby("parent")["node"].transform("size")
    g["n_sib"] = np.where(g["parent"] >= 0, sib, 0)
    return g


def add_features(nodes: pd.DataFrame) -> pd.DataFrame:
    df = nodes.copy()
    df["log_area"] = np.log(df["area"])
    df["edge"] = df["p_mean"] - df["p_ring"]
    df["p_over_thr"] = df["p_mean"] - df["thr"]
    df["site"] = pd.Categorical(df["stem"].str[14:], categories=SITES)
    df["n_roots"] = df.groupby("stem")["parent"].transform(lambda s: int((s < 0).sum()))
    df = pd.concat([_tree_context(g) for _, g in df.groupby("stem", sort=False)])
    if "u_sum" in df:
        df["u"] = df["u_sum"] / df["R"]
        df["hit"] = (df["hits"] > 0).astype(float)
    return df


def fit(train: pd.DataFrame, kind: str = "xent", rounds: int = 500, feats=FEATURES):
    import lightgbm as lgb
    if kind == "xent":
        return ("xent", lgb.train(PARAMS, lgb.Dataset(train[feats], train["u"], weight=train["R"]), rounds))
    mc = lgb.train(dict(PARAMS, objective="binary"), lgb.Dataset(train[feats], train["hit"], weight=train["R"]), rounds)
    h = train[train["hit"] > 0]
    mr = lgb.train(PARAMS, lgb.Dataset(h[feats], h["u"], weight=h["R"]), max(100, rounds // 2))
    return ("hurdle", (mc, mr))


def predict(model, df: pd.DataFrame, feats=FEATURES) -> np.ndarray:
    kind, m = model
    if kind == "xent":
        return m.predict(df[feats])
    mc, mr = m
    return mc.predict(df[feats]) * mr.predict(df[feats])


def cross_fit(df: pd.DataFrame, fold_of_stem: dict, kind: str = "xent", rounds: int = 500, feats=FEATURES) -> np.ndarray:
    """Out-of-fold u_hat: each fold scored by a model trained on the other folds."""
    folds = df["stem"].map(fold_of_stem).to_numpy()
    out = np.zeros(len(df))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        out[va] = predict(fit(df[tr], kind, rounds, feats), df[va], feats)
    return out
