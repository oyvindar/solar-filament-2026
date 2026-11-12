"""Unit tests for the metric, the selection and the submission checks.

Run: python -m pytest tests/   (or: python tests/test_pipeline.py)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from fsc.metric import H, W, check_submission, pq_fast, pq_official, rle_from_binary  # noqa: E402
from fsc.select import pq_of_selection, tree_dp  # noqa: E402


def _box(r0, r1, c0, c1):
    m = np.zeros((H, W), np.uint8)
    m[r0:r1, c0:c1] = 1
    return rle_from_binary(m)


def _frames():
    """Two annotators on image A, one on image B; predictions with a hit, a near miss and a false positive."""
    gt = pd.DataFrame([
        ("a1-A_1", _box(100, 140, 100, 200)), ("a1-A_2", _box(500, 520, 500, 600)),
        ("a2-A_1", _box(102, 142, 100, 210)),
        ("a1-B_1", _box(1000, 1050, 1000, 1100)),
    ], columns=["filament_id", "segmentation_rle"])
    pred = pd.DataFrame([
        ("A_1", _box(100, 140, 105, 205)),      # hit for both annotators
        ("A_2", _box(500, 520, 500, 540)),      # IoU 0.4: near miss -> FP and FN
        ("B_1", _box(1500, 1520, 1500, 1520)),  # false positive
    ], columns=["filament_id", "segmentation_rle"])
    return gt, pred


def test_fast_metric_matches_official_port():
    gt, pred = _frames()
    a, b = pq_official(gt, pred), pq_fast(gt, pred)
    assert abs(a - b.pq) < 1e-6  # the dense port works in float32, like the organisers' torch code
    assert (b.tp, b.fp, b.fn) == (2, 3, 2)


def test_pq_hand_computed():
    gt, pred = _frames()
    r = pq_fast(gt, pred)
    iou1 = (40 * 95) / (40 * 100 + 40 * 100 - 40 * 95)    # annotator a1 vs prediction A_1
    iou2 = (38 * 100) / (40 * 110 + 40 * 100 - 38 * 100)  # annotator a2 vs prediction A_1
    assert abs(r.pq - (iou1 + iou2) / (2 + 0.5 * 3 + 0.5 * 2)) < 1e-9


def test_tree_dp_picks_best_antichain():
    # root 0 with children 1 and 2; child 1 has child 3
    df = pd.DataFrame(dict(stem="A", node=[0, 1, 2, 3], parent=[-1, 0, 0, 1], level=[0, 1, 1, 2]))
    assert tree_dp(df, np.array([0.5, 0.2, 0.4, 0.3])).tolist() == [False, False, True, True]  # 0.4 + 0.3 > 0.5
    assert tree_dp(df, np.array([0.9, 0.2, 0.4, 0.3])).tolist() == [True, False, False, False]
    assert not tree_dp(df, np.array([-0.1, -0.2, -0.4, -0.3])).any()


def test_pq_of_selection_matches_definition():
    df = pd.DataFrame(dict(stem="A", node=[0, 1], parent=[-1, -1], level=[0, 0], R=[2, 2], u_sum=[1.4, 0.0], hits=[2, 0]))
    r = pq_of_selection(df, np.array([True, True]), G_total=3)
    assert abs(r["pq"] - 1.4 / (0.5 * 4 + 0.5 * 3)) < 1e-12


def test_check_submission_flags_overlap_and_empty():
    ok = pd.DataFrame([("A_1", _box(0, 10, 0, 10)), ("A_2", _box(20, 30, 0, 10))], columns=["filament_id", "segmentation_rle"])
    assert check_submission(ok)["images_with_overlap"] == 0
    bad = pd.DataFrame([("A_1", _box(0, 10, 0, 10)), ("A_2", _box(5, 15, 0, 10)),
                        ("B_1", rle_from_binary(np.zeros((H, W), np.uint8)))], columns=["filament_id", "segmentation_rle"])
    r = check_submission(bad)
    assert r["images_with_overlap"] == 1 and r["empty"] == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
