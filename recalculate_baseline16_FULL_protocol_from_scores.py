#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recalculate Baseline-16 metrics from saved scores_npz using the CURRENT MSDS-Net FULL
final evaluation protocol — WITHOUT retraining any baseline model.

This script is intentionally standalone: it does NOT import frozen_msds_v4.py and it does
NOT use the old baseline evaluator. It reads the already-saved per-method/per-seed scores,
then applies the same final evaluation chain used by the current FULL experiment:

    1) Threshold selection from clean/reference score:
       candidate thresholds = Q[0.90 ... 0.999] (300 points)
       select the candidate with FAR <= 0.05 that maximizes validation F1;
       fallback = configured quantile Q0.99.

    2) Evaluation score post-processing:
       scipy.ndimage.median_filter(score, size=3)

    3) Point metrics / Adj metrics:
       same definitions as current FULL.

    4) Event metrics:
       point alarms -> temporal_extend(window=30)
       -> contiguous episodes inside eval interval
       -> merge episodes with gap <= 120 min
       -> one-to-one event matching.

Expected baseline output directory layout:

    baseline_output_dir/
      ├─ scores_npz/
      │   ├─ PCA_seed_42.npz
      │   ├─ ...
      ├─ frozen_10_events.csv
      ├─ baseline16_protocol_audit.json
      └─ baseline16_all_seed_results.csv   # optional; metadata only

Each NPZ must contain:
    score       : evaluation score over retained_n points
    train_score : clean/reference score over retained_n points
    threshold   : old threshold (ignored for recalculation; kept only for audit if desired)

Usage:
    Easiest: put this .py in the SAME directory as scores_npz and run it.

    It also auto-detects one child directory containing scores_npz, so it can be placed
    one level above the baseline output folder.

Outputs are written to:
    baseline_output_dir/FULLprotocol_recalculated/

No model training is performed.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score


# =============================================================================
# User-editable settings
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Leave as None for automatic detection.
# If needed, replace None with an explicit path, e.g.:
# BASELINE_OUTPUT_DIR = Path(r"C:\Users\MSI-NB\PycharmProjects\A1\930eswaR1\03baseline\baseline16_v20_deepfaithful_outputs")
BASELINE_OUTPUT_DIR: Optional[Path] = None

OUTPUT_SUBDIR = "FULLprotocol_recalculated"
THRESHOLD_QUANTILE = 0.99
FAR_TARGET = 0.05
MEDIAN_FILTER_SIZE = 3
EVENT_EXTEND_WINDOW = 30
EVENT_MERGE_GAP_MIN = 120
EXPECTED_SEEDS = [42, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032]

PAPER_METRICS = [
    ("PointF1", "Point-F1"),
    ("EventF1", "Event-F1"),
    ("AdjF1", "Adj-F1"),
    ("ROC_AUC", "ROC-AUC"),
    ("PR_AUC_AP", "PR-AUC"),
    ("FullFAR", "Full-test FAR"),
    ("EventRecall", "EventRecall"),
    ("Delay", "Delay(min)"),
]

FULL_METRIC_COLUMNS = [
    "PointPrecision", "PointRecall", "PointF1",
    "ROC-AUC", "PR_AUC", "FAR", "MDR", "Detected",
    "ROC_AUC", "PR_AUC_AP", "FullFAR",
    "AdjPrecision", "AdjRecall", "AdjF1",
    "EventPrecision", "EventRecall", "EventF1",
    "PredEventCount", "FalsePredEventCount", "MatchedEventCount",
    "DetectedEvents", "NumEvents",
    "DelayMeanMin", "DelayMedianMin", "IndependentMaxDelayMin",
    "AlarmEpisodes", "MatchedAlarmEpisodes", "Delay", "DelayDetectedOnly",
    "Merge120EventPrecision", "Merge120EventRecall", "Merge120EventF1",
    "Merge120PredEventCount", "Merge120FalsePredEventCount",
    "Merge120MatchedEventCount", "Merge120DelayMeanMin",
    "Threshold", "EvalPoints",
]


# =============================================================================
# Directory / protocol loading
# =============================================================================

def resolve_baseline_output_dir() -> Path:
    if BASELINE_OUTPUT_DIR is not None:
        p = Path(BASELINE_OUTPUT_DIR).expanduser().resolve()
        if not (p / "scores_npz").is_dir():
            raise FileNotFoundError(f"scores_npz not found under BASELINE_OUTPUT_DIR: {p}")
        return p

    # Case 1: script is placed directly in baseline output directory.
    if (SCRIPT_DIR / "scores_npz").is_dir():
        return SCRIPT_DIR

    # Case 2: script is one level above the output directory.
    children = [p for p in SCRIPT_DIR.iterdir() if p.is_dir() and (p / "scores_npz").is_dir()]
    if len(children) == 1:
        return children[0].resolve()
    if len(children) > 1:
        names = "\n  - ".join(str(p) for p in children)
        raise RuntimeError(
            "Multiple child directories contain scores_npz. Please set BASELINE_OUTPUT_DIR explicitly:\n  - " + names
        )

    raise FileNotFoundError(
        "Could not find scores_npz. Put this script in the baseline output directory, "
        "or one level above it, or set BASELINE_OUTPUT_DIR at the top of this file."
    )


def load_protocol(root: Path):
    audit_path = root / "baseline16_protocol_audit.json"
    events_path = root / "frozen_10_events.csv"

    if not audit_path.exists():
        raise FileNotFoundError(f"Missing protocol audit: {audit_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Missing frozen events table: {events_path}")

    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    events_df = pd.read_csv(events_path, encoding="utf-8-sig")

    retained_n = int(audit["retained_n"])
    tr = audit.get("train_reference", [0, int(audit["full_n"] * 0.55)])
    va = audit.get("validation", [int(audit["full_n"] * 0.55), retained_n])
    train_start, train_end = int(tr[0]), int(tr[1])
    eval_start, eval_end = int(va[0]), int(va[1])

    if train_start != 0:
        raise RuntimeError(f"Unexpected train_reference start: {train_start}")
    if not (0 < train_end < eval_end <= retained_n):
        raise RuntimeError(
            f"Invalid split in audit: train_end={train_end}, eval=[{eval_start},{eval_end}), retained_n={retained_n}"
        )
    if eval_start != train_end:
        raise RuntimeError(
            f"Current FULL protocol expects validation to start at train_end, got {eval_start} vs {train_end}."
        )

    # Normalize event columns to the current FULL evaluator format.
    if "start_idx" not in events_df.columns or "end_idx" not in events_df.columns:
        raise RuntimeError("frozen_10_events.csv must contain start_idx and end_idx.")
    events_df["start_idx"] = events_df["start_idx"].astype(int)
    events_df["end_idx"] = events_df["end_idx"].astype(int)

    labels = np.zeros(retained_n, dtype=int)
    for _, row in events_df.iterrows():
        s, e = int(row["start_idx"]), int(row["end_idx"])
        if not (0 <= s < e <= retained_n):
            raise RuntimeError(f"Invalid event interval [{s},{e}) for retained_n={retained_n}")
        labels[s:e] = 1

    eval_mask = np.zeros(retained_n, dtype=bool)
    eval_mask[eval_start:eval_end] = True

    # Frozen benchmark integrity checks used by the baseline experiment.
    if len(events_df) != 10:
        raise RuntimeError(f"Expected 10 frozen events, found {len(events_df)}")
    positive_points = int(labels[eval_mask].sum())
    if positive_points != 1200:
        raise RuntimeError(f"Expected 1200 positive evaluation points, found {positive_points}")

    methods = list(audit.get("methods", []))
    seeds = [int(x) for x in audit.get("formal_seeds", EXPECTED_SEEDS)]
    if not methods:
        raise RuntimeError("No methods found in baseline16_protocol_audit.json")

    return audit, events_df, labels, eval_mask, train_end, eval_end, retained_n, methods, seeds


# =============================================================================
# CURRENT FULL threshold + evaluation protocol
# =============================================================================

def robust_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, score))
    except Exception:
        return np.nan


def robust_pr_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(average_precision_score(y_true, score))
    except Exception:
        return np.nan


def choose_threshold_full_protocol(
    train_score: np.ndarray,
    quantile: float,
    far_target: float = 0.05,
    val_score: Optional[np.ndarray] = None,
    val_label: Optional[np.ndarray] = None,
) -> float:
    """Exact current FULL threshold-selection logic."""
    x = np.asarray(train_score, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("Empty score array.")

    candidates = np.quantile(x, np.linspace(0.90, 0.999, 300))

    if val_score is None or val_label is None:
        return float(np.quantile(x, quantile))

    best_thr = None
    best_f1 = -1.0
    val_score = np.asarray(val_score, dtype=float)
    val_label = np.asarray(val_label).astype(int)

    if len(val_score) != len(val_label):
        raise ValueError(f"val_score/val_label length mismatch: {len(val_score)} vs {len(val_label)}")

    for thr in candidates:
        pred = (val_score > thr).astype(int)
        normal = (val_label == 0)
        far = pred[normal].sum() / max(normal.sum(), 1)

        if far <= far_target:
            _, _, f1, _ = precision_recall_fscore_support(
                val_label, pred, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1 = float(f1)
                best_thr = float(thr)

    if best_thr is None:
        best_thr = float(np.quantile(x, quantile))

    return best_thr


def compute_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (score > threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true.astype(int), pred, average="binary", zero_division=0
    )
    far = float(((pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1))
    mdr = float(((pred == 0) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1))
    return {
        "PointPrecision": float(p),
        "PointRecall": float(r),
        "PointF1": float(f1),
        "ROC-AUC": robust_auc(y_true, score),
        "PR_AUC": robust_pr_auc(y_true, score),
        "FAR": far,
        "MDR": mdr,
        "Detected": int(pred.sum()),
    }


def _extract_alarm_episodes(pred_full: np.ndarray, eval_start: int, eval_end: int) -> List[Tuple[int, int]]:
    pred = np.asarray(pred_full, dtype=int)
    episodes: List[Tuple[int, int]] = []
    in_ep = False
    s = None
    for idx in range(int(eval_start), int(eval_end)):
        val = int(pred[idx])
        if val == 1 and not in_ep:
            s = idx
            in_ep = True
        elif val == 0 and in_ep:
            episodes.append((int(s), int(idx)))
            in_ep = False
            s = None
    if in_ep:
        episodes.append((int(s), int(eval_end)))
    return episodes


def _merge_alarm_episodes(episodes: List[Tuple[int, int]], max_gap_min: int = 120) -> List[Tuple[int, int]]:
    if not episodes:
        return []
    merged = [list(episodes[0])]
    for s, e in episodes[1:]:
        gap = int(s) - int(merged[-1][1])
        if gap <= int(max_gap_min):
            merged[-1][1] = max(int(merged[-1][1]), int(e))
        else:
            merged.append([int(s), int(e)])
    return [(int(s), int(e)) for s, e in merged]


def _event_matching_metrics(
    pred_full: np.ndarray,
    events_df: pd.DataFrame,
    episodes: List[Tuple[int, int]],
) -> Dict[str, float]:
    if events_df is None or len(events_df) == 0:
        return {
            "EventPrecision": np.nan, "EventRecall": np.nan, "EventF1": np.nan,
            "PredEventCount": 0, "FalsePredEventCount": 0, "MatchedEventCount": 0,
            "DetectedEvents": 0, "NumEvents": 0,
            "DelayMeanMin": np.nan, "DelayMedianMin": np.nan, "IndependentMaxDelayMin": np.nan,
        }

    true_events = [
        (int(r["start_idx"]), int(r["end_idx"]))
        for _, r in events_df.sort_values("start_idx").iterrows()
    ]

    matched_episode_ids = set()
    delays: List[float] = []
    detected_events = 0

    for ts, te in true_events:
        inside = np.flatnonzero(np.asarray(pred_full[ts:te], dtype=int) > 0)
        if inside.size == 0:
            continue
        detected_events += 1
        delays.append(float(inside[0]))

        for j, (ps, pe) in enumerate(episodes):
            if j in matched_episode_ids:
                continue
            if ps < te and pe > ts:
                matched_episode_ids.add(j)
                break

    matched = int(len(matched_episode_ids))
    pred_count = int(len(episodes))
    true_count = int(len(true_events))

    ep = float(matched / pred_count) if pred_count > 0 else 0.0
    er = float(detected_events / true_count) if true_count > 0 else 0.0
    ef1 = float(2.0 * ep * er / (ep + er)) if (ep + er) > 0 else 0.0

    return {
        "EventPrecision": ep,
        "EventRecall": er,
        "EventF1": ef1,
        "PredEventCount": pred_count,
        "FalsePredEventCount": int(max(pred_count - matched, 0)),
        "MatchedEventCount": matched,
        "DetectedEvents": int(detected_events),
        "NumEvents": true_count,
        "DelayMeanMin": float(np.mean(delays)) if delays else np.nan,
        "DelayMedianMin": float(np.median(delays)) if delays else np.nan,
        "IndependentMaxDelayMin": float(np.max(delays)) if delays else np.nan,
    }


def temporal_extend(pred: np.ndarray, window: int = 20) -> np.ndarray:
    out = pred.copy()
    idx = np.where(pred == 1)[0]
    for i in idx:
        s = max(0, i - window)
        e = min(len(pred), i + window + 1)
        out[s:e] = 1
    return out


def compute_event_level_metrics(
    labels_full: np.ndarray,
    score_full: np.ndarray,
    threshold: float,
    events_df: pd.DataFrame,
    eval_mask: np.ndarray,
) -> Dict[str, float]:
    pred_full = (np.asarray(score_full) > float(threshold)).astype(int)

    # CURRENT FULL protocol: extend each point alarm by +/-30 points first.
    pred_full = temporal_extend(pred_full, window=EVENT_EXTEND_WINDOW)

    eval_indices = np.flatnonzero(np.asarray(eval_mask, dtype=bool))
    if eval_indices.size == 0:
        raise ValueError("Empty evaluation mask.")
    eval_start = int(eval_indices[0])
    eval_end = int(eval_indices[-1]) + 1

    raw_eps = _extract_alarm_episodes(pred_full, eval_start, eval_end)

    # CURRENT FULL protocol: EventF1 itself is calculated after 120-min merging.
    merged_eps = _merge_alarm_episodes(raw_eps, max_gap_min=EVENT_MERGE_GAP_MIN)
    raw = _event_matching_metrics(pred_full, events_df, merged_eps)

    # Keep the legacy-named Merge120 fields for report compatibility.
    merge120_eps = _merge_alarm_episodes(raw_eps, max_gap_min=EVENT_MERGE_GAP_MIN)
    merged = _event_matching_metrics(pred_full, events_df, merge120_eps)

    raw.update({
        "AlarmEpisodes": int(raw["PredEventCount"]),
        "MatchedAlarmEpisodes": int(raw["MatchedEventCount"]),
        "Delay": raw["DelayMeanMin"],
        "DelayDetectedOnly": raw["DelayMeanMin"],
        "Merge120EventPrecision": merged["EventPrecision"],
        "Merge120EventRecall": merged["EventRecall"],
        "Merge120EventF1": merged["EventF1"],
        "Merge120PredEventCount": int(merged["PredEventCount"]),
        "Merge120FalsePredEventCount": int(merged["FalsePredEventCount"]),
        "Merge120MatchedEventCount": int(merged["MatchedEventCount"]),
        "Merge120DelayMeanMin": merged["DelayMeanMin"],
    })
    return raw


def compute_event_adjusted_metrics(
    labels_full: np.ndarray,
    score_full: np.ndarray,
    threshold: float,
    events_df: pd.DataFrame,
    eval_mask: np.ndarray,
) -> Dict[str, float]:
    pred_full = (np.asarray(score_full) > float(threshold)).astype(int)
    pred_adj = pred_full.copy()

    for _, row in events_df.iterrows():
        s, e = int(row["start_idx"]), int(row["end_idx"])
        if pred_full[s:e].any():
            pred_adj[s:e] = 1

    y = np.asarray(labels_full)[eval_mask].astype(int)
    score = np.asarray(score_full)[eval_mask]
    pred_adj_eval = pred_adj[eval_mask]

    p, r, f1, _ = precision_recall_fscore_support(
        y, pred_adj_eval, average="binary", zero_division=0
    )

    base = compute_metrics(y, score, threshold)

    # Same explicit point-metric overwrite used in current FULL.
    pred_eval = pred_full[eval_mask]
    p0, r0, f10, _ = precision_recall_fscore_support(
        y, pred_eval, average="binary", zero_division=0
    )
    base["PointPrecision"] = float(p0)
    base["PointRecall"] = float(r0)
    base["PointF1"] = float(f10)

    base["ROC_AUC"] = base["ROC-AUC"]
    base["PR_AUC_AP"] = base["PR_AUC"]
    base["FullFAR"] = base["FAR"]
    base["AdjPrecision"] = float(p)
    base["AdjRecall"] = float(r)
    base["AdjF1"] = float(f1)

    event_met = compute_event_level_metrics(
        labels_full, score_full, threshold, events_df, eval_mask
    )
    base.update(event_met)
    return base


# =============================================================================
# Recalculation + reporting
# =============================================================================

def method_slug(method: str) -> str:
    return method.replace(" ", "_").replace("/", "_")


def sample_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0 if values.size == 1 else np.nan
    return float(np.std(values, ddof=1))


def load_old_metadata(root: Path) -> Dict[Tuple[str, int], Dict[str, object]]:
    p = root / "baseline16_all_seed_results.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out = {}
    for _, row in df.iterrows():
        try:
            key = (str(row["Variant"]), int(row["Seed"]))
        except Exception:
            continue
        keep = {}
        for c in ["TrainingSeconds", "TotalRunSeconds", "PackageVersion"]:
            if c in df.columns:
                keep[c] = row.get(c)
        out[key] = keep
    return out


def build_eventwise_rows(
    method: str,
    seed: int,
    score_filtered: np.ndarray,
    threshold: float,
    events_df: pd.DataFrame,
) -> List[Dict[str, object]]:
    # Diagnostic only: point-level first crossing inside each true event.
    pred = (np.asarray(score_filtered) > float(threshold)).astype(int)
    rows = []
    for _, r in events_df.sort_values("start_idx").iterrows():
        s, e = int(r["start_idx"]), int(r["end_idx"])
        inside = np.flatnonzero(pred[s:e] > 0)
        rows.append({
            "Variant": method,
            "Seed": int(seed),
            "EventID": r.get("EventID", r.get("event_id", r.get("short", ""))),
            "Type": r.get("type", r.get("short", "")),
            "StartIdx": s,
            "EndIdx": e,
            "DetectedByRawPointAlarm": int(inside.size > 0),
            "RawPointDelayMin": float(inside[0]) if inside.size else np.nan,
            "PeakScore": float(np.nanmax(score_filtered[s:e])) if e > s else np.nan,
            "Threshold": float(threshold),
        })
    return rows


def summarize_metrics(all_df: pd.DataFrame, method_order: List[str]) -> pd.DataFrame:
    rows = []
    for method in method_order:
        part = all_df[all_df["Variant"] == method]
        if part.empty:
            continue
        for metric in FULL_METRIC_COLUMNS:
            if metric not in part.columns:
                continue
            vals = pd.to_numeric(part[metric], errors="coerce").to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            rows.append({
                "Method": method,
                "Metric": metric,
                "Mean": float(np.mean(finite)) if finite.size else np.nan,
                "Std": sample_std(finite),
                "Runs": int(finite.size),
            })
    return pd.DataFrame(rows)


def build_paper_table(summary: pd.DataFrame, method_order: List[str]) -> pd.DataFrame:
    rows = []
    for method in method_order:
        d = summary[summary["Method"] == method]
        if d.empty:
            continue
        row = {"Method": method}
        for metric, display in PAPER_METRICS:
            q = d[d["Metric"] == metric]
            if len(q):
                mean = float(q.iloc[0]["Mean"])
                std = float(q.iloc[0]["Std"])
                row[display] = f"{mean:.4f} ± {std:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def build_wide_full_table(summary: pd.DataFrame, method_order: List[str]) -> pd.DataFrame:
    rows = []
    for method in method_order:
        d = summary[summary["Method"] == method]
        if d.empty:
            continue
        row = {"Method": method}
        for _, rr in d.iterrows():
            metric = str(rr["Metric"])
            mean = float(rr["Mean"])
            std = float(rr["Std"])
            row[metric] = f"{mean:.6f} ± {std:.6f}"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    root = resolve_baseline_output_dir()
    score_dir = root / "scores_npz"
    out_dir = root / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    audit, events_df, labels, eval_mask, train_end, eval_end, retained_n, methods, seeds = load_protocol(root)
    old_meta = load_old_metadata(root)

    print("=" * 88)
    print("Baseline-16 score-only recalculation using CURRENT FULL evaluation protocol")
    print("=" * 88)
    print(f"Baseline output : {root}")
    print(f"Scores          : {score_dir}")
    print(f"Output          : {out_dir}")
    print(f"retained_n      : {retained_n}")
    print(f"train/reference : [0, {train_end})")
    print(f"validation/eval : [{train_end}, {eval_end})")
    print(f"events          : {len(events_df)} | positive eval points={int(labels[eval_mask].sum())}")
    print(f"methods         : {len(methods)}")
    print(f"seeds           : {seeds}")
    print("NO TRAINING will be performed.\n")

    rows: List[Dict[str, object]] = []
    eventwise: List[Dict[str, object]] = []
    missing = []

    for method in methods:
        slug = method_slug(method)
        print(f"\n[{method}]")
        for seed in seeds:
            npz_path = score_dir / f"{slug}_seed_{seed}.npz"
            if not npz_path.exists():
                print(f"  MISSING: {npz_path.name}")
                missing.append(str(npz_path))
                continue

            with np.load(npz_path) as z:
                if "score" not in z.files or "train_score" not in z.files:
                    raise RuntimeError(f"{npz_path.name} lacks score/train_score. Keys={z.files}")
                score_raw = np.asarray(z["score"], dtype=float).reshape(-1)
                train_score = np.asarray(z["train_score"], dtype=float).reshape(-1)
                old_threshold = float(np.asarray(z["threshold"]).reshape(-1)[0]) if "threshold" in z.files else np.nan

            if len(score_raw) != retained_n:
                raise RuntimeError(
                    f"{npz_path.name}: score length={len(score_raw)}, expected retained_n={retained_n}"
                )
            if len(train_score) != retained_n:
                raise RuntimeError(
                    f"{npz_path.name}: train_score length={len(train_score)}, expected retained_n={retained_n}"
                )

            # ---- exact CURRENT FULL threshold call ----
            threshold = choose_threshold_full_protocol(
                train_score=train_score[:train_end],
                quantile=THRESHOLD_QUANTILE,
                far_target=FAR_TARGET,
                val_score=train_score[train_end:],
                val_label=labels[train_end:],
            )

            # ---- exact CURRENT FULL evaluation score post-processing ----
            score_filtered = median_filter(score_raw, size=MEDIAN_FILTER_SIZE)

            # ---- exact CURRENT FULL metrics ----
            met = compute_event_adjusted_metrics(
                labels_full=labels,
                score_full=score_filtered,
                threshold=threshold,
                events_df=events_df,
                eval_mask=eval_mask,
            )
            met["Threshold"] = float(threshold)
            met["EvalPoints"] = int(eval_mask.sum())
            met["Variant"] = method
            met["Seed"] = int(seed)
            met["OldSavedThreshold"] = float(old_threshold)
            met["ThresholdChanged"] = float(threshold - old_threshold) if np.isfinite(old_threshold) else np.nan
            met["EvaluationProtocol"] = "CURRENT_FULL: val/FAR-threshold + median3 + event_extend30 + merge120"

            # Preserve old training-time metadata only; NEVER reuse old metrics.
            meta = old_meta.get((method, int(seed)), {})
            if "TrainingSeconds" in meta:
                met["TrainingSeconds"] = meta["TrainingSeconds"]
            if "TotalRunSeconds" in meta:
                met["OriginalTotalRunSeconds"] = meta["TotalRunSeconds"]
            if "PackageVersion" in meta:
                met["OriginalPackageVersion"] = meta["PackageVersion"]

            rows.append(met)
            eventwise.extend(build_eventwise_rows(method, seed, score_filtered, threshold, events_df))

            print(
                f"  seed={seed:<4d} | thr={threshold:.6g} (old {old_threshold:.6g}) | "
                f"PointF1={met['PointF1']:.4f} | EventF1={met['EventF1']:.4f} | "
                f"AdjF1={met['AdjF1']:.4f} | FAR={met['FullFAR']:.4f} | "
                f"EventR={met['EventRecall']:.4f} | Delay={met['Delay']:.2f}"
            )

    if missing:
        (out_dir / "missing_score_files.txt").write_text("\n".join(missing), encoding="utf-8")
        print(f"\nWARNING: {len(missing)} score files were missing. See missing_score_files.txt")

    if not rows:
        raise RuntimeError("No score files were recalculated.")

    all_df = pd.DataFrame(rows)
    rank = {m: i for i, m in enumerate(methods)}
    all_df["_rank"] = all_df["Variant"].map(rank)
    all_df = all_df.sort_values(["_rank", "Seed"]).drop(columns="_rank").reset_index(drop=True)

    # Main seedwise output.
    all_path = out_dir / "baseline16_all_seed_results_FULLprotocol_recalc.csv"
    all_df.to_csv(all_path, index=False, encoding="utf-8-sig")

    # Eventwise diagnostic output.
    event_df = pd.DataFrame(eventwise)
    event_path = out_dir / "baseline16_eventwise_results_FULLprotocol_recalc.csv"
    event_df.to_csv(event_path, index=False, encoding="utf-8-sig")

    # Long mean/std summary.
    summary = summarize_metrics(all_df, methods)
    summary_path = out_dir / "baseline16_mean_std_FULLprotocol_recalc.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # Paper table: exactly the columns currently used in the manuscript/revision tables.
    paper = build_paper_table(summary, methods)
    paper_path = out_dir / "baseline16_paper_table_FULLprotocol_recalc.csv"
    paper.to_csv(paper_path, index=False, encoding="utf-8-sig")

    # Wide table with every FULL metric as mean ± std.
    wide = build_wide_full_table(summary, methods)
    wide_path = out_dir / "baseline16_full_metrics_mean_std_wide_FULLprotocol_recalc.csv"
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")

    # Protocol audit for revision traceability.
    new_audit = {
        "purpose": "Post-hoc metric recalculation from saved baseline score arrays; no retraining.",
        "source_baseline_output": str(root),
        "score_source": str(score_dir),
        "source_protocol_audit": str(root / "baseline16_protocol_audit.json"),
        "source_events": str(root / "frozen_10_events.csv"),
        "retained_n": retained_n,
        "train_reference": [0, train_end],
        "validation_eval": [train_end, eval_end],
        "event_count": int(len(events_df)),
        "positive_eval_points": int(labels[eval_mask].sum()),
        "threshold_protocol": {
            "candidate_source": "clean/reference train_score[0:train_end]",
            "candidate_quantiles": "np.linspace(0.90, 0.999, 300)",
            "validation_score": "train_score[train_end:]",
            "validation_label": "frozen B1-R5 labels[train_end:]",
            "constraint": f"validation normal FAR <= {FAR_TARGET}",
            "selection": "maximize validation binary F1 among feasible candidates",
            "fallback_quantile": THRESHOLD_QUANTILE,
        },
        "eval_score_postprocess": f"scipy.ndimage.median_filter(size={MEDIAN_FILTER_SIZE})",
        "point_adj_protocol": "same as current FULL; no temporal extension for Point/Adj metrics",
        "event_protocol": {
            "temporal_extend_window": EVENT_EXTEND_WINDOW,
            "episode_merge_gap_min": EVENT_MERGE_GAP_MIN,
            "event_f1": "computed after temporal extension and <=120-min episode merging",
            "matching": "one-to-one predicted episode / true event overlap",
        },
        "methods": methods,
        "seeds_expected": seeds,
        "score_files_missing": missing,
        "outputs": [
            all_path.name,
            event_path.name,
            summary_path.name,
            paper_path.name,
            wide_path.name,
        ],
    }
    audit_out = out_dir / "baseline16_FULLprotocol_recalc_audit.json"
    audit_out.write_text(json.dumps(new_audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # Full JSON report.
    report = {
        "audit": new_audit,
        "seedwise_results": all_df.to_dict("records"),
        "mean_std_summary": summary.to_dict("records"),
        "paper_table": paper.to_dict("records"),
    }
    report_path = out_dir / "baseline16_full_report_FULLprotocol_recalc.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("\n" + "=" * 88)
    print("DONE — no baseline model was retrained.")
    print("=" * 88)
    print(f"Seedwise metrics : {all_path}")
    print(f"Mean ± std       : {summary_path}")
    print(f"Paper table      : {paper_path}")
    print(f"Wide full table  : {wide_path}")
    print(f"Full JSON report : {report_path}")
    print(f"Protocol audit   : {audit_out}")


if __name__ == "__main__":
    main()
