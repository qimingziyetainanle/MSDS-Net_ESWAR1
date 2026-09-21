# -*- coding: utf-8 -*-
"""
MSDS-Net for electro-thermal time-series anomaly detection — Protocol v2.1 B1-R5.

V4.0: exact Exp25 frozen 10-event benchmark with Exp30B B1-R5 temporal profile; clean 0%-55% training/reference, 55%-70% internal validation, pure Q0.99 threshold.

This script contains:
1) FULL MSDS-Net training and detection.
2) Synthetic anomaly injection for controlled simulation validation.
3) Ablation experiments, with FULL model listed first.
4) Publication-style figures for anomaly-score, L/E/A separation, residual heatmap,
   residual-graph attention, score distribution, threshold sensitivity, and ablation results.
5) Final score aligned with Section 3.4: S_t = sum_k theta_k r_t^(k) + theta_g ||q_t||_2.

Expected files in the same directory:
    930T.csv
    930V.csv

Run:
    python msds_net_930_full_ablation_v20.py
"""

import math
import warnings
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gc
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import matplotlib

# Suppress harmless layout warnings caused by manually placed colorbars / polar axes.
warnings.filterwarnings(
    "ignore",
    message=r".*figure includes Axes that are not compatible with tight_layout.*",
    category=UserWarning,
)


def _set_external_matplotlib_backend() -> str:
    """Use a real GUI window rather than PyCharm's Scientific Plots pane when possible."""
    preferred = ["TkAgg", "QtAgg", "Qt5Agg"]
    for b in preferred:
        try:
            matplotlib.use(b, force=True)
            return b
        except Exception:
            pass
    return matplotlib.get_backend()


MATPLOTLIB_BACKEND = _set_external_matplotlib_backend()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib import patches

def set_paper_style() -> None:
    """Clean IEEE/AAAI-like plotting style with readable labels."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,

        # 全局字体放大，适合论文插图缩放后阅读
        "font.size": 12.0,
        "axes.labelsize": 13.0,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 12.0,

        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,

        "legend.frameon": True,
        "legend.framealpha": 0.94,
        "legend.edgecolor": "0.72",
        "legend.fancybox": False,
    })


def expand_ylim_for_legend(ax, top: float = 0.22, bottom: float = 0.04) -> None:
    """Expand y-axis range so that an upper-right legend does not cover curves."""
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    if not np.isfinite(span) or span <= 1e-12:
        span = 1.0
    ax.set_ylim(ymin - bottom * span, ymax + top * span)


def apply_paper_axis(ax, tick_size: float = 11.5) -> None:
    """Use full frame, light y-grid, and readable ticks."""
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.9)

    ax.grid(axis="y", color="0.88", lw=0.55, alpha=0.75)
    ax.tick_params(axis="both", labelsize=tick_size, width=0.8, length=3.5)
# =========================
# 1. Configuration
# =========================

@dataclass
class Config:
    temperature_file: str = "930T.csv"
    vi_file: str = "930V.csv"
    encoding: str = "gbk"

    seed: int = 42
    train_ratio: float = 0.70
    internal_train_ratio: float = 0.55
    reference_end_ratio: float = 0.70
    seq_len: int = 60
    stride: int = 5

    hidden_dim: int = 64
    graph_dim: int = 32
    dropout: float = 0.10

    batch_size: int = 128
    epochs: int = 30
    ablation_epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-5

    force_gpu: bool = True
    use_amp: bool = False
    num_workers: int = 0

    lambda_trend: float = 1.75
    lambda_dist: float = 0.10
    lambda_res: float = 0.0008
    a_residual_mix: float = 0.50

    lambda_graph: float = 0.018
    graph_score_weight: float = 0.031

    subspace_theta: Tuple[float, float, float, float, float] = (1.45, 0.70, 0.70, 1.35, 1.80)
    threshold_quantile: float = 0.990
    eval_context_multiplier: int = 2

    run_injection_eval: bool = True
    run_ablation: bool = False  # 临时调图：只跑 FULL，跳过消融  True/False
    make_threshold_sensitivity_figure: bool = False
    show_figures: bool = False
    save_figures: bool = False
    output_dir: str = "02full_outputs"

    # To avoid figures with too many points. 1 means no downsampling.
    plot_step: int = 5


CFG = Config()
torch.set_num_threads(min(4, max(1, torch.get_num_threads())))


# =========================
# 2. General utilities
# =========================

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def get_training_device(cfg: Config) -> torch.device:
    if cfg.force_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available, but force_gpu=True.\n"
            "Install a CUDA-enabled PyTorch build or set force_gpu=False temporarily."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"Using device: cuda | GPU: {name} | VRAM: {mem:.2f} GB", flush=True)
    else:
        print("Using device: cpu", flush=True)
    return device


def ensure_output_dir(cfg: Config) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_or_show(fig: plt.Figure, cfg: Config, filename: str) -> None:
    if cfg.save_figures:
        out = ensure_output_dir(cfg) / filename

        # PNG for preview
        fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.02)

        # PDF for paper
        out_pdf = out.with_suffix(".pdf")
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)

        print(f"Saved figure: {out}")
        print(f"Saved figure: {out_pdf}")


def finish_figures(cfg: Config) -> None:
    print(f"Matplotlib backend: {MATPLOTLIB_BACKEND}")
    if cfg.show_figures:
        plt.show(block=True)
    else:
        plt.close("all")


def progress_bar(current: int, total: int, width: int = 30) -> str:
    ratio = current / max(total, 1)
    filled = int(width * ratio)
    return "█" * filled + "-" * (width - filled)


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
    """Return contiguous alarm episodes [start,end) inside the evaluation interval."""
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
    """Merge predicted alarm episodes whose inter-episode gap is <= max_gap_min."""
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
    """
    One-to-one event matching used by the Protocol-v2.1 benchmark.
    A true event is detected if at least one predicted point lies in its interval.
    """
    if events_df is None or len(events_df) == 0:
        return {
            "EventPrecision": np.nan,
            "EventRecall": np.nan,
            "EventF1": np.nan,
            "PredEventCount": 0,
            "FalsePredEventCount": 0,
            "MatchedEventCount": 0,
            "DetectedEvents": 0,
            "NumEvents": 0,
            "DelayMeanMin": np.nan,
            "DelayMedianMin": np.nan,
            "IndependentMaxDelayMin": np.nan,
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

        # One-to-one predicted-episode match for event precision.
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


def temporal_extend(pred, window=20):

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
    pred_full = (
            np.asarray(score_full) > float(threshold)
    ).astype(int)

    pred_full = temporal_extend(
        pred_full,
        window=30
    )


    eval_indices = np.flatnonzero(np.asarray(eval_mask, dtype=bool))
    if eval_indices.size == 0:
        raise ValueError("Empty evaluation mask.")
    eval_start = int(eval_indices[0])
    eval_end = int(eval_indices[-1]) + 1

    raw_eps = _extract_alarm_episodes(
        pred_full,
        eval_start,
        eval_end
    )

    merged_eps = _merge_alarm_episodes(
        raw_eps,
        max_gap_min=120
    )

    raw = _event_matching_metrics(
        pred_full,
        events_df,
        merged_eps
    )

    merge120_eps = _merge_alarm_episodes(raw_eps, max_gap_min=120)
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
        y,
        pred_adj_eval,
        average="binary",
        zero_division=0
    )

    base = compute_metrics(
        y,
        score,
        threshold
    )

    # overwrite point metrics with alarm decision output
    pred_eval = pred_full[eval_mask]

    p0, r0, f10, _ = precision_recall_fscore_support(
        y,
        pred_eval,
        average="binary",
        zero_division=0
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


# =========================
# 3. Data loading, features, synthetic anomalies
# =========================

def read_csv_auto(path: str, encoding: str = "gbk") -> pd.DataFrame:
    return pd.read_csv(path, encoding=encoding)


def load_raw_inputs(cfg: Config) -> pd.DataFrame:
    df_t = read_csv_auto(cfg.temperature_file, cfg.encoding)
    df_vi = read_csv_auto(cfg.vi_file, cfg.encoding)

    df_t = df_t.rename(columns={"时间": "time", "AI01240112.PV": "T"})
    df_vi = df_vi.rename(columns={
        "时间": "time",
        "AI01240151.PV": "U51",
        "AI01240152.PV": "U52",
        "AI01240153.PV": "U53",
        "AI01240154.PV": "I54",
        "AI01240155.PV": "I55",
        "AI01240156.PV": "I56",
    })

    df_t["time"] = pd.to_datetime(df_t["time"])
    df_vi["time"] = pd.to_datetime(df_vi["time"])

    keep_t = ["time", "T"]
    keep_vi = ["time", "U51", "U52", "U53", "I54", "I55", "I56"]
    df = pd.merge(df_t[keep_t], df_vi[keep_vi], on="time", how="inner")
    df = df.sort_values("time").reset_index(drop=True)

    raw_cols = ["T", "U51", "U52", "U53", "I54", "I55", "I56"]
    for c in raw_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[raw_cols] = df[raw_cols].ffill().bfill()
    return df


def add_engineered_features(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]], List[str]]:
    df = df_raw.copy()
    eps = 1e-6

    df["U_mean"] = (df["U51"] + df["U52"] + df["U53"]) / 3.0
    df["I_mean"] = (df["I54"] + df["I55"] + df["I56"]) / 3.0

    df["B_U"] = np.sqrt(
        ((df["U51"] - df["U_mean"]) ** 2 +
         (df["U52"] - df["U_mean"]) ** 2 +
         (df["U53"] - df["U_mean"]) ** 2) / 3.0
    ) / (df["U_mean"].abs() + eps)

    df["B_I"] = np.sqrt(
        ((df["I54"] - df["I_mean"]) ** 2 +
         (df["I55"] - df["I_mean"]) ** 2 +
         (df["I56"] - df["I_mean"]) ** 2) / 3.0
    ) / (df["I_mean"].abs() + eps)

    df["P"] = (
        (df["U51"] * df["I54"]).abs() +
        (df["U52"] * df["I55"]).abs() +
        (df["U53"] * df["I56"]).abs()
    ) / 3.0

    df["dT"] = df["T"].diff().fillna(0.0)
    df["dP"] = df["P"].diff().fillna(0.0)

    subspace_cols = {
        "T":  ["T", "dT"],
        "U":  ["U51", "U52", "U53", "U_mean", "B_U"],
        "I":  ["I54", "I55", "I56", "I_mean", "B_I"],
        "B":  ["B_U", "B_I"],
        "ET": ["T", "dT", "P", "dP", "U_mean", "I_mean"],
    }
    feature_cols = [
        "T", "U51", "U52", "U53", "I54", "I55", "I56",
        "U_mean", "I_mean", "B_U", "B_I", "P", "dT", "dP"
    ]
    return df, subspace_cols, feature_cols


def scale_features(
    df_feature: pd.DataFrame,
    feature_cols: List[str],
    train_end: int,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    df_scaled = df_feature.copy()
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df_scaled.loc[:train_end - 1, feature_cols].values)
    df_scaled.loc[:, feature_cols] = scaler.transform(df_scaled.loc[:, feature_cols].values)
    return df_scaled, scaler


def robust_scale_value(x: pd.Series) -> float:
    """Robust scale used for synthetic anomaly amplitude."""
    med = float(np.nanmedian(x.values))
    mad = float(np.nanmedian(np.abs(x.values - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.nanstd(x.values))
    return max(scale, 1e-6)


def inject_synthetic_anomalies(df_raw: pd.DataFrame, train_end: int, cfg: Config) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Inject five relation-oriented anomaly types into a copy of the real data.

    V18 intentionally avoids obvious single-variable spikes.  The injected anomalies are
    mainly weak structural inconsistencies: single channels are not made extremely large,
    but the electro-thermal or phase-balance relations are changed.  This setting is more
    suitable for validating MQ-RGA because the graph module is designed to capture
    cross-subspace residual-relation deviations.
    """
    df = df_raw.copy()
    n = len(df)
    labels = np.zeros(n, dtype=int)
    events = []

    # Put events after the normal reference section.  Longer, smoother intervals make the
    # anomalies relation-oriented rather than single-point spikes.
    start_base = max(train_end + 4 * cfg.seq_len, int(n * 0.74))
    available = n - start_base - 4 * cfg.seq_len
    if available <= 7 * cfg.seq_len:
        start_base = min(max(train_end + 2 * cfg.seq_len, int(n * 0.72)), n - 8 * cfg.seq_len)
        available = max(n - start_base - 4 * cfg.seq_len, 1)

    gap = max(cfg.seq_len * 3, available // 6)
    length = max(cfg.seq_len * 3, 220)

    ref = df.loc[:train_end - 1]
    t_scale = robust_scale_value(ref["T"])
    u_scale = np.nanmedian([robust_scale_value(ref[c]) for c in ["U51", "U52", "U53"]])
    i_scale = np.nanmedian([robust_scale_value(ref[c]) for c in ["I54", "I55", "I56"]])
    t_med = max(float(np.nanmedian(ref["T"])), 1.0)
    u_med = max(float(np.nanmedian(ref[["U51", "U52", "U53"]].values)), 1.0)
    i_med = max(float(np.nanmedian(ref[["I54", "I55", "I56"]].values)), 1.0)

    # Smaller amplitudes than the earlier versions: the difficulty comes from relation
    # mismatch, not from obvious outliers.
    t_amp = max(1.10 * t_scale, 0.035 * t_med)
    u_amp = max(1.05 * u_scale, 0.020 * u_med)
    i_amp = max(1.05 * i_scale, 0.020 * i_med)

    def smooth_step(m: int) -> np.ndarray:
        z = np.linspace(-3.0, 3.0, m)
        y = 1.0 / (1.0 + np.exp(-z))
        return (y - y.min()) / max(y.max() - y.min(), 1e-8)

    def mark(s: int, e: int, name: str, mech: str, short: str) -> None:
        s = int(max(0, min(s, n - 1)))
        e = int(max(s + 1, min(e, n)))
        labels[s:e] = 1
        events.append({
            "type": name,
            "short": short,
            "mechanism": mech,
            "start_idx": s,
            "end_idx": e,
            "start_time": df.loc[s, "time"],
            "end_time": df.loc[e - 1, "time"],
        })

    starts = [start_base + k * gap for k in range(5)]
    starts = [min(max(0, s), n - length - 1) for s in starts]

    # Hard negative controls: relation-consistent normal disturbances.
    # These windows remain label=0 and fall into the evaluation context around injected
    # events. They are visually strong but physically coordinated: U and I move together,
    # phase balance is preserved, and T follows with a mild delay. This evaluates whether
    # the model can avoid false alarms under normal strong disturbances.
    normal_len = max(cfg.seq_len * 3, 220)
    for s0 in starts:
        ns = int(max(train_end, s0 - int(2.60 * cfg.seq_len)))
        ne = int(min(s0 - int(0.18 * cfg.seq_len), ns + normal_len, n))
        if ne - ns < cfg.seq_len:
            continue
        m = ne - ns
        bump = np.sin(np.linspace(0.0, np.pi, m))
        delay = max(3, int(0.18 * m))
        t_bump = np.zeros(m)
        t_bump[delay:] = bump[:-delay]
        df.loc[ns:ne - 1, ["U51", "U52", "U53"]] += 0.95 * u_amp * bump[:, None]
        df.loc[ns:ne - 1, ["I54", "I55", "I56"]] += 0.95 * i_amp * bump[:, None]
        df.loc[ns:ne - 1, "T"] += 0.62 * t_amp * t_bump


    # A1: thermal response mismatch.  Temperature drifts smoothly while electrical inputs
    # remain almost unchanged; the anomaly is mainly in T--ET relation.
    s, e = starts[0], starts[0] + length
    w = smooth_step(e - s)
    df.loc[s:e - 1, "T"] -= t_amp * (0.25 + 0.75 * w)
    # A weak delayed recovery makes it look like response mismatch rather than a hard jump.
    rec_start = s + int(0.62 * (e - s))
    rec_w = smooth_step(e - rec_start)
    df.loc[rec_start:e - 1, "T"] += 0.35 * t_amp * rec_w
    mark(s, e, "A1_Thermal_response_mismatch", "T / ET", "A1")

    # A2: voltage-balance relation anomaly.  The mean voltage is nearly preserved, but the
    # phase relation changes gradually, activating U--B instead of simple U magnitude.
    s, e = starts[1], starts[1] + length
    w = smooth_step(e - s)
    df.loc[s:e - 1, "U51"] += u_amp * w
    df.loc[s:e - 1, "U52"] -= 0.85 * u_amp * w
    df.loc[s:e - 1, "U53"] -= 0.15 * u_amp * w
    mark(s, e, "A2_Voltage_balance_relation", "U / B", "A2")

    # A3: current-balance relation anomaly.  The mean current is approximately unchanged,
    # while the phase distribution drifts, activating I--B.
    s, e = starts[2], starts[2] + length
    w = smooth_step(e - s)
    df.loc[s:e - 1, "I54"] += 0.90 * i_amp * w
    df.loc[s:e - 1, "I55"] -= 0.75 * i_amp * w
    df.loc[s:e - 1, "I56"] -= 0.15 * i_amp * w
    mark(s, e, "A3_Current_balance_relation", "I / B", "A3")

    # A4: coupled balance inconsistency.  Voltage and current are both only mildly changed,
    # but their balance directions conflict, giving a structural B-related anomaly.
    s, e = starts[3], starts[3] + length
    w = smooth_step(e - s)
    df.loc[s:e - 1, "U51"] += 0.70 * u_amp * w
    df.loc[s:e - 1, "U52"] -= 0.35 * u_amp * w
    df.loc[s:e - 1, "U53"] -= 0.35 * u_amp * w
    df.loc[s:e - 1, "I54"] -= 0.70 * i_amp * w
    df.loc[s:e - 1, "I55"] += 0.35 * i_amp * w
    df.loc[s:e - 1, "I56"] += 0.35 * i_amp * w
    mark(s, e, "A4_Coupled_balance_inconsistency", "B", "A4")

    # A5: electro-thermal coupling mismatch.  Load-related electrical inputs change in a
    # coordinated way, but temperature reacts in the wrong direction / with insufficient response.
    s, e = starts[4], starts[4] + length
    w = smooth_step(e - s)
    df.loc[s:e - 1, ["U51", "U52", "U53"]] += 0.35 * u_amp * w[:, None]
    df.loc[s:e - 1, ["I54", "I55", "I56"]] += 0.65 * i_amp * w[:, None]
    df.loc[s:e - 1, "T"] -= 0.55 * t_amp * w
    mark(s, e, "A5_Electrothermal_coupling_mismatch", "ET", "A5")

    events_df = pd.DataFrame(events)
    return df, labels, events_df


def build_controlled_eval_mask(labels: np.ndarray, events_df: pd.DataFrame, n: int, cfg: Config) -> np.ndarray:
    """Evaluate injected-anomaly metrics only in controlled windows around the injected events."""
    mask = np.zeros(n, dtype=bool)
    context = int(cfg.eval_context_multiplier * cfg.seq_len)
    for _, row in events_df.iterrows():
        s = int(row["start_idx"])
        e = int(row["end_idx"])
        mask[max(0, s - context): min(n, e + context)] = True
    # Always include injected samples even if a malformed event table is given.
    mask = mask | (labels.astype(bool))
    return mask

def build_subspace_arrays(df_scaled: pd.DataFrame, subspace_cols: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    return {k: df_scaled[v].values.astype(np.float32) for k, v in subspace_cols.items()}


def make_start_indices(n: int, seq_len: int, stride: int, end_limit: Optional[int] = None) -> np.ndarray:
    max_start = n - seq_len if end_limit is None else end_limit - seq_len
    if max_start < 0:
        raise ValueError("seq_len is larger than the available data length.")
    starts = list(range(0, max_start + 1, stride))
    if len(starts) == 0 or starts[-1] != max_start:
        starts.append(max_start)
    return np.array(starts, dtype=np.int64)


# =========================
# 4. Dataset
# =========================

class WindowDataset(Dataset):
    def __init__(self, subspace_arrays: Dict[str, np.ndarray], starts: np.ndarray, seq_len: int) -> None:
        self.subspace_arrays = subspace_arrays
        self.starts = starts
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = int(self.starts[idx])
        item = {name: torch.tensor(arr[s:s + self.seq_len], dtype=torch.float32)
                for name, arr in self.subspace_arrays.items()}
        item["__start__"] = torch.tensor(s, dtype=torch.long)
        return item


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    starts = batch["__start__"]
    x = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "__start__"}
    return x, starts


# =========================
# 5. Model modules
# =========================

class SubspaceBranch(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, use_separation: bool = True) -> None:
        super().__init__()
        self.use_separation = use_separation
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        if use_separation:
            self.head_L = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, in_dim))
            self.head_E = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, in_dim))
            self.head_A = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, in_dim))
        else:
            self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, in_dim))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(z)
        if self.use_separation:
            L = self.head_L(h)
            E = self.head_E(h)
            A_head = self.head_A(h)
            mix = float(getattr(CFG, "a_residual_mix", 0.0))
            if mix > 0:
                A_res = x - L - E
                A = mix * A_res + (1.0 - mix) * A_head
            else:
                A = A_head
            x_hat = L + E + A
            return {"H": h, "L": L, "E": E, "A": A, "X_hat": x_hat, "has_LEA": torch.tensor(1, device=x.device)}
        else:
            x_hat = self.decoder(h)
            # For the no-separation variant, residual is ordinary reconstruction error.
            A = x - x_hat
            Z = torch.zeros_like(x)
            return {"H": h, "L": x_hat, "E": Z, "A": A, "X_hat": x_hat, "has_LEA": torch.tensor(0, device=x.device)}


class MQRGA(nn.Module):
    def __init__(
        self,
        num_nodes: int = 5,
        graph_dim: int = 32,
        use_gate: bool = True,
        use_edge_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.graph_dim = graph_dim
        self.use_gate = use_gate
        self.use_edge_bias = use_edge_bias

        self.node_mlp = nn.Sequential(nn.Linear(1, graph_dim), nn.GELU(), nn.Linear(graph_dim, graph_dim))
        self.W_Q = nn.Linear(graph_dim, graph_dim, bias=False)
        self.W_K = nn.Linear(graph_dim, graph_dim, bias=False)
        self.W_V = nn.Linear(graph_dim, graph_dim, bias=False)
        # Mild non-zero priors for edge types. Kept small to avoid the V22-style over-amplification.
        self.edge_bias = nn.Parameter(torch.tensor([0.015, 0.0075, 0.018], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(0.0295, dtype=torch.float32))
        self.gate_mlp = nn.Sequential(nn.Linear(graph_dim * 2 + 1, graph_dim), nn.GELU(), nn.Linear(graph_dim, 1))

        # Only the mechanism-guided five-node graph is defined here.
        undirected_edges = [(0, 4, 0, "T-ET"), (1, 4, 1, "U-ET"), (2, 4, 1, "I-ET"),
                            (1, 3, 2, "U-B"), (2, 3, 2, "I-B")]
        self.physical_edges = undirected_edges
        neighbors: Dict[int, List[Tuple[int, int, int]]] = {i: [] for i in range(num_nodes)}
        for edge_id, (i, j, etype, _) in enumerate(undirected_edges):
            if i < num_nodes and j < num_nodes:
                neighbors[i].append((j, etype, edge_id))
                neighbors[j].append((i, etype, edge_id))
        self.neighbors = neighbors
        self.edge_names = [x[3] for x in undirected_edges]

    def forward(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # r: [B, L, K]
        node_h = self.node_mlp(r.unsqueeze(-1))
        Q, K, V = self.W_Q(node_h), self.W_K(node_h), self.W_V(node_h)
        Bsz, L, K_num, D = node_h.shape
        h_prime = torch.zeros_like(node_h)
        edge_sum = torch.zeros(Bsz, L, len(self.edge_names), device=r.device)
        edge_count = torch.zeros(len(self.edge_names), device=r.device)

        for i in range(self.num_nodes):
            if len(self.neighbors[i]) == 0:
                h_prime[:, :, i, :] = node_h[:, :, i, :]
                continue
            score_list, value_list, edge_ids = [], [], []
            for j, etype, edge_id in self.neighbors[i]:
                qk = (Q[:, :, i, :] * K[:, :, j, :]).sum(dim=-1) / math.sqrt(D)
                diff = (r[:, :, i] - r[:, :, j]).abs().unsqueeze(-1)
                if self.use_gate:
                    gate_in = torch.cat([node_h[:, :, i, :], node_h[:, :, j, :], diff], dim=-1)
                    gate = torch.sigmoid(self.gate_mlp(gate_in)).squeeze(-1)
                else:
                    gate = torch.zeros_like(qk)
                bias = self.edge_bias[etype] if self.use_edge_bias else 0.0
                score = qk + bias + self.gamma * gate
                score_list.append(score)
                value_list.append(V[:, :, j, :])
                edge_ids.append(edge_id)
            scores = torch.stack(score_list, dim=-1)
            alpha = torch.softmax(scores, dim=-1)
            values = torch.stack(value_list, dim=-2)
            h_prime[:, :, i, :] = F.elu((alpha.unsqueeze(-1) * values).sum(dim=-2))
            for local_idx, edge_id in enumerate(edge_ids):
                edge_sum[:, :, edge_id] += alpha[:, :, local_idx]
                edge_count[edge_id] += 1.0

        edge_count = torch.clamp(edge_count, min=1.0)
        edge_attention = edge_sum / edge_count.view(1, 1, -1)

        # Attention-weighted residual-relation readout.
        # This is a concrete implementation of q_t = Readout(H'_t): each physical edge
        # contributes a relation-deviation feature. It avoids using an arbitrary untrained
        # hidden-state norm as the graph anomaly term.
        edge_q = []
        for edge_id, (i, j, _, _) in enumerate(self.physical_edges):
            mismatch = (r[:, :, i] - r[:, :, j]).abs()
            magnitude = 0.5 * (r[:, :, i] + r[:, :, j])
            edge_q.append(edge_attention[:, :, edge_id] * (mismatch + 0.5 * magnitude))
        q = torch.stack(edge_q, dim=-1)  # [B, L, |E_r|]
        return q, h_prime, edge_attention


class MSDSNet(nn.Module):
    def __init__(
        self,
        subspace_dims: Dict[str, int],
        hidden_dim: int,
        graph_dim: int,
        dropout: float,
        graph_score_weight: float,
        use_separation: bool = True,
        use_graph: bool = True,
        use_gate: bool = True,
        use_edge_bias: bool = True,
    ) -> None:
        super().__init__()
        self.subspace_names = list(subspace_dims.keys())
        self.use_graph = use_graph and len(self.subspace_names) > 1
        self.use_separation = use_separation
        self.graph_score_weight = graph_score_weight
        self.branches = nn.ModuleDict({
            name: SubspaceBranch(dim, hidden_dim, dropout, use_separation=use_separation)
            for name, dim in subspace_dims.items()
        })
        if self.use_graph:
            self.rga = MQRGA(len(self.subspace_names), graph_dim, use_gate=use_gate, use_edge_bias=use_edge_bias)
        else:
            self.rga = None

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        comps = {}
        r_list = []
        rec_list = []
        a_list = []
        for name in self.subspace_names:
            comp = self.branches[name](batch[name])
            comps[name] = comp
            a_norm = torch.linalg.vector_norm(comp["A"], ord=2, dim=-1)
            rec_norm = torch.linalg.vector_norm(batch[name] - comp["X_hat"], ord=2, dim=-1)

            # Text-consistent residual strength.
            # FULL model: r_t^(k) = ||a_t^(k)||_2.
            # w/o Separation has no A-head, so its residual proxy is the ordinary reconstruction error.
            if self.use_separation:
                r = a_norm
            else:
                r = rec_norm

            r_list.append(r)
            rec_list.append(rec_norm)
            a_list.append(a_norm)

        r_all = torch.stack(r_list, dim=-1)
        rec_all = torch.stack(rec_list, dim=-1)
        a_all = torch.stack(a_list, dim=-1)

        # Text-consistent residual term: sum_k theta_k r_t^(k).
        # For the five mechanism-guided subspaces [T, U, I, B, ET], use configured theta.
        # Other variants, such as w/o Subspace with one branch, fall back to equal weights.
        if r_all.shape[-1] == 5:
            theta = torch.tensor(CFG.subspace_theta, dtype=r_all.dtype, device=r_all.device)
            theta = theta / torch.clamp(theta.sum(), min=1e-8)
            top2_score = torch.topk(
                r_all,
                k=2,
                dim=-1
            ).values.mean(dim=-1)

            max_score = torch.max(
                r_all,
                dim=-1
            ).values

            residual_score = (
                    0.8 * top2_score
                    +
                    0.2 * max_score
            )
        else:
            residual_score = r_all.mean(dim=-1)

        if self.use_graph:
            q, h_prime, edge_attention = self.rga(r_all)
            # Text-consistent graph term: ||q_t||_2. Here q_t is an attention-weighted
            # physical residual-relation readout returned by MQ-RGA.
            graph_score = torch.linalg.vector_norm(q, ord=2, dim=-1)
            # Edge response is saved only for mechanism visualization, not for scoring.
            edge_responses = []
            for edge_id, (i, j, _, _) in enumerate(self.rga.physical_edges):
                edge_responses.append(edge_attention[:, :, edge_id] * 0.5 * (r_all[:, :, i] + r_all[:, :, j]))
            edge_response = torch.stack(edge_responses, dim=-1)
            score = residual_score + self.graph_score_weight * graph_score
        else:
            q = torch.zeros(r_all.shape[0], r_all.shape[1], 1, device=r_all.device)
            h_prime = torch.zeros(r_all.shape[0], r_all.shape[1], r_all.shape[2], 1, device=r_all.device)
            edge_attention = torch.zeros(r_all.shape[0], r_all.shape[1], 5, device=r_all.device)
            edge_response = torch.zeros(r_all.shape[0], r_all.shape[1], 5, device=r_all.device)
            graph_score = torch.zeros_like(residual_score)
            score = residual_score

        return {"components": comps, "r_all": r_all, "rec_all": rec_all, "a_all": a_all,
                "q": q, "h_prime": h_prime, "edge_attention": edge_attention,
                "edge_response": edge_response, "residual_score": residual_score,
                "graph_score": graph_score, "score": score}


# =========================
# 6. Loss, training, inference
# =========================

def msds_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: Config) -> Tuple[torch.Tensor, Dict[str, float]]:
    rec_loss = torch.tensor(0.0, device=out["score"].device)
    trend_loss = torch.tensor(0.0, device=out["score"].device)
    dist_loss = torch.tensor(0.0, device=out["score"].device)
    res_loss = torch.tensor(0.0, device=out["score"].device)
    graph_loss = torch.tensor(0.0, device=out["score"].device)
    lea_count = 0

    for name, comp in out["components"].items():
        x = batch[name]
        L, E, A, x_hat = comp["L"], comp["E"], comp["A"], comp["X_hat"]
        rec_loss = rec_loss + F.mse_loss(x_hat, x)
        if bool(comp["has_LEA"].item()):
            trend_loss = trend_loss + ((L[:, 1:, :] - L[:, :-1, :]) ** 2).mean()
            dist_loss = dist_loss + (E ** 2).mean()
            res_loss = res_loss + torch.sqrt((A ** 2).sum(dim=-1) + 1e-8).mean()
            lea_count += 1

    if "graph_score" in out:
        graph_loss = out["graph_score"].mean()

    total = (rec_loss
             + cfg.lambda_trend * trend_loss
             + cfg.lambda_dist * dist_loss
             + cfg.lambda_res * res_loss
             + cfg.lambda_graph * graph_loss)
    logs = {"total": float(total.detach().cpu()), "rec": float(rec_loss.detach().cpu()),
            "trend": float(trend_loss.detach().cpu()), "dist": float(dist_loss.detach().cpu()),
            "res": float(res_loss.detach().cpu()), "graph": float(graph_loss.detach().cpu())}
    return total, logs


def train_model(model: MSDSNet, train_loader: DataLoader, cfg: Config, device: torch.device, epochs: int, title: str) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = bool(cfg.use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"\n[{title}] Training start | epochs={epochs} | batches/epoch={len(train_loader)} | AMP={use_amp}", flush=True)
    model.train()
    for epoch in range(1, epochs + 1):
        meter = {"total": 0.0, "rec": 0.0, "trend": 0.0, "dist": 0.0, "res": 0.0, "graph": 0.0}
        nb = 0
        for batch_idx, raw in enumerate(train_loader, start=1):
            batch, _ = move_batch_to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(batch)
                loss, logs = msds_loss(out, batch, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            for k in meter:
                meter[k] += logs[k]
            nb += 1
            bar = progress_bar(batch_idx, len(train_loader))
            print(f"\r[{title}] Epoch {epoch:03d}/{epochs} [{bar}] {batch_idx:03d}/{len(train_loader):03d} | running_loss={meter['total']/nb:.5f}",
                  end="", flush=True)
        avg = {k: v / max(nb, 1) for k, v in meter.items()}
        print(f"\r[{title}] Epoch {epoch:03d}/{epochs} [██████████████████████████████] {len(train_loader):03d}/{len(train_loader):03d} | "
              f"loss={avg['total']:.5f} | rec={avg['rec']:.5f} | trend={avg['trend']:.5f} | dist={avg['dist']:.5f} | res={avg['res']:.5f} | graph={avg['graph']:.5f}", flush=True)


@torch.no_grad()
def aggregate_scores(
    model: MSDSNet,
    loader: DataLoader,
    n_points: int,
    cfg: Config,
    device: torch.device,
    collect_components: bool = True,
) -> Dict[str, np.ndarray]:
    model.eval()
    score_sum = np.zeros(n_points, dtype=np.float64)
    graph_sum = np.zeros(n_points, dtype=np.float64)
    count = np.zeros(n_points, dtype=np.float64)
    names = model.subspace_names
    r_sum = np.zeros((n_points, len(names)), dtype=np.float64)
    edge_sum = np.zeros((n_points, 5), dtype=np.float64)
    edge_resp_sum = np.zeros((n_points, 5), dtype=np.float64)
    rec_sum = np.zeros((n_points, len(names)), dtype=np.float64)
    a_sum = np.zeros((n_points, len(names)), dtype=np.float64)

    comp_sum = {}
    if collect_components:
        for name in names:
            # Dimensions inferred lazily.
            comp_sum[name] = {"L": None, "E": None, "A": None, "X_hat": None, "count": np.zeros(n_points, dtype=np.float64)}

    for raw in loader:
        starts = raw["__start__"].cpu().numpy()
        batch, _ = move_batch_to_device(raw, device)
        out = model(batch)
        score = out["score"].detach().cpu().numpy()
        graph = out["graph_score"].detach().cpu().numpy()
        r_all = out["r_all"].detach().cpu().numpy()
        edge = out["edge_attention"].detach().cpu().numpy()
        edge_resp = out.get("edge_response", torch.zeros_like(out["edge_attention"])).detach().cpu().numpy()
        rec_all = out.get("rec_all", torch.zeros_like(out["r_all"])).detach().cpu().numpy()
        a_all = out.get("a_all", torch.zeros_like(out["r_all"])).detach().cpu().numpy()

        comps_np = None
        if collect_components:
            comps_np = {
                name: {key: out["components"][name][key].detach().cpu().numpy() for key in ["L", "E", "A", "X_hat"]}
                for name in names
            }
            for name in names:
                d = comps_np[name]["L"].shape[-1]
                for key in ["L", "E", "A", "X_hat"]:
                    if comp_sum[name][key] is None:
                        comp_sum[name][key] = np.zeros((n_points, d), dtype=np.float64)

        for b, s in enumerate(starts):
            e = s + cfg.seq_len
            score_sum[s:e] += score[b]
            graph_sum[s:e] += graph[b]
            r_sum[s:e] += r_all[b]
            edge_sum[s:e] += edge[b]
            edge_resp_sum[s:e] += edge_resp[b]
            rec_sum[s:e] += rec_all[b]
            a_sum[s:e] += a_all[b]
            count[s:e] += 1.0
            if collect_components:
                for name in names:
                    for key in ["L", "E", "A", "X_hat"]:
                        comp_sum[name][key][s:e] += comps_np[name][key][b]
                    comp_sum[name]["count"][s:e] += 1.0

    count[count == 0] = 1.0
    result = {"score": score_sum / count, "graph_score": graph_sum / count,
              "r_all": r_sum / count[:, None],
              "rec_all": rec_sum / count[:, None],
              "a_all": a_sum / count[:, None],
              "edge_attention": edge_sum / count[:, None],
              "edge_response": edge_resp_sum / count[:, None],
              "subspace_names": np.array(names)}
    if collect_components:
        final_comp = {}
        for name in names:
            c = comp_sum[name]["count"]
            c[c == 0] = 1.0
            final_comp[name] = {key: comp_sum[name][key] / c[:, None] for key in ["L", "E", "A", "X_hat"]}
        result["components"] = final_comp
    return result


def choose_threshold(
        train_score: np.ndarray,
        quantile: float,
        far_target: float = 0.05,
        val_score: np.ndarray = None,
        val_label: np.ndarray = None
) -> float:

    x = np.asarray(train_score, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        raise ValueError(
            "Empty score array."
        )


    candidates = np.quantile(
        x,
        np.linspace(0.90, 0.999, 300)
    )


    if val_score is None or val_label is None:
        return float(np.quantile(x, quantile))


    best_thr = None
    best_f1 = -1


    for thr in candidates:

        pred = (
            np.asarray(val_score) > thr
        ).astype(int)


        normal = (
            np.asarray(val_label)==0
        )


        far = (
            pred[normal].sum()
            /
            max(normal.sum(),1)
        )


        if far <= far_target:


            _,_,f1,_ = precision_recall_fscore_support(
                val_label,
                pred,
                average="binary",
                zero_division=0
            )


            if f1 > best_f1:
                best_f1=float(f1)
                best_thr=float(thr)



    if best_thr is None:

        best_thr=float(
            np.quantile(x,quantile)
        )


    return best_thr


# =========================
# 7. Plotting
# =========================

def thin(df: pd.DataFrame, arr: Optional[np.ndarray], cfg: Config):
    step = max(1, cfg.plot_step)
    if arr is None:
        return df.iloc[::step]
    return df.iloc[::step], arr[::step]


def shade_events(ax, events_df: Optional[pd.DataFrame], alpha: float = 0.15, label: bool = False) -> None:
    """Draw injected anomaly regions in a consistent orange color."""
    if events_df is None or len(events_df) == 0:
        return
    for idx, (_, row) in enumerate(events_df.iterrows()):
        ax.axvspan(row["start_time"], row["end_time"], color="#F39C12", alpha=alpha,
                   lw=0, label="Injected region" if label and idx == 0 else None)
        if "short" in row:
            ymax = ax.get_ylim()[1]
            ax.text(row["start_time"], ymax, str(row["short"]), fontsize=8, va="top", ha="left", color="#8A4B08")



def set_time_xlim_tight(ax, times) -> None:
    """Remove default x-axis padding for time-series figures."""
    try:
        if len(times) > 1:
            ax.set_xlim(pd.to_datetime(times).iloc[0], pd.to_datetime(times).iloc[-1])
        ax.margins(x=0)
    except Exception:
        pass

def fix_edge_time_labels(fig, ax) -> None:
    """
    Keep the first and last x tick labels inside the figure.
    This only changes text alignment. It does not change the time range,
    window length, or tick locations.
    """
    fig.canvas.draw()

    labels = [
        lab for lab in ax.get_xticklabels()
        if lab.get_visible() and lab.get_text() != ""
    ]

    if len(labels) >= 1:
        labels[0].set_ha("left")
        labels[0].set_clip_on(False)

    if len(labels) >= 2:
        labels[-1].set_ha("right")
        labels[-1].set_clip_on(False)

    fig.canvas.draw_idle()


def robust_row_normalize(mat: np.ndarray, q: float = 99.0, log1p: bool = True) -> np.ndarray:
    x = np.asarray(mat, dtype=float).copy()
    if log1p:
        x = np.log1p(np.maximum(x, 0.0))
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        hi = np.nanpercentile(x[i], q)
        lo = np.nanpercentile(x[i], 1.0)
        den = max(hi - lo, 1e-8)
        out[i] = np.clip((x[i] - lo) / den, 0.0, 1.0)
    return out


def plot_raw_variables(
    df_raw: pd.DataFrame,
    labels: Optional[np.ndarray],
    events_df: Optional[pd.DataFrame],
    cfg: Config,
    filename: str
) -> None:
    d = df_raw.iloc[::max(1, cfg.plot_step)].copy()

    fig, axes = plt.subplots(
        3, 1,
        figsize=(10.8, 6.2),   # 比之前更紧凑、更适合论文，也更清楚
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0], "hspace": 0.08}
    )

    event_color = "#F2C879"

    def shade_events_light(ax, add_label=False):
        if events_df is None or len(events_df) == 0:
            return
        for idx, (_, row) in enumerate(events_df.iterrows()):
            ax.axvspan(
                row["start_time"],
                row["end_time"],
                color=event_color,
                alpha=0.18,
                lw=0,
                label="Injected region" if add_label and idx == 0 else None
            )

    # ---------- 1) Temperature ----------
    axes[0].plot(
        d["time"], d["T"],
        lw=0.8,
        color="#2F6B9A",
        label="Temperature"
    )
    axes[0].set_ylabel("Temp.", fontsize=18)

    # ---------- 2) Voltage ----------
    voltage_colors = ["#4E79A7", "#F28E2B", "#59A14F"]
    voltage_label_map = {
        "U51": r"$U_a$",
        "U52": r"$U_b$",
        "U53": r"$U_c$",
    }

    for c, col in zip(["U51", "U52", "U53"], voltage_colors):
        axes[1].plot(
            d["time"], d[c],
            lw=0.7,
            alpha=0.95,
            color=col,
            label=voltage_label_map[c]
        )
    axes[1].set_ylabel("Voltage", fontsize=18)

    # ---------- 3) Current ----------
    current_colors = ["#4E79A7", "#F28E2B", "#59A14F"]
    current_label_map = {
        "I54": r"$I_a$",
        "I55": r"$I_b$",
        "I56": r"$I_c$",
    }

    for c, col in zip(["I54", "I55", "I56"], current_colors):
        axes[2].plot(
            d["time"], d[c],
            lw=0.7,
            alpha=0.95,
            color=col,
            label=current_label_map[c]
        )
    axes[2].set_ylabel("Current", fontsize=18)
    axes[2].set_xlabel("Time", fontsize=18)

    # ---------- common style ----------
    for i, ax in enumerate(axes):
        shade_events_light(ax, add_label=(i == 0))

        apply_paper_axis(ax, tick_size=11.5)
        set_time_xlim_tight(ax, d["time"])

        if i == 0:
            expand_ylim_for_legend(ax, top=0.22, bottom=0.03)
        else:
            expand_ylim_for_legend(ax, top=0.45, bottom=0.03)

    # ---------- event labels only on the first panel ----------
    if events_df is not None and len(events_df) > 0:
        ymin, ymax = axes[0].get_ylim()
        y_text = ymax - 0.05 * (ymax - ymin)
        for _, row in events_df.iterrows():
            mid = row["start_time"] + (row["end_time"] - row["start_time"]) / 2
            axes[0].text(
                mid,
                y_text,
                str(row.get("short", "")),
                ha="center",
                va="top",
                fontsize=9.5,
                color="#8A4B08"
            )

    # ---------- legends ----------
    # 第一行图例放左上角
    axes[0].legend(
        loc="upper left",
        ncol=2,
        fontsize=18,
        frameon=True,
        framealpha=0.94,
        edgecolor="0.72",
        borderpad=0.30,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    axes[1].legend(
        loc="upper right",
        ncol=3,
        fontsize=18,
        frameon=True,
        framealpha=0.94,
        edgecolor="0.72",
        borderpad=0.30,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=0.9,
    )

    axes[2].legend(
        loc="upper right",
        ncol=3,
        fontsize=18,
        frameon=True,
        framealpha=0.94,
        edgecolor="0.72",
        borderpad=0.30,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=0.9,
    )

    # ---------- x axis ----------
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate(rotation=0, ha="center")

    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.985,
        bottom=0.11,
        hspace=0.08
    )

    # 单独保存 Fig. 1 的 SVG 版本
    svg_path = ensure_output_dir(cfg) / Path(filename).with_suffix(".svg")
    fig.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.02
    )
    print(f"Saved figure: {svg_path}")

    # 原有的 PNG 和 PDF 保存
    save_or_show(fig, cfg, filename)


def plot_score_curve(df: pd.DataFrame, result: Dict[str, np.ndarray], threshold: float, cfg: Config, filename: str,
                     events_df: Optional[pd.DataFrame] = None, title: str = "MSDS-Net anomaly score") -> None:
    """Publication-style anomaly response timeline.

    The upper panel gives the anomaly score and threshold. The lower event row uses fixed-size
    event markers instead of duration-scaled blocks, because the injected intervals are short
    compared with the full-month record and otherwise appear as broken vertical stripes.
    """
    d, score = thin(df, result["score"], cfg)
    pred = score > threshold
    fig, (ax, ax_band) = plt.subplots(
        2, 1, figsize=(14, 4.8), sharex=True,
        gridspec_kw={"height_ratios": [4.0, 0.55], "hspace": 0.08}
    )
    shade_events(ax, events_df, alpha=0.08, label=False)
    ax.plot(d["time"], score, linewidth=1.05, label="Anomaly score", color="#1F4E79")
    ax.axhline(threshold, linestyle="--", linewidth=1.15, label="Threshold", color="#C0392B")
    if pred.any():
        ax.scatter(d.loc[pred, "time"], score[pred], s=11, label="Detected points", color="#C0392B", zorder=3)
    ax.set_ylabel("Score", fontsize=18)
    #ax.set_title(title)
    expand_ylim_for_legend(ax, top=0.28, bottom=0.03)

    ax.legend(
        loc="upper right",
        ncol=3,
        fontsize=18,
        frameon=True,
        framealpha=0.94,
        edgecolor="0.72",
        borderpad=0.35,
        handlelength=1.8,
        handletextpad=0.5
    )
    ax.grid(axis="y", alpha=0.18)
    set_time_xlim_tight(ax, d["time"])

    ax_band.set_ylim(0, 1)
    ax_band.set_yticks([])
    ax_band.set_ylabel("Events", rotation=0, labelpad=28, va="center")
    event_colors = ["#F6C85F", "#6F4E7C", "#9DD866", "#CA472F", "#4E79A7"]
    if events_df is not None and len(events_df) > 0:
        for idx, (_, row) in enumerate(events_df.iterrows()):
            c = event_colors[idx % len(event_colors)]
            mid = row["start_time"] + (row["end_time"] - row["start_time"]) / 2
            ax_band.axvline(mid, ymin=0.23, ymax=0.76, color=c, lw=2.0, alpha=0.50, zorder=2)
            ax_band.text(
                mid, 0.82, str(row.get("short", f"A{idx+1}")),
                ha="center", va="bottom", fontsize=8.0,
                color=c, fontweight="normal", zorder=4
            )
    ax_band.set_xlabel("Time", fontsize=18)
    set_time_xlim_tight(ax_band, d["time"])
    ax_band.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.065, right=0.985, top=0.90, bottom=0.16, hspace=0.08)
    save_or_show(fig, cfg, filename)

def plot_lea_decomposition(
    df_scaled: pd.DataFrame,
    result: Dict[str, np.ndarray],
    cfg: Config,
    filename: str,
    events_df: Optional[pd.DataFrame] = None,
    threshold: Optional[float] = None,
    subspace: str = "T",
    feature_idx: int = 0,
    center_idx: Optional[int] = None
) -> None:

    if "components" not in result or subspace not in result["components"]:
        return

    n = len(df_scaled)
    if center_idx is None:
        center_idx = int(np.argmax(result["score"]))

    # 保持你之前较好的时间范围，不乱拉长
    half = max(cfg.seq_len * 3, 180)
    s = max(0, center_idx - half)
    e = min(n, center_idx + half)

    time = df_scaled.loc[s:e - 1, "time"].reset_index(drop=True)
    comp = result["components"][subspace]

    # 原始观测 X
    if subspace == "T":
        obs_col = "T" if feature_idx == 0 else "dT"
        obs = df_scaled.loc[s:e - 1, obs_col].to_numpy(dtype=float)
    else:
        # 非 T 子空间时，直接用重构和残差近似显示
        obs = (comp["X_hat"][s:e, feature_idx] + comp["A"][s:e, feature_idx] * 0.0).astype(float)

    L = comp["L"][s:e, feature_idx].astype(float)
    E = comp["E"][s:e, feature_idx].astype(float)
    A = comp["A"][s:e, feature_idx].astype(float)

    # X 面板里顺便画一条 L+E 参考线（你现在图里就是这种效果）
    LE = L + E

    fig, axes = plt.subplots(
        4, 1,
        figsize=(10.8, 6.6),
        sharex=True,
        gridspec_kw={"hspace": 0.10}
    )

    panel_face = "#F7F3E8"

    # ---------- X ----------
    axes[0].set_facecolor(panel_face)
    axes[0].plot(time, obs, lw=1.2, color="#1f77b4")
    axes[0].plot(time, LE, lw=1.1, ls="--", color="0.35")
    axes[0].set_ylabel(r"$X$", fontsize=18)

    # ---------- L ----------
    axes[1].set_facecolor(panel_face)
    axes[1].plot(time, L, lw=1.2, color="#2ca02c")
    axes[1].set_ylabel(r"$L$", fontsize=18)

    # ---------- E ----------
    axes[2].set_facecolor(panel_face)
    axes[2].plot(time, E, lw=1.2, color="#ff7f0e")
    axes[2].set_ylabel(r"$E$", fontsize=18)

    # ---------- A ----------
    axes[3].set_facecolor(panel_face)
    axes[3].plot(time, A, lw=1.2, color="#d62728")
    axes[3].set_ylabel(r"$A$", fontsize=18)
    axes[3].set_xlabel("Time", fontsize=18)

    # --------- 统一风格 ----------
    for ax in axes:
        apply_paper_axis(ax, tick_size=11.0)
        ax.grid(axis="y", color="0.85", lw=0.55, alpha=0.75)

        # 不再画竖虚线
        # ax.axvline(...)

        # y 轴数字统一成两位小数，更整齐
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))

    # X 与 A 的 y 轴范围统一，增强可比性
    xa_min = min(np.nanmin(obs), np.nanmin(A))
    xa_max = max(np.nanmax(obs), np.nanmax(A))
    pad = 0.08 * max(xa_max - xa_min, 1e-6)
    axes[0].set_ylim(xa_min - pad, xa_max + pad)
    axes[3].set_ylim(xa_min - pad, xa_max + pad)

    # 给最右边时间标签留一点点白边，避免显示不全
    if len(time) > 1:
        dt = time.iloc[-1] - time.iloc[0]
        right_pad = dt * 0.03
        for ax in axes:
            ax.set_xlim(time.iloc[0], time.iloc[-1] + right_pad)

    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    fig.subplots_adjust(left=0.10, right=0.985, top=0.985, bottom=0.12, hspace=0.10)
    save_or_show(fig, cfg, filename)


def plot_residual_heatmap(df: pd.DataFrame, result: Dict[str, np.ndarray], cfg: Config, filename: str,
                          events_df: Optional[pd.DataFrame] = None) -> None:
    step = max(1, cfg.plot_step)
    r_raw = result["r_all"][::step].T
    r = robust_row_normalize(r_raw, q=99.0, log1p=True)
    times = df["time"].iloc[::step]
    names = result["subspace_names"]
    fig, ax = plt.subplots(figsize=(14, 3.6))
    im = ax.imshow(r, aspect="auto", interpolation="nearest", origin="lower", cmap="magma",
                   extent=[mdates.date2num(times.iloc[0]), mdates.date2num(times.iloc[-1]), -0.5, len(names) - 0.5])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"r_{n}" for n in names])
    ax.set_xlabel("Time")
    #ax.set_title("Normalized residual contribution of mechanism-guided subspaces")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    if events_df is not None:
        for _, row in events_df.iterrows():
            ax.axvline(row["start_time"], color="#F39C12", lw=0.9, alpha=0.9)
            ax.axvline(row["end_time"], color="#F39C12", lw=0.9, alpha=0.45)
            ax.text(row["start_time"], len(names) - 0.25, str(row.get("short", "")), fontsize=8, color="white", va="top")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row-normalized residual response")
    fig.autofmt_xdate()
    fig.tight_layout()
    save_or_show(fig, cfg, filename)



def _window_variation_score(df_raw: pd.DataFrame, a: int, b: int) -> float:
    seg = df_raw.iloc[a:b]
    if len(seg) < 5:
        return -np.inf
    u_mean = seg[["U51", "U52", "U53"]].mean(axis=1).to_numpy(dtype=float)
    i_mean = seg[["I54", "I55", "I56"]].mean(axis=1).to_numpy(dtype=float)
    t = seg["T"].to_numpy(dtype=float)
    def scale(x):
        return (x - np.nanmedian(x)) / (np.nanmedian(np.abs(x - np.nanmedian(x))) * 1.4826 + 1e-8)
    return float(np.nanstd(scale(t)) + np.nanstd(np.diff(scale(u_mean))) + np.nanstd(np.diff(scale(i_mean))))


def _normalize_for_local_plot(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    z = (x - med) / max(mad, 1e-8)
    return np.clip(z, -4, 4)


def _radar(ax, values: np.ndarray, labels: List[str], color: str, title: str) -> None:
    vals = np.asarray(values, dtype=float)
    vals = vals / max(np.nanmax(vals), 1e-8)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    vals = np.concatenate([vals, vals[:1]])
    ax.plot(angles, vals, color=color, lw=1.8)
    ax.fill(angles, vals, color=color, alpha=0.24)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(["0.5", "1.0"], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=10, pad=12)


def _polar_bar_profile(ax, values: np.ndarray, labels: List[str], color: str, title: str) -> None:
    """Compact polar bar profile used in Fig.4.

    It is easier to read than a radar polygon: each radial bar is one mechanism-guided
    residual subspace, and the bar height indicates the normalized residual strength.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals / max(np.nanmax(vals), 1e-8)
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / max(n, 1) * 0.70
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.bar(
        angles,
        vals,
        width=width,
        bottom=0.04,
        color=color,
        alpha=0.72,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.plot(np.r_[angles, angles[0]], np.r_[vals + 0.04, vals[0] + 0.04], color=color, lw=1.1, alpha=0.80)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11.5)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(["0.5", "1.0"], fontsize=10.0)
    ax.set_ylim(0, 1.08)
    ax.grid(color="0.82", linestyle="--", linewidth=0.65, alpha=0.85)
    ax.spines["polar"].set_color("0.35")
    ax.spines["polar"].set_linewidth(0.9)
    #ax.set_title(title, fontsize=10.2, pad=11)



def plot_normal_vs_structural_contrast(
    df_raw: pd.DataFrame,
    result: Dict[str, np.ndarray],
    labels: np.ndarray,
    events_df: pd.DataFrame,
    threshold: float,
    cfg: Config,
    filename: str
) -> None:
    """
    Fig.8: normal disturbance vs structural anomaly.

    Final version:
    - save four independent subfigures
    - both windows are about 90 minutes
    - structural window prefers A3 if detected
    - normal score y-axis fixed to 0--2
    - no inward-aligned edge tick labels
    - keep centered time labels and reserve right blank space
    """
    if events_df is None or len(events_df) == 0:
        return

    n = len(df_raw)
    score = np.asarray(result["score"], dtype=float)
    times = pd.to_datetime(df_raw["time"]).reset_index(drop=True)
    base = Path(filename).stem

    window_delta = pd.Timedelta(minutes=90)
    step_delta = pd.Timedelta(minutes=10)
    min_points = max(20, int(cfg.seq_len * 0.5))

    # ============================================================
    # 1. helper functions
    # ============================================================
    def time_to_index(t, side="left"):
        return int(np.searchsorted(times.values, np.datetime64(t), side=side))

    def get_window_by_time(start_time):
        end_time = start_time + window_delta
        a = time_to_index(start_time, side="left")
        b = time_to_index(end_time, side="right")
        a = max(0, min(a, n - 1))
        b = max(a + 1, min(b, n))
        return a, b

    def get_series(a: int, b: int):
        seg = df_raw.iloc[a:b]
        t = pd.to_datetime(seg["time"]).reset_index(drop=True)
        T = seg["T"].to_numpy(dtype=float)
        U = seg[["U51", "U52", "U53"]].mean(axis=1).to_numpy(dtype=float)
        I = seg[["I54", "I55", "I56"]].mean(axis=1).to_numpy(dtype=float)
        return seg, t, T, U, I

    def relative_normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            return x

        k = max(5, int(0.20 * len(x)))
        baseline = float(np.nanmedian(x[:k]))

        p05, p95 = np.nanpercentile(x, [5, 95])
        scale = float(p95 - p05)

        if not np.isfinite(scale) or scale < 1e-8:
            scale = float(np.nanstd(x))

        scale = max(scale, 1e-8)
        z = (x - baseline) / scale
        return np.clip(z, -2.5, 2.5)

    def window_shape_score(a: int, b: int):
        _, _, T, U, I = get_series(a, b)

        zT = relative_normalize(T)
        zU = relative_normalize(U)
        zI = relative_normalize(I)

        if len(zT) < 5:
            return -np.inf, np.inf, np.inf

        dT = np.diff(zT)
        dU = np.diff(zU)
        dI = np.diff(zI)

        visual_score = (
            0.35 * float(np.nanstd(zT)) +
            0.50 * float(np.nanstd(zU)) +
            0.50 * float(np.nanstd(zI)) +
            0.35 * float(np.nanstd(dU)) +
            0.35 * float(np.nanstd(dI))
        )

        max_jump = max(
            float(np.nanmax(np.abs(dT))) if len(dT) else 0.0,
            float(np.nanmax(np.abs(dU))) if len(dU) else 0.0,
            float(np.nanmax(np.abs(dI))) if len(dI) else 0.0
        )

        diff_all = np.concatenate([
            np.abs(dT), np.abs(dU), np.abs(dI)
        ]) if len(dT) + len(dU) + len(dI) > 0 else np.asarray([0.0])

        jump_ratio = max_jump / (float(np.nanmedian(diff_all)) + 1e-6)

        return visual_score, max_jump, jump_ratio

    # ============================================================
    # 2. select non-injected disturbance window
    # ============================================================
    forbidden = labels.astype(bool).copy()
    buffer = int(max(cfg.seq_len, 60))

    for _, row in events_df.iterrows():
        s0, e0 = int(row["start_idx"]), int(row["end_idx"])
        forbidden[max(0, s0 - buffer):min(n, e0 + buffer)] = True

    normal_candidates = []

    t_start = times.iloc[0]
    t_end = times.iloc[-1] - window_delta
    cur = t_start

    while cur <= t_end:
        a, b = get_window_by_time(cur)

        if b - a < min_points:
            cur += step_delta
            continue

        if forbidden[a:b].any():
            cur += step_delta
            continue

        max_score = float(np.nanmax(score[a:b]))
        q95_score = float(np.nanpercentile(score[a:b], 95))
        visual_score, max_jump, jump_ratio = window_shape_score(a, b)

        if (
            max_score <= threshold * 0.98
            and q95_score <= threshold * 0.80
            and max_jump <= 1.80
            and jump_ratio <= 20.0
        ):
            normal_candidates.append(
                (a, b, visual_score, max_score, q95_score, max_jump, jump_ratio)
            )

        cur += step_delta

    if len(normal_candidates) == 0:
        print("[Fig.8] Warning: no suitable non-injected disturbance window found.")
        return

    normal_a, normal_b, normal_var, normal_max_score, normal_q95_score, normal_jump, normal_jump_ratio = max(
        normal_candidates,
        key=lambda z: z[2] - 0.20 * z[5] - 0.015 * z[6]
    )

    # ============================================================
    # 3. build structural candidates
    # ============================================================
    structural_candidates = []

    for _, row in events_df.iterrows():
        event_start = int(row["start_idx"])
        event_end = int(row["end_idx"])
        event_start_time = pd.Timestamp(row["start_time"])
        short = str(row.get("short", ""))
        mech = str(row.get("mechanism", ""))

        local_score = score[event_start:event_end]
        if len(local_score) == 0:
            continue

        above = local_score > threshold
        if not above.any():
            continue

        first_det = event_start + int(np.argmax(above))
        peak_local = int(np.nanargmax(local_score))
        peak_idx = event_start + peak_local

        peak_score = float(local_score[peak_local])
        det_ratio = float(above.mean())

        first_det_time = times.iloc[first_det]
        peak_time = times.iloc[peak_idx]

        start_options = [
            event_start_time - pd.Timedelta(minutes=40),
            event_start_time - pd.Timedelta(minutes=35),
            event_start_time - pd.Timedelta(minutes=30),
            first_det_time - pd.Timedelta(minutes=45),
            peak_time - pd.Timedelta(minutes=50),
        ]

        for start_time in start_options:
            a, b = get_window_by_time(start_time)

            if b - a < min_points:
                continue

            if not (a <= event_start < b):
                continue
            if not (a <= first_det < b):
                continue
            if not (a <= peak_idx < b):
                continue

            local_win_score = score[a:b]

            pre_a = a
            pre_b = max(a + 1, min(event_start, b))

            if pre_b > pre_a + 5:
                pre_median = float(np.nanmedian(score[pre_a:pre_b]))
                pre_max = float(np.nanmax(score[pre_a:pre_b]))
            else:
                pre_median = float(np.nanmedian(local_win_score))
                pre_max = float(np.nanmax(local_win_score))

            peak_pos = (peak_idx - a) / max(b - a - 1, 1)
            first_det_pos = (first_det - a) / max(b - a - 1, 1)
            event_start_pos = (event_start - a) / max(b - a - 1, 1)

            _, _, T, U, I = get_series(a, b)
            zT = relative_normalize(T)
            zU = relative_normalize(U)
            zI = relative_normalize(I)

            range_T = float(np.nanpercentile(zT, 95) - np.nanpercentile(zT, 5))
            range_U = float(np.nanpercentile(zU, 95) - np.nanpercentile(zU, 5))
            range_I = float(np.nanpercentile(zI, 95) - np.nanpercentile(zI, 5))

            relation_visibility = 0.35 * range_T + 0.45 * range_U + 0.45 * range_I
            score_contrast = peak_score - pre_median

            peak_position_penalty = abs(peak_pos - 0.62)
            det_position_penalty = abs(first_det_pos - 0.55)
            onset_position_penalty = abs(event_start_pos - 0.42)
            pre_score_penalty = max(0.0, pre_max - threshold * 0.75)

            candidate_score = (
                2.20 * score_contrast
                + 0.80 * peak_score
                + 0.50 * det_ratio
                + 0.35 * relation_visibility
                - 0.70 * peak_position_penalty
                - 0.45 * det_position_penalty
                - 0.45 * onset_position_penalty
                - 0.80 * pre_score_penalty
            )

            structural_candidates.append({
                "short": short,
                "mechanism": mech,
                "event_start": event_start,
                "event_end": event_end,
                "first_det": first_det,
                "peak_idx": peak_idx,
                "window_start": a,
                "window_end": b,
                "peak_score": peak_score,
                "det_ratio": det_ratio,
                "pre_median": pre_median,
                "pre_max": pre_max,
                "score_contrast": score_contrast,
                "peak_pos": peak_pos,
                "first_det_pos": first_det_pos,
                "event_start_pos": event_start_pos,
                "relation_visibility": relation_visibility,
                "candidate_score": candidate_score
            })

    if len(structural_candidates) == 0:
        print("[Fig.8] Warning: no suitable detected structural anomaly window found.")
        return

    cand_df = pd.DataFrame(structural_candidates)
    cand_df["window_start_time"] = cand_df["window_start"].apply(lambda x: times.iloc[int(x)])
    cand_df["window_end_time"] = cand_df["window_end"].apply(lambda x: times.iloc[int(x) - 1])
    cand_df["event_start_time"] = cand_df["event_start"].apply(lambda x: times.iloc[int(x)])
    cand_df["event_end_time"] = cand_df["event_end"].apply(lambda x: times.iloc[int(x) - 1])
    cand_df.to_csv(
        ensure_output_dir(cfg) / "fig8_structural_event_candidates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ============================================================
    # 4. prefer A3
    # ============================================================
    preferred_short = "A3"

    preferred_candidates = [
        c for c in structural_candidates
        if str(c.get("short", "")) == preferred_short
    ]

    if len(preferred_candidates) > 0:
        structural_candidates_sorted = sorted(
            preferred_candidates,
            key=lambda z: z["candidate_score"],
            reverse=True
        )
        print(f"[Fig.8] Prefer structural event: {preferred_short}")
    else:
        structural_candidates_sorted = sorted(
            structural_candidates,
            key=lambda z: z["candidate_score"],
            reverse=True
        )
        print(f"[Fig.8] Preferred event {preferred_short} not available or not detected. Use best candidate instead.")

    ev = structural_candidates_sorted[0]
    struct_a = int(ev["window_start"])
    struct_b = int(ev["window_end"])

    selected_info = pd.DataFrame([
        {
            "Window": "non_injected_disturbance",
            "StartIndex": normal_a,
            "EndIndex": normal_b,
            "StartTime": df_raw.loc[normal_a, "time"],
            "EndTime": df_raw.loc[normal_b - 1, "time"],
            "MaxScore": normal_max_score,
            "Q95Score": normal_q95_score,
            "Threshold": threshold,
            "Event": "",
            "Mechanism": ""
        },
        {
            "Window": "detected_structural_anomaly",
            "StartIndex": struct_a,
            "EndIndex": struct_b,
            "StartTime": df_raw.loc[struct_a, "time"],
            "EndTime": df_raw.loc[struct_b - 1, "time"],
            "MaxScore": float(np.nanmax(score[struct_a:struct_b])),
            "Q95Score": float(np.nanpercentile(score[struct_a:struct_b], 95)),
            "Threshold": threshold,
            "Event": ev["short"],
            "Mechanism": ev["mechanism"],
            "CandidateScore": ev["candidate_score"]
        }
    ])

    selected_info.to_csv(
        ensure_output_dir(cfg) / "selected_windows_for_fig8.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "[Fig.8] Selected non-injected window:",
        df_raw.loc[normal_a, "time"], "->", df_raw.loc[normal_b - 1, "time"],
        "| max score =", f"{normal_max_score:.4f}",
        "| threshold =", f"{threshold:.4f}"
    )

    print(
        "[Fig.8] Selected structural window:",
        ev["short"], ev["mechanism"],
        "| event =", df_raw.loc[ev["event_start"], "time"], "->", df_raw.loc[ev["event_end"] - 1, "time"],
        "| plotted =", df_raw.loc[struct_a, "time"], "->", df_raw.loc[struct_b - 1, "time"],
        "| peak score =", f"{ev['peak_score']:.4f}",
        "| threshold =", f"{threshold:.4f}",
        "| candidate score =", f"{ev['candidate_score']:.4f}"
    )

    # ============================================================
    # 5. plotting
    # ============================================================
    def beautify_axis(ax):
        apply_paper_axis(ax, tick_size=11.5)

    def shade_event_region(ax):
        visible_start = df_raw.loc[struct_a, "time"]
        visible_end = df_raw.loc[struct_b - 1, "time"]

        es = max(df_raw.loc[ev["event_start"], "time"], visible_start)
        ee = min(df_raw.loc[ev["event_end"] - 1, "time"], visible_end)

        if es < ee:
            ax.axvspan(es, ee, color="#F2C879", alpha=0.16, lw=0)


    def save_traj_plot(a: int, b: int, out_name: str, shade_event: bool = False):
        seg, t, T, U, I = get_series(a, b)

        fig, ax = plt.subplots(figsize=(6.2, 3.0))

        if shade_event:
            shade_event_region(ax)

        ax.plot(t, relative_normalize(T), lw=1.10, label="T", color="#1F77B4")
        ax.plot(t, relative_normalize(U), lw=1.00, label="mean U", color="#FF7F0E")
        ax.plot(t, relative_normalize(I), lw=1.00, label="mean I", color="#2CA02C")

        ax.set_ylabel("Relative normalized change", fontsize=15)
        ax.set_xlabel("Time", fontsize=16)

        beautify_axis(ax)
        set_time_xlim_tight(ax, t)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

        expand_ylim_for_legend(ax, top=0.22, bottom=0.04)

        ax.legend(
            loc="upper right",
            ncol=3,
            fontsize=11.5,
            frameon=True,
            framealpha=0.94,
            edgecolor="0.72",
            borderpad=0.30,
            handlelength=1.6,
            handletextpad=0.5,
            columnspacing=0.7
        )

        fig.subplots_adjust(
            left=0.16,
            right=0.90,
            top=0.965,
            bottom=0.25
        )

        save_or_show(fig, cfg, out_name)

    def save_score_plot(
        a: int,
        b: int,
        out_name: str,
        shade_event: bool = False,
        fixed_ylim: Optional[Tuple[float, float]] = None
    ):
        seg = df_raw.iloc[a:b].copy()
        t = pd.to_datetime(seg["time"]).reset_index(drop=True)
        score_local = score[a:b]

        fig, ax = plt.subplots(figsize=(6.2, 2.8))

        if shade_event:
            shade_event_region(ax)

        ax.plot(t, score_local, color="#2F6B9A", lw=1.35, label="Anomaly score")
        ax.axhline(threshold, ls="--", color="#C0392B", lw=1.25, label="Threshold")

        pred = score_local > threshold
        if pred.any():
            ax.scatter(
                t[pred],
                score_local[pred],
                s=16,
                color="#C0392B",
                zorder=3
            )

        ax.set_ylabel("Anomaly score", fontsize=16)
        ax.set_xlabel("Time", fontsize=16)

        beautify_axis(ax)
        set_time_xlim_tight(ax, t)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

        if fixed_ylim is not None:
            ax.set_ylim(fixed_ylim)
        else:
            expand_ylim_for_legend(ax, top=0.22, bottom=0.04)

        ax.legend(
            loc="upper left",
            ncol=2,
            fontsize=11.5,
            frameon=True,
            framealpha=0.94,
            edgecolor="0.72",
            borderpad=0.30,
            handlelength=1.6,
            handletextpad=0.5,
            columnspacing=0.8
        )

        fig.subplots_adjust(
            left=0.16,
            right=0.90,
            top=0.965,
            bottom=0.27
        )

        save_or_show(fig, cfg, out_name)

    # ============================================================
    # 6. save four subfigures
    # ============================================================
    save_traj_plot(
        normal_a,
        normal_b,
        out_name=f"{base}_normal_traj.png",
        shade_event=False
    )

    save_score_plot(
        normal_a,
        normal_b,
        out_name=f"{base}_normal_score.png",
        shade_event=False,
        fixed_ylim=(0.0, 2.0)
    )

    save_traj_plot(
        struct_a,
        struct_b,
        out_name=f"{base}_structural_traj.png",
        shade_event=True
    )

    save_score_plot(
        struct_a,
        struct_b,
        out_name=f"{base}_structural_score.png",
        shade_event=True
    )


def plot_event_residual_contribution(result: Dict[str, np.ndarray], events_df: pd.DataFrame, labels: np.ndarray, cfg: Config, filename: str) -> None:
    """Publication-style event-by-subspace residual fingerprint map.

    Rows denote injected events A1--A5; columns denote mechanism-guided residual subspaces.
    Cell color denotes row-normalized residual contribution within each event.
    """
    if events_df is None or len(events_df) == 0:
        return
    names = list(result["subspace_names"])
    vals = []
    event_labels = []
    for _, row in events_df.iterrows():
        s_idx, e_idx = int(row["start_idx"]), int(row["end_idx"])
        vals.append(np.nanpercentile(result["r_all"][s_idx:e_idx], 90, axis=0))
        short = str(row.get("short", ""))
        mech = str(row.get("mechanism", ""))
        event_labels.append(f"{short}\n{mech}")
    vals = np.asarray(vals, dtype=float)
    vals_pct = vals / np.maximum(vals.sum(axis=1, keepdims=True), 1e-8)

    pd.DataFrame(vals_pct, columns=[f"r_{n}" for n in names], index=event_labels).to_csv(
        ensure_output_dir(cfg) / "event_residual_contribution_matrix.csv", encoding="utf-8-sig"
    )

    cols = [f"r$_{{{n}}}$" for n in names]
    n_events, n_sub = vals_pct.shape
    cmap = matplotlib.colormaps["YlGnBu"]
    vmax = max(0.40, float(np.nanmax(vals_pct)))
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ax.set_xlim(-0.5, n_sub - 0.5)
    ax.set_ylim(n_events - 0.5, -0.5)

    for i in range(n_events):
        for j in range(n_sub):
            val = float(vals_pct[i, j])
            color = cmap(norm(val))
            rect = patches.FancyBboxPatch(
                (j - 0.43, i - 0.37), 0.86, 0.74,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=color, edgecolor="white", linewidth=1.2
            )
            ax.add_patch(rect)
            text_color = "white" if norm(val) > 0.62 else "#1A1A1A"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=12.0,
                    color=text_color, fontweight="bold" if val >= np.nanpercentile(vals_pct, 75) else "normal")

    ax.set_xticks(np.arange(n_sub))
    ax.set_xticklabels(cols, fontsize=12)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", pad=8, length=0)
    ax.set_yticks(np.arange(n_events))
    ax.set_yticklabels(event_labels, fontsize=12.0)
    ax.tick_params(axis="y", length=0)
    #ax.set_xlabel("Mechanism-guided residual subspace", labelpad=24, fontsize=18)
    #ax.set_ylabel("Injected anomaly event", labelpad=16, fontsize=18)
    #ax.set_title("Event-level residual fingerprint by physical subspace", pad=40, fontsize=12.6)

    for x in np.arange(0.5, n_sub - 0.5, 1.0):
        ax.axvline(x, color="0.92", lw=0.8, zorder=0)
    for y in np.arange(0.5, n_events - 0.5, 1.0):
        ax.axhline(y, color="0.92", lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.035)
    cbar.set_label("Relative residual contribution", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    fig.subplots_adjust(left=0.17, right=0.91, top=0.82, bottom=0.16)
    save_or_show(fig, cfg, filename)


def plot_event_edge_response(
    result: Dict[str, np.ndarray],
    events_df: pd.DataFrame,
    cfg: Config,
    filename: str
) -> None:
    """
    Save five independent MQ-RGA residual-relation snapshots.
    Output files:
        fig06_event_edge_response_A1.png/.pdf
        fig06_event_edge_response_A2.png/.pdf
        ...
        fig06_event_edge_response_A5.png/.pdf
        fig06_event_edge_response_legend.png/.pdf
    """
    if events_df is None or len(events_df) == 0:
        return

    edge_pairs = [(0, 4), (1, 4), (2, 4), (1, 3), (2, 3)]
    edge_names = ["T-ET", "U-ET", "I-ET", "U-B", "I-B"]
    node_names = list(result["subspace_names"])

    r = result["r_all"]
    edge_resp = result.get("edge_response", None)

    if edge_resp is None:
        edge = result["edge_attention"]
        edge_resp = np.zeros((len(edge), len(edge_pairs)), dtype=float)
        for k, (i, j) in enumerate(edge_pairs):
            edge_resp[:, k] = edge[:, k] * 0.5 * (r[:, i] + r[:, j])

    node_mat, edge_mat, event_tags = [], [], []

    for _, row in events_df.iterrows():
        s_idx, e_idx = int(row["start_idx"]), int(row["end_idx"])

        node_mat.append(np.nanpercentile(r[s_idx:e_idx], 90, axis=0))
        edge_mat.append(np.nanpercentile(edge_resp[s_idx:e_idx], 90, axis=0))

        short = str(row.get("short", ""))
        mech = str(row.get("mechanism", ""))
        event_tags.append((short, mech))

    node_mat = np.asarray(node_mat, dtype=float)
    edge_mat = np.asarray(edge_mat, dtype=float)

    node_norm = np.clip(
        node_mat / max(np.nanpercentile(node_mat, 95), 1e-8),
        0, 1
    )
    edge_norm = np.clip(
        edge_mat / max(np.nanpercentile(edge_mat, 95), 1e-8),
        0, 1
    )

    pd.DataFrame(
        edge_norm,
        columns=edge_names,
        index=[f"{s}: {m}" for s, m in event_tags]
    ).to_csv(
        ensure_output_dir(cfg) / "event_edge_response_matrix.csv",
        encoding="utf-8-sig"
    )

    pos = {
        "T":  (0.00, 1.06),
        "U":  (-1.12, 0.18),
        "I":  (-0.92, -0.78),
        "B":  (0.48, -0.88),
        "ET": (1.12, 0.18),
    }

    cmap = matplotlib.colormaps["YlOrRd"]

    base = Path(filename).stem

    def draw_single_snapshot(ev_idx: int) -> None:
        short, mech = event_tags[ev_idx]

        fig, ax = plt.subplots(figsize=(2.35, 2.10))
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.set_xlim(-1.42, 1.42)
        ax.set_ylim(-1.10, 1.24)

        # edges
        for edge_idx, (i, j) in enumerate(edge_pairs):
            ni, nj = node_names[i], node_names[j]
            x0, y0 = pos[ni]
            x1, y1 = pos[nj]

            val = float(edge_norm[ev_idx, edge_idx])
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=cmap(0.12 + 0.86 * val),
                lw=1.10 + 2.20 * val,
                alpha=0.96,
                solid_capstyle="round",
                zorder=1
            )

        # nodes
        for n_idx, name in enumerate(node_names):
            x0, y0 = pos[name]
            val = float(node_norm[ev_idx, n_idx])

            ax.scatter(
                [x0], [y0],
                s=340,
                color=cmap(0.12 + 0.86 * val),
                edgecolor="white",
                linewidth=1.15,
                zorder=3
            )

            ax.text(
                x0, y0,
                name,
                ha="center",
                va="center",
                fontsize=8.4,
                fontweight="bold",
                color="black" if val < 0.62 else "white",
                zorder=4
            )

        # only keep small inner label, not a top title
        ax.text(
            -1.34, 1.16,
            short,
            ha="left",
            va="top",
            fontsize=8.6,
            fontweight="bold",
            color="0.15"
        )

        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

        out_name = f"{base}_{short}.png"
        save_or_show(fig, cfg, out_name)

    for ev_idx in range(len(event_tags)):
        draw_single_snapshot(ev_idx)

    # shared legend
    fig, ax = plt.subplots(figsize=(3.2, 0.62))
    ax.axis("off")

    grad = np.linspace(0, 1, 256).reshape(1, -1)
    cax = ax.inset_axes([0.08, 0.44, 0.82, 0.20])
    cax.imshow(grad, aspect="auto", cmap=cmap, origin="lower")
    cax.set_xticks([0, 128, 255])
    cax.set_xticklabels(["low", "medium", "high"], fontsize=7.6)
    cax.set_yticks([])

    for spine in cax.spines.values():
        spine.set_visible(False)

    ax.text(
        0.08, 0.08,
        "Node color: residual strength; edge color/width: relation response",
        transform=ax.transAxes,
        fontsize=7.4,
        ha="left",
        va="bottom"
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
    save_or_show(fig, cfg, f"{base}_legend.png")

def _event_window_scores(score: np.ndarray, labels: np.ndarray, events_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return matched normal-window and injected-event scores using the 95th percentile per window."""
    normal_scores, event_scores = [], []
    n = len(score)
    for _, row in events_df.iterrows():
        s_idx, e_idx = int(row["start_idx"]), int(row["end_idx"])
        length = max(e_idx - s_idx, 1)
        event_scores.append(float(np.nanpercentile(score[s_idx:e_idx], 95)))
        candidates = [(max(0, s_idx - length), s_idx), (e_idx, min(n, e_idx + length))]
        for a, b in candidates:
            if b - a >= max(5, length // 3) and labels[a:b].sum() == 0:
                normal_scores.append(float(np.nanpercentile(score[a:b], 95)))
    return np.asarray(normal_scores, dtype=float), np.asarray(event_scores, dtype=float)


def plot_score_distribution(result: Dict[str, np.ndarray], labels: np.ndarray, threshold: float, cfg: Config, filename: str,
                            eval_mask: Optional[np.ndarray] = None, events_df: Optional[pd.DataFrame] = None) -> None:
    """Raincloud-style event-level score separability plot."""
    score = result["score"]
    if events_df is not None and len(events_df) > 0:
        normal, abnormal = _event_window_scores(score, labels, events_df)
        if len(normal) == 0:
            score_eval = score if eval_mask is None else score[eval_mask]
            lab = labels if eval_mask is None else labels[eval_mask]
            normal = score_eval[lab == 0]
            abnormal = score_eval[lab == 1]
        pd.DataFrame({
            "Group": ["Normal context"] * len(normal) + ["Injected anomaly"] * len(abnormal),
            "WindowScore": np.concatenate([normal, abnormal]) if len(normal) + len(abnormal) > 0 else []
        }).to_csv(ensure_output_dir(cfg) / "event_window_score_separability.csv", index=False, encoding="utf-8-sig")
        ylabel = "Window-level anomaly score (95th percentile)"
        #title = "Score separability in controlled event windows"
    else:
        score_eval = score if eval_mask is None else score[eval_mask]
        lab = labels if eval_mask is None else labels[eval_mask]
        normal = score_eval[lab == 0]
        abnormal = score_eval[lab == 1]
        ylabel = "Point-level anomaly score"
        title = "Point-level score separability"

    data = [np.asarray(normal, dtype=float), np.asarray(abnormal, dtype=float)]
    all_scores = np.concatenate([x for x in data if len(x) > 0]) if any(len(x) > 0 for x in data) else np.asarray([threshold])
    upper = max(float(np.nanpercentile(all_scores, 99.0)), threshold * 1.2, 1e-6)
    data_clip = [np.clip(x, 0, upper) for x in data]
    colors = ["#4E79A7", "#E15759"]
    labels_txt = ["Normal context", "Injected anomaly"]

    def kde_manual(arr, grid):
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0:
            return np.zeros_like(grid)
        std = max(float(np.nanstd(arr)), 1e-6)
        bw = max(1.06 * std * (len(arr) ** (-1/5)), upper / 80.0, 1e-6)
        z = (grid[:, None] - arr[None, :]) / bw
        den = np.exp(-0.5 * z * z).sum(axis=1) / (len(arr) * bw * np.sqrt(2 * np.pi))
        return den

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ygrid = np.linspace(0, upper, 240)
    rng = np.random.default_rng(123)
    for idx, arr in enumerate(data_clip):
        pos = idx + 1
        den = kde_manual(arr, ygrid)
        den = den / max(den.max(), 1e-8) * 0.34
        # Half violin cloud on the right side.
        ax.fill_betweenx(ygrid, pos, pos + den, color=colors[idx], alpha=0.35, lw=0)
        if len(arr):
            jitter = rng.normal(-0.10, 0.035, size=len(arr))
            ax.scatter(np.full(len(arr), pos) + jitter, arr, s=28, alpha=0.70,
                       color=colors[idx], edgecolor="white", linewidth=0.4, zorder=3)
            q1, med, q3 = np.nanpercentile(arr, [25, 50, 75])
            lo, hi = np.nanpercentile(arr, [5, 95])
            ax.plot([pos - 0.03, pos + 0.24], [med, med], color="black", lw=1.4, zorder=4)
            ax.add_patch(plt.Rectangle((pos + 0.03, q1), 0.18, q3 - q1,
                                       facecolor="white", edgecolor="black", lw=0.9, alpha=0.85, zorder=4))
            ax.plot([pos + 0.12, pos + 0.12], [lo, hi], color="black", lw=0.9, zorder=4)
    ax.axhline(min(threshold, upper), linestyle="--", linewidth=1.15, color="#C0392B", label="Threshold")
    ax.text(
        0.97,
        0.88,
        " ",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=18,
        color="#C0392B",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88),
        zorder=5
    )
    ax.set_xlim(0.55, 2.65)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels_txt)
    ax.set_ylabel(ylabel)
    #ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    save_or_show(fig, cfg, filename)


def plot_threshold_sensitivity(result: Dict[str, np.ndarray], labels: np.ndarray, train_score: np.ndarray, cfg: Config, filename: str,
                               eval_mask: Optional[np.ndarray] = None) -> None:
    qs = np.array([0.900, 0.930, 0.950, 0.970, 0.980, 0.990])
    y = labels if eval_mask is None else labels[eval_mask]
    score = result["score"] if eval_mask is None else result["score"][eval_mask]
    rows = []
    for q in qs:
        th = choose_threshold(train_score, float(q))
        m = compute_metrics(y, score, th)
        m["Quantile"] = q
        rows.append(m)
    sens = pd.DataFrame(rows)
    sens.to_csv(ensure_output_dir(cfg) / "threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(sens["Quantile"], sens["Precision"], marker="o", label="Precision")
    ax.plot(sens["Quantile"], sens["Recall"], marker="o", label="Recall")
    ax.plot(sens["Quantile"], sens["F1"], marker="o", label="F1")
    ax.set_xlabel("Reference-score threshold quantile")
    ax.set_ylabel("Metric")
    ax.set_ylim(-0.02, 1.02)
    #ax.set_title("Threshold sensitivity in controlled injected-anomaly validation")
    ax.legend()
    fig.tight_layout()
    save_or_show(fig, cfg, filename)




def plot_ablation_rose(metrics_df: pd.DataFrame, cfg: Config, filename: str) -> None:
    """
    Wind-rose style ablation chart.

    Each sector is one MSDS-Net variant. Sector radius indicates event-adjusted F1.
    Stacked color bands indicate F1 intervals, similar to a wind-rose plot.
    """
    df = metrics_df.copy()
    variants = df["Variant"].tolist()
    f_col = "AdjF1" if "AdjF1" in df.columns else "F1"
    f1 = df[f_col].fillna(0).to_numpy(dtype=float)

    n = len(df)
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    sector_width = 2 * np.pi / max(n, 1)
    width = sector_width * 0.96

    # Magnify the high-score region. If all variants are high, start at 0.70.
    min_f = float(np.nanmin(f1))
    r0 = 0.0
    rmax = 1.00

    # Stacked F1 bands, following a wind-rose-like visual language.
    step = 0.05
    levels = np.arange(r0, rmax + step, step)
    if levels[-1] < rmax:
        levels = np.append(levels, rmax)

    cmap = matplotlib.colormaps["Spectral_r"]
    norm = matplotlib.colors.Normalize(vmin=0.65, vmax=1)

    fig = plt.figure(figsize=(8.8, 7.8))
    ax = fig.add_axes([0.06, 0.06, 0.74, 0.84], projection="polar")

    # Put FULL at the top and arrange variants clockwise.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for i, val in enumerate(f1):
        val_clip = float(np.clip(val, r0, rmax))

        # Stack each sector by F1 interval.
        for lower, upper in zip(levels[:-1], levels[1:]):
            seg_top = min(val_clip, upper)
            if seg_top <= lower:
                continue
            ax.bar(
                theta[i],
                seg_top - lower,
                bottom=lower,
                width=width,
                color=cmap(norm(np.clip((lower + seg_top) / 2, 0.65, 1.00))),
                edgecolor="white",
                linewidth=0.25,
                align="center",
                alpha=0.96,
            )

        # Uniform outline for all sectors; FULL is not highlighted by a special border.
        ax.bar(
            theta[i],
            max(val_clip - r0, 0.0),
            bottom=r0,
            width=width,
            facecolor="none",
            edgecolor="0.20",
            linewidth=0.55,
            align="center",
        )

        # Value inside the sector; avoids overlap with outer labels.
        value_r = min(max(val_clip - 0.035, 0.12), 0.96)
        ax.text(
            theta[i],
            value_r,
            f"{val:.3f}", #################################################################
            ha="center",
            va="center",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.78),
        )

    # Unified wrapped labels.
    label_map = {
        "FULL": "FULL",
        "w/o Subspace": "w/o\nSubspace",
        "w/o Separation": "w/o\nSeparation",
        "w/o MQ-RGA": "w/o\nMQ-RGA",
        "w/o Gate": "w/o\nGate",
        "w/o EdgeBias": "w/o\nEdgeBias",
    }
    labels = [label_map.get(v, v.replace(" ", "\n")) for v in variants]
    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=9.4)
    ax.tick_params(axis="x", pad=14)

    ax.set_ylim(r0, rmax)

    # Keep only key reference rings; no center baseline annotation.
    yticks = [0.25, 0.50, 0.75, 0.90, 1.00]
    ax.set_yticks(yticks)
    ax.set_yticklabels(["", "", "", "", "1.00"], fontsize=8.5)
    ax.set_rlabel_position(72)

    ax.grid(color="0.78", linestyle="--", linewidth=0.75, alpha=0.85)
    ax.spines["polar"].set_linewidth(0.9)
    ax.set_title(
        "Ablation study: event-adjusted F1 of MSDS-Net variants",
        pad=14,
        fontsize=12.5,
    )

    # Colorbar for F1 bands, similar to wind-rose legends.
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.86, 0.25, 0.026, 0.48])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Event-adjusted F1 level", fontsize=9)
    ticks = [0.65, 0.75, 0.85, 0.95, 1.00]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
    cbar.ax.tick_params(labelsize=8.5)


    ####fig.subplots_adjust(left=0.10, right=0.84, top=0.88, bottom=0.09)
    save_or_show(fig, cfg, filename)


def plot_ablation_bar(metrics_df: pd.DataFrame, cfg: Config, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(metrics_df))
    width = 0.25
    p_col = "AdjPrecision" if "AdjPrecision" in metrics_df.columns else "Precision"
    r_col = "AdjRecall" if "AdjRecall" in metrics_df.columns else "Recall"
    f_col = "AdjF1" if "AdjF1" in metrics_df.columns else "F1"
    ax.bar(x - width, metrics_df[p_col], width, label="Precision")
    ax.bar(x, metrics_df[r_col], width, label="Recall")
    ax.bar(x + width, metrics_df[f_col], width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["Variant"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric")
    #ax.set_title("Ablation study of MSDS-Net components")
    ax.legend(ncol=3)
    fig.tight_layout()
    save_or_show(fig, cfg, filename)



def plot_ablation_degradation(metrics_df: pd.DataFrame, cfg: Config, filename: str) -> None:
    """Radial degradation map: performance drop after removing each component."""
    if metrics_df.empty or "FULL" not in set(metrics_df["Variant"]):
        return
    df = metrics_df.copy()
    f_col = "AdjF1" if "AdjF1" in df.columns else "F1"
    full_val = float(df.loc[df["Variant"] == "FULL", f_col].iloc[0])
    sub = df[df["Variant"] != "FULL"].copy()
    sub["Drop"] = full_val - sub[f_col].astype(float)
    sub.to_csv(ensure_output_dir(cfg) / "ablation_degradation_from_full.csv", index=False, encoding="utf-8-sig")

    variants = sub["Variant"].tolist()
    drops = sub["Drop"].to_numpy(dtype=float)
    n = len(drops)
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / max(n, 1) * 0.72
    max_abs = max(float(np.nanmax(np.abs(drops))), 1e-6)
    rmax = max_abs * 1.35
    fig = plt.figure(figsize=(7.2, 6.2))
    ax = plt.subplot(111, projection="polar")
    colors = ["#2E86AB" if d >= 0 else "#C0392B" for d in drops]
    ax.bar(theta, np.abs(drops), width=width, bottom=0.0, color=colors, alpha=0.86,
           edgecolor="white", linewidth=1.1)
    ax.set_ylim(0, rmax)
    ax.set_xticks(theta)
    ax.set_xticklabels(variants, fontsize=9)
    ax.tick_params(axis='x', pad=15)
    tick_vals = np.linspace(0, rmax, 4)[1:]
    ax.set_yticks(tick_vals)
    ax.set_yticklabels([f"{t:.2f}" for t in tick_vals], fontsize=8)
    #ax.set_title(f"Ablation degradation from FULL ({f_col})", pad=28)
    for ang, drop in zip(theta, drops):
        ax.text(ang, min(abs(drop) + rmax * 0.10, rmax * 0.96), f"{drop:+.2f}",
                ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
    ax.text(0.5, 0.5, f"FULL\n{full_val:.2f}", transform=ax.transAxes,
            ha="center", va="center", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.35", fc="#F7F7F7", ec="0.6", alpha=0.95))
    fig.tight_layout()
    save_or_show(fig, cfg, filename)


def make_model_for_variant(variant: str, subspace_cols: Dict[str, List[str]], feature_cols: List[str], cfg: Config, device: torch.device) -> Tuple[MSDSNet, Dict[str, List[str]]]:
    """Create model and subspace definition for a given ablation variant."""
    if variant == "w/o Subspace":
        # Fair ablation for removing mechanism-guided subspaces:
        # use only raw measurements as one direct-concatenation branch.
        # Engineered mechanism variables (U_mean, I_mean, B_U, B_I, P, dT, dP) are
        # intentionally excluded here; otherwise the no-subspace variant would still
        # retain much of the mechanism-guided feature construction.
        raw_cols = ["T", "U51", "U52", "U53", "I54", "I55", "I56"]
        cols = {"ALL": raw_cols}
        model = MSDSNet({"ALL": len(raw_cols)}, cfg.hidden_dim, cfg.graph_dim, cfg.dropout,
                        cfg.graph_score_weight, use_separation=True, use_graph=False)
    else:
        cols = subspace_cols
        dims = {k: len(v) for k, v in cols.items()}
        if variant == "FULL":
            model = MSDSNet(dims, cfg.hidden_dim, cfg.graph_dim, cfg.dropout, cfg.graph_score_weight,
                            use_separation=True, use_graph=True, use_gate=True, use_edge_bias=True)
        elif variant == "w/o Separation":
            model = MSDSNet(dims, cfg.hidden_dim, cfg.graph_dim, cfg.dropout, cfg.graph_score_weight,
                            use_separation=False, use_graph=True, use_gate=True, use_edge_bias=True)
        elif variant == "w/o MQ-RGA":
            model = MSDSNet(dims, cfg.hidden_dim, cfg.graph_dim, cfg.dropout, cfg.graph_score_weight,
                            use_separation=True, use_graph=False, use_gate=True, use_edge_bias=True)
        elif variant == "w/o Gate":
            model = MSDSNet(dims, cfg.hidden_dim, cfg.graph_dim, cfg.dropout, cfg.graph_score_weight,
                            use_separation=True, use_graph=True, use_gate=False, use_edge_bias=True)
        elif variant == "w/o EdgeBias":
            model = MSDSNet(dims, cfg.hidden_dim, cfg.graph_dim, cfg.dropout, cfg.graph_score_weight,
                            use_separation=True, use_graph=True, use_gate=True, use_edge_bias=False)
        else:
            raise ValueError(f"Unknown variant: {variant}")
    return model.to(device), cols


def build_loaders_for_arrays(arrays_train: Dict[str, np.ndarray], arrays_eval: Dict[str, np.ndarray], train_end: int, cfg: Config, device: torch.device):
    n = len(next(iter(arrays_train.values())))
    train_starts = make_start_indices(n, cfg.seq_len, cfg.stride, end_limit=train_end)
    all_starts = make_start_indices(n, cfg.seq_len, cfg.stride, end_limit=None)
    train_loader = DataLoader(WindowDataset(arrays_train, train_starts, cfg.seq_len), batch_size=cfg.batch_size,
                              shuffle=True, drop_last=False, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
    eval_loader = DataLoader(WindowDataset(arrays_eval, all_starts, cfg.seq_len), batch_size=cfg.batch_size,
                             shuffle=False, drop_last=False, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
    train_eval_loader = DataLoader(WindowDataset(arrays_train, all_starts, cfg.seq_len), batch_size=cfg.batch_size,
                                   shuffle=False, drop_last=False, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
    return train_loader, eval_loader, train_eval_loader


def save_result_csv(df_eval: pd.DataFrame, result: Dict[str, np.ndarray], threshold: float, cfg: Config, filename: str, labels: Optional[np.ndarray] = None) -> None:
    out = pd.DataFrame({
        "time": df_eval["time"],
        "T": df_eval["T"],
        "score": result["score"],
        "graph_score": result["graph_score"],
        "threshold": threshold,
        "pred_anomaly": (result["score"] > threshold).astype(int),
    })
    if labels is not None:
        out["label"] = labels.astype(int)
    for i, name in enumerate(result["subspace_names"]):
        out[f"r_{name}"] = result["r_all"][:, i]
    edge_names = ["T_ET", "U_ET", "I_ET", "U_B", "I_B"]
    for i, name in enumerate(edge_names):
        out[f"att_{name}"] = result["edge_attention"][:, i]
    path = ensure_output_dir(cfg) / filename
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved result CSV: {path}")


def save_event_level_metrics(
    events_df: pd.DataFrame,
    labels: np.ndarray,
    result: Dict[str, np.ndarray],
    threshold: float,
    cfg: Config,
    variant: str,
) -> None:
    """Save one row per injected event to diagnose which mechanism is or is not detected."""
    if events_df is None or len(events_df) == 0:
        return
    rows = []
    score = result["score"]
    pred = (score > threshold).astype(int)
    names = list(result["subspace_names"])
    for _, row in events_df.iterrows():
        s, e = int(row["start_idx"]), int(row["end_idx"])
        loc = slice(s, e)
        det_ratio = float(pred[loc].mean())
        score_mean = float(score[loc].mean())
        score_max = float(score[loc].max())
        item = {
            "Variant": variant,
            "Event": row.get("short", ""),
            "Mechanism": row.get("mechanism", ""),
            "DetectionRatio": det_ratio,
            "MeanScore": score_mean,
            "MaxScore": score_max,
        }
        for i, n in enumerate(names):
            item[f"Mean_r_{n}"] = float(result["r_all"][loc, i].mean())
        rows.append(item)
    path = ensure_output_dir(cfg) / f"{variant.replace('/', 'without').replace(' ', '_')}_event_level_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved event-level metrics: {path}")


# =========================
# 9. Main pipeline
# =========================


# ============================================================
# ============================================================
# V4.0 — Protocol v2.1 / B1-R5 / frozen 10-event benchmark
# ============================================================
#
# Exact benchmark source:
#   - event locations/durations/severities: Experiment 25 selected_10_events
#   - temporal profile: Experiment 30B B1_R5
#   - A1: original temporal mechanism
#   - A2-A5: 5-min half-cosine ramp + stable plateau
#   - event strengths: B1_R5 recalibrated multipliers at frozen TargetGamma
#   - threshold: pure Q0.99 on clean 0%-55%
#   - evaluation: complete 55%-70% internal validation interval
#   - final 30% is physically excluded
#
# IMPORTANT PROVENANCE:
# Runtime injection does NOT consult model scores, labels, thresholds, EventRecall,
# or Delay. The frozen locations themselves originate from the Exp25 internal
# completion/design stage and are not re-selected during formal reruns.

PROTOCOL_VERSION_V4 = "MSDSNET_REVISION_EXPERIMENT_PROTOCOL_v2.1_B1_R5"
FROZEN_EVENT_SOURCE = "EXP25_SELECTED_10_EVENTS"
TEMPORAL_PROFILE = "B1_R5"
RAMP_MIN = 5
MERGE_GAP_DESCRIPTIVE_MIN = 120

FROZEN_B1R5_EVENTS = [{'event_id': 'A2_1', 'short': 'A2', 'instance': 1, 'duration_min': 150, 'severity': 0.85, 'start_idx': 23819, 'end_idx': 23969, 'target_gamma': 4.0, 'amplitude_multiplier': 34.97298551612349, 'actual_effect_q90': 3.991308998758543, 'event_max_injected_abs_z': 9.52679088831246}, {'event_id': 'A4_2', 'short': 'A4', 'instance': 2, 'duration_min': 90, 'severity': 1.15, 'start_idx': 24394, 'end_idx': 24484, 'target_gamma': 4.0, 'amplitude_multiplier': 27.162298437613043, 'actual_effect_q90': 3.957648552222099, 'event_max_injected_abs_z': 8.331536553843684}, {'event_id': 'A1_1', 'short': 'A1', 'instance': 1, 'duration_min': 90, 'severity': 0.85, 'start_idx': 25409, 'end_idx': 25499, 'target_gamma': 4.0, 'amplitude_multiplier': 37.262646539273575, 'actual_effect_q90': 4.000000000000001, 'event_max_injected_abs_z': 8.48963711349021}, {'event_id': 'A3_1', 'short': 'A3', 'instance': 1, 'duration_min': 90, 'severity': 0.85, 'start_idx': 26716, 'end_idx': 26806, 'target_gamma': 4.0, 'amplitude_multiplier': 17.17112183530599, 'actual_effect_q90': 4.222033989449009, 'event_max_injected_abs_z': 9.064899386126704}, {'event_id': 'A1_2', 'short': 'A1', 'instance': 2, 'duration_min': 150, 'severity': 1.15, 'start_idx': 27799, 'end_idx': 27949, 'target_gamma': 4.0, 'amplitude_multiplier': 27.47117001509355, 'actual_effect_q90': 4.000000000000004, 'event_max_injected_abs_z': 7.816006955471749}, {'event_id': 'A3_2', 'short': 'A3', 'instance': 2, 'duration_min': 150, 'severity': 1.15, 'start_idx': 28239, 'end_idx': 28389, 'target_gamma': 4.0, 'amplitude_multiplier': 15.951121971155038, 'actual_effect_q90': 3.873809969378731, 'event_max_injected_abs_z': 7.127361835302503}, {'event_id': 'A4_1', 'short': 'A4', 'instance': 1, 'duration_min': 150, 'severity': 0.85, 'start_idx': 28665, 'end_idx': 28815, 'target_gamma': 4.0, 'amplitude_multiplier': 22.11649956210145, 'actual_effect_q90': 3.9401083049818406, 'event_max_injected_abs_z': 8.240699607808613}, {'event_id': 'A5_1', 'short': 'A5', 'instance': 1, 'duration_min': 90, 'severity': 0.85, 'start_idx': 29419, 'end_idx': 29509, 'target_gamma': 4.0, 'amplitude_multiplier': 57.7508157906105, 'actual_effect_q90': 3.964146405571789, 'event_max_injected_abs_z': 12.477007992788154}, {'event_id': 'A2_2', 'short': 'A2', 'instance': 2, 'duration_min': 90, 'severity': 1.15, 'start_idx': 29734, 'end_idx': 29824, 'target_gamma': 4.0, 'amplitude_multiplier': 18.940231722020343, 'actual_effect_q90': 3.98898545135976, 'event_max_injected_abs_z': 7.637325406557952}, {'event_id': 'A5_2', 'short': 'A5', 'instance': 2, 'duration_min': 150, 'severity': 1.15, 'start_idx': 30004, 'end_idx': 30154, 'target_gamma': 5.0, 'amplitude_multiplier': 52.0037122690956, 'actual_effect_q90': 4.929473667309277, 'event_max_injected_abs_z': 13.908095353291706}]


def _reference_amplitudes_v4(df_raw: pd.DataFrame, train_end: int) -> Dict[str, float]:
    """Reference amplitudes from the original injection mechanism, fitted on clean 0%-55% only."""
    ref = df_raw.iloc[:int(train_end)].copy()

    t_scale = robust_scale_value(ref["T"])
    u_scale = float(np.nanmedian([
        robust_scale_value(ref[c]) for c in ["U51", "U52", "U53"]
    ]))
    i_scale = float(np.nanmedian([
        robust_scale_value(ref[c]) for c in ["I54", "I55", "I56"]
    ]))

    t_med = max(float(np.nanmedian(ref["T"].values)), 1.0)
    u_med = max(float(np.nanmedian(ref[["U51","U52","U53"]].values)), 1.0)
    i_med = max(float(np.nanmedian(ref[["I54","I55","I56"]].values)), 1.0)

    return {
        "t_amp": float(max(1.10 * t_scale, 0.035 * t_med)),
        "u_amp": float(max(1.05 * u_scale, 0.020 * u_med)),
        "i_amp": float(max(1.05 * i_scale, 0.020 * i_med)),
    }


def _original_smooth_step_v4(m: int) -> np.ndarray:
    """Original logistic-like temporal envelope retained for A1."""
    z = np.linspace(-3.0, 3.0, int(m))
    y = 1.0 / (1.0 + np.exp(-z))
    return (y - y.min()) / max(float(y.max() - y.min()), 1e-8)


def _fixed_ramp_plateau_v4(duration: int, ramp_min: int = 5) -> np.ndarray:
    """Half-cosine ramp over exactly ramp_min samples, then a stable plateau at 1."""
    d = int(duration)
    r = min(int(ramp_min), d)
    if d <= 0:
        raise ValueError(f"Invalid duration: {d}")
    if r <= 0:
        return np.ones(d, dtype=np.float64)

    w = np.ones(d, dtype=np.float64)
    if r == 1:
        w[0] = 1.0
        return w

    x = np.linspace(0.0, 1.0, r, dtype=np.float64)
    w[:r] = 0.5 * (1.0 - np.cos(np.pi * x))
    return w


def _inject_one_b1r5_event(
    df: pd.DataFrame,
    event: Dict[str, float],
    amps: Dict[str, float],
) -> None:
    short = str(event["short"])
    s = int(event["start_idx"])
    e = int(event["end_idx"])
    duration = int(event["duration_min"])
    if e - s != duration:
        raise RuntimeError(f"Frozen event duration mismatch for {event['event_id']}.")

    eta = float(event["severity"]) * float(event["amplitude_multiplier"])

    # B1: A1 retains the exact original temporal mechanism.
    if short == "A1":
        w = _original_smooth_step_v4(duration)
        df.loc[s:e-1, "T"] -= eta * amps["t_amp"] * (0.25 + 0.75 * w)

        rs = s + int(0.62 * duration)
        rw = _original_smooth_step_v4(e - rs)
        df.loc[rs:e-1, "T"] += 0.35 * eta * amps["t_amp"] * rw
        return

    # R5: A2-A5 use 5-min half-cosine transition + stable plateau.
    w = _fixed_ramp_plateau_v4(duration, ramp_min=RAMP_MIN)

    if short == "A2":
        df.loc[s:e-1, "U51"] += eta * amps["u_amp"] * w
        df.loc[s:e-1, "U52"] -= 0.85 * eta * amps["u_amp"] * w
        df.loc[s:e-1, "U53"] -= 0.15 * eta * amps["u_amp"] * w

    elif short == "A3":
        df.loc[s:e-1, "I54"] += 0.90 * eta * amps["i_amp"] * w
        df.loc[s:e-1, "I55"] -= 0.75 * eta * amps["i_amp"] * w
        df.loc[s:e-1, "I56"] -= 0.15 * eta * amps["i_amp"] * w

    elif short == "A4":
        df.loc[s:e-1, "U51"] += 0.70 * eta * amps["u_amp"] * w
        df.loc[s:e-1, "U52"] -= 0.35 * eta * amps["u_amp"] * w
        df.loc[s:e-1, "U53"] -= 0.35 * eta * amps["u_amp"] * w
        df.loc[s:e-1, "I54"] -= 0.70 * eta * amps["i_amp"] * w
        df.loc[s:e-1, "I55"] += 0.35 * eta * amps["i_amp"] * w
        df.loc[s:e-1, "I56"] += 0.35 * eta * amps["i_amp"] * w

    elif short == "A5":
        df.loc[s:e-1, ["U51","U52","U53"]] += (
            0.35 * eta * amps["u_amp"] * w[:, None]
        )
        df.loc[s:e-1, ["I54","I55","I56"]] += (
            0.65 * eta * amps["i_amp"] * w[:, None]
        )
        df.loc[s:e-1, "T"] -= 0.55 * eta * amps["t_amp"] * w

    else:
        raise ValueError(f"Unknown frozen event type: {short}")


def inject_synthetic_anomalies(
    df_raw: pd.DataFrame,
    train_end: int,
    cfg: Config,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Exact Protocol-v2.1 B1-R5 ten-event controlled benchmark."""
    df = df_raw.copy()
    n = len(df)
    labels = np.zeros(n, dtype=int)
    amps = _reference_amplitudes_v4(df_raw, train_end)

    mechanism_map = {
        "A1": "T / ET",
        "A2": "U / B",
        "A3": "I / B",
        "A4": "U / I / B",
        "A5": "T / U / I / ET",
    }

    rows = []
    for ev in FROZEN_B1R5_EVENTS:
        s = int(ev["start_idx"])
        e = int(ev["end_idx"])

        if s < int(train_end):
            raise RuntimeError(
                f"Frozen event {ev['event_id']} starts before 55% validation boundary."
            )
        if e > n:
            raise RuntimeError(
                f"Frozen event {ev['event_id']} exceeds the physically retained 0%-70% sequence."
            )

        _inject_one_b1r5_event(df, ev, amps)
        labels[s:e] = 1

        rows.append({
            "EventID": ev["event_id"],
            "type": ev["short"],
            "short": ev["short"],
            "instance": ev["instance"],
            "mechanism": mechanism_map[ev["short"]],
            "start_idx": s,
            "end_idx": e,
            "duration_min": ev["duration_min"],
            "severity": ev["severity"],
            "TargetGamma": ev["target_gamma"],
            "AmplitudeMultiplier": ev["amplitude_multiplier"],
            "ActualEffectQ90_Exp30B": ev["actual_effect_q90"],
            "EventMaxInjectedAbsZ_Exp30B": ev["event_max_injected_abs_z"],
            "start_time": df.loc[s, "time"],
            "end_time": df.loc[e-1, "time"],
            "protocol": PROTOCOL_VERSION_V4,
            "temporal_profile": TEMPORAL_PROFILE,
            "ramp_min": 0 if ev["short"] == "A1" else RAMP_MIN,
            "runtime_model_output_used_for_injection": False,
            "location_source": FROZEN_EVENT_SOURCE,
        })

    events_df = pd.DataFrame(rows).sort_values("start_idx").reset_index(drop=True)

    # Frozen benchmark integrity checks.
    if len(events_df) != 10:
        raise RuntimeError(f"Protocol-v2.1 requires exactly 10 events, got {len(events_df)}.")
    counts = events_df["short"].value_counts().to_dict()
    if any(counts.get(k, 0) != 2 for k in ["A1","A2","A3","A4","A5"]):
        raise RuntimeError(f"Protocol-v2.1 requires two events per type, got {counts}.")
    if int(labels.sum()) != 1200:
        raise RuntimeError(f"Expected 1200 positive validation points, got {int(labels.sum())}.")

    return df, labels, events_df


# overwrite old experimental injector
inject_synthetic_anomalies = inject_synthetic_anomalies

# Final protocol constants
CFG.a_residual_mix = 0.50
CFG.lambda_trend = 1.75
CFG.lambda_dist = 0.10
CFG.lambda_res = 0.0008
CFG.lambda_graph = 0.018
CFG.graph_score_weight = 0.031
CFG.threshold_quantile = 0.99
CFG.seed = 42
CFG.run_ablation = False


SEEDS_10 = [42, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032]


def _sample_std(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return str(obj)


def run_single_seed_full(seed: int,
                         cfg: Config,
                         device: torch.device,
                         scaled_df: pd.DataFrame,
                         injected_scaled_df: pd.DataFrame,
                         injected_feature_df: pd.DataFrame,
                         subspace_cols: Dict[str, List[str]],
                         feature_cols: List[str],
                         train_end: int,
                         n: int,
                         labels: np.ndarray,
                         events_df: pd.DataFrame,
                         eval_mask: np.ndarray,
                         out_dir: Path) -> Dict[str, float]:
    set_seed(seed)
    cfg.seed = int(seed)
    model, variant_subspace_cols = make_model_for_variant("FULL", subspace_cols, feature_cols, cfg, device)
    train_arrays = build_subspace_arrays(scaled_df, variant_subspace_cols)
    eval_arrays = build_subspace_arrays(injected_scaled_df, variant_subspace_cols)
    train_loader, eval_loader, train_eval_loader = build_loaders_for_arrays(train_arrays, eval_arrays, train_end, cfg, device)
    train_model(model, train_loader, cfg, device, epochs=cfg.epochs, title=f"FULL_seed_{seed}")

    train_result = aggregate_scores(model, train_eval_loader, n_points=n, cfg=cfg, device=device, collect_components=False)
    threshold = choose_threshold(
        train_result["score"][:train_end],
        cfg.threshold_quantile
    )
    eval_result = aggregate_scores(model, eval_loader, n_points=n, cfg=cfg, device=device, collect_components=False)

    from scipy.ndimage import median_filter

    eval_result["score"] = median_filter(
        eval_result["score"],
        size=3
    )

    met = compute_event_adjusted_metrics(labels, eval_result["score"], threshold, events_df, eval_mask)
    met["Threshold"] = float(threshold)
    met["EvalPoints"] = int(eval_mask.sum())
    met["Seed"] = int(seed)
    # save per-seed lightweight csv only, no figures
    seed_scores = pd.DataFrame({
        "time": injected_feature_df["time"],
        "label": labels.astype(int),
        "eval_mask": eval_mask.astype(int),
        "score": eval_result["score"],
        "pred": (eval_result["score"] > threshold).astype(int),
        "threshold": float(threshold),
    })
    seed_scores.to_csv(out_dir / f"seed_{seed}_scores.csv", index=False, encoding="utf-8-sig")
    del model, train_loader, eval_loader, train_eval_loader, train_result, eval_result, train_arrays, eval_arrays
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return met


def main() -> None:
    set_paper_style()
    cfg = CFG

    # Formal frozen settings.
    cfg.lambda_trend = 1.75
    cfg.lambda_dist = 0.10
    cfg.lambda_res = 0.0008
    cfg.lambda_graph = 0.018
    cfg.a_residual_mix = 0.8
    cfg.graph_score_weight = 0.031
    cfg.threshold_quantile = 0.99

    cfg.run_ablation = False
    cfg.run_injection_eval = True
    cfg.show_figures = False
    cfg.save_figures = False
    cfg.make_threshold_sensitivity_figure = False
    cfg.output_dir = "02full_v4_protocol_v2p1_B1R5_10event_10seed_outputs"

    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent

    if not Path(cfg.temperature_file).exists():
        alt_t = parent_dir / "930T.csv"
        if alt_t.exists():
            cfg.temperature_file = str(alt_t)

    if not Path(cfg.vi_file).exists():
        alt_v = parent_dir / "930V.csv"
        if alt_v.exists():
            cfg.vi_file = str(alt_v)

    out_dir = ensure_output_dir(cfg)
    device = get_training_device(cfg)

    # ------------------------------------------------------------
    # Protocol-v2.1 physical split
    # clean train/reference = original 0%-55%
    # internal validation   = original 55%-70%
    # final 30%             = physically excluded from this run
    # ------------------------------------------------------------
    raw_full = load_raw_inputs(cfg)
    full_n = len(raw_full)

    train_end = int(full_n * 0.55)
    eval_end = int(full_n * 0.70)
    eval_start = int(train_end)

    raw_df = raw_full.iloc[:eval_end].copy().reset_index(drop=True)
    del raw_full

    n = len(raw_df)

    print(
        f"Loaded full sequence: {full_n} samples | "
        f"retained first 70%: {n} | "
        f"train/reference=[0,{train_end}) | "
        f"validation=[{eval_start},{eval_end}) | "
        f"final30 excluded=[{eval_end},{full_n})",
        flush=True,
    )

    if full_n != 43199:
        print(
            f"[Protocol note] full_n={full_n}; frozen Exp25 indices were derived on the "
            "930 dataset. Integrity checks below will stop the run if intervals are invalid.",
            flush=True,
        )

    feature_df, subspace_cols, feature_cols = add_engineered_features(raw_df)
    scaled_df, scaler = scale_features(
        feature_df, feature_cols, train_end, scaler=None
    )

    injected_raw, labels, events_df = inject_synthetic_anomalies(
        raw_df, train_end, cfg
    )
    injected_feature_df, _, _ = add_engineered_features(injected_raw)
    injected_scaled_df, _ = scale_features(
        injected_feature_df, feature_cols, train_end, scaler=scaler
    )

    # Full 55%-70% internal-validation interval, not local controlled windows.
    eval_mask = np.zeros(n, dtype=bool)
    eval_mask[eval_start:eval_end] = True

    # ------------------------------------------------------------
    # Static protocol audit before any training.
    # ------------------------------------------------------------
    static_model, _ = make_model_for_variant(
        "FULL", subspace_cols, feature_cols, cfg, device
    )
    param_count = int(sum(p.numel() for p in static_model.parameters() if p.requires_grad))
    del static_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    audit = {
        "protocol": PROTOCOL_VERSION_V4,
        "temporal_profile": TEMPORAL_PROFILE,
        "event_count": int(len(events_df)),
        "two_events_per_type": bool(
            all(events_df["short"].value_counts().get(k, 0) == 2 for k in ["A1","A2","A3","A4","A5"])
        ),
        "positive_points": int(labels[eval_mask].sum()),
        "train_0_55": True,
        "validation_55_70": True,
        "final_30_physically_excluded": True,
        "threshold_rule": "pure_Q0.99_clean_reference",
        "rho": float(cfg.a_residual_mix),
        "lambda_trend": float(cfg.lambda_trend),
        "lambda_dist": float(cfg.lambda_dist),
        "lambda_res": float(cfg.lambda_res),
        "lambda_graph": float(cfg.lambda_graph),
        "graph_score_weight": float(cfg.graph_score_weight),
        "parameter_count": int(param_count),
        "expected_parameter_count": 264001,
        "formal_seeds": list(SEEDS_10),
        "runtime_model_output_used_for_injection": False,
        "location_source": FROZEN_EVENT_SOURCE,
        "location_provenance_note": (
            "Frozen Exp25 ten-event locations are reproduced exactly; "
            "V4 performs no location search or model-guided re-selection."
        ),
    }

    if audit["event_count"] != 10:
        raise RuntimeError("Static audit failed: event_count != 10.")
    if audit["positive_points"] != 1200:
        raise RuntimeError("Static audit failed: positive_points != 1200.")
    if param_count != 264001:
        raise RuntimeError(
            f"Static audit failed: parameter_count={param_count}, expected 264001."
        )

    with open(out_dir / "protocol_v2p1_B1R5_static_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2, default=_json_default)

    events_df.to_csv(
        out_dir / "selected_injection_sites_and_events.csv",
        index=False, encoding="utf-8-sig"
    )

    pd.DataFrame({
        "time": injected_raw["time"],
        "label": labels.astype(int),
        "eval_mask": eval_mask.astype(int),
    }).to_csv(
        out_dir / "internal_validation_mask_55_70.csv",
        index=False, encoding="utf-8-sig"
    )

    # Protocol container useful for later baseline/ablation reuse.
    protocol_container = {
        "protocol": PROTOCOL_VERSION_V4,
        "source_chain": [
            "Experiment12_lambda_dist_0.10",
            "Experiment25_frozen_10_event_completion",
            "Experiment30B_B1_R5",
            "Experiment31_delay_source_audit",
        ],
        "split": {
            "train_reference": "0%-55%",
            "internal_validation": "55%-70%",
            "final_test": "70%-100% physically excluded",
            "train_end": int(train_end),
            "eval_start": int(eval_start),
            "eval_end": int(eval_end),
            "full_n": int(full_n),
        },
        "threshold": {
            "rule": "pure quantile",
            "q": 0.99,
            "source": "clean 0%-55% score only",
        },
        "temporal_profile": {
            "name": "B1_R5",
            "A1": "ORIGINAL",
            "A2_A5": "5-min half-cosine ramp + stable plateau",
        },
        "events": events_df.to_dict(orient="records"),
        "formal_seeds": list(SEEDS_10),
        "merge_gap_descriptive_min": 120,
    }
    with open(out_dir / "injection_protocol_v2p1_B1R5.json", "w", encoding="utf-8") as f:
        json.dump(protocol_container, f, ensure_ascii=False, indent=2, default=_json_default)

    results = []

    for seed in SEEDS_10:
        print(f"\n===== Running FULL seed={seed} =====", flush=True)

        row = run_single_seed_full(
            seed, cfg, device, scaled_df, injected_scaled_df,
            injected_feature_df, subspace_cols, feature_cols,
            train_end, n, labels, events_df, eval_mask, out_dir
        )

        results.append(row)

        print(
            f"[seed={seed}] "
            f"P={row['PointPrecision']:.4f} "
            f"R={row['PointRecall']:.4f} "
            f"F1={row['PointF1']:.4f} | "
            f"ROC={row['ROC_AUC']:.4f} "
            f"AP={row['PR_AUC_AP']:.4f} | "
            f"FAR={row['FullFAR']:.4f} | "
            f"EventRecall={row['EventRecall']:.4f} "
            f"EventF1={row['EventF1']:.4f} | "
            f"Merge120F1={row['Merge120EventF1']:.4f} | "
            f"Delay={row['DelayMeanMin']:.2f} min",
            flush=True,
        )

    results_df = pd.DataFrame(results)

    if "Seed" in results_df.columns:
        results_df = results_df[
            [c for c in results_df.columns if c != "Seed"] + ["Seed"]
        ]

    results_df.to_csv(
        out_dir / "MSDSNet_R1_full_10seed_seedwise_results.csv",
        index=False, encoding="utf-8-sig"
    )

    numeric_cols = [
        c for c in results_df.columns
        if pd.api.types.is_numeric_dtype(results_df[c])
    ]

    summary_rows = []
    for c in numeric_cols:
        if c == "Seed":
            continue
        vals = results_df[c].astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            mean = np.nan
            std = np.nan
        else:
            mean = float(np.mean(vals))
            std = _sample_std(vals.tolist())
        summary_rows.append({
            "Metric": c,
            "Mean": mean,
            "Std": std,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        out_dir / "MSDSNet_R1_full_10seed_mean_std.csv",
        index=False, encoding="utf-8-sig"
    )

    # Reference values are NOT used by the run; they are recorded only for
    # post-run reproduction checking against Experiment 30B/31.
    exp30b_reference = {
        "PointPrecision": 0.7963376726405919,
        "PointRecall": 0.124,
        "PointF1": 0.2145435929638327,
        "FullFAR": 0.0071969696969696965,
        "ROC_AUC": 0.9613502840909092,
        "PR_AUC_AP": 0.7602052090129263,
        "EventRecall": 1.0,
        "EventF1": 0.3847007598394946,
        "Merge120EventF1": 0.6473118279569893,
        "DelayMeanMin": 67.4,
    }

    report = {
        "experiment": "MSDS-Net_R1_FULL_10SEEDS_PROTOCOL_v2.1_B1_R5",
        "protocol": PROTOCOL_VERSION_V4,
        "seeds": list(SEEDS_10),
        "config": {
            "internal_train_ratio": 0.55,
            "reference_end_ratio": 0.70,
            "seq_len": cfg.seq_len,
            "stride": cfg.stride,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "lambda_trend": cfg.lambda_trend,
            "lambda_dist": cfg.lambda_dist,
            "lambda_res": cfg.lambda_res,
            "a_residual_mix": cfg.a_residual_mix,
            "lambda_graph": cfg.lambda_graph,
            "graph_score_weight": cfg.graph_score_weight,
            "threshold_quantile": cfg.threshold_quantile,
            "temperature_file": cfg.temperature_file,
            "vi_file": cfg.vi_file,
        },
        "full_num_samples": int(full_n),
        "retained_num_samples": int(n),
        "train_end": int(train_end),
        "eval_start": int(eval_start),
        "eval_end": int(eval_end),
        "final_30_excluded": True,
        "event_count": int(len(events_df)),
        "positive_points": int(labels[eval_mask].sum()),
        "events": events_df.to_dict(orient="records"),
        "static_audit": audit,
        "per_seed_results": results_df.to_dict(orient="records"),
        "mean_std_summary": summary_df.to_dict(orient="records"),
        "exp30b_exp31_reference_means_for_reproduction_check_only": exp30b_reference,
    }

    with open(
        out_dir / "MSDSNet_R1_full_10seed_report.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            report, f, ensure_ascii=False, indent=2, default=_json_default
        )

    print(
        f"\nSaved seedwise results: "
        f"{out_dir / 'MSDSNet_R1_full_10seed_seedwise_results.csv'}"
    )
    print(
        f"Saved mean±std summary: "
        f"{out_dir / 'MSDSNet_R1_full_10seed_mean_std.csv'}"
    )
    print(
        f"Saved final json report: "
        f"{out_dir / 'MSDSNet_R1_full_10seed_report.json'}"
    )
    print(
        f"Saved protocol audit: "
        f"{out_dir / 'protocol_v2p1_B1R5_static_audit.json'}"
    )


if __name__ == "__main__":
    main()
