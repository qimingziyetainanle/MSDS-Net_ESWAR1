# -*- coding: utf-8 -*-
"""
Formal 16-baseline comparison for MSDS-Net revision.

Common benchmark is locked to the frozen MSDS-Net V4 Protocol-v2.1 / B1-R5:
  * original 0%-55%: clean training + threshold reference
  * original 55%-70%: fixed 10-event internal validation
  * original 70%-100%: physically excluded
  * exact Exp25 ten event locations
  * A1 original temporal mechanism
  * A2-A5 five-minute half-cosine transition + stable plateau
  * each method's native anomaly score -> clean Q0.99 -> common V4 evaluator
  * formal seeds: 42, 2024, ..., 2032

The 16 baseline methods are:
PCA, Isolation Forest, USAD, OmniAnomaly, MTAD-GAT, InterFusion, GDN,
TranAD, DCdetector, TimesNet, MEMTO, CATCH, MDGAD, MSHTrans, CKDGAT,
MAD-DGTD.

All V4 metrics are retained; the compact paper table is only a selected view.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset

import frozen_msds_v4 as base
from baseline_models import BaseDetector, CATCH, MEMTO, USAD, make_model
from method_configs import METHOD_CONFIGS, METHOD_ORDER, PAPER_METRICS, SEEDS_10, V4_METRICS

RAW_COLS = ["T", "U51", "U52", "U53", "I54", "I55", "I56"]
PROTOCOL_NAME = "MSDSNET_REVISION_EXPERIMENT_PROTOCOL_v2.1_B1_R5"
PACKAGE_VERSION = "Baseline16_v2.0_DeepFaithful_30Epoch_ProtocolV2p1"

# Benign PyTorch warning caused by TranAD's source-faithful nhead=n_features=7.
# It only disables a nested-tensor optimization path; it does not change the model.
warnings.filterwarnings(
    "ignore",
    message=r".*enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd.*",
    category=UserWarning,
)


# -----------------------------------------------------------------------------
# Reproducibility / data windows
# -----------------------------------------------------------------------------

def _configure_deterministic_cuda_attention() -> None:
    """Force the deterministic math SDPA backend for formal repeated runs.

    On some Windows/PyTorch builds, MultiheadAttention may otherwise select the
    memory-efficient scaled-dot-product-attention CUDA kernel, whose backward
    pass is non-deterministic.  The math backend is slower but appropriate for
    the frozen 10-seed paper protocol.
    """
    if not torch.cuda.is_available():
        return
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is None:
        return
    for name, value in (
        ("enable_flash_sdp", False),
        ("enable_mem_efficient_sdp", False),
        ("enable_math_sdp", True),
    ):
        fn = getattr(cuda_backend, name, None)
        if callable(fn):
            fn(value)
    # Avoid TF32 changing numerical paths across hardware/settings.
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _configure_deterministic_cuda_attention()
    # Formal runs should fail loudly if another non-deterministic CUDA op is encountered.
    torch.use_deterministic_algorithms(True, warn_only=False)             ##False


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, starts: np.ndarray, seq_len: int):
        self.x = np.asarray(x, dtype=np.float32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.seq_len = int(seq_len)
    def __len__(self): return len(self.starts)
    def __getitem__(self, idx):
        s = int(self.starts[idx])
        return torch.from_numpy(self.x[s:s+self.seq_len]), torch.tensor(s, dtype=torch.long)


def make_loaders(x_clean: np.ndarray, x_eval: np.ndarray, train_end: int,
                 seq_len: int, stride: int, batch_size: int, seed: int,
                 device: torch.device):
    n = len(x_clean)
    train_starts = base.make_start_indices(n, seq_len, stride, end_limit=train_end)
    all_starts = base.make_start_indices(n, seq_len, stride, end_limit=None)
    g = torch.Generator(); g.manual_seed(int(seed))
    train_loader = DataLoader(
        WindowDataset(x_clean, train_starts, seq_len), batch_size=batch_size,
        shuffle=True, generator=g, num_workers=0, pin_memory=(device.type == "cuda"), drop_last=False,
    )
    clean_eval_loader = DataLoader(
        WindowDataset(x_clean, all_starts, seq_len), batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"), drop_last=False,
    )
    anomaly_eval_loader = DataLoader(
        WindowDataset(x_eval, all_starts, seq_len), batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"), drop_last=False,
    )
    return train_loader, clean_eval_loader, anomaly_eval_loader


@torch.no_grad()
def aggregate_scores(model: BaseDetector, loader: DataLoader, n_points: int,
                     seq_len: int, device: torch.device) -> np.ndarray:
    model.eval()
    score_sum = np.zeros(n_points, dtype=np.float64)
    count = np.zeros(n_points, dtype=np.float64)
    for x, starts in loader:
        x = x.to(device, non_blocking=True)
        sw = model.score_points(x).detach().cpu().numpy()
        for b, s in enumerate(starts.numpy()):
            s = int(s); e = min(n_points, s + seq_len)
            score_sum[s:e] += sw[b, :e-s]
            count[s:e] += 1.0
    count[count == 0] = 1.0
    return score_sum / count


# -----------------------------------------------------------------------------
# Method training
# -----------------------------------------------------------------------------

def initialize_memto_memory(model: MEMTO, loader: DataLoader, device: torch.device,
                            max_vectors: int = 12000) -> None:
    model.eval(); vecs = []
    with torch.no_grad():
        for x, _ in loader:
            z = model.encode(x.to(device, non_blocking=True)).detach().cpu().numpy().reshape(-1, model.memory.size(1))
            if len(z) > 0: vecs.append(z)
            if sum(len(v) for v in vecs) >= max_vectors: break
    arr = np.concatenate(vecs, axis=0)[:max_vectors]
    k = model.memory.size(0)
    km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=min(2048, len(arr)), n_init=3)
    centers = km.fit(arr).cluster_centers_.astype(np.float32)
    with torch.no_grad(): model.memory.copy_(torch.from_numpy(centers).to(device))
    model.memory_initialized = True


def train_usad(model: USAD, loader: DataLoader, device: torch.device, epochs: int,
               lr: float, wd: float, name: str) -> None:
    """Two-optimizer USAD training, matching the public implementation.

    We intentionally use Adam (not AdamW) and no weight decay for USAD.
    The input to this routine is clean-train-fitted MinMax data, matching the
    sigmoid decoder range. Both objective components are printed separately.
    """
    opt1 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder1.parameters()), lr=lr
    )
    opt2 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder2.parameters()), lr=lr
    )
    for ep in range(1, epochs + 1):
        model.train(); total1 = 0.0; total2 = 0.0; nb = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)

            opt1.zero_grad(set_to_none=True)
            l1 = model.loss1(x, ep)
            if not torch.isfinite(l1):
                raise RuntimeError(f"Non-finite USAD loss1 at epoch {ep}: {l1}")
            l1.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.encoder.parameters()) + list(model.decoder1.parameters()), 5.0
            )
            opt1.step()

            opt2.zero_grad(set_to_none=True)
            l2 = model.loss2(x, ep)
            if not torch.isfinite(l2):
                raise RuntimeError(f"Non-finite USAD loss2 at epoch {ep}: {l2}")
            l2.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.encoder.parameters()) + list(model.decoder2.parameters()), 5.0
            )
            opt2.step()

            total1 += float(l1.detach().cpu())
            total2 += float(l2.detach().cpu())
            nb += 1

        m1 = total1 / max(nb, 1)
        m2 = total2 / max(nb, 1)
        if not np.isfinite(m1) or not np.isfinite(m2):
            raise RuntimeError(f"USAD diverged at epoch {ep}: loss1={m1}, loss2={m2}")
        # Large positive runaway like the previous package is a hard failure.
        if abs(m1) > 1e3 or abs(m2) > 1e3:
            raise RuntimeError(
                f"USAD unstable objective at epoch {ep}: loss1={m1:.6f}, loss2={m2:.6f}. "
                "Stop before producing invalid baseline results."
            )
        print(
            f"[{name}] epoch {ep:03d}/{epochs} loss1={m1:.6f} loss2={m2:.6f}",
            flush=True,
        )


def _make_optimizer(kind: str, params, lr: float, wd: float):
    kind = str(kind).lower()
    if kind == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if kind == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {kind}")


def train_catch(model: CATCH, loader: DataLoader, device: torch.device, epochs: int,
                lr: float, wd: float, name: str, grad_clip: float) -> None:
    """CATCH source-guided bi-level-style optimization.

    The spectral reconstruction network and CFM mask generator use separate
    optimizers; the latter uses the smaller mask learning rate described by the
    public implementation family. Evaluation/threshold remain common V4.
    """
    mask_ids = {id(p) for p in model.mask_parameters()}
    main_params = [p for p in model.parameters() if id(p) not in mask_ids]
    mask_params = model.mask_parameters()
    opt_main = torch.optim.Adam(main_params, lr=lr, weight_decay=wd)
    opt_mask = torch.optim.Adam(mask_params, lr=min(1e-5, lr), weight_decay=0.0)
    for ep in range(1, epochs + 1):
        model.train(); tm = 0.0; ta = 0.0; nb = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            # Inner/model step.
            opt_main.zero_grad(set_to_none=True)
            rec, info = model.forward(x)
            main_loss = torch.nn.functional.mse_loss(rec, x) + 0.01 * info["aux"]
            if not torch.isfinite(main_loss):
                raise RuntimeError(f"Non-finite CATCH main loss at epoch {ep}: {main_loss}")
            main_loss.backward()
            torch.nn.utils.clip_grad_norm_(main_params, grad_clip)
            opt_main.step()
            # Outer/mask-generator step on a fresh graph.
            opt_mask.zero_grad(set_to_none=True)
            _, info2 = model.forward(x)
            aux_loss = info2["aux"]
            if not torch.isfinite(aux_loss):
                raise RuntimeError(f"Non-finite CATCH auxiliary loss at epoch {ep}: {aux_loss}")
            aux_loss.backward()
            torch.nn.utils.clip_grad_norm_(mask_params, grad_clip)
            opt_mask.step()
            tm += float(main_loss.detach().cpu()); ta += float(aux_loss.detach().cpu()); nb += 1
        print(f"[{name}] epoch {ep:03d}/{epochs} main={tm/max(nb,1):.6f} aux={ta/max(nb,1):.6f}", flush=True)


def train_generic(model: BaseDetector, loader: DataLoader, device: torch.device,
                  epochs: int, lr: float, wd: float, name: str,
                  optimizer_kind: str, grad_clip: float) -> None:
    opt = _make_optimizer(optimizer_kind, model.parameters(), lr, wd)
    memto_warmup = max(5, epochs // 4) if isinstance(model, MEMTO) else None
    for ep in range(1, epochs + 1):
        if isinstance(model, MEMTO) and ep == memto_warmup + 1:
            print(f"[{name}] initializing memory with K-means after warm-up...", flush=True)
            initialize_memto_memory(model, loader, device)
        model.train(); total = 0.0; nb = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = model.training_loss(x, ep)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss for {name} at epoch {ep}: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total += float(loss.detach().cpu()); nb += 1
        avg = total / max(nb, 1)
        note = " (stop-gradient contrastive objective may be numerically near zero)" if name == "DCdetector" else ""
        print(f"[{name}] epoch {ep:03d}/{epochs} loss={avg:.6f}{note}", flush=True)


def train_model(model: BaseDetector, loader: DataLoader, device: torch.device,
                name: str, epochs: int, lr: float, wd: float,
                optimizer_kind: str = "Adam", grad_clip: float = 5.0):
    if isinstance(model, USAD):
        return train_usad(model, loader, device, epochs, lr, wd, name)
    if isinstance(model, CATCH):
        return train_catch(model, loader, device, epochs, lr, wd, name, grad_clip)
    return train_generic(model, loader, device, epochs, lr, wd, name, optimizer_kind, grad_clip)


# -----------------------------------------------------------------------------
# Efficiency profiling
# -----------------------------------------------------------------------------

def trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def estimate_macs(model: nn.Module, sample: torch.Tensor) -> float:
    """Approximate trainable-layer MACs for one 60x7 window.

    Counts Linear/Conv/GRU/LSTM/MultiheadAttention modules. Tensor-only graph/FFT
    operations are not fully represented, therefore this value is explicitly saved
    as an estimate rather than an exact hardware FLOP count.
    """
    total = [0.0]
    hooks = []
    def linear_hook(m, inp, out):
        x = inp[0]; n = x.numel() / max(m.in_features, 1)
        total[0] += n * m.in_features * m.out_features
    def conv1_hook(m, inp, out):
        y = out; k = m.kernel_size[0]; cin = m.in_channels / m.groups
        total[0] += y.numel() * k * cin
    def conv2_hook(m, inp, out):
        y = out; k = m.kernel_size[0] * m.kernel_size[1]; cin = m.in_channels / m.groups
        total[0] += y.numel() * k * cin
    def gru_hook(m, inp, out):
        x = inp[0]; B,L,_ = x.shape; gates = 3
        total[0] += B * L * m.num_layers * gates * (m.input_size*m.hidden_size + m.hidden_size*m.hidden_size)
    def lstm_hook(m, inp, out):
        x = inp[0]; B,L,_ = x.shape; gates = 4
        total[0] += B * L * m.num_layers * gates * (m.input_size*m.hidden_size + m.hidden_size*m.hidden_size)
    def mha_hook(m, inp, out):
        q = inp[0]; B,L,D = q.shape
        total[0] += B * (4 * L * D * D + 2 * L * L * D)
    for mod in model.modules():
        if isinstance(mod, nn.Linear): hooks.append(mod.register_forward_hook(linear_hook))
        elif isinstance(mod, nn.Conv1d): hooks.append(mod.register_forward_hook(conv1_hook))
        elif isinstance(mod, nn.Conv2d): hooks.append(mod.register_forward_hook(conv2_hook))
        elif isinstance(mod, nn.GRU): hooks.append(mod.register_forward_hook(gru_hook))
        elif isinstance(mod, nn.LSTM): hooks.append(mod.register_forward_hook(lstm_hook))
        elif isinstance(mod, nn.MultiheadAttention): hooks.append(mod.register_forward_hook(mha_hook))
    was_training = model.training; model.eval()
    with torch.no_grad(): model.score_points(sample)
    for h in hooks: h.remove()
    model.train(was_training)
    return float(total[0])


def profile_inference(model: BaseDetector, device: torch.device, seq_len: int, input_dim: int) -> Dict[str, float]:
    sample = torch.zeros(1, seq_len, input_dim, device=device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(10): model.score_points(sample)
        if device.type == "cuda": torch.cuda.synchronize(device)
        times = []
        for _ in range(50):
            if device.type == "cuda": torch.cuda.synchronize(device)
            t0 = time.perf_counter(); model.score_points(sample)
            if device.type == "cuda": torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000.0)
    peak = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else float("nan")
    macs = estimate_macs(model, sample)
    return {
        "TrainableParams": trainable_params(model),
        "EstimatedMACs_1x60": macs,
        "EstimatedFLOPs_1x60": 2.0 * macs,
        "InferenceLatencyMs_B1_mean": float(np.mean(times)),
        "InferenceLatencyMs_B1_std": float(np.std(times, ddof=1)),
        "InferencePeakGPUMB": peak,
    }


def _hook_estimate_macs_from_forward(model: nn.Module, forward_fn) -> float:
    total = [0.0]; hooks = []
    def linear_hook(m, inp, out):
        x = inp[0]; total[0] += (x.numel() / max(m.in_features, 1)) * m.in_features * m.out_features
    def conv1_hook(m, inp, out):
        total[0] += out.numel() * m.kernel_size[0] * (m.in_channels / m.groups)
    def gru_hook(m, inp, out):
        x=inp[0]; B,L,_=x.shape
        total[0] += B*L*m.num_layers*3*(m.input_size*m.hidden_size + m.hidden_size*m.hidden_size)
    for mod in model.modules():
        if isinstance(mod, nn.Linear): hooks.append(mod.register_forward_hook(linear_hook))
        elif isinstance(mod, nn.Conv1d): hooks.append(mod.register_forward_hook(conv1_hook))
        elif isinstance(mod, nn.GRU): hooks.append(mod.register_forward_hook(gru_hook))
    with torch.no_grad(): forward_fn()
    for h in hooks: h.remove()
    return float(total[0])


def profile_msds_full(device: torch.device, cfg) -> Dict[str, float]:
    """Profile the exact frozen FULL architecture without retraining it."""
    subspace_cols = {
        "T": ["T","dT"],
        "U": ["U51","U52","U53","U_mean","B_U"],
        "I": ["I54","I55","I56","I_mean","B_I"],
        "B": ["B_U","B_I"],
        "ET": ["T","dT","P","dP","U_mean","I_mean"],
    }
    feature_cols = ["T","U51","U52","U53","I54","I55","I56","U_mean","I_mean","B_U","B_I","P","dT","dP"]
    model, cols = base.make_model_for_variant("FULL", subspace_cols, feature_cols, cfg, device)
    batch = {k: torch.zeros(1, 60, len(v), device=device) for k,v in cols.items()}
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(10): model(batch)
        if device.type == "cuda": torch.cuda.synchronize(device)
        times=[]
        for _ in range(50):
            if device.type == "cuda": torch.cuda.synchronize(device)
            t0=time.perf_counter(); model(batch)
            if device.type == "cuda": torch.cuda.synchronize(device)
            times.append((time.perf_counter()-t0)*1000.0)
    macs = _hook_estimate_macs_from_forward(model, lambda: model(batch))
    peak = float(torch.cuda.max_memory_allocated(device)/(1024**2)) if device.type == "cuda" else np.nan
    row = {
        "Method":"MSDS-Net", "Runs":1,
        "TrainableParams_mean":float(sum(p.numel() for p in model.parameters() if p.requires_grad)), "TrainableParams_std":0.0,
        "EstimatedMACs_1x60_mean":macs, "EstimatedMACs_1x60_std":0.0,
        "EstimatedFLOPs_1x60_mean":2.0*macs, "EstimatedFLOPs_1x60_std":0.0,
        "InferenceLatencyMs_B1_mean":float(np.mean(times)), "InferenceLatencyMs_B1_std":float(np.std(times,ddof=1)),
        "InferencePeakGPUMB_mean":peak, "InferencePeakGPUMB_std":0.0,
        "TrainingSeconds_mean":np.nan, "TrainingSeconds_std":np.nan,
        "TrainingPeakGPUMB_mean":np.nan, "TrainingPeakGPUMB_std":np.nan,
        "Note":"Exact frozen FULL architecture profiled without retraining; training-time cost should be reported from a dedicated timing rerun if required."
    }
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return row


def profile_cpu_callable(fn, repeats: int = 100) -> Tuple[float,float]:
    for _ in range(10): fn()
    vals=[]
    for _ in range(repeats):
        t0=time.perf_counter(); fn(); vals.append((time.perf_counter()-t0)*1000.0)
    return float(np.mean(vals)), float(np.std(vals,ddof=1))


# -----------------------------------------------------------------------------
# Evaluation/reporting
# -----------------------------------------------------------------------------

def evaluate_score(name: str, seed: int, train_score: np.ndarray, eval_score: np.ndarray,
                   train_end: int, labels: np.ndarray, events_df: pd.DataFrame,
                   eval_mask: np.ndarray, q: float = 0.99) -> Dict[str, float]:
    th = base.choose_threshold(train_score[:train_end], q)
    met = base.compute_event_adjusted_metrics(labels, eval_score, th, events_df, eval_mask)
    met["Threshold"] = float(th); met["EvalPoints"] = int(eval_mask.sum())
    met["Variant"] = name; met["Seed"] = int(seed)
    return met


def eventwise_rows(name: str, seed: int, score: np.ndarray, threshold: float,
                   events_df: pd.DataFrame) -> List[Dict[str, object]]:
    pred = (score > threshold).astype(int); rows = []
    for _, r in events_df.iterrows():
        s,e = int(r["start_idx"]), int(r["end_idx"])
        inside = np.flatnonzero(pred[s:e] > 0)
        rows.append({
            "Variant": name, "Seed": int(seed), "EventID": r.get("EventID", ""),
            "Type": r.get("short", r.get("type", "")), "Mechanism": r.get("mechanism", ""),
            "StartIdx": s, "EndIdx": e, "DurationMin": int(e-s),
            "Detected": int(inside.size > 0),
            "FirstDetectionDelayMin": float(inside[0]) if inside.size else float(e-s),
            "DetectionRatio": float(pred[s:e].mean()),
            "MeanScore": float(np.mean(score[s:e])), "MaxScore": float(np.max(score[s:e])),
        })
    return rows


def sample_std(a: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    return float(np.std(a, ddof=1)) if a.size > 1 else 0.0


def summarize_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [c for c in df.columns if c not in {"Variant", "Seed"} and pd.api.types.is_numeric_dtype(df[c])]
    for method in METHOD_ORDER:
        p = df[df["Variant"] == method]
        if p.empty: continue
        for metric in metric_cols:
            vals = pd.to_numeric(p[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            rows.append({"Method": method, "Metric": metric, "Mean": float(np.mean(vals)) if vals.size else np.nan,
                         "Std": sample_std(vals), "Runs": int(vals.size)})
    return pd.DataFrame(rows)


def build_paper_table(summary_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        d = summary_long[summary_long["Method"] == method]
        if d.empty: continue
        row = {"Method": method}
        for m in PAPER_METRICS:
            q = d[d["Metric"] == m]
            if len(q):
                mean, std = float(q.iloc[0]["Mean"]), float(q.iloc[0]["Std"])
                row[m] = f"{mean:.4f} ± {std:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def load_msds_reference(path: Path) -> Tuple[pd.DataFrame, Dict]:
    if not path.exists(): return pd.DataFrame(), {}
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for r in report.get("mean_std_summary", []):
        rows.append({"Method": "MSDS-Net", "Metric": r["Metric"], "Mean": r["Mean"], "Std": r["Std"], "Runs": 10})
    return pd.DataFrame(rows), report


def provenance() -> List[Dict[str, str]]:
    common = "Independent PyTorch reproduction of architecture/core objective; common V4 split/B1-R5/Q99/evaluator intentionally replaces repository-specific threshold/evaluation."
    return [
        {"Method":"PCA","Fidelity":"library-standard","Source":"scikit-learn PCA","Core":"95%-variance PCA reconstruction error","URL":"https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html","ProtocolAdaptation":"common V4 only"},
        {"Method":"Isolation Forest","Fidelity":"library-standard","Source":"scikit-learn IsolationForest","Core":"300-tree isolation forest; negative decision_function anomaly score","URL":"https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html","ProtocolAdaptation":"common V4 only"},
        {"Method":"USAD","Fidelity":"public-repo-guided core reproduction","Source":"USAD public implementation","Core":"shared encoder + two mirrored sigmoid decoders + two epoch-dependent adversarial objectives","URL":"https://github.com/manigalati/usad","ProtocolAdaptation":common + " Clean-train MinMax retained because of sigmoid decoder."},
        {"Method":"OmniAnomaly","Fidelity":"official-repo/paper-guided core reproduction","Source":"OmniAnomaly official repository","Core":"stochastic GRU recurrent VAE + conditional prior/posterior + planar flows + reconstruction probability","URL":"https://github.com/NetManAIOps/OmniAnomaly","ProtocolAdaptation":common},
        {"Method":"MTAD-GAT","Fidelity":"paper + public-reproduction-guided core reproduction","Source":"MTAD-GAT paper and public PyTorch reproductions","Core":"temporal Conv1D + parallel feature/time GAT + GRU + forecasting + VAE reconstruction","URL":"https://arxiv.org/abs/2009.02040","ProtocolAdaptation":common},
        {"Method":"InterFusion","Fidelity":"official-repo/paper-guided core reproduction","Source":"InterFusion official repository","Core":"hierarchical VAE with inter-metric stochastic embedding and temporal conditional stochastic embedding","URL":"https://github.com/zhhlee/InterFusion","ProtocolAdaptation":common},
        {"Method":"GDN","Fidelity":"official-repo-guided core reproduction","Source":"GDN official repository","Core":"learned node embeddings + cosine top-k graph + graph attention + sensor-history prediction + output MLP","URL":"https://github.com/d-ailin/GDN","ProtocolAdaptation":common},
        {"Method":"TranAD","Fidelity":"official-repo-guided core reproduction","Source":"TranAD official repository","Core":"d_model=2F, nhead=F, one Transformer encoder + two self-conditioned decoders, epoch-weighted two-phase loss","URL":"https://github.com/imperial-qore/TranAD","ProtocolAdaptation":common + " Clean-train MinMax retained because public output is sigmoid-bounded."},
        {"Method":"DCdetector","Fidelity":"official-repo-semantics-guided core reproduction","Source":"DCdetector official repository","Core":"channel-independent multi-scale patch/in-patch dual association + pure stop-gradient symmetric-KL contrastive objective, d_model=256, 3 levels","URL":"https://github.com/DAMO-DI-ML/KDD2023-DCdetector","ProtocolAdaptation":common + " Repository threshold/point adjustment not used."},
        {"Method":"TimesNet","Fidelity":"official-code-core reproduction","Source":"THUML Time-Series-Library","Core":"FFT_for_Period + 3 TimesBlocks + two Inception_Block_V1 per block, d_model=d_ff=128, top_k=3, six kernels","URL":"https://github.com/thuml/Time-Series-Library","ProtocolAdaptation":common},
        {"Method":"MEMTO","Fidelity":"official-repo/paper-guided core reproduction","Source":"MEMTO official repository","Core":"memory-guided Transformer + two-phase K-means memory initialization + gated memory read/update + input/latent deviation score","URL":"https://github.com/gunny97/MEMTO","ProtocolAdaptation":common},
        {"Method":"CATCH","Fidelity":"official-repo/paper-guided core reproduction","Source":"CATCH official repository","Core":"frequency patching + Channel Fusion Module + patch-wise mask generator + masked channel attention + spectral reconstruction + auxiliary CFM optimization","URL":"https://github.com/decisionintelligence/CATCH","ProtocolAdaptation":common + " Frequency patch size adapted to seq_len=60."},
        {"Method":"MDGAD","Fidelity":"paper-guided core reproduction","Source":"AAAI 2024 MDGAD student abstract","Core":"multi-scale evolving dynamic graph learning + bilateral forward/backward prediction errors","URL":"https://ojs.aaai.org/index.php/AAAI/article/view/30456","ProtocolAdaptation":common + " No reliable official repository identified."},
        {"Method":"MSHTrans","Fidelity":"official-repo/paper-guided core reproduction","Source":"MSHTrans official repository / KDD 2025 paper","Core":"multi-scale downsampling + trainable adaptive hypergraphs + time-series decomposition + hypergraph Transformer + scale fusion reconstruction","URL":"https://github.com/chenzl23/MSHTrans","ProtocolAdaptation":common},
        {"Method":"CKDGAT","Fidelity":"paper-guided core reproduction","Source":"Computers in Industry 2026 CKDGAT paper","Core":"data-oriented GAT + knowledge-oriented stochastic multi-head GAT + two-layer composite encoder + temporal fusion + reconstruction","URL":"https://www.sciencedirect.com/science/article/pii/S0166361526000126","ProtocolAdaptation":common + " Generic seven-sensor process graph supplied as method-required knowledge input; no reliable official repo identified."},
        {"Method":"MAD-DGTD","Fidelity":"paper-guided core reproduction","Source":"Neurocomputing 2025 MAD-DGTD paper","Core":"TDDG global-static/dynamic/delay graph fusion + stacked dilated multi-scale TDIE + GCIP + prediction-based anomaly score","URL":"https://www.sciencedirect.com/science/article/pii/S0925231225005594","ProtocolAdaptation":common + " No reliable official repository identified."},
    ]


def write_checkpoint(rows: List[Dict], events: List[Dict], eff: List[Dict], out: Path):
    pd.DataFrame(rows).to_csv(out / "baseline16_all_seed_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(events).to_csv(out / "baseline16_eventwise_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eff).to_csv(out / "baseline16_efficiency_all_runs.csv", index=False, encoding="utf-8-sig")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_csv_arg(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", help="Directory containing 930T.csv and 930V.csv")
    ap.add_argument("--output-dir", default="baseline16_v20_deepfaithful_outputs")
    ap.add_argument("--methods", default="all", help="all or comma-separated method names")
    ap.add_argument("--seeds", default="all", help="all or comma-separated integer seeds")
    ap.add_argument("--device", default="auto", choices=["auto","cuda","cpu"])
    ap.add_argument("--quick-check", action="store_true", help="1 seed, 1 epoch for code-path validation only; NOT paper results")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-save-scores", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).resolve()
    # Convenience fallback: current requested directory -> script directory -> parent.
    if not ((data_dir / "930T.csv").exists() and (data_dir / "930V.csv").exists()):
        for cand in (root, root.parent):
            if (cand / "930T.csv").exists() and (cand / "930V.csv").exists():
                data_dir = cand
                break
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    score_dir = out / "scores_npz"; score_dir.mkdir(exist_ok=True)

    if not all(cfg_.epochs == 30 for cfg_ in METHOD_CONFIGS.values()):
        bad = {name: cfg_.epochs for name, cfg_ in METHOD_CONFIGS.items() if cfg_.epochs != 30}
        raise RuntimeError(f"Formal baseline protocol requires exactly 30 epochs for every deep method; found {bad}")

    methods = list(METHOD_ORDER) if args.methods == "all" else parse_csv_arg(args.methods)
    unknown = [m for m in methods if m not in METHOD_ORDER]
    if unknown: raise ValueError(f"Unknown methods: {unknown}")
    seeds = list(SEEDS_10) if args.seeds == "all" else [int(x) for x in parse_csv_arg(args.seeds)]
    if args.quick_check:
        seeds = [42]; methods = methods[:1] if args.methods == "all" else methods
        print("*** QUICK CHECK MODE: results are NOT valid paper results. ***")

    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    print(f"Device: {device}")

    # Exact V4 benchmark preparation.
    cfg = base.CFG
    cfg.temperature_file = str(data_dir / "930T.csv")
    cfg.vi_file = str(data_dir / "930V.csv")
    cfg.threshold_quantile = 0.99
    cfg.a_residual_mix = 0.50
    cfg.lambda_dist = 0.10
    if not Path(cfg.temperature_file).exists() or not Path(cfg.vi_file).exists():
        raise FileNotFoundError(f"Expected 930T.csv and 930V.csv in {data_dir}")

    full_raw = base.load_raw_inputs(cfg)
    full_n = len(full_raw); train_end = int(full_n * 0.55); eval_end = int(full_n * 0.70)
    raw = full_raw.iloc[:eval_end].copy().reset_index(drop=True); n = len(raw)
    injected_raw, labels, events_df = base.inject_synthetic_anomalies(raw, train_end, cfg)
    eval_mask = np.zeros(n, dtype=bool); eval_mask[train_end:eval_end] = True

    if len(events_df) != 10 or int(labels[eval_mask].sum()) != 1200:
        raise RuntimeError("Frozen benchmark integrity check failed (10 events / 1200 positive points expected).")

    # Common raw 7-channel representation. StandardScaler is the common view.
    # USAD and TranAD receive a clean-train-fitted MinMax view because their
    # public implementations use sigmoid-bounded reconstruction outputs. This
    # changes only method-native normalization, never split/events/threshold/evaluator.
    train_raw = raw.loc[:train_end-1, RAW_COLS].to_numpy(dtype=float)
    scaler = StandardScaler().fit(train_raw)
    x_clean = scaler.transform(raw[RAW_COLS].to_numpy(dtype=float)).astype(np.float32)
    x_eval = scaler.transform(injected_raw[RAW_COLS].to_numpy(dtype=float)).astype(np.float32)

    bounded_scaler = MinMaxScaler(feature_range=(0.0, 1.0)).fit(train_raw)
    x_clean_bounded = bounded_scaler.transform(raw[RAW_COLS].to_numpy(dtype=float)).astype(np.float32)
    x_eval_bounded = bounded_scaler.transform(injected_raw[RAW_COLS].to_numpy(dtype=float)).astype(np.float32)

    protocol_audit = {
        "protocol": PROTOCOL_NAME, "package_version": PACKAGE_VERSION, "full_n": full_n, "retained_n": n,
        "train_reference": [0, train_end], "validation": [train_end, eval_end],
        "final_30_physically_excluded": True, "raw_input_channels": RAW_COLS,
        "seq_len": 60, "stride": 5, "threshold": "per-method clean Q0.99",
        "event_count": len(events_df), "positive_points": int(labels[eval_mask].sum()),
        "formal_seeds": list(SEEDS_10), "methods": list(METHOD_ORDER),
        "deep_baseline_epochs": 30,
        "all_deep_methods_exactly_30_epochs": all(METHOD_CONFIGS[m].epochs == 30 for m in METHOD_CONFIGS),
        "normalization": {
            "default": "StandardScaler fitted on clean 0%-55% only",
            "USAD": "MinMaxScaler [0,1] fitted on clean 0%-55% only (public sigmoid decoder)",
            "TranAD": "MinMaxScaler [0,1] fitted on clean 0%-55% only (public sigmoid output)",
        },
        "all_v4_metrics_saved": True,
    }
    (out / "baseline16_protocol_audit.json").write_text(json.dumps(protocol_audit, indent=2, ensure_ascii=False), encoding="utf-8")
    events_df.to_csv(out / "frozen_10_events.csv", index=False, encoding="utf-8-sig")
    (out / "baseline16_method_provenance.json").write_text(json.dumps(provenance(), indent=2, ensure_ascii=False), encoding="utf-8")

    all_path = out / "baseline16_all_seed_results.csv"
    ev_path = out / "baseline16_eventwise_results.csv"
    eff_path = out / "baseline16_efficiency_all_runs.csv"
    resume = not args.no_resume
    rows = pd.read_csv(all_path).to_dict("records") if resume and all_path.exists() else []
    event_rows = pd.read_csv(ev_path).to_dict("records") if resume and ev_path.exists() else []
    eff_rows = pd.read_csv(eff_path).to_dict("records") if resume and eff_path.exists() else []

    def completed(method, seed):
        safe_method = (
            str(method)
            .replace(" ", "_")
            .replace("/", "_")
        )

        npz_path = (
                score_dir
                / f"{safe_method}_seed_{int(seed)}.npz"
        )

        if not npz_path.exists():
            return False

        try:
            with np.load(
                    npz_path,
                    allow_pickle=False
            ) as z:

                if "score" not in z.files:
                    return False

                if "train_score" not in z.files:
                    return False

                if z["score"].size == 0:
                    return False

                if z["train_score"].size == 0:
                    return False

            return True

        except Exception:
            return False

    for mi, method in enumerate(methods, 1):
        print("\n" + "="*88); print(f"METHOD {mi}/{len(methods)}: {method}"); print("="*88)
        for si, seed in enumerate(seeds, 1):
            if resume and completed(method, seed):
                print(f"[{method}] seed={seed} completed; skip."); continue
            print(f"--- {method} | seed {seed} ({si}/{len(seeds)}) ---", flush=True)
            set_seed(seed)
            if device.type == "cuda": torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
            t_run = time.perf_counter()
            score = train_score = None; model = None

            if method == "PCA":
                t0 = time.perf_counter()
                pca = PCA(n_components=0.95, svd_solver="full").fit(x_clean[:train_end])
                tr = pca.inverse_transform(pca.transform(x_clean)); ev = pca.inverse_transform(pca.transform(x_eval))
                train_score = ((x_clean - tr)**2).mean(1); score = ((x_eval - ev)**2).mean(1)
                train_seconds = time.perf_counter() - t0
                pca_sample = x_clean[:60]
                lat_m, lat_s = profile_cpu_callable(lambda: pca.inverse_transform(pca.transform(pca_sample)))
                pca_macs = float(2 * len(pca_sample) * pca_sample.shape[1] * int(pca.n_components_))
                efficiency = {"TrainableParams":0,"EstimatedMACs_1x60":pca_macs,"EstimatedFLOPs_1x60":2*pca_macs,
                              "InferenceLatencyMs_B1_mean":lat_m,"InferenceLatencyMs_B1_std":lat_s,"InferencePeakGPUMB":np.nan,
                              "ClassicalModelUnits":int(pca.n_components_)}
            elif method == "Isolation Forest":
                t0 = time.perf_counter()
                iso = IsolationForest(n_estimators=300, max_samples="auto", contamination="auto", random_state=seed, n_jobs=-1).fit(x_clean[:train_end])
                train_score = -iso.decision_function(x_clean); score = -iso.decision_function(x_eval)
                train_seconds = time.perf_counter() - t0
                iso_sample = x_clean[:60]
                lat_m, lat_s = profile_cpu_callable(lambda: iso.decision_function(iso_sample))
                efficiency = {"TrainableParams":0,"EstimatedMACs_1x60":np.nan,"EstimatedFLOPs_1x60":np.nan,
                              "InferenceLatencyMs_B1_mean":lat_m,"InferenceLatencyMs_B1_std":lat_s,"InferencePeakGPUMB":np.nan,
                              "ClassicalModelUnits":int(sum(t.tree_.node_count for t in iso.estimators_))}
            else:
                mc = METHOD_CONFIGS[method]
                epochs = 1 if args.quick_check else mc.epochs
                bounded_methods = {"USAD", "TranAD"}
                method_x_clean = x_clean_bounded if method in bounded_methods else x_clean
                method_x_eval = x_eval_bounded if method in bounded_methods else x_eval
                train_loader, clean_loader, eval_loader = make_loaders(
                    method_x_clean, method_x_eval, train_end, 60, 5, mc.batch_size, seed, device
                )
                model = make_model(method, 60, len(RAW_COLS), mc.hidden, mc.latent, mc.dropout).to(device)
                t0 = time.perf_counter(); train_model(
                    model, train_loader, device, method, epochs, mc.lr, mc.weight_decay,
                    optimizer_kind=mc.optimizer, grad_clip=mc.grad_clip
                )
                train_seconds = time.perf_counter() - t0
                train_score = aggregate_scores(model, clean_loader, n, 60, device)
                score = aggregate_scores(model, eval_loader, n, 60, device)
                efficiency = profile_inference(model, device, 60, len(RAW_COLS))
                del train_loader, clean_loader, eval_loader

            met = evaluate_score(method, seed, train_score, score, train_end, labels, events_df, eval_mask, q=0.99)
            met["TrainingSeconds"] = float(train_seconds)
            met["TotalRunSeconds"] = float(time.perf_counter() - t_run)
            met["PackageVersion"] = PACKAGE_VERSION
            rows.append(met)
            event_rows.extend(eventwise_rows(method, seed, score, float(met["Threshold"]), events_df))
            er = {"Variant": method, "Seed": int(seed), "PackageVersion": PACKAGE_VERSION, **efficiency, "TrainingSeconds": float(train_seconds)}
            if device.type == "cuda": er["TrainingPeakGPUMB"] = float(torch.cuda.max_memory_allocated(device)/(1024**2))
            else: er["TrainingPeakGPUMB"] = np.nan
            eff_rows.append(er)

            if not args.no_save_scores:
                np.savez_compressed(score_dir / f"{method.replace(' ','_').replace('/','_')}_seed_{seed}.npz",
                                    score=score.astype(np.float32), train_score=train_score.astype(np.float32),
                                    threshold=np.asarray([met["Threshold"]], dtype=np.float64))

            write_checkpoint(rows, event_rows, eff_rows, out)
            print(
                f"[{method}|{seed}] PointF1={met['PointF1']:.4f} ROC={met['ROC_AUC']:.4f} "
                f"AP={met['PR_AUC_AP']:.4f} FAR={met['FullFAR']:.4f} "
                f"EventR={met['EventRecall']:.4f} EventF1={met['EventF1']:.4f} "
                f"Delay={met['DelayMeanMin']:.1f} Merge120F1={met['Merge120EventF1']:.4f}", flush=True
            )
            if model is not None: del model
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    all_df = pd.DataFrame(rows).drop_duplicates(subset=["Variant","Seed"], keep="last")
    rank = {m:i for i,m in enumerate(METHOD_ORDER)}
    all_df["_rank"] = all_df["Variant"].map(rank); all_df = all_df.sort_values(["_rank","Seed"]).drop(columns="_rank")
    all_df.to_csv(all_path, index=False, encoding="utf-8-sig")

    summary = summarize_all_metrics(all_df)
    summary.to_csv(out / "baseline16_mean_std.csv", index=False, encoding="utf-8-sig")
    paper = build_paper_table(summary); paper.to_csv(out / "baseline16_paper_table.csv", index=False, encoding="utf-8-sig")

    # Wide table containing every numeric V4 metric (plus timing fields) as mean ± std.
    full_wide_rows = []
    for m in METHOD_ORDER:
        d = summary[summary["Method"] == m]
        if d.empty: continue
        row = {"Method": m}
        for _, rr in d.iterrows():
            row[str(rr["Metric"])] = f"{float(rr['Mean']):.6f} ± {float(rr['Std']):.6f}"
        full_wide_rows.append(row)
    pd.DataFrame(full_wide_rows).to_csv(
        out / "baseline16_full_metrics_mean_std_wide.csv", index=False, encoding="utf-8-sig"
    )

    eff_df = pd.DataFrame(eff_rows).drop_duplicates(subset=["Variant","Seed"], keep="last")
    eff_summary = []
    efficiency_fields = [
        ("TrainableParams", "TrainableParams"),
        ("EstimatedMACs_1x60", "EstimatedMACs_1x60"),
        ("EstimatedFLOPs_1x60", "EstimatedFLOPs_1x60"),
        ("InferenceLatencyMs_B1_mean", "InferenceLatencyMs_B1"),
        ("InferencePeakGPUMB", "InferencePeakGPUMB"),
        ("TrainingSeconds", "TrainingSeconds"),
        ("TrainingPeakGPUMB", "TrainingPeakGPUMB"),
    ]
    for m in METHOD_ORDER:
        part = eff_df[eff_df["Variant"] == m]
        if part.empty: continue
        row = {"Method":m,"Runs":len(part)}
        for src, out_name in efficiency_fields:
            if src in part:
                v = pd.to_numeric(part[src], errors="coerce").to_numpy(float); v=v[np.isfinite(v)]
                row[out_name+"_mean"] = float(v.mean()) if len(v) else np.nan
                row[out_name+"_std"] = sample_std(v)
        eff_summary.append(row)
    baseline_eff_df = pd.DataFrame(eff_summary)
    baseline_eff_df.to_csv(out / "baseline16_efficiency.csv", index=False, encoding="utf-8-sig")




    report = {
        "experiment":"MSDSNet_Baseline16_ProtocolV2p1_FINAL",
        "protocol":protocol_audit,
        "methods":list(METHOD_ORDER),
        "seeds":seeds,
        "method_provenance":provenance(),
        "all_v4_metrics":list(V4_METRICS),
        "paper_metrics":list(PAPER_METRICS),
        "baseline_all_seed_results":all_df.to_dict("records"),
        "baseline_mean_std":summary.to_dict("records"),
        "baseline_efficiency":baseline_eff_df.to_dict("records"),
    }
    (out / "baseline16_full_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\nDONE")
    print("All V4 metrics are stored in baseline16_all_seed_results.csv and baseline16_full_report.json")
    print("Compact paper view: comparison17_paper_table.csv")


if __name__ == "__main__":
    main()
