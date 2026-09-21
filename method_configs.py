from dataclasses import dataclass
from typing import Dict, Tuple

SEEDS_10: Tuple[int, ...] = (42, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032)

METHOD_ORDER: Tuple[str, ...] = (
    "PCA", "Isolation Forest",
    "USAD", "OmniAnomaly", "MTAD-GAT", "InterFusion", "GDN", "TranAD",
    "DCdetector", "TimesNet", "MEMTO", "CATCH",
    "MDGAD", "MSHTrans", "CKDGAT", "MAD-DGTD",
)

@dataclass(frozen=True)
class MethodConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    hidden: int
    latent: int
    dropout: float
    optimizer: str = "Adam"
    grad_clip: float = 5.0
    notes: str = ""

# Method-internal architecture settings are source-guided. The benchmark itself
# is NOT configurable here: raw 7 channels, 60-min windows, 0-55 clean train,
# 55-70 evaluation, frozen B1-R5 events, per-method clean Q0.99, and V4 metrics.
# All deep methods are deliberately locked to 30 epochs for this revision study.
METHOD_CONFIGS: Dict[str, MethodConfig] = {
    "USAD": MethodConfig(30, 128, 1e-3, 0.0, 256, 64, 0.0, "Adam", 5.0,
        "public-repo-guided: shared encoder; two mirrored sigmoid decoders; two epoch-dependent adversarial objectives; clean-train MinMax input"),
    "OmniAnomaly": MethodConfig(30, 128, 1e-3, 0.0, 64, 16, 0.0, "Adam", 5.0,
        "official-repo/paper-guided stochastic recurrent VAE with GRU, conditional prior/posterior, planar normalizing flows, reconstruction NLL"),
    "MTAD-GAT": MethodConfig(30, 128, 1e-4, 0.0, 32, 8, 0.30, "Adam", 0.1,
        "paper/reproduction-guided: Conv1D; parallel feature- and time-oriented complete GATs; GRU; forecasting + VAE reconstruction"),
    "InterFusion": MethodConfig(30, 128, 5e-4, 0.0, 32, 4, 0.30, "Adam", 5.0,
        "official-repo/paper-guided hierarchical VAE with jointly trained inter-metric and temporal stochastic embeddings"),
    "GDN": MethodConfig(30, 128, 1e-3, 0.0, 16, 8, 0.30, "Adam", 5.0,
        "official-repo-guided: learned node embeddings, cosine top-k graph, graph attention, sensor-wise 10-step history, 256-unit output MLP"),
    "TranAD": MethodConfig(30, 128, 1e-3, 0.0, 64, 32, 0.20, "AdamW", 5.0,
        "official-repo-guided: d_model=2*n_features, nhead=n_features, one encoder and two self-conditioned Transformer decoders; clean-train MinMax input"),
    "DCdetector": MethodConfig(30, 128, 1e-4, 0.0, 256, 32, 0.10, "Adam", 5.0,
        "official-repo-semantics: channel-independent multi-scale patch/in-patch dual association; 3 encoder levels; d_model=256; pure KL contrastive objective"),
    "TimesNet": MethodConfig(30, 128, 1e-3, 0.0, 32, 64, 0.10, "Adam", 5.0,
        "official Time-Series-Library core: FFT period discovery; 3 TimesBlocks; d_model=128,d_ff=128,top_k=3,num_kernels=6"),
    "MEMTO": MethodConfig(30, 128, 1e-4, 0.0, 128, 32, 0.10, "Adam", 5.0,
        "official-repo-guided: 3-layer Transformer encoder; learnable memory update/read gate; two-phase K-means memory init; input+latent deviation"),
    "CATCH": MethodConfig(30, 128, 1e-4, 0.0, 128, 64, 0.10, "Adam", 5.0,
        "official-repo-guided: frequency patching; CFM patch-wise mask generator; masked channel attention; 3 spectral Transformer layers; auxiliary CFM objective"),
    "MDGAD": MethodConfig(30, 128, 1e-5, 0.0, 64, 16, 0.30, "Adam", 5.0,
        "paper-guided (no reliable official repo found): multi-scale evolving graph learners + bidirectional prediction errors"),
    "MSHTrans": MethodConfig(30, 128, 1e-3, 0.0, 128, 32, 0.10, "Adam", 5.0,
        "official-repo/paper-guided: multi-scale downsampling; adaptive trainable hypergraphs; decomposition; hypergraph Transformer; multi-scale reconstruction fusion"),
    "CKDGAT": MethodConfig(30, 128, 1e-3, 0.0, 128, 32, 0.10, "Adam", 5.0,
        "paper-guided (2026): data-oriented GAT + knowledge-oriented stochastic multi-head GAT + temporal fusion + reconstruction"),
    "MAD-DGTD": MethodConfig(30, 128, 1e-3, 0.0, 64, 16, 0.20, "Adam", 5.0,
        "paper-guided (2025): TDDG static/dynamic/delay graph fusion + stacked multi-scale dilated TDIE + GCIP + prediction"),
}

PAPER_METRICS: Tuple[str, ...] = (
    "PointPrecision", "PointRecall", "PointF1", "ROC_AUC", "PR_AUC_AP",
    "FullFAR", "EventRecall", "EventF1", "DelayMeanMin", "Merge120EventF1",
)

V4_METRICS: Tuple[str, ...] = (
    "PointPrecision", "PointRecall", "PointF1", "ROC-AUC", "PR_AUC", "FAR", "MDR", "Detected",
    "ROC_AUC", "PR_AUC_AP", "FullFAR",
    "AdjPrecision", "AdjRecall", "AdjF1",
    "EventPrecision", "EventRecall", "EventF1", "PredEventCount", "FalsePredEventCount", "MatchedEventCount",
    "DetectedEvents", "NumEvents", "DelayMeanMin", "DelayMedianMin", "IndependentMaxDelayMin",
    "AlarmEpisodes", "MatchedAlarmEpisodes", "Delay", "DelayDetectedOnly",
    "Merge120EventPrecision", "Merge120EventRecall", "Merge120EventF1",
    "Merge120PredEventCount", "Merge120FalsePredEventCount", "Merge120MatchedEventCount", "Merge120DelayMeanMin",
    "Threshold", "EvalPoints",
)
