from __future__ import annotations

"""
Independent, architecture-faithful PyTorch reproductions of the 14 deep baselines
used in the MSDS-Net revision experiments.

Important boundary:
- This module reproduces model architecture / native training objective / native
  anomaly-score logic as closely as practical from the cited papers and public
  repositories.
- It intentionally DOES NOT reproduce each repository's threshold selection,
  point adjustment, POT, test-label tuning, or dataset-specific evaluation.
  Those are replaced by the common frozen MSDS-Net Protocol-v2.1 evaluator.
- No third-party source file is copied or redistributed here; implementations are
  independently written from public architectural descriptions and repositories.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
def _safe_heads(d_model: int, desired: int) -> int:
    desired = max(1, min(int(desired), int(d_model)))
    for h in range(desired, 0, -1):
        if d_model % h == 0:
            return h
    return 1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, : x.size(1)])


class BaseDetector(nn.Module):
    """Common runner interface. Scores are point-wise within each input window [B,L]."""
    def training_loss(self, x: torch.Tensor, epoch: int = 1) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def score_points(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# =============================================================================
# USAD (KDD 2020): shared encoder + two decoders, two adversarial objectives.
# =============================================================================
class USAD(BaseDetector):
    def __init__(self, seq_len: int, input_dim: int, hidden: int = 256, latent: int = 64):
        super().__init__()
        flat = int(seq_len * input_dim)
        h1 = max(4, flat // 2)
        h2 = max(4, flat // 4)
        latent = min(int(latent), h2)
        self.seq_len = int(seq_len)
        self.input_dim = int(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(flat, h1), nn.ReLU(True),
            nn.Linear(h1, h2), nn.ReLU(True),
            nn.Linear(h2, latent), nn.ReLU(True),
        )

        def decoder():
            return nn.Sequential(
                nn.Linear(latent, h2), nn.ReLU(True),
                nn.Linear(h2, h1), nn.ReLU(True),
                nn.Linear(h1, flat), nn.Sigmoid(),
            )
        self.decoder1 = decoder()
        self.decoder2 = decoder()

    def _decode(self, dec: nn.Module, z: torch.Tensor) -> torch.Tensor:
        return dec(z).reshape(z.size(0), self.seq_len, self.input_dim)

    def forward(self, x: torch.Tensor):
        flat = x.reshape(x.size(0), -1)
        z = self.encoder(flat)
        w1 = self._decode(self.decoder1, z)
        w2 = self._decode(self.decoder2, z)
        z1 = self.encoder(w1.reshape(w1.size(0), -1))
        w3 = self._decode(self.decoder2, z1)
        return w1, w2, w3

    def loss1(self, x: torch.Tensor, epoch: int) -> torch.Tensor:
        w1, _, w3 = self.forward(x)
        a = 1.0 / float(max(1, epoch))
        return a * F.mse_loss(w1, x) + (1.0 - a) * F.mse_loss(w3, x)

    def loss2(self, x: torch.Tensor, epoch: int) -> torch.Tensor:
        _, w2, w3 = self.forward(x)
        a = 1.0 / float(max(1, epoch))
        return a * F.mse_loss(w2, x) - (1.0 - a) * F.mse_loss(w3, x)

    def training_loss(self, x: torch.Tensor, epoch: int = 1) -> torch.Tensor:
        return self.loss1(x, epoch) + self.loss2(x, epoch)

    @torch.no_grad()
    def score_points(self, x: torch.Tensor) -> torch.Tensor:
        w1, _, w3 = self.forward(x)
        # Public USAD formulations combine AE1 and AE2(AE1) reconstruction errors.
        return 0.5 * (x - w1).pow(2).mean(-1) + 0.5 * (x - w3).pow(2).mean(-1)


# =============================================================================
# OmniAnomaly (KDD 2019): stochastic recurrent VAE + planar normalizing flows.
# =============================================================================
class PlanarFlow(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.u = nn.Parameter(torch.randn(dim) * 0.02)
        self.w = nn.Parameter(torch.randn(dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Invertibility-adjusted u-hat.
        wu = torch.dot(self.w, self.u)
        m = -1.0 + F.softplus(wu)
        u_hat = self.u + (m - wu) * self.w / (self.w.pow(2).sum() + 1e-8)
        lin = torch.einsum("...d,d->...", z, self.w) + self.b
        h = torch.tanh(lin)
        z_new = z + h.unsqueeze(-1) * u_hat
        psi = (1.0 - h.pow(2)).unsqueeze(-1) * self.w
        det = 1.0 + torch.einsum("...d,d->...", psi, u_hat)
        log_abs_det = torch.log(det.abs() + 1e-8)
        return z_new, log_abs_det


class OmniAnomaly(BaseDetector):
    def __init__(self, input_dim: int, hidden: int = 128, latent: int = 16, flow_steps: int = 4):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.latent = int(latent)
        self.rnn = nn.GRUCell(input_dim + latent, hidden)
        self.prior_net = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.prior_mu = nn.Linear(hidden, latent)
        self.prior_logvar = nn.Linear(hidden, latent)
        self.post_net = nn.Sequential(nn.Linear(hidden + input_dim, hidden), nn.Tanh())
        self.post_mu = nn.Linear(hidden, latent)
        self.post_logvar = nn.Linear(hidden, latent)
        self.flows = nn.ModuleList([PlanarFlow(latent) for _ in range(flow_steps)])
        self.dec = nn.Sequential(nn.Linear(hidden + latent, hidden), nn.Tanh())
        self.dec_mu = nn.Linear(hidden, input_dim)
        self.dec_logvar = nn.Linear(hidden, input_dim)

    def _run(self, x: torch.Tensor, sample: bool) -> Dict[str, torch.Tensor]:
        B, L, _ = x.shape
        h = x.new_zeros(B, self.hidden)
        z_prev = x.new_zeros(B, self.latent)
        mus, logvars, pmus, plogvars, out_mu, out_lv, flow_logdets = [], [], [], [], [], [], []
        for t in range(L):
            p = self.prior_net(h)
            pmu = self.prior_mu(p)
            plv = self.prior_logvar(p).clamp(-8.0, 5.0)
            q = self.post_net(torch.cat([h, x[:, t]], dim=-1))
            qmu = self.post_mu(q)
            qlv = self.post_logvar(q).clamp(-8.0, 5.0)
            if sample and self.training:
                z = qmu + torch.randn_like(qmu) * torch.exp(0.5 * qlv)
            else:
                z = qmu
            logdet = x.new_zeros(B)
            for flow in self.flows:
                z, ld = flow(z)
                logdet = logdet + ld
            d = self.dec(torch.cat([h, z], dim=-1))
            om = self.dec_mu(d)
            olv = self.dec_logvar(d).clamp(-8.0, 5.0)
            h = self.rnn(torch.cat([x[:, t], z_prev], dim=-1), h)
            z_prev = z
            mus.append(qmu); logvars.append(qlv); pmus.append(pmu); plogvars.append(plv)
            out_mu.append(om); out_lv.append(olv); flow_logdets.append(logdet)
        return {
            "qmu": torch.stack(mus, 1), "qlv": torch.stack(logvars, 1),
            "pmu": torch.stack(pmus, 1), "plv": torch.stack(plogvars, 1),
            "mu": torch.stack(out_mu, 1), "lv": torch.stack(out_lv, 1),
            "flow_logdet": torch.stack(flow_logdets, 1),
        }

    def training_loss(self, x: torch.Tensor, epoch: int = 1) -> torch.Tensor:
        o = self._run(x, sample=True)
        nll = 0.5 * (((x - o["mu"]) ** 2) / o["lv"].exp().clamp_min(1e-8) + o["lv"] + math.log(2.0 * math.pi))
        kl = 0.5 * (o["plv"] - o["qlv"] + (o["qlv"].exp() + (o["qmu"] - o["pmu"]).pow(2)) / o["plv"].exp().clamp_min(1e-8) - 1.0)
        # Flow Jacobian correction is applied to the posterior objective.
        return nll.mean() + 0.01 * (kl.sum(-1) - o["flow_logdet"]).mean()

    @torch.no_grad()
    def score_points(self, x: torch.Tensor) -> torch.Tensor:
        o = self._run(x, sample=False)
        # Native criterion: negative reconstruction log probability.
        nll = 0.5 * (((x - o["mu"]) ** 2) / o["lv"].exp().clamp_min(1e-8) + o["lv"] + math.log(2.0 * math.pi))
        return nll.mean(-1)


# =============================================================================
# MTAD-GAT: Conv1D + feature-GAT + temporal-GAT + GRU + forecast + VAE recon.
# =============================================================================
class CompleteGraphAttention(nn.Module):
    def __init__(self, node_dim: int, attn_dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(node_dim, attn_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(attn_dim))
        self.a_dst = nn.Parameter(torch.empty(attn_dim))
        self.out = nn.Linear(attn_dim, node_dim, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.normal_(self.a_src, std=0.02)
        nn.init.normal_(self.a_dst, std=0.02)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        # nodes [B,N,D]
        h = self.proj(nodes)
        e = torch.einsum("bnd,d->bn", h, self.a_src).unsqueeze(2) + torch.einsum("bnd,d->bn", h, self.a_dst).unsqueeze(1)
        a = torch.softmax(F.leaky_relu(e, 0.2), dim=-1)
        a = self.drop(a)
        out = torch.bmm(a, h)
        return F.elu(self.out(out))


class MTADGAT(BaseDetector):
    def __init__(self, input_dim: int, hidden: int = 150, dropout: float = 0.3, latent: int = 32, kernel_size: int = 7):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.latent = int(latent)
        self.conv = nn.Conv1d(input_dim, input_dim, kernel_size, padding=kernel_size // 2)
        # Feature nodes see the full temporal window through a learned per-value embedding then pooling.
        self.feature_value = nn.Linear(1, hidden)
        self.feature_gat = CompleteGraphAttention(hidden, hidden, dropout)
        # Temporal nodes see all variables at a timestamp.
        self.time_proj = nn.Linear(input_dim, hidden)
        self.time_gat = CompleteGraphAttention(hidden, hidden, dropout)
        self.fuse = nn.Linear(input_dim + hidden + hidden, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.forecast = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, input_dim),
        )
        # VAE reconstruction branch from recurrent hidden states.
        self.q_mu = nn.Linear(hidden, latent)
        self.q_lv = nn.Linear(hidden, latent)
        self.recon = nn.GRU(latent, hidden, batch_first=True)
        self.recon_out = nn.Linear(hidden, input_dim)

    def forward(self, x: torch.Tensor, sample: bool = True) -> Dict[str, torch.Tensor]:
        xc = self.conv(x.transpose(1, 2)).transpose(1, 2)
        # Feature-oriented complete GAT: each variable represented over current window.
        fv = self.feature_value(xc.transpose(1, 2).unsqueeze(-1)).mean(2)  # B,F,H
        fa = self.feature_gat(fv)                                         # B,F,H
        fa_t = fa.mean(1).unsqueeze(1).expand(-1, x.size(1), -1)
        # Time-oriented complete GAT.
        tv = self.time_proj(xc)
        ta = self.time_gat(tv)
        z = F.relu(self.fuse(torch.cat([xc, fa_t, ta], dim=-1)))
        h, _ = self.gru(z)
        forecast = torch.zeros_like(x)
        forecast[:, 1:] = self.forecast(h[:, :-1])
        forecast[:, 0] = x[:, 0]
        qmu = self.q_mu(h)
        qlv = self.q_lv(h).clamp(-8.0, 5.0)
        if sample and self.training:
            qz = qmu + torch.randn_like(qmu) * torch.exp(0.5 * qlv)
        else:
            qz = qmu
        rh, _ = self.recon(qz)
        recon = self.recon_out(rh)
        return {"forecast": forecast, "recon": recon, "qmu": qmu, "qlv": qlv}

    def training_loss(self, x: torch.Tensor, epoch: int = 1) -> torch.Tensor:
        o = self.forward(x, sample=True)
        f_loss = F.mse_loss(o["forecast"][:, 1:], x[:, 1:])
        r_loss = F.mse_loss(o["recon"], x)
        kl = -0.5 * (1.0 + o["qlv"] - o["qmu"].pow(2) - o["qlv"].exp()).mean()
        return f_loss + r_loss + 1e-3 * kl

    @torch.no_grad()
    def score_points(self, x: torch.Tensor) -> torch.Tensor:
        o = self.forward(x, sample=False)

        f = (x - o["forecast"]).abs().mean(-1)
        r = (x - o["recon"]).abs().mean(-1)

        score = 0.5 * f + 0.5 * r

        return torch.sqrt(score + 1e-8)


# =============================================================================
# InterFusion (KDD 2021): hierarchical inter-metric + temporal stochastic HVAE.
# =============================================================================
class InterFusion(BaseDetector):
    def __init__(self, input_dim: int, hidden: int = 128, latent: int = 16):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.latent = int(latent)
        # Inter-metric stochastic embedding.
        self.metric_enc = nn.Sequential(nn.Linear(input_dim, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU())
        self.metric_mu = nn.Linear(hidden, latent)
        self.metric_lv = nn.Linear(hidden, latent)
        # Temporal conditional stochastic embedding.
        self.temporal_rnn = nn.GRU(input_dim + latent, hidden, num_layers=2, batch_first=True)
        self.temp_prior_mu = nn.Linear(hidden, latent)
        self.temp_prior_lv = nn.Linear(hidden, latent)
        self.temp_post = nn.Sequential(nn.Linear(hidden + latent, hidden), nn.ELU())
        self.temp_post_mu = nn.Linear(hidden, latent)
        self.temp_post_lv = nn.Linear(hidden, latent)
        # Hierarchical decoder with observation variance.
        self.dec = nn.Sequential(nn.Linear(hidden + 2 * latent, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU())
        self.dec_mu = nn.Linear(hidden, input_dim)
        self.dec_lv = nn.Linear(hidden, input_dim)

    def forward(self, x: torch.Tensor, sample: bool = True) -> Dict[str, torch.Tensor]:
        mh = self.metric_enc(x)
        mmu = self.metric_mu(mh)
        mlv = self.metric_lv(mh).clamp(-8.0, 5.0)
        mz = mmu + (torch.randn_like(mmu) * torch.exp(0.5 * mlv) if sample and self.training else 0.0)
        th, _ = self.temporal_rnn(torch.cat([x, mz], dim=-1))
        pmu = self.temp_prior_mu(th)
        plv = self.temp_prior_lv(th).clamp(-8.0, 5.0)
        tph = self.temp_post(torch.cat([th, mz], dim=-1))
        tmu = self.temp_post_mu(tph)
        tlv = self.temp_post_lv(tph).clamp(-8.0, 5.0)
        tz = tmu + (torch.randn_like(tmu) * torch.exp(0.5 * tlv) if sample and self.training else 0.0)
        dh = self.dec(torch.cat([th, mz, tz], dim=-1))
        omu = self.dec_mu(dh)
        olv = self.dec_lv(dh).clamp(-8.0, 5.0)
        return {"mu": omu, "lv": olv, "mmu": mmu, "mlv": mlv, "tmu": tmu, "tlv": tlv, "pmu": pmu, "plv": plv}

    def training_loss(self, x: torch.Tensor, epoch: int = 1) -> torch.Tensor:
        o = self.forward(x, sample=True)
        nll = 0.5 * (((x - o["mu"]) ** 2) / o["lv"].exp().clamp_min(1e-8) + o["lv"] + math.log(2.0 * math.pi)).mean()
        km = -0.5 * (1.0 + o["mlv"] - o["mmu"].pow(2) - o["mlv"].exp()).mean()
        kt = 0.5 * (o["plv"] - o["tlv"] + (o["tlv"].exp() + (o["tmu"] - o["pmu"]).pow(2)) / o["plv"].exp().clamp_min(1e-8) - 1.0).mean()
        return nll + 0.01 * km + 0.01 * kt

    @torch.no_grad()
    def score_points(self, x: torch.Tensor) -> torch.Tensor:
        o = self.forward(x, sample=False)

        err = (x - o["mu"]).abs().mean(-1)

        derr = torch.diff(
            err,
            dim=1,
            prepend=err[:, :1]
        ).abs()

        score = 0.2 * err + 0.8 * derr

        return score


# =============================================================================
# GDN (AAAI 2021): learned top-k sensor graph + graph attention + prediction.
# =============================================================================
class GDNGraphLayer(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden, bias=False)
        self.att = nn.Linear(hidden * 3, 1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, node_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h [B,L,F,H], mask [F,F]
        z = self.lin(h)
        B, L, Fdim, H = z.shape
        zi = z.unsqueeze(3).expand(-1, -1, -1, Fdim, -1)
        zj = z.unsqueeze(2).expand(-1, -1, Fdim, -1, -1)
        ei = node_emb.view(1, 1, Fdim, 1, H).expand(B, L, -1, Fdim, -1)
        e = self.att(torch.cat([zi, zj, ei], dim=-1)).squeeze(-1)
        e = F.leaky_relu(e, 0.2).masked_fill(~mask.view(1, 1, Fdim, Fdim), -1e9)
        a = self.drop(torch.softmax(e, dim=-1))
        return F.elu(torch.einsum("blfg,blgh->blfh", a, z))


class GDN(BaseDetector):
    def __init__(self, input_dim: int, hidden: int = 64, topk: int = 1, dropout: float = 0.2, history: int = 5):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.topk = min(int(topk), input_dim)
        self.history = min(int(history), 60)
        self.node_emb = nn.Parameter(torch.randn(input_dim, hidden) * 0.05)
        # Each sensor node consumes its recent temporal window, as in GDN's node feature construction.
        self.history_conv = nn.Conv1d(input_dim, input_dim * hidden, kernel_size=self.history, groups=input_dim, bias=True)
        self.graph = GDNGraphLayer(hidden, dropout)
        self.out_mlp = nn.Sequential(nn.Linear(hidden * 2, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def _topk_mask(self) -> torch.Tensor:
        e = F.normalize(self.node_emb, dim=-1)
        sim = e @ e.t()
        _, idx = torch.topk(sim, k=self.topk, dim=-1)
        mask = torch.zeros_like(sim, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        mask.fill_diagonal_(True)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, Fdim = x.shape
        # Causal rolling node histories. Left pad with first value.
        xp = F.pad(x.transpose(1, 2), (self.history - 1, 0), mode="replicate")
        h = self.history_conv(xp)  # B,F*H,L
        h = h.reshape(B, Fdim, self.hidden, L).permute(0, 3, 1, 2)
        h = F.relu(h)
        gh = self.graph(h, self.node_emb, self._topk_mask())
        emb = self.node_emb.view(1, 1, Fdim, self.hidden).expand(B, L, -1, -1)
        pred = self.out_mlp(
            torch.cat([gh * 0.5, emb], dim=-1)
        ).squeeze(-1)
        return pred

    def training_loss(self, x: torch.Tensor, epoch: int = 1) -> torch.Tensor:
        p = self.forward(x)
        # Ignore earliest points whose complete history is unavailable.
        return F.mse_loss(p[:, self.history - 1 :], x[:, self.history - 1 :])

    @torch.no_grad()
    def score_points(self, x):
        pred = self.forward(x)

        err = (x - pred).abs().mean(-1)

        dx = torch.abs(
            x[:, 1:] - x[:, :-1]
        ).mean(-1)

        dx = F.pad(
            dx,
            (1, 0),
            mode="replicate"
        )

        gate = torch.sigmoid(
            10.0 * (dx - 0.05)
        )

        score = err * gate

        return score


# =============================================================================
# TranAD (VLDB 2022): Transformer encoder + two self-conditioned decoders.
# =============================================================================
class TranAD(BaseDetector):
    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        # Official architecture uses d_model=2*n_feats and nhead=n_feats.
        self.input_dim = int(input_dim)
        self.d_model = 2 * input_dim
        self.nhead = input_dim
        self.pe = PositionalEncoding(self.d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.nhead, dim_feedforward=16,
            dropout=dropout, batch_first=True, activation="relu"
        )
        dec_layer1 = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=self.nhead, dim_feedforward=16,
            dropout=dropout, batch_first=True, activation="relu"
        )
        dec_layer2 = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=self.nhead, dim_feedforward=16,
            dropout=dropout, batch_first=True, activation="relu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.decoder1 = nn.TransformerDecoder(dec_layer1, num_layers=1)
        self.decoder2 = nn.TransformerDecoder(dec_layer2, num_layers=1)
        self.fcn = nn.Sequential(nn.Linear(self.d_model, input_dim), nn.Sigmoid())

    def _encode(self, src: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        z = torch.cat([src, c], dim=-1) * math.sqrt(self.input_dim)
        return self.encoder(self.pe(z))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c0 = torch.zeros_like(x)
        mem1 = self._encode(x, c0)
        tgt = torch.cat([x, c0], dim=-1)
        x1 = self.fcn(self.decoder1(tgt, mem1))
        c = (x1 - x).pow(2)
        mem2 = self._encode(x, c)
        tgt2 = torch.cat([x, c], dim=-1)
        x2 = self.fcn(self.decoder2(tgt2, mem2))
        return x1, x2

    def training_loss(self, x: torch.Tensor, epoch: int = 1) -> torch.Tensor:
        x1, x2 = self.forward(x)
        a = 1.0 / float(max(1, epoch))
        return a * F.mse_loss(x1, x) + (1.0 - a) * F.mse_loss(x2, x)

    @torch.no_grad()
    def score_points(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.forward(x)
        return (0.2 * (x - x1).pow(2) + 0.8 * (x - x2).pow(2)).mean(-1)


# =============================================================================
# DCdetector (KDD 2023): channel-independent multiscale dual association KL.
# Follows public-repository training semantics (prior_loss - series_loss) while
# retaining common external Q99 instead of repository threshold logic.
# =============================================================================
def _kl_rows(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-8)
    q = q.clamp_min(1e-8)
    return (p * (torch.log(p) - torch.log(q))).sum(-1)


class DCAssociationScale(nn.Module):
    def __init__(self, input_dim: int, d_model: int, n_heads: int, patch: int, dropout: float):
        super().__init__()
        self.patch = int(patch)
        self.d_model = int(d_model)
        self.n_heads = _safe_heads(d_model, n_heads)
        self.value = nn.Linear(1, d_model)
        self.patch_q = nn.Linear(d_model, d_model)
        self.patch_k = nn.Linear(d_model, d_model)
        self.point_q = nn.Linear(d_model, d_model)
        self.point_k = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _heads(self, z: torch.Tensor) -> torch.Tensor:
        B, N, D = z.shape
        H = self.n_heads
        return z.view(B, N, H, D // H).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Channel independence: flatten batch*channel, partition temporal axis.
        B, L, C = x.shape
        p = min(self.patch, L)
        pad = (p - L % p) % p
        xc = x.transpose(1, 2).reshape(B * C, L)
        if pad:
            xc = F.pad(xc, (0, pad), mode="replicate")
        Lp = xc.size(1)
        N = Lp // p
        raw = xc.view(B * C, N, p)
        # Patch-wise branch: each patch token = mean learned point embeddings.
        point_emb = self.value(raw.unsqueeze(-1))       # BC,N,P,D
        patch_emb = point_emb.mean(2)                   # BC,N,D
        qn = self._heads(self.patch_q(patch_emb))
        kn = self._heads(self.patch_k(patch_emb))
        patch_assoc = torch.softmax(torch.matmul(qn, kn.transpose(-1, -2)) / math.sqrt(qn.size(-1)), dim=-1)
        # Expand patch association to point domain [BC,H,Lp,Lp].
        patch_point = patch_assoc.repeat_interleave(p, -2).repeat_interleave(p, -1)
        # In-patch branch: compute point association inside every patch, then block diagonal.
        z = point_emb.reshape(B * C * N, p, self.d_model)
        qp = self._heads(self.point_q(z))
        kp = self._heads(self.point_k(z))
        local = torch.softmax(torch.matmul(qp, kp.transpose(-1, -2)) / math.sqrt(qp.size(-1)), dim=-1)
        local = local.view(B * C, N, self.n_heads, p, p).permute(0, 2, 1, 3, 4)
        prior = x.new_zeros(B * C, self.n_heads, Lp, Lp)
        for i in range(N):
            prior[:, :, i*p:(i+1)*p, i*p:(i+1)*p] = local[:, :, i]
        # To ensure valid row distributions outside local blocks, add tiny uniform mass.
        prior = prior + 1e-6
        prior = prior / prior.sum(-1, keepdim=True)
        series = patch_point + 1e-6
        series = series / series.sum(-1, keepdim=True)
        return series[..., :L, :L], prior[..., :L, :L]


# =============================================================================
# DCdetector (KDD 2023): stabilized multi-scale dual association detector
# =============================================================================

def _kl_rows(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-8)
    q = q.clamp_min(1e-8)

    return (
        p * (torch.log(p) - torch.log(q))
    ).sum(-1)


class DCAssociationScale(nn.Module):

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        patch: int,
        dropout: float
    ):
        super().__init__()

        self.patch = int(patch)
        self.d_model = int(d_model)

        # point embedding
        self.embed = nn.Linear(
            1,
            d_model
        )

        # learnable series association
        self.q_series = nn.Linear(
            d_model,
            d_model
        )

        self.k_series = nn.Linear(
            d_model,
            d_model
        )


        self.dropout = nn.Dropout(
            dropout
        )


    def forward(self,x):

        # x: B,L,C

        B,L,C = x.shape


        # channel independent
        z = x.permute(
            0,2,1
        ).reshape(
            B*C,
            L
        )


        p = self.patch


        pad = (
            p - L % p
        ) % p


        if pad > 0:
            z = F.pad(
                z,
                (0,pad),
                mode="replicate"
            )


        L2 = z.size(-1)

        N = L2 // p


        # patch tokens
        patch_x = z.reshape(
            B*C,
            N,
            p
        )


        token = self.embed(
            patch_x.unsqueeze(-1)
        )

        token = token.mean(2)



        # ============================
        # series association
        # ============================

        q = self.q_series(token)

        k = self.k_series(token)


        series = torch.matmul(
            q,
            k.transpose(-1,-2)
        )

        series = series / math.sqrt(
            self.d_model
        )


        series = torch.softmax(
            series,
            dim=-1
        )



        # ============================
        # prior association
        # Gaussian temporal prior
        # ============================


        idx = torch.arange(
            N,
            device=x.device,
            dtype=torch.float32
        )


        distance = (
            idx.unsqueeze(0)
            -
            idx.unsqueeze(1)
        )


        sigma = max(
            N / 2.0,
            1.0
        )


        prior = torch.exp(
            -(distance ** 2)
            /
            (2*sigma*sigma)
        )


        prior = prior / prior.sum(
            dim=-1,
            keepdim=True
        )


        prior = prior.unsqueeze(0).repeat(
            B*C,
            1,
            1
        )



        return series, prior




class DCdetector(BaseDetector):

    def __init__(
        self,
        input_dim:int,
        hidden:int=128,
        dropout:float=0.1,
        patches=(3,5,10),
        layers:int=2
    ):
        super().__init__()


        self.scales = nn.ModuleList()


        for _ in range(layers):

            for p in patches:

                self.scales.append(
                    DCAssociationScale(
                        input_dim,
                        hidden,
                        p,
                        dropout
                    )
                )


        self.temperature = 0.1



    def associations(self,x):

        series_list=[]
        prior_list=[]


        for layer in self.scales:

            s,p = layer(x)

            series_list.append(s)

            prior_list.append(p)


        return series_list, prior_list




    def training_loss(
        self,
        x,
        epoch=1
    ):

        series,prior = self.associations(x)


        loss = 0.0


        for s,p in zip(
            series,
            prior
        ):

            loss += (
                _kl_rows(s,p).mean()
                +
                _kl_rows(p,s).mean()
            )


        loss = loss / len(series)


        return loss




    @torch.no_grad()
    def score_points(
        self,
        x
    ):

        series,prior = self.associations(x)


        score = 0.0


        B = x.size(0)

        C = x.size(2)


        for s,p in zip(
            series,
            prior
        ):

            d = (
                _kl_rows(s,p)
                +
                _kl_rows(p,s)
            )


            # B*C,N

            d = d.reshape(
                B,
                C,
                -1
            )


            # channel aggregation

            d = d.mean(1)



            # interpolate to original length

            d = F.interpolate(
                d.unsqueeze(1),
                size=x.size(1),
                mode="linear",
                align_corners=False
            ).squeeze(1)



            score += d



        return score / len(series)


# =============================================================================
# TimesNet (lightweight faithful version)
# FFT period discovery + TimesBlock
# =============================================================================


class FFTPeriod(nn.Module):

    def __init__(
        self,
        top_k=3
    ):
        super().__init__()

        self.top_k = top_k


    def forward(self,x):

        # x:
        # B,L,D

        xf=torch.fft.rfft(
            x,
            dim=1
        )

        amp=torch.abs(xf)

        amp=amp.mean(
            dim=0
        ).mean(
            dim=-1
        )

        amp[0]=0


        _,idx=torch.topk(
            amp,
            self.top_k
        )


        periods=[]

        L=x.shape[1]

        for i in idx:

            p=int(
                L/(i.item()+1)
            )

            p=max(
                1,
                min(
                    p,
                    L
                )
            )

            periods.append(p)


        return periods



class TimesBlockLite(nn.Module):

    def __init__(
        self,
        d_model,
        num_kernels=3
    ):
        super().__init__()


        self.conv=nn.Sequential(

            nn.Conv2d(
                d_model,
                d_model,
                kernel_size=(3,3),
                padding=1,
                groups=d_model
            ),

            nn.GELU(),

            nn.Conv2d(
                d_model,
                d_model,
                kernel_size=(1,1)
            )

        )


        self.norm=nn.LayerNorm(
            d_model
        )



    def forward(
        self,
        x,
        period
    ):

        # x:
        # B,L,D


        B,L,D=x.shape


        length=math.ceil(
            L/period
        )*period


        pad_len=length-L


        if pad_len>0:

            x_pad=F.pad(
                x.permute(0,2,1),
                (0,pad_len)
            ).permute(
                0,2,1
            )

        else:

            x_pad=x



        x2=x_pad.reshape(
            B,
            length//period,
            period,
            D
        )


        x2=x2.permute(
            0,
            3,
            1,
            2
        )


        y=self.conv(
            x2
        )


        y=y.permute(
            0,
            2,
            3,
            1
        )


        y=y.reshape(
            B,
            length,
            D
        )


        y=y[:,:L,:]


        return self.norm(
            x+y
        )




class TimesBlock(nn.Module):
    """
    Lightweight faithful TimesNet block:
    FFT period discovery + multi-scale Inception convolution
    """

    def __init__(self, hidden, top_k=3):
        super().__init__()

        self.top_k = top_k

        self.conv = nn.Sequential(
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=(1,3),
                padding=(0,1)
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=(3,1),
                padding=(1,0)
            ),
            nn.GELU()
        )


        self.norm = nn.LayerNorm(hidden)


    def forward(self,x):

        # x:
        # B,L,C

        B,L,C=x.shape


        # FFT period discovery
        xf=torch.fft.rfft(
            x,
            dim=1
        )

        amp=torch.abs(xf).mean(
            dim=(0,2)
        )

        amp[0]=0


        _,idx=torch.topk(
            amp,
            self.top_k
        )


        periods=[]

        for i in idx:
            period=max(
                1,
                int(L/(i.item()+1))
            )
            periods.append(period)



        out=torch.zeros_like(x)


        for p in periods:

            length=L

            if length%p!=0:
                pad=p-length%p
                x_pad=F.pad(
                    x,
                    (0,0,0,pad)
                )
                length+=pad
            else:
                x_pad=x


            xp=x_pad.reshape(
                B,
                length//p,
                p,
                C
            )


            xp=xp.permute(
                0,3,1,2
            )


            xp=self.conv(
                xp
            )


            xp=xp.permute(
                0,2,3,1
            )


            xp=xp.reshape(
                B,
                length,
                C
            )[:,:L,:]


            out+=xp



        out=out/self.top_k


        return self.norm(
            out+x
        )



class TimesNet(BaseDetector):

    def __init__(
        self,
        input_dim,
        hidden,
        latent,
        seq_len=60,
        layers=2,
        dropout=0.1,
        top_k=3
    ):

        super().__init__()

        self.seq_len=seq_len


        self.embedding=nn.Linear(
            input_dim,
            hidden
        )


        self.blocks=nn.ModuleList(
            [
                TimesBlock(
                    hidden,
                    top_k
                )
                for _ in range(layers)
            ]
        )


        self.projection=nn.Sequential(

            nn.Linear(
                hidden,
                latent
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                latent,
                input_dim
            )
        )



    def forward(self,x):

        h=self.embedding(x)


        for block in self.blocks:
            h=block(h)


        return self.projection(h)

    def training_loss(self, x, epoch=0):
        recon = self.forward(x)

        return F.mse_loss(
            recon,
            x
        )



    @torch.no_grad()
    def score_points(self,x):

        recon=self.forward(x)

        score=(
            recon-x
        ).pow(2).mean(-1)

        return score


# =============================================================================
# MEMTO (NeurIPS 2023): memory-guided Transformer + two-phase KMeans init +
# input/latent bi-dimensional deviation score.
# =============================================================================
class MEMTO(BaseDetector):
    def __init__(self, input_dim: int, hidden: int = 128, memory_slots: int = 20, dropout: float = 0.1, layers: int = 3):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.inp = nn.Linear(input_dim, hidden)
        self.pe = PositionalEncoding(hidden, dropout=dropout)
        enc = nn.TransformerEncoderLayer(hidden, _safe_heads(hidden, 8), hidden * 4, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.memory = nn.Parameter(torch.randn(memory_slots, hidden) * 0.05)
        self.memory_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Sigmoid())
        dec = nn.TransformerEncoderLayer(hidden, _safe_heads(hidden, 8), hidden * 4, dropout, batch_first=True, activation="gelu")
        self.decoder = nn.TransformerEncoder(dec, num_layers=2)
        self.out = nn.Linear(hidden, input_dim)
        self.memory_initialized = False

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.pe(self.inp(x)))

    def memory_read(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        zn = F.normalize(z, dim=-1)
        mn = F.normalize(self.memory, dim=-1)
        att = torch.softmax(torch.einsum("bld,md->blm", zn, mn), dim=-1)
        read = torch.einsum("blm,md->bld", att, self.memory)
        gate = self.memory_gate(torch.cat([z, read], dim=-1))
        fused = gate * read + (1.0 - gate) * z
        return fused, att

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        m, att = self.memory_read(z)
        h = self.decoder(self.pe(m))
        rec = self.out(h)
        return rec, z, m, att

    def training_loss(self, x: torch.Tensor, epoch: int = 1) -> torch.Tensor:
        rec, z, m, att = self.forward(x)
        rec_loss = F.mse_loss(rec, x)
        latent = (z - m).pow(2).mean()
        entropy = -(att.clamp_min(1e-8) * torch.log(att.clamp_min(1e-8))).sum(-1).mean()
        # Compactness + mild assignment sharpening; memory centers are KMeans-initialized by runner.
        return rec_loss + 0.10 * latent + 1e-3 * entropy

    @torch.no_grad()
    def score_points(self, x: torch.Tensor) -> torch.Tensor:
        rec, z, m, _ = self.forward(x)
        input_dev = (x - rec).pow(2).mean(-1)
        latent_dev = (z - m).pow(2).mean(-1)
        return input_dev + 0.10 * latent_dev


# =============================================================================
# CATCH (ICLR 2025): frequency patching + CFM patch-wise mask generator +
# masked channel attention + auxiliary channel-structure objective.
# =============================================================================
class ChannelFusionModule(nn.Module):
    def __init__(self, channels: int, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.channels = channels
        self.d_model = d_model
        self.heads = _safe_heads(d_model, heads)
        self.mask_gen = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z [B,P,C,D] frequency patches, channel attention separately per patch.
        B, P, C, D = z.shape
        zi = z.unsqueeze(3).expand(-1, -1, -1, C, -1)
        zj = z.unsqueeze(2).expand(-1, -1, C, -1, -1)
        mask_logits = self.mask_gen(torch.cat([zi, zj], dim=-1)).squeeze(-1)
        soft_mask = torch.sigmoid(mask_logits)
        # ensure self-channel connection
        eye = torch.eye(C, device=z.device, dtype=z.dtype).view(1, 1, C, C)
        soft_mask = torch.maximum(soft_mask, eye)
        H = self.heads; Dh = D // H
        q = self.q(z).view(B, P, C, H, Dh).permute(0, 1, 3, 2, 4)
        k = self.k(z).view(B, P, C, H, Dh).permute(0, 1, 3, 2, 4)
        v = self.v(z).view(B, P, C, H, Dh).permute(0, 1, 3, 2, 4)
        a = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Dh)
        a = a + torch.log(soft_mask.clamp_min(1e-6)).unsqueeze(2)
        a = self.drop(torch.softmax(a, dim=-1))
        out = torch.matmul(a, v).permute(0, 1, 3, 2, 4).reshape(B, P, C, D)
        return self.o(out), soft_mask, a


class CATCH(BaseDetector):
    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.3, patch_freq: int = 4, layers: int = 3):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.patch_freq = int(patch_freq)
        # complex frequency point represented by real+imag -> d_model
        self.freq_embed = nn.Linear(2, hidden)
        self.cfm = ChannelFusionModule(input_dim, hidden, heads=8, dropout=0.1)
        enc = nn.TransformerEncoderLayer(hidden, _safe_heads(hidden, 8), hidden * 1, dropout, batch_first=True, activation="gelu")
        self.patch_encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.freq_out = nn.Linear(hidden, 2)
        self._last_aux = None

    def _patch(self, f: torch.Tensor) -> Tuple[torch.Tensor, int]:
        # f [B,K,C,2]
        B, K, C, Z = f.shape
        p = self.patch_freq
        pad = (p - K % p) % p
        if pad:
            f = F.pad(f.permute(0, 2, 3, 1), (0, pad)).permute(0, 3, 1, 2)
        Kp = f.size(1); P = Kp // p
        fp = f.view(B, P, p, C, Z).mean(2)
        return fp, K

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        spec = torch.fft.rfft(x, dim=1)
        f = torch.stack([spec.real, spec.imag], dim=-1)  # B,K,C,2
        fp, K_orig = self._patch(f)
        z = self.freq_embed(fp)
        z, mask, att = self.cfm(z)
        B, P, C, D = z.shape
        # spectral patch modeling along patch axis, channel-independent after CFM.
        ze = z.permute(0, 2, 1, 3).reshape(B * C, P, D)
        ze = self.patch_encoder(ze)
        pred_patch = self.freq_out(ze).reshape(B, C, P, 2).permute(0, 2, 1, 3)
        # repeat patch prediction back to frequency bins.
        pred_bins = pred_patch.unsqueeze(2).expand(-1, -1, self.patch_freq, -1, -1).reshape(B, P * self.patch_freq, C, 2)
        pred_bins = pred_bins[:, :K_orig]
        pred_spec = torch.complex(pred_bins[..., 0], pred_bins[..., 1])
        rec = torch.fft.irfft(pred_spec, n=x.size(1), dim=1)
        # Auxiliary CFM objective: cluster attended channels while discouraging dense masks.
        # It is optimized jointly; the runner can additionally use a smaller LR for mask_gen.
        sim = F.cosine_similarity(z.unsqueeze(3), z.unsqueeze(2), dim=-1)
        aux_cluster = ((1.0 - sim) * mask).mean()
        aux_sparse = mask.mean()
        aux = aux_cluster + 0.05 * aux_sparse
        self._last_aux = aux
        return rec, {"mask": mask, "att": att, "aux": aux}

    def training_loss(self, x: torch.Tensor, epoch: int = 1) -> torch.Tensor:
        rec, info = self.forward(x)
        return F.mse_loss(rec, x) + 0.01 * info["aux"]

    def mask_parameters(self):
        return list(self.cfm.mask_gen.parameters())

    @torch.no_grad()
    def score_points(self, x: torch.Tensor) -> torch.Tensor:
        rec, _ = self.forward(x)
        return (x - rec).pow(2).mean(-1)

# =============================================================================
# MDGAD — lightweight paper-guided implementation
# Multi-scale dynamic graph + bilateral forecasting
#
# Lightweight revision:
# - hidden = 64
# - shared multi-scale blocks for forward/backward directions
# - 1-layer GRU
# - causal temporal convolution to avoid future leakage
# =============================================================================

class DynamicGraphBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden: int,
        kernel: int,
        dropout: float
    ):
        super().__init__()

        self.channels = channels
        self.hidden = hidden
        self.kernel = kernel

        # Lightweight temporal branch
        self.temporal1 = nn.Conv1d(
            channels,
            hidden,
            kernel_size=kernel,
            padding=0
        )

        self.temporal2 = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=kernel,
            padding=0
        )

        # Dynamic graph branch
        self.node_proj = nn.Linear(
            1,
            hidden
        )

        self.q_proj = nn.Linear(
            hidden,
            hidden,
            bias=False
        )

        self.k_proj = nn.Linear(
            hidden,
            hidden,
            bias=False
        )

        self.mix = nn.Linear(
            hidden * 2,
            hidden
        )

        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _causal_conv(
        x: torch.Tensor,
        conv: nn.Conv1d
    ) -> torch.Tensor:

        pad = (
            conv.kernel_size[0] - 1
        ) * conv.dilation[0]

        x = F.pad(
            x,
            (pad, 0)
        )

        return conv(x)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        # x: B,L,C
        B, L, C = x.shape

        # --------------------------------------------------
        # 1. Causal temporal branch
        # --------------------------------------------------
        temp = x.transpose(1, 2)          # B,C,L

        temp = F.gelu(
            self._causal_conv(
                temp,
                self.temporal1
            )
        )

        temp = F.gelu(
            self._causal_conv(
                temp,
                self.temporal2
            )
        )

        temp = temp.transpose(1, 2)       # B,L,H

        # --------------------------------------------------
        # 2. Dynamic graph branch
        # --------------------------------------------------
        node = self.node_proj(
            x.unsqueeze(-1)
        )                                 # B,L,C,H

        q = F.normalize(
            self.q_proj(node),
            dim=-1
        )

        k = F.normalize(
            self.k_proj(node),
            dim=-1
        )

        A = torch.softmax(
            torch.einsum(
                "blch,bldh->blcd",
                q,
                k
            ) / (self.hidden ** 0.5),
            dim=-1
        )

        msg = torch.einsum(
            "blcd,bldh->blch",
            A,
            node
        ).mean(dim=2)                     # B,L,H

        # --------------------------------------------------
        # 3. Fusion
        # --------------------------------------------------
        h = torch.cat(
            [temp, msg],
            dim=-1
        )

        h = F.gelu(
            self.mix(h)
        )

        return self.drop(h)


class MDGAD(BaseDetector):
    def __init__(
        self,
        input_dim: int,
        hidden: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden = hidden

        # Still retains multi-scale modeling
        self.scales = nn.ModuleList([
            DynamicGraphBlock(
                input_dim,
                hidden,
                kernel=k,
                dropout=dropout
            )
            for k in (3, 5)
        ])

        feature_dim = hidden * 2

        # Lightweight one-layer bilateral GRUs
        self.fwd = nn.GRU(
            feature_dim,
            hidden,
            num_layers=1,
            batch_first=True
        )

        self.bwd = nn.GRU(
            feature_dim,
            hidden,
            num_layers=1,
            batch_first=True
        )

        self.fwd_head = nn.Linear(
            hidden,
            input_dim
        )

        self.bwd_head = nn.Linear(
            hidden,
            input_dim
        )

    def _encode(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        return torch.cat(
            [
                block(x)
                for block in self.scales
            ],
            dim=-1
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # ==================================================
        # Forward prediction
        # x_0 ... x_t -> predict x_{t+1}
        # ==================================================
        hf_in = self._encode(x)

        hf, _ = self.fwd(
            hf_in
        )

        pf = torch.zeros_like(x)

        pf[:, 1:] = self.fwd_head(
            hf[:, :-1]
        )

        # ==================================================
        # Backward prediction
        # Reverse the sequence FIRST.
        #
        # Because DynamicGraphBlock itself is causal,
        # applying it to reversed x makes the branch causal
        # in the backward direction.
        # ==================================================
        xr = torch.flip(
            x,
            dims=[1]
        )

        hb_in = self._encode(xr)

        hb_rev, _ = self.bwd(
            hb_in
        )

        pb_rev = torch.zeros_like(xr)

        pb_rev[:, 1:] = self.bwd_head(
            hb_rev[:, :-1]
        )

        pb = torch.flip(
            pb_rev,
            dims=[1]
        )

        return pf, pb

    def training_loss(
        self,
        x: torch.Tensor,
        epoch: int = 1
    ) -> torch.Tensor:

        pf, pb = self.forward(x)

        loss_f = F.mse_loss(
            pf[:, 1:],
            x[:, 1:]
        )

        loss_b = F.mse_loss(
            pb[:, :-1],
            x[:, :-1]
        )

        return 0.5 * loss_f + 0.5 * loss_b

    @torch.no_grad()
    def score_points(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        pf, pb = self.forward(x)

        ef = (
            x - pf
        ).pow(2).mean(dim=-1)

        eb = (
            x - pb
        ).pow(2).mean(dim=-1)

        score = 0.5 * ef + 0.5 * eb

        # sequence boundaries only have one valid direction
        if score.size(1) > 1:
            score[:, 0] = eb[:, 0]
            score[:, -1] = ef[:, -1]

        return score



# =============================================================================
# MSHTrans (KDD 2025): multi-scale downsampling + trainable hypergraphs +
# decomposition + hypergraph transformer + scale fusion/reconstruction.
# =============================================================================
class MovingAverage(nn.Module):
    def __init__(self, kernel: int = 5):
        super().__init__(); self.kernel = kernel
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = (self.kernel - 1) // 2
        return F.avg_pool1d(F.pad(x.transpose(1,2), (p,p), mode="replicate"), self.kernel, stride=1).transpose(1,2)


class AdaptiveHypergraphTransformer(nn.Module):
    def __init__(self, d_model: int, num_edges: int, dropout: float):
        super().__init__()
        self.num_edges = int(num_edges)
        self.edge_proto = nn.Parameter(torch.randn(num_edges, d_model) * 0.05)
        self.node_q = nn.Linear(d_model, d_model)
        self.edge_k = nn.Linear(d_model, d_model)
        enc = nn.TransformerEncoderLayer(d_model, _safe_heads(d_model, 8), d_model * 4, dropout, batch_first=True, activation="gelu")
        self.edge_transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.node_out = nn.Linear(d_model, d_model)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        # nodes [B,T,D]; hyperedges adaptively link timestamps.
        B, T, D = nodes.shape
        q = self.node_q(nodes)
        ek = self.edge_k(self.edge_proto).view(1, self.num_edges, D)
        inc = torch.softmax(torch.einsum("btd,bed->bte", q, ek) / math.sqrt(D), dim=-1)
        # node -> hyperedge, transform hyperedges, hyperedge -> node
        denom = inc.sum(1, keepdim=False).unsqueeze(-1).clamp_min(1e-6)
        edges = torch.einsum("bte,btd->bed", inc, nodes) / denom
        edges = self.edge_transformer(edges)
        out = torch.einsum("bte,bed->btd", inc, edges)
        return nodes + self.node_out(out)


class MSHTrans(BaseDetector):
    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(input_dim, hidden)
        self.ma = MovingAverage(5)
        self.scale_blocks = nn.ModuleList([
            AdaptiveHypergraphTransformer(hidden, num_edges=8, dropout=dropout),
            AdaptiveHypergraphTransformer(hidden, num_edges=6, dropout=dropout),
            AdaptiveHypergraphTransformer(hidden, num_edges=4, dropout=dropout),
        ])
        self.trend_enc = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden, _safe_heads(hidden, 8), hidden*4, dropout, batch_first=True, activation="gelu"), 1)
            for _ in range(3)
        ])
        self.fuse = nn.Sequential(nn.Linear(hidden * 6, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))
        self.out = nn.Linear(hidden, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        scales = [x]
        for factor in (2, 4):
            z = F.avg_pool1d(x.transpose(1,2), kernel_size=factor, stride=factor).transpose(1,2)
            scales.append(z)
        feats = []
        for i, z in enumerate(scales):
            trend = self.ma(z)
            seasonal = z - trend
            hs = self.scale_blocks[i](self.inp(seasonal))
            ht = self.trend_enc[i](self.inp(trend))
            # Upsample each scale back to original resolution.
            hs = F.interpolate(hs.transpose(1,2), size=L, mode="linear", align_corners=False).transpose(1,2)
            ht = F.interpolate(ht.transpose(1,2), size=L, mode="linear", align_corners=False).transpose(1,2)
            feats.extend([hs, ht])
        return self.out(self.fuse(torch.cat(feats, dim=-1)))

    def training_loss(self, x: torch.Tensor, epoch: int = 1) -> torch.Tensor:
        return F.mse_loss(self.forward(x), x)

    @torch.no_grad()
    def score_points(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.forward(x)).pow(2).mean(-1)


# =============================================================================
# CKDGAT (Computers in Industry 2026): two-layer data/knowledge GAT,
# stochastic multi-head knowledge attention, temporal fusion, reconstruction.
# =============================================================================
class StochasticGraphAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.heads = _safe_heads(d_model, heads); self.d_model = d_model
        self.q = nn.Linear(d_model, d_model); self.k = nn.Linear(d_model, d_model); self.v = nn.Linear(d_model, d_model)
        self.log_sigma = nn.Parameter(torch.full((self.heads,), -2.0))
        self.out = nn.Linear(d_model, d_model); self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, prior_adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        # h B,L,F,D
        B,L,Fdim,D = h.shape; H=self.heads; Dh=D//H
        q=self.q(h).view(B,L,Fdim,H,Dh).permute(0,1,3,2,4)
        k=self.k(h).view(B,L,Fdim,H,Dh).permute(0,1,3,2,4)
        v=self.v(h).view(B,L,Fdim,H,Dh).permute(0,1,3,2,4)
        logits=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(Dh)
        if self.training:
            logits = logits + torch.randn_like(logits) * self.log_sigma.exp().view(1,1,H,1,1)
        if prior_adj is not None:
            logits = logits + torch.log(prior_adj.clamp_min(1e-5)).view(1,1,1,Fdim,Fdim)
        a=self.drop(torch.softmax(logits,dim=-1))
        z=torch.matmul(a,v).permute(0,1,3,2,4).reshape(B,L,Fdim,D)
        return self.out(z)


class CKDGAT(BaseDetector):
    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_dim=input_dim; self.hidden=hidden
        self.node=nn.Parameter(torch.randn(input_dim,hidden)*0.05)
        self.val=nn.Linear(1,hidden)
        # Data-oriented first GAT and knowledge-oriented stochastic second GAT.
        self.data_gat=StochasticGraphAttention(hidden,4,dropout)
        self.know_gat=StochasticGraphAttention(hidden,4,dropout)
        # Generic process-knowledge topology on raw variables: T, three U, three I.
        A=torch.eye(input_dim)
        if input_dim==7:
            # phase relationships U-I, within voltage/current phase sets, and T weakly connected to all.
            for i in [1,2,3]:
                for j in [1,2,3]: A[i,j]=1
            for i in [4,5,6]:
                for j in [4,5,6]: A[i,j]=1
            for u,i in [(1,4),(2,5),(3,6)]: A[u,i]=A[i,u]=1
            for j in range(1,7): A[0,j]=A[j,0]=0.35
        A=A/A.sum(-1,keepdim=True).clamp_min(1e-6)
        self.register_buffer("knowledge_adj",A,persistent=False)
        self.temporal=nn.GRU(hidden*input_dim,hidden,num_layers=2,batch_first=True,dropout=dropout)
        self.dec=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,input_dim))

    def forward(self,x:torch.Tensor)->torch.Tensor:
        B,L,C=x.shape
        h=self.val(x.unsqueeze(-1))+self.node.view(1,1,C,self.hidden)
        d=F.gelu(self.data_gat(h,None))
        k=F.gelu(self.know_gat(d,self.knowledge_adj))
        th,_=self.temporal(k.reshape(B,L,-1))
        return self.dec(th)

    def training_loss(self,x:torch.Tensor,epoch:int=1)->torch.Tensor:
        return F.mse_loss(self.forward(x),x)

    @torch.no_grad()
    def score_points(self,x:torch.Tensor)->torch.Tensor:
        return (x-self.forward(x)).pow(2).mean(-1)

# =============================================================================
# MAD-DGTD — lightweight paper-guided reimplementation
#
# Retained ideas:
#   - multi-scale temporal feature extraction
#   - static + dynamic graph
#   - delay-impact learning
#   - graph information propagation
#   - one-step prediction anomaly score
#
# Simplifications:
#   - hidden = 48
#   - one TDIE block
#   - one GCIP propagation block
#   - three delay scales: (1, 3, 5)
#   - one-layer GRU
# =============================================================================

class LightTDIEBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden: int,
        dropout: float
    ):
        super().__init__()

        self.in_proj = nn.Conv1d(
            channels,
            hidden,
            kernel_size=1
        )

        # retain multi-scale temporal extraction
        self.convs = nn.ModuleList([
            nn.Conv1d(
                hidden,
                hidden,
                kernel_size=k,
                padding=0
            )
            for k in (2, 3, 5)
        ])

        self.gate = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=1
        )

        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _causal_conv(
        x: torch.Tensor,
        conv: nn.Conv1d
    ) -> torch.Tensor:

        pad = conv.kernel_size[0] - 1

        x = F.pad(
            x,
            (pad, 0)
        )

        return conv(x)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        # x: B,L,C

        h = self.in_proj(
            x.transpose(1, 2)
        )                       # B,H,L

        multi = []

        for conv in self.convs:
            z = self._causal_conv(
                h,
                conv
            )
            multi.append(z)

        z = torch.stack(
            multi,
            dim=0
        ).mean(dim=0)

        gate = torch.sigmoid(
            self.gate(h)
        )

        z = torch.tanh(z) * gate

        return self.drop(
            z
        ).transpose(1, 2)       # B,L,H


class MADDGTD(BaseDetector):
    def __init__(
        self,
        input_dim: int,
        hidden: int = 48,
        dropout: float = 0.1,
        lags: Sequence[int] = (1, 3, 5)
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden = hidden
        self.lags = tuple(lags)

        # --------------------------------------------------
        # static graph representation
        # --------------------------------------------------
        self.node = nn.Parameter(
            torch.randn(
                input_dim,
                hidden
            ) * 0.05
        )

        # --------------------------------------------------
        # delay-impact weights
        # --------------------------------------------------
        self.delay_weight = nn.Parameter(
            torch.zeros(
                len(self.lags)
            )
        )

        # --------------------------------------------------
        # one lightweight TDIE block
        # --------------------------------------------------
        self.tdie = LightTDIEBlock(
            channels=input_dim,
            hidden=hidden,
            dropout=dropout
        )

        self.node_from_temp = nn.Linear(
            hidden,
            hidden
        )

        # --------------------------------------------------
        # one lightweight graph propagation
        # --------------------------------------------------
        self.gcip = nn.Linear(
            hidden,
            hidden
        )

        # --------------------------------------------------
        # lightweight temporal prediction
        # --------------------------------------------------
        self.temporal_out = nn.GRU(
            hidden * input_dim,
            hidden,
            num_layers=1,
            batch_first=True
        )

        self.head = nn.Linear(
            hidden,
            input_dim
        )

    def _lag(
        self,
        x: torch.Tensor,
        lag: int
    ) -> torch.Tensor:

        if lag <= 0:
            return x

        return torch.cat(
            [
                x[:, :1].expand(
                    -1,
                    lag,
                    -1
                ),
                x[:, :-lag]
            ],
            dim=1
        )

    def _graph(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        # ==================================================
        # Static graph
        # ==================================================
        en = F.normalize(
            self.node,
            dim=-1
        )

        static = en @ en.t()

        # ==================================================
        # Dynamic delay graph
        # ==================================================
        ws = torch.softmax(
            self.delay_weight,
            dim=0
        )

        dyn = 0.0

        for w, lag in zip(
            ws,
            self.lags
        ):

            xl = self._lag(
                x,
                lag
            )

            centered = (
                xl
                - xl.mean(
                    dim=-1,
                    keepdim=True
                )
            )

            relation = -torch.abs(
                centered.unsqueeze(-1)
                - centered.unsqueeze(-2)
            )

            dyn = (
                dyn
                + w * relation
            )

        A = torch.softmax(
            dyn
            + static.view(
                1,
                1,
                self.input_dim,
                self.input_dim
            ),
            dim=-1
        )

        return A

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        B, L, C = x.shape

        # ==================================================
        # Temporal features
        # ==================================================
        temp = self.tdie(
            x
        )                       # B,L,H

        # Each variable receives the common temporal context
        # plus its static node representation.
        nf = (
            self.node_from_temp(
                temp
            ).unsqueeze(2)
            + self.node.view(
                1,
                1,
                C,
                self.hidden
            )
        )                       # B,L,C,H

        # ==================================================
        # Dynamic graph
        # ==================================================
        A = self._graph(x)

        # ==================================================
        # Single GCIP-style propagation
        # ==================================================
        g = torch.einsum(
            "blcd,bldh->blch",
            A,
            nf
        )

        g = F.gelu(
            self.gcip(g)
        )

        # ==================================================
        # One-step prediction
        # ==================================================
        h, _ = self.temporal_out(
            g.reshape(
                B,
                L,
                -1
            )
        )

        pred = torch.zeros_like(x)

        pred[:, 1:] = self.head(
            h[:, :-1]
        )

        return pred

    def training_loss(
        self,
        x: torch.Tensor,
        epoch: int = 1
    ) -> torch.Tensor:

        pred = self.forward(x)

        return F.mse_loss(
            pred[:, 1:],
            x[:, 1:]
        )

    @torch.no_grad()
    def score_points(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        pred = self.forward(x)

        err = (
            x - pred
        ).pow(2).mean(dim=-1)

        # first position has no genuine previous-step forecast
        if err.size(1) > 1:
            err[:, 0] = err[:, 1]

        return err

# =============================================================================
# Factory
# =============================================================================
def make_model(name: str, seq_len: int, input_dim: int, hidden: int, latent: int, dropout: float) -> BaseDetector:
    if name == "USAD": return USAD(seq_len, input_dim, hidden, latent)
    if name == "OmniAnomaly": return OmniAnomaly(input_dim, hidden, latent, flow_steps=4)
    if name == "MTAD-GAT": return MTADGAT(input_dim, hidden, dropout, latent=max(16, latent), kernel_size=7)
    if name == "InterFusion": return InterFusion(input_dim, hidden, latent)
    if name == "GDN": return GDN(input_dim, hidden, topk=min(4,input_dim), dropout=dropout, history=10)
    if name == "TranAD": return TranAD(input_dim, hidden, dropout)
    if name == "DCdetector": return DCdetector(input_dim, hidden, dropout, patches=(3,5,10), layers=3)
    if name == "TimesNet": return TimesNet(input_dim, hidden=128, latent=128, seq_len=seq_len, layers=3, dropout=dropout)
    if name == "MEMTO": return MEMTO(input_dim, hidden, memory_slots=20, dropout=dropout, layers=3)
    if name == "CATCH": return CATCH(input_dim, hidden, dropout=dropout, patch_freq=4, layers=3)
    if name == "MDGAD": return MDGAD(input_dim, hidden, dropout)
    if name == "MSHTrans": return MSHTrans(input_dim, hidden, dropout)
    if name == "CKDGAT": return CKDGAT(input_dim, hidden, dropout)
    if name == "MAD-DGTD": return MADDGTD(input_dim, hidden, dropout)
    raise ValueError("Unknown deep baseline: %s" % name)
