"""
DelphiCrossAttention — two-stream encoder with cross-attention fusion.

Architecture string (specifies structure only):

    CrossAttention(
        [group_A_domains] : (hN dD lL_A),
        [group_B_domains] : (hN dD lL_B),
        xattn             : (hN dD),
    ) : (hN dD lL_trunk)

The attention *policy* (which tokens attend to which) is passed separately,
using the same syntax as ``Delphi``::

    [hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)

The policy is applied globally and then restricted per stage:

    Encoder A     — global_mask AND (query ∈ A  AND key ∈ A)
    Encoder B     — global_mask AND (query ∈ B  AND key ∈ B)
    Cross-attn    — global_mask AND (query ∈ A  AND key ∈ B)  OR  vice-versa
    Trunk         — global_mask  (unrestricted)

Tokens that belong to both groups (e.g. ``sex`` in both A and B) participate
in both encoders; their representations are averaged before cross-attention.
All stages share the same ``n_embd``.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from delphi.model import (
    DelphiConfig,
    AgeEncoding, LayerNorm, Block,
    AttentionMaskBuilder,
    initialize_weights,
    Delphi,
)
from delphi.embedding import MultiDomainEmbedding

_log = logging.getLogger(__name__)


# ── Parser ───────────────────────────────────────────────────────────────────

def _split_top_level(s: str, sep: str = ",") -> List[str]:
    parts, buf, d_brack, d_paren = [], "", 0, 0
    for ch in s:
        if ch == "[":   d_brack += 1
        elif ch == "]": d_brack -= 1
        elif ch == "(": d_paren += 1
        elif ch == ")": d_paren -= 1
        if ch == sep and d_brack == 0 and d_paren == 0:
            parts.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append(buf.strip())
    return parts


def _parse_arch_str(s: str) -> Dict[str, int]:
    """``'h24d240l3'`` → ``{'n_head': 24, 'n_embd': 240, 'n_layer': 3}``."""
    out: Dict[str, int] = {}
    for key, pat in [("n_head", r"h(\d+)"), ("n_embd", r"d(\d+)"), ("n_layer", r"l(\d+)")]:
        m = re.search(pat, s)
        if m:
            out[key] = int(m.group(1))
    return out


@dataclass
class EncoderSpec:
    domains: List[str]
    n_head: int
    n_embd: int
    n_layer: int


@dataclass
class CrossAttentionScheme:
    encoder_A: EncoderSpec
    encoder_B: EncoderSpec
    xattn_n_head: int
    xattn_n_embd: int
    trunk_n_head: int
    trunk_n_embd: int
    trunk_n_layer: int


def parse_cross_attention_scheme(scheme: str) -> CrossAttentionScheme:
    """
    Parse a CrossAttention architecture string into a :class:`CrossAttentionScheme`.

    Expected format::

        CrossAttention([g1]:(hNdDlL),[g2]:(hNdDlL),xattn:(hNdD)):(hNdDlL)
    """
    scheme = scheme.strip()
    outer = re.match(r"^CrossAttention\((.+)\):\(([^)]+)\)$", scheme, re.DOTALL)
    if not outer:
        raise ValueError(f"Cannot parse CrossAttention scheme: {scheme!r}")

    trunk_cfg = _parse_arch_str(outer.group(2))
    encoders: List[EncoderSpec] = []
    xattn: Dict[str, int] = {}

    for part in _split_top_level(outer.group(1).strip()):
        part = part.strip()
        if part.lower().startswith("xattn:"):
            m = re.search(r"\(([^)]+)\)", part)
            if not m:
                raise ValueError(f"Invalid xattn spec: {part!r}")
            xattn = _parse_arch_str(m.group(1))
        elif part.startswith("["):
            m = re.match(r"^\[([^\]]+)\]:\(([^)]+)\)$", part)
            if not m:
                raise ValueError(f"Invalid encoder spec: {part!r}")
            domains = [d.strip() for d in m.group(1).split(",")]
            cfg = _parse_arch_str(m.group(2))
            encoders.append(EncoderSpec(
                domains=domains,
                n_head=cfg["n_head"],
                n_embd=cfg["n_embd"],
                n_layer=cfg.get("n_layer", 1),
            ))
        else:
            raise ValueError(f"Unexpected fragment in CrossAttention spec: {part!r}")

    if len(encoders) != 2:
        raise ValueError(f"Expected exactly 2 encoder groups, got {len(encoders)}")
    if not xattn:
        raise ValueError("Missing xattn:(hNdD) in CrossAttention spec")

    return CrossAttentionScheme(
        encoder_A=encoders[0],
        encoder_B=encoders[1],
        xattn_n_head=xattn["n_head"],
        xattn_n_embd=xattn["n_embd"],
        trunk_n_head=trunk_cfg["n_head"],
        trunk_n_embd=trunk_cfg["n_embd"],
        trunk_n_layer=trunk_cfg.get("n_layer", 1),
    )


# ── Model ────────────────────────────────────────────────────────────────────

@dataclass
class _StageConfig:
    """Minimal config consumed by Block / MaskedSelfAttention / MLP."""
    n_embd: int
    n_head: int
    dropout: float
    bias: bool


class DelphiCrossAttention(nn.Module):
    """
    Two-stream encoder variant of Delphi.

    Parameters
    ----------
    config:
        Standard :class:`~delphi.model.DelphiConfig` (domains, dropout, bias, …).
        ``n_layer`` / ``n_head`` / ``n_embd`` are ignored — those come from *scheme*.
    scheme:
        Architecture spec (encoder sizes, trunk size).  Build with
        :func:`parse_cross_attention_scheme`.
    attention_scheme:
        Global attention policy string, same syntax as Delphi, e.g.
        ``"[hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)"``.
        The ``at_birth`` alias is resolved automatically.
    """

    def __init__(
        self,
        config: DelphiConfig,
        scheme: CrossAttentionScheme,
        attention_scheme: str,
    ):
        super().__init__()

        n_embds = {
            "encoder_A": scheme.encoder_A.n_embd,
            "encoder_B": scheme.encoder_B.n_embd,
            "xattn":     scheme.xattn_n_embd,
            "trunk":     scheme.trunk_n_embd,
        }
        if len(set(n_embds.values())) != 1:
            raise ValueError(f"All stages must share the same n_embd: {n_embds}")

        self._config  = config
        self._scheme  = scheme
        n_embd  = scheme.trunk_n_embd
        dropout = config.dropout
        bias    = config.bias

        # ── Embedding (shared with Delphi) ────────────────────────────────
        self.domain_to_int = Delphi._build_domain_to_int(list(config.domains.keys()))
        self.int_to_domain = {v: k for k, v in self.domain_to_int.items()}
        self.domain_offsets, global_vocab_size = Delphi._build_domain_offsets(
            self.domain_to_int, config.domains
        )
        self.global_vocab_size = global_vocab_size

        self.embed = MultiDomainEmbedding(
            config=config,
            domain_offsets=self.domain_offsets,
            global_vocab_size=global_vocab_size,
            domain_to_int=self.domain_to_int,
        )
        self.age_encoding = AgeEncoding(n_embd=n_embd)
        self.drop = nn.Dropout(dropout)

        # ── Transformer stages ────────────────────────────────────────────
        def _blocks(n_head: int, n_layer: int) -> nn.ModuleList:
            cfg = _StageConfig(n_embd=n_embd, n_head=n_head, dropout=dropout, bias=bias)
            return nn.ModuleList([Block(cfg) for _ in range(n_layer)])

        self.encoder_A        = _blocks(scheme.encoder_A.n_head, scheme.encoder_A.n_layer)
        self.encoder_B        = _blocks(scheme.encoder_B.n_head, scheme.encoder_B.n_layer)
        self.cross_attn_block = Block(
            _StageConfig(n_embd=n_embd, n_head=scheme.xattn_n_head, dropout=dropout, bias=bias)
        )
        self.trunk = _blocks(scheme.trunk_n_head, scheme.trunk_n_layer)
        self.ln_f  = LayerNorm(n_embd, bias=bias)

        # ── Global attention policy ───────────────────────────────────────
        # Resolve the at_birth alias before storing.
        attention_scheme = self._resolve_at_birth(attention_scheme, config)
        self.attention_scheme = attention_scheme
        self.mask_builder = AttentionMaskBuilder(attention_scheme, self.domain_to_int)

        # Group-membership buffers (follow .to(device) automatically).
        self.register_buffer("_group_A_ids", torch.tensor([
            self.domain_to_int[d] for d in scheme.encoder_A.domains if d in self.domain_to_int
        ]))
        self.register_buffer("_group_B_ids", torch.tensor([
            self.domain_to_int[d] for d in scheme.encoder_B.domains if d in self.domain_to_int
        ]))

        # ── Weight initialisation ─────────────────────────────────────────
        total_layers = (
            scheme.encoder_A.n_layer + scheme.encoder_B.n_layer
            + 1  # cross-attention block
            + scheme.trunk_n_layer
        )
        torch.manual_seed(config.seed)
        initialize_weights(self, config=replace(config, n_layer=total_layers))
        self.embed._zero_placeholder_rows()

        self.block_size = config.block_size

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_at_birth(attention_scheme: str, config: DelphiConfig) -> str:
        if "at_birth" in attention_scheme:
            at_birth = ",".join(
                name for name, cfg in config.domains.items() if cfg.at_birth
            )
            attention_scheme = attention_scheme.replace("at_birth", at_birth)
        return attention_scheme

    def _age_emb(self, batch) -> torch.Tensor:
        ae = self.age_encoding(batch.ages.T)   # [T, B, n_embd] or [T, n_embd]
        if ae.ndim == 2:
            ae = ae.unsqueeze(1)
        return ae.transpose(0, 1)              # [B, T, n_embd]

    @staticmethod
    def _apply_self_attn_fallback(mask: torch.Tensor) -> torch.Tensor:
        """
        Ensure no row in ``mask`` [B, T, T] is all-zero.

        When a group restriction is ANDed with the global mask, tokens that
        don't belong to the group lose all valid attention targets.  Without
        this fallback those rows produce softmax(-inf) → NaN.
        """
        no_target = ~mask.bool().any(dim=-1)          # [B, T]
        b_idx, t_idx = no_target.nonzero(as_tuple=True)
        mask[b_idx, t_idx, t_idx] = 1
        return mask

    def _build_masks(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Derive the four per-stage attention masks from the global policy.

        Returns ``(mask_A, mask_B, mask_xattn, mask_trunk)`` each shaped
        ``[B, 1, T, T]`` (broadcast-ready over heads).
        """
        global_mask = self.mask_builder.build(
            batch.domain_ids, batch.global_token_ids, batch.ages
        )  # [B, T, T]  int

        is_A = torch.isin(batch.domain_ids, self._group_A_ids)  # [B, T]
        is_B = torch.isin(batch.domain_ids, self._group_B_ids)

        # Within-group masks — re-apply fallback after ANDing with group pair,
        # because out-of-group tokens lose all valid targets.
        pair_A = is_A.unsqueeze(2) & is_A.unsqueeze(1)   # [B, T, T]
        pair_B = is_B.unsqueeze(2) & is_B.unsqueeze(1)

        mask_A = self._apply_self_attn_fallback(global_mask & pair_A)
        mask_B = self._apply_self_attn_fallback(global_mask & pair_B)

        # Cross-group mask: A→B and B→A only (no within-group)
        cross = (
            (is_A.unsqueeze(2) & is_B.unsqueeze(1)) |
            (is_B.unsqueeze(2) & is_A.unsqueeze(1))
        )
        mask_x = self._apply_self_attn_fallback(global_mask & cross)

        return mask_A.unsqueeze(1), mask_B.unsqueeze(1), mask_x.unsqueeze(1), global_mask.unsqueeze(1)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def config(self) -> DelphiConfig:
        return self._config

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def predicted_domains(self):
        return self.embed.predicted_domains

    @property
    def domains(self):
        return list(self._config.domains.keys())

    def set_block_size(self, block_size: int):
        self.block_size = block_size

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, batch, return_attention: bool = False, return_embeddings: bool = False):
        # 1. Embed + age encoding
        h = self.embed(batch) + self._age_emb(batch)   # [B, T, n_embd]
        h = self.drop(h)

        # 2. Build all masks at once
        mask_A, mask_B, mask_xattn, mask_trunk = self._build_masks(batch)

        # 3. Independent encoders (parallel, on cloned copies)
        h_A = h.clone()
        for block in self.encoder_A:
            h_A, _ = block(h_A, mask_A)

        h_B = h.clone()
        for block in self.encoder_B:
            h_B, _ = block(h_B, mask_B)

        # 4. Merge encoder outputs
        is_A = torch.isin(batch.domain_ids, self._group_A_ids)
        is_B = torch.isin(batch.domain_ids, self._group_B_ids)

        h = h.clone()
        h[is_A & ~is_B] = h_A[is_A & ~is_B]
        h[is_B & ~is_A] = h_B[is_B & ~is_A]
        overlap = is_A & is_B
        if overlap.any():
            h[overlap] = 0.5 * (h_A[overlap] + h_B[overlap])

        # 5. Cross-attention (cross-group only, policy-filtered)
        h, att_x = self.cross_attn_block(h, mask_xattn)

        # 6. Causal trunk (full global policy)
        att_list = [att_x] if return_attention else []
        for block in self.trunk:
            h, att = block(h, mask_trunk)
            if return_attention:
                att_list.append(att)

        h = self.ln_f(h)
        logits = self.embed.to_logits(h)

        attention = torch.stack(att_list) if return_attention else None
        if return_embeddings:
            return logits, attention, h
        return logits, attention

    # ── Loss functions (identical to Delphi) ─────────────────────────────────

    def cross_entropy_loss(self, logits, targets, agg=None):
        import pandas as pd
        n_classes = logits.size(-1)
        if agg == "per_token":
            lsm = F.log_softmax(logits.view(-1, n_classes), dim=-1)
            return lsm[torch.arange(lsm.size(0)), targets.view(-1)]
        elif agg == "per_disease":
            flat = targets.view(-1)
            lsm  = F.log_softmax(logits.view(-1, n_classes), dim=-1)
            log_p = lsm[torch.arange(flat.size(0), device=logits.device), flat]
            unique_ids, inverse, counts = torch.unique(flat, return_inverse=True, return_counts=True)
            sums = torch.zeros(unique_ids.size(0), dtype=log_p.dtype, device=logits.device)
            sums.scatter_add_(0, inverse, log_p)
            result = pd.Series(
                (sums / counts.float()).detach().cpu().numpy(),
                index=unique_ids.cpu().numpy(),
                name="log_p",
            )
            result.index.name = "token_id"
            return result
        elif agg is None:
            return F.cross_entropy(
                logits.reshape(-1, n_classes), targets.reshape(-1), ignore_index=-1
            )
        raise ValueError(f"agg must be None, 'per_token', or 'per_disease'")

    def time_to_event_loss(self, logits, time_to_next, t_min, agg=None):
        lse = torch.logsumexp(logits, -1)
        lse = -torch.log(torch.exp(-lse) + t_min)
        log_dt = -torch.log(torch.clamp(time_to_next, min=1.0) + t_min).view(-1)
        loss_dt = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - log_dt))
        if agg == "mean":  return loss_dt.mean()
        if agg == "sum":   return loss_dt.sum()
        if agg is None:    return loss_dt
        raise NotImplementedError(f"agg={agg!r}")

    # ── Inference ────────────────────────────────────────────────────────────

    def run_inference(self, dataloader, block_size: Optional[int] = None):
        old_bs = self.block_size
        if block_size is not None:
            self.set_block_size(block_size)
        all_logits = []
        for batch in dataloader:
            batch = batch.to(self.device)
            with torch.no_grad():
                logits_dict, _ = self(batch)
            all_logits.append(torch.cat([logits_dict[d] for d in self.predicted_domains], dim=-1))
        if block_size is not None:
            self.set_block_size(old_bs)
        return torch.cat(all_logits, dim=0)

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_scheme_string(
        cls,
        arch_str: str,
        attention_scheme: str,
        config: DelphiConfig,
    ) -> "DelphiCrossAttention":
        """
        Instantiate from strings.

        Parameters
        ----------
        arch_str:
            CrossAttention architecture string, e.g.
            ``"CrossAttention([hla_alleles,sex]:(h24d240l3),...):(h24d240l6)"``.
        attention_scheme:
            Global attention policy, e.g.
            ``"[hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)"``.
        config:
            Base :class:`~delphi.model.DelphiConfig` (domains, dropout, …).
        """
        return cls(config, parse_cross_attention_scheme(arch_str), attention_scheme)
