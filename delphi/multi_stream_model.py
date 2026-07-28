"""
DelphiMultiStream — N-stream encoder with trunk fusion.

Architecture string (specifies structure only):

    MultiStream(
        [group_1_domains] : (hN dD lL_1),
        [group_2_domains] : (hN dD lL_2),
        ...
        [group_K_domains] : (hN dD lL_K),
    ) : (hN dD lL_trunk)

The attention *policy* (which tokens attend to which) is passed separately,
using the same syntax as ``Delphi``::

    [hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)

The policy is applied globally and then restricted per stage:

    Encoder i     — global_mask AND (query ∈ group_i  AND key ∈ group_i)
    Trunk         — global_mask  (unrestricted; this is where cross-group mixing happens)

Tokens that belong to more than one group (e.g. ``sex`` in both group 1 and
group 2) participate in every encoder whose group they belong to; their
representations are averaged before the trunk. All stages share the same
``n_embd``.
"""
from __future__ import annotations

import re
import logging
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from delphi.model import (
    DomainConfig,
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
class MultiStreamScheme:
    encoders: List[EncoderSpec]
    trunk_n_head: int
    trunk_n_embd: int
    trunk_n_layer: int


def parse_multi_stream_scheme(scheme: str) -> MultiStreamScheme:
    """
    Parse a MultiStream architecture string into a :class:`MultiStreamScheme`.

    Expected format (any number ``K >= 2`` of encoder groups)::

        MultiStream([g1]:(hNdDlL),[g2]:(hNdDlL),...,[gK]:(hNdDlL)):(hNdDlL)
    """
    scheme = scheme.strip()
    outer = re.match(r"^MultiStream\((.+)\):\(([^)]+)\)$", scheme, re.DOTALL)
    if not outer:
        raise ValueError(f"Cannot parse MultiStream scheme: {scheme!r}")

    trunk_cfg = _parse_arch_str(outer.group(2))
    encoders: List[EncoderSpec] = []

    for part in _split_top_level(outer.group(1).strip()):
        part = part.strip()
        if part.startswith("["):
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
            raise ValueError(f"Unexpected fragment in MultiStream spec: {part!r}")

    if len(encoders) < 2:
        raise ValueError(f"Expected at least 2 encoder groups, got {len(encoders)}")

    return MultiStreamScheme(
        encoders=encoders,
        trunk_n_head=trunk_cfg["n_head"],
        trunk_n_embd=trunk_cfg["n_embd"],
        trunk_n_layer=trunk_cfg.get("n_layer", 1),
    )


# ── Model ────────────────────────────────────────────────────────────────────

@dataclass
class DelphiMultiStreamConfig:
    """
    Config for :class:`DelphiMultiStream`.

    Deliberately has no ``n_layer`` / ``n_head`` / ``n_embd`` / ``attention_scheme``
    fields, unlike :class:`~delphi.model.DelphiConfig`. Per-stage architecture
    (per-encoder / trunk sizes) comes from :class:`MultiStreamScheme` instead,
    and the attention policy string is passed as a separate constructor
    argument. A generic ``DelphiConfig`` used to be reused here, but its
    ``n_embd`` silently fed the embedding table while the transformer blocks
    used ``scheme``'s ``n_embd`` — two disconnected sources of truth that broke
    with a confusing shape-mismatch error whenever they disagreed. This config
    has only one place ``n_embd`` can come from.
    """
    domains: Dict[str, DomainConfig] = field(default_factory=dict)
    dropout: float = 0.1
    token_dropout: float = 0.1
    bias: bool = True
    block_size: int = 64
    seed: int = 42

    def items(self):
        # Duck-types this as dict-like for mlflow.log_params (called as
        # `logger.log_params(self.model.config)` in Trainer.train), mirroring
        # DelphiConfig.items().
        return asdict(self).items()

    def set_dropout(self, dropout: float):
        self.dropout = dropout
        return self

    def set_token_dropout(self, token_dropout: float):
        self.token_dropout = token_dropout
        return self

    def set_block_size(self, block_size: int):
        self.block_size = block_size
        return self

    def add_domain(self, domain_name: str, domain_config: DomainConfig):
        self.domains[domain_name] = domain_config
        return self

    def remove_domain(self, domain_name: str):
        if domain_name in self.domains:
            del self.domains[domain_name]
        return self


@dataclass
class _StageConfig:
    """Minimal config consumed by Block / MaskedSelfAttention / MLP."""
    n_embd: int
    n_head: int
    dropout: float
    bias: bool


class DelphiMultiStream(nn.Module):
    """
    N-stream encoder variant of Delphi.

    Parameters
    ----------
    config:
        :class:`DelphiMultiStreamConfig` (domains, dropout, bias, …).
        Architecture (``n_layer`` / ``n_head`` / ``n_embd`` per stage) is not
        part of this config at all — it comes from *scheme*.
    scheme:
        Architecture spec (per-encoder sizes, trunk size). Build with
        :func:`parse_multi_stream_scheme`.
    attention_scheme:
        Global attention policy string, same syntax as Delphi, e.g.
        ``"[hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)"``.
        The ``at_birth`` alias is resolved automatically.
    """

    def __init__(
        self,
        config: DelphiMultiStreamConfig,
        scheme: MultiStreamScheme,
        attention_scheme: str,
    ):
        super().__init__()

        n_embds = {f"encoder_{i}": spec.n_embd for i, spec in enumerate(scheme.encoders)}
        n_embds["trunk"] = scheme.trunk_n_embd
        if len(set(n_embds.values())) != 1:
            raise ValueError(f"All stages must share the same n_embd: {n_embds}")

        self._config  = config
        self._scheme  = scheme
        self.n_streams = len(scheme.encoders)
        n_embd  = scheme.trunk_n_embd
        dropout = config.dropout
        bias    = config.bias

        # ── Embedding (shared with Delphi) ──────────────────────────────  ──
        self.domain_to_int = Delphi._build_domain_to_int(list(config.domains.keys()))
        self.int_to_domain = {v: k for k, v in self.domain_to_int.items()}
        self.domain_offsets, global_vocab_size = Delphi._build_domain_offsets(
            self.domain_to_int, config.domains
        )
        self.global_vocab_size = global_vocab_size

        # MultiDomainEmbedding is duck-typed (needs .domains/.n_embd/.token_dropout);
        # n_embd comes from `scheme` (validated above), never from `config`, so
        # there is exactly one place n_embd is read from.
        self.embed = MultiDomainEmbedding(
            config=SimpleNamespace(
                domains=config.domains, n_embd=n_embd, token_dropout=config.token_dropout
            ),
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

        self.encoders = nn.ModuleList([
            _blocks(spec.n_head, spec.n_layer) for spec in scheme.encoders
        ])
        self.trunk = _blocks(scheme.trunk_n_head, scheme.trunk_n_layer)
        self.ln_f  = LayerNorm(n_embd, bias=bias)

        # ── Global attention policy ───────────────────────────────────────
        # Resolve the at_birth alias before storing.
        attention_scheme = self._resolve_at_birth(attention_scheme, config)
        self.attention_scheme = attention_scheme

        # Group aliases (e.g. "hla_alleles" standing in for its per-locus
        # children via `group` or `parent`), same resolution as Delphi._build_model.
        # Without this, an attention_scheme bracket referencing an alias name
        # that isn't itself a real domain (e.g. ESM2 per-locus configs, where
        # "hla_alleles" is abstract) raises a KeyError in AttentionMaskBuilder.build.
        group_to_ints: Dict[str, list] = {}
        for dname, dcfg in config.domains.items():
            alias = dcfg.group or dcfg.parent
            if alias:
                group_to_ints.setdefault(alias, []).append(self.domain_to_int[dname])

        self.mask_builder = AttentionMaskBuilder(attention_scheme, self.domain_to_int, group_to_ints)

        # Group-membership buffers, one per stream (follow .to(device) automatically).
        # Registered under indexed names since ModuleList/buffers can't hold a
        # plain Python list of tensors; use group_ids(i) to retrieve.
        for i, spec in enumerate(scheme.encoders):
            self.register_buffer(f"_group_ids_{i}", torch.tensor([
                self.domain_to_int[d] for d in spec.domains if d in self.domain_to_int
            ]))

        # ── Weight initialisation ─────────────────────────────────────────
        total_layers = (
            sum(spec.n_layer for spec in scheme.encoders) + scheme.trunk_n_layer
        )
        torch.manual_seed(config.seed)
        # initialize_weights only reads .n_layer (for scaled residual init);
        # config has no n_layer field, so pass a minimal stand-in.
        initialize_weights(self, config=SimpleNamespace(n_layer=total_layers))
        self.embed._zero_placeholder_rows()

        self.block_size = config.block_size

    # ── Helpers ───────────────────────────────────────────────────────────────

    def group_ids(self, stream_idx: int) -> torch.Tensor:
        return getattr(self, f"_group_ids_{stream_idx}")

    @staticmethod
    def _resolve_at_birth(attention_scheme: str, config: DelphiMultiStreamConfig) -> str:
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

    def _is_in_group(self, domain_ids: torch.Tensor, stream_idx: int) -> torch.Tensor:
        return torch.isin(domain_ids, self.group_ids(stream_idx))

    def _build_masks(self, batch) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Derive the per-stage attention masks from the global policy.

        Returns ``(masks, mask_trunk)`` where ``masks`` is a list of one
        ``[B, 1, T, T]`` mask per stream, and ``mask_trunk`` is ``[B, 1, T, T]``
        (broadcast-ready over heads).
        """
        global_mask = self.mask_builder.build(
            batch.domain_ids, batch.global_token_ids, batch.ages
        )  # [B, T, T]  int

        masks = []
        for i in range(self.n_streams):
            is_i = self._is_in_group(batch.domain_ids, i)         # [B, T]
            pair = is_i.unsqueeze(2) & is_i.unsqueeze(1)           # [B, T, T]
            # Re-apply fallback after ANDing with group pair, because
            # out-of-group tokens lose all valid targets.
            mask_i = self._apply_self_attn_fallback(global_mask & pair)
            masks.append(mask_i.unsqueeze(1))

        return masks, global_mask.unsqueeze(1)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def config(self) -> DelphiMultiStreamConfig:
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
        masks, mask_trunk = self._build_masks(batch)

        # 3. Independent encoders (parallel, on cloned copies)
        is_in = [self._is_in_group(batch.domain_ids, i) for i in range(self.n_streams)]  # [B, T] each

        acc   = torch.zeros_like(h)
        count = torch.zeros(batch.domain_ids.shape, dtype=h.dtype, device=h.device)  # [B, T]
        for i in range(self.n_streams):
            h_i = h.clone()
            for block in self.encoders[i]:
                h_i, _ = block(h_i, masks[i])
            m = is_in[i].unsqueeze(-1).to(h.dtype)   # [B, T, 1]
            acc = acc + h_i * m
            count = count + is_in[i].to(h.dtype)

        # 4. Merge encoder outputs: average over whichever streams a token
        # belongs to; tokens in no stream keep their original embedding.
        any_membership = (count > 0).unsqueeze(-1)             # [B, T, 1]
        merged = acc / count.clamp(min=1).unsqueeze(-1)
        h = torch.where(any_membership, merged, h)

        # 5. Causal trunk (full global policy) — cross-stream mixing happens here
        att_list = []
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

    def cross_entropy_loss(self, logits, targets, agg=None, token_weights=None):
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
            w = token_weights.to(dtype=logits.dtype, device=logits.device) if token_weights is not None else None
            return F.cross_entropy(
                logits.reshape(-1, n_classes), targets.reshape(-1), ignore_index=-1, weight=w
            )
        raise ValueError(f"agg must be None, 'per_token', or 'per_disease'")

    def time_to_event_loss(self, logits, time_to_next, t_min, agg=None, pos_weights=None):
        lse = torch.logsumexp(logits, -1)
        lse = -torch.log(torch.exp(-lse) + t_min)
        log_dt = -torch.log(torch.clamp(time_to_next, min=1.0) + t_min).view(-1)
        loss_dt = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - log_dt))
        if agg == "mean":
            if pos_weights is not None:
                w = pos_weights.to(dtype=loss_dt.dtype, device=loss_dt.device)
                return (loss_dt * w).sum() / w.sum()
            return loss_dt.mean()
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
        config: DelphiMultiStreamConfig,
    ) -> "DelphiMultiStream":
        """
        Instantiate from strings.

        Parameters
        ----------
        arch_str:
            MultiStream architecture string, e.g.
            ``"MultiStream([hla_alleles,sex]:(h24d240l3),...):(h24d240l6)"``.
        attention_scheme:
            Global attention policy, e.g.
            ``"[hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)"``.
        config:
            :class:`DelphiMultiStreamConfig` (domains, dropout, …).
        """
        return cls(config, parse_multi_stream_scheme(arch_str), attention_scheme)
