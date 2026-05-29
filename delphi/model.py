"""
Delphi model v2.

Key changes vs. the original model.py:
- forward() receives a DelphiBatch (produced by the DataLoader/CollateFn)
- Uses the new MultiDomainEmbedding with global embedding table
- All preprocessing (no-event tokens, truncation, padding, sorting,
  global ID computation) has been moved to the DataLoader collate.
- AgeSampler and all prepare_input logic removed from the model.

Components kept unchanged:
- AttentionMaskBuilder
- AgeEncoding
- Block, MaskedSelfAttention, MLP, LayerNorm
- DelphiConfig, DomainConfig
- Loss functions
"""

from __future__ import annotations

import inspect
import logging
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, TypedDict

import torch
import torch.nn as nn
import yaml
from torch.nn import functional as F

from data.dataset import DelphiBatch
from delphi.embedding import MultiDomainEmbedding

logger = logging.getLogger(__name__)

DAYS_PER_YEAR = 365.25

# ═══════════════════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DomainConfig:
    projector: str = "embed"
    n_layers: int | None = None
    n_hidden: int | None = None
    input_size: int | None = None
    pretrained_path: str | None = None
    freeze: bool = False
    path: str | None = None
    predict: bool = False
    age_jitter: bool = False
    type: str = "categorical"
    at_birth: bool = False
    n_latent_tokens: int | None = None
    subdomain: str | None = None  # filter tokens by metadata (e.g. "hla_a")
    group: str | None = None  # alias for attention mask (e.g. "hla_alleles")
    dropout_mode: str | None = None  # "token" (random tokens) | "block" (entire domain per subject)
    dropout_rate: float = 0.0  # probability of dropping; 0 = disabled

    def set_freeze(self, freeze: bool):
        self.freeze = freeze
        return self

    def unfreeze(self):
        self.set_freeze(freeze=False)

    def set_predict(self, predict: bool):
        self.predict = predict
        return self


@dataclass
class DelphiConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 120
    domains: dict[str, DomainConfig] = field(default_factory=dict)
    attention_scheme: str | list = "all:causal(mask_ties=True)"
    dropout: float = 0.1
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    token_dropout: float = 0.1
    bias: bool = True
    block_size: int = 64
    no_event_token_rate: float = 5
    no_event_token_insertion_mode: str = "random"
    seed: int = 42

    def items(self):
        return asdict(self).items()

    def set_dropout(self, dropout: float):
        self.dropout = dropout
        self.resid_pdrop = dropout
        self.embd_pdrop = dropout
        self.attn_pdrop = dropout
        return self

    def set_token_dropout(self, token_dropout: float):
        self.token_dropout = token_dropout
        return self

    def set_block_size(self, block_size: int):
        self.block_size = block_size
        return self

    def set_attention_scheme(self, attention_scheme: str | list):
        self.attention_scheme = attention_scheme
        return self

    def add_domain(self, domain_name: str, embed_config: DomainConfig):
        self.domains[domain_name] = embed_config
        return self

    def remove_domain(self, domain_name: str):
        if domain_name in self.domains:
            del self.domains[domain_name]
        return self


# ═══════════════════════════════════════════════════════════════════════════════
#  Attention Mask Builder  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════


class AttentionRule(TypedDict):
    type: Literal["causal", "bidirectional"]
    mask_ties: bool


class AttentionMaskBuilder(nn.Module):
    def __init__(self, scheme_str: str, domain2id: dict):
        super().__init__()
        self.scheme = AttentionMaskBuilder._parse_scheme(scheme_str)
        self.domain2id = domain2id

    @staticmethod
    def _split_top_level(s: str, sep: str = ",") -> list[str]:
        parts: list[str] = []
        buf, depth_brack, depth_paren = "", 0, 0
        for ch in s:
            if ch == "[":
                depth_brack += 1
            elif ch == "]":
                depth_brack -= 1
            elif ch == "(":
                depth_paren += 1
            elif ch == ")":
                depth_paren -= 1
            if ch == sep and depth_brack == 0 and depth_paren == 0:
                if buf.strip():
                    parts.append(buf.strip())
                buf = ""
            else:
                buf += ch
        if buf.strip():
            parts.append(buf.strip())
        return parts

    @staticmethod
    def _parse_scheme(scheme_str: str) -> dict[tuple[str, ...], AttentionRule]:
        rule_type: Literal["causal", "bidirectional"]
        mask_ties: bool

        scheme: dict[tuple[str, ...], AttentionRule] = {}
        parts = AttentionMaskBuilder._split_top_level(scheme_str)
        for part in parts:
            if ":" not in part:
                raise ValueError(f"Invalid rule fragment: {part}")
            domain_part, rule_part = part.split(":", 1)
            domain_part, rule_part = domain_part.strip(), rule_part.strip()
            if domain_part.startswith("[") and domain_part.endswith("]"):
                domains = [d.strip() for d in domain_part[1:-1].split(",")]
            else:
                domains = [domain_part]

            if rule_part.startswith("causal"):
                rule_type = "causal"
                mask_ties = "mask_ties=True" in rule_part
            elif rule_part.startswith("bidirectional"):
                rule_type = "bidirectional"
                mask_ties = False
            else:
                raise ValueError(f"Unknown rule type: {rule_part}")
            scheme[tuple(domains)] = {"type": rule_type, "mask_ties": mask_ties}
        return scheme

    def build(self, domains: torch.Tensor, local_token_ids: torch.Tensor, ages: torch.Tensor):
        B, L = ages.shape
        device = ages.device
        mask = torch.zeros(B, L, L, device=device)
        age_row = ages.unsqueeze(2)
        age_col = ages.unsqueeze(1)

        all_domain_ids = torch.tensor(list(self.domain2id.values()), device=device)

        for dom_names, cfg in self.scheme.items():
            # Resolve "all" to every domain
            if dom_names == ("all",):
                dom_ids = all_domain_ids
            else:
                dom_ids = torch.tensor([self.domain2id[d] for d in dom_names], device=device)
            dom_mask = torch.isin(domains, dom_ids)
            pair_mask = dom_mask.unsqueeze(2) & dom_mask.unsqueeze(1)

            if cfg["type"] == "bidirectional":
                mask[pair_mask] = 1
            elif cfg["type"] == "causal":
                causal = age_row > age_col if cfg.get("mask_ties", False) else age_row >= age_col
                final = pair_mask & causal
                mask[final] = 1
            else:
                raise ValueError(f"Unknown attention type: {cfg['type']}")

        mask = mask.bool()
        # Padding tokens have age == PADDING_AGE (-10000); no-event tokens share
        # the padding domain but carry real positive ages, so age-based detection
        # is more robust than checking local_token_ids (which may be global IDs).
        is_padding = ages < 0.0
        mask &= ~(is_padding.unsqueeze(2) | is_padding.unsqueeze(1))
        mask = mask.int()

        # Fallback self-attention to avoid NaNs
        row_has_any = mask.any(dim=-1)
        needs_fallback = ~row_has_any
        b_idx, i_idx = needs_fallback.nonzero(as_tuple=True)
        mask[b_idx, i_idx, i_idx] = 1

        return mask

    def forward(self, domains, local_token_ids, ages):
        return self.build(domains, local_token_ids, ages)


# ═══════════════════════════════════════════════════════════════════════════════
#  Age Encoding  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════


class AgeEncoding(nn.Module):
    def __init__(self, n_embd: int, norm_factor: float = 365.25, max_wavelen: float = 10000.0):
        super().__init__()
        div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(max_wavelen) / n_embd))
        self.register_buffer("div_term", div_term)
        self.n_embd = n_embd
        self.linear = nn.Linear(n_embd, n_embd, bias=False)
        self.norm_factor = norm_factor

    def forward(self, age: torch.Tensor):
        age = age.float() / self.norm_factor
        if age.ndim == 0:
            age = age.view(1, 1)
        elif age.ndim == 1:
            age = age.unsqueeze(1)

        seq_len, batch_size = age.shape
        age_expanded = age.unsqueeze(-1)
        div_term = self.div_term.view(1, 1, -1)
        sin_part = torch.sin(age_expanded * div_term)
        cos_part = torch.cos(age_expanded * div_term)
        y = torch.zeros(seq_len, batch_size, self.n_embd, device=age.device)
        y[..., 0::2] = sin_part
        y[..., 1::2] = cos_part
        if batch_size == 1:
            y = y.squeeze(1)
        return self.linear(y)


# ═══════════════════════════════════════════════════════════════════════════════
#  Transformer components  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MaskedSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        hs = k.size(-1)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        if attn_mask is not None:
            att = att.masked_fill(attn_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, att


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MaskedSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask):
        y, att = self.attn(self.ln_1(x), attn_mask)
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, att


# ═══════════════════════════════════════════════════════════════════════════════
#  Weight initialization
# ═══════════════════════════════════════════════════════════════════════════════


def initialize_weights(model: nn.Module, config: DelphiConfig):
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    model.apply(_init_weights)
    for pn, p in model.named_parameters():
        if pn.endswith("c_proj.weight"):
            torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))


# ═══════════════════════════════════════════════════════════════════════════════
#  Delphi
# ═══════════════════════════════════════════════════════════════════════════════


class Delphi(nn.Module):
    def __init__(self, config: DelphiConfig):
        super().__init__()

        self.config = config
        self._attention_schemes = self._adapt_attention_scheme(config.attention_scheme)

        # Derive shared metadata from config.domains
        self.domain_to_int = self._build_domain_to_int(list(config.domains.keys()))
        self.int_to_domain = {v: k for k, v in self.domain_to_int.items()}
        self.domain_offsets, global_vocab_size = self._build_domain_offsets(self.domain_to_int, config.domains)
        self.global_vocab_size = global_vocab_size

        self._build_model(config)
        self.set_block_size(config.block_size)

        torch.manual_seed(config.seed)
        initialize_weights(self, config=config)

        # Re-zero placeholder rows after init (init overwrote them)
        self.embed._zero_placeholder_rows()

    # ── Domain metadata (static, reusable by dataset/collate) ─────────────

    @staticmethod
    def _build_domain_to_int(domain_names: list[str]) -> dict[str, int]:
        """Stable mapping: all domains except padding first, padding last."""
        ordered = [d for d in domain_names if d != "padding"] + ["padding"]
        return {name: i for i, name in enumerate(ordered)}

    @staticmethod
    def _resolve_vocab_size(cfg) -> int:
        """Get vocab size, reading tokenizer.yaml if input_size is None."""
        if cfg.input_size is not None:
            return cfg.input_size
        tokenizer_path = Path(cfg.path) / "tokenizer.yaml"
        with tokenizer_path.open() as f:
            return len(yaml.safe_load(f))

    @staticmethod
    def _build_domain_offsets(
        domain_to_int: dict[str, int],
        domain_cfg: dict[str, DomainConfig],
    ) -> tuple[dict[int, int], int]:
        """Compute global embedding offsets. Returns (offsets, global_vocab_size)."""
        offsets = {}
        running = 0
        for dname in domain_to_int:
            d_int = domain_to_int[dname]
            cfg = domain_cfg.get(dname)

            if dname == "padding":
                offsets[d_int] = running
                running += 2  # PADDING_TOKEN=0, NO_EVENT_TOKEN=1
            elif cfg is None:
                offsets[d_int] = running
            elif cfg.type == "categorical" and cfg.projector in ("embed", "Embed"):
                offsets[d_int] = running
                running += Delphi._resolve_vocab_size(cfg)
            else:
                # projected domain: placeholder slots
                offsets[d_int] = running
                n_slots = getattr(cfg, "n_latent_tokens", 1) or 1
                running += n_slots

        return offsets, running

    # ── Model construction ────────────────────────────────────────────────

    def _adapt_attention_scheme(self, attention_scheme):
        if isinstance(attention_scheme, str):
            attention_scheme = [attention_scheme]
        if len(attention_scheme) == 1:
            attention_scheme = self.config.n_layer * attention_scheme
        if any("at_birth" in s for s in attention_scheme):
            at_birth = ",".join(name for name, cfg in self.config.domains.items() if cfg.at_birth)
            attention_scheme = [s.replace("at_birth", at_birth) for s in attention_scheme]
        return attention_scheme

    def _build_model(self, config):

        self.embed = MultiDomainEmbedding(
            config=config,
            domain_offsets=self.domain_offsets,
            global_vocab_size=self.global_vocab_size,
            domain_to_int=self.domain_to_int,
        )

        self.transformer = nn.ModuleDict(
            dict(
                age_embedding=AgeEncoding(n_embd=config.n_embd),
                drop=nn.Dropout(config.dropout),
                attn_mask_builder=nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                AttentionMaskBuilder(self._attention_schemes[i], self.domain_to_int)
                                for i in range(config.n_layer)
                            ]
                        )
                        for j in range(config.n_head)
                    ]
                ),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

    # ── Properties ────────────────────────────────────────────────────────

    def set_block_size(self, block_size):
        self.block_size = block_size

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def domain_cfg(self):
        return self.config.domains

    @property
    def domains(self):
        return list(self.config.domains.keys())

    @property
    def categorical_domains(self):
        return [k for k, v in self.config.domains.items() if v.type == "categorical"]

    @property
    def continuous_domains(self):
        return [k for k, v in self.config.domains.items() if v.type == "continuous"]

    @property
    def predicted_domains(self):
        return self.embed.predicted_domains

    # ── Attention mask ────────────────────────────────────────────────────

    def build_attn_mask(self, batch):
        """
        Build attention mask from a DelphiBatch.
        Returns [n_layer, B, n_head, T, T].
        """
        single_mask = self.transformer.attn_mask_builder[0][0].build(
            batch.domain_ids, batch.global_token_ids, batch.ages
        )
        attn_mask = (
            single_mask.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, self.config.n_layer, self.config.n_head, -1, -1)
            .permute(1, 0, 2, 3, 4)
        )
        return attn_mask

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        batch,
        return_attention: bool = False,
        return_embeddings: bool = False,
    ):
        """
        Parameters
        ----------
        batch : DelphiBatch
            Pre-processed batch from the DataLoader.
        return_attention : bool
            If True, return stacked attention matrices.
        return_embeddings : bool
            If True, return hidden states before logit projection.

        Returns
        -------
        logits : dict[str, Tensor]
            {domain_name: [B, T, vocab_size]} for predicted domains.
        attention : Tensor or None
            [n_layer, B, n_head, T, T] if requested.
        embeddings : Tensor or None
            [B, T, n_embd] if requested.
        """
        # 1. Embed all tokens
        emb = self.embed(batch)  # [B, T, n_embd]

        # 2. Add age encoding
        #    AgeEncoding expects (T, B) or (T,), but batch.ages is [B, T]
        #    Transpose, encode, transpose back.
        age_emb = self.transformer.age_embedding(
            batch.ages.T  # [T, B]
        )  # [T, B, n_embd] or [T, n_embd]
        if age_emb.ndim == 2:
            age_emb = age_emb.unsqueeze(1)  # [T, 1, n_embd]
        age_emb = age_emb.transpose(0, 1)  # [B, T, n_embd]

        emb = emb + age_emb
        emb = self.transformer.drop(emb)

        # 3. Build attention mask
        attn_mask = self.build_attn_mask(batch)  # [n_layer, B, n_head, T, T]

        # 4. Transformer blocks
        h = emb
        att_list = []
        for i, block in enumerate(self.transformer.h):
            h, att = block(h, attn_mask=attn_mask[i])
            att_list.append(att)

        h = self.transformer.ln_f(h)  # [B, T, n_embd]

        # 5. Logits (tied weights, predicted domains only)
        logits = self.embed.to_logits(h)

        # 6. Build return tuple
        attention = torch.stack(att_list) if return_attention else None

        if return_embeddings:
            return logits, attention, h
        else:
            return logits, attention

    # ── Loss functions ────────────────────────────────────────────────────

    def cross_entropy_loss(self, logits, targets, agg=None):
        import pandas as pd

        n_classes = logits.size(-1)
        if agg == "per_token":
            log_softmax = F.log_softmax(logits.view(-1, n_classes), dim=-1)
            loss_ce_per_token = log_softmax[torch.arange(log_softmax.size(0)), targets.view(-1)]
            return loss_ce_per_token
        elif agg == "per_disease":
            # Returns a pd.Series indexed by token_id with mean log-probability per token.
            # Negative log-prob = CE loss; more negative = harder to predict.
            # Uses scatter_add on GPU to avoid a pandas groupby round-trip.
            flat_targets = targets.view(-1)
            log_softmax = F.log_softmax(logits.view(-1, n_classes), dim=-1)
            log_p = log_softmax[torch.arange(flat_targets.size(0), device=logits.device), flat_targets]

            unique_ids, inverse, counts = torch.unique(flat_targets, return_inverse=True, return_counts=True)
            sum_log_p = torch.zeros(unique_ids.size(0), dtype=log_p.dtype, device=logits.device)
            sum_log_p.scatter_add_(0, inverse, log_p)
            mean_log_p = sum_log_p / counts.float()

            result = pd.Series(
                mean_log_p.detach().cpu().numpy(),
                index=unique_ids.cpu().numpy(),
                name="log_p",
            )
            result.index.name = "token_id"
            return result
        elif agg is None:
            loss_ce = F.cross_entropy(
                logits.reshape(-1, n_classes),
                targets.reshape(-1),
                ignore_index=-1,
            )
        else:
            raise ValueError("agg should be in [None, 'per_token', 'per_disease']")
        return loss_ce

    def time_to_event_loss(self, logits, time_to_next, t_min, agg=None):
        lse = torch.logsumexp(logits, -1)
        lse = -torch.log(torch.exp(-lse) + t_min)
        dt = torch.clamp(time_to_next, min=1.0)
        log_dt = -torch.log(dt + t_min).view(-1)
        loss_dt = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - log_dt.reshape(-1)))

        if agg is None:
            pass
        elif agg == "mean":
            loss_dt = loss_dt.mean()
        elif agg == "sum":
            loss_dt = loss_dt.sum()
        else:
            raise NotImplementedError
        return loss_dt

    # ── Inference convenience ─────────────────────────────────────────────

    def run_inference(self, dataloader: Iterable[DelphiBatch], block_size=None, return_token_df=False):
        """
        Run inference over an entire dataloader.
        The dataloader should use DelphiCollateFn.
        """
        if return_token_df:
            raise NotImplementedError

        old_block_size = self.block_size
        if block_size is not None:
            logger.warning(f"Temporarily changing block_size from {self.block_size} to {block_size}")
            self.set_block_size(block_size)

        all_logits: list[torch.Tensor] = []
        for batch in dataloader:
            batch = batch.to(self.device)
            with torch.no_grad():
                logits_dict = self.forward(batch, return_embeddings=False)[0]

            _logits_all_domains: list[torch.Tensor] = []
            for dom in self.predicted_domains:
                assert dom in logits_dict
                _logits_all_domains.append(logits_dict[dom])
            logits_all_domains = torch.cat(_logits_all_domains, dim=-1)
            all_logits.append(logits_all_domains)

        logits = torch.cat(all_logits, dim=0)

        if block_size is not None:
            self.set_block_size(old_block_size)

        return logits

    # ── Checkpoint loading ────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(cls, ckpt_path, device=None):
        """
        Load a model from a checkpoint.
        domain_to_int, offsets and global_vocab_size are derived
        automatically from config.domains.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(ckpt_path, map_location=device)

        sig = inspect.signature(DelphiConfig.__init__)
        valid_keys = set(sig.parameters.keys()) - {"self"}
        conf_dict = {k: v for k, v in checkpoint["model_args"].items() if k in valid_keys}
        conf = DelphiConfig(**conf_dict)

        model = cls(config=conf)

        state_dict = checkpoint["model"]
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)

        return model
