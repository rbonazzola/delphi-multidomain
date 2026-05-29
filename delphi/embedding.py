"""
MultiDomainEmbedding v2.

Single global embedding table for all "simple" categorical domains,
with scatter_add injection for projected domains (continuous, pretrained).

Usage:
    embed = MultiDomainEmbedding(config, domain_offsets, global_vocab_size)
    h = embed(batch)          # batch: DelphiBatch -> [B, T, n_embd]
    logits = embed.to_logits(h)  # -> {domain_name: [B, T, vocab_size]}
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Projectors for non-simple domains
# ═══════════════════════════════════════════════════════════════════════════════


class LinearProjector(nn.Module):
    """
    Projects a continuous input vector to (n_latent_tokens, n_embd).
    input_size -> n_latent_tokens * n_embd, then unflatten.
    """

    def __init__(self, input_size: int, n_latent_tokens: int, n_embd: int):
        super().__init__()
        self.n_latent_tokens = n_latent_tokens
        self.n_embd = n_embd
        self.projector = nn.Sequential(
            nn.Linear(input_size, n_latent_tokens * n_embd, bias=False),
            nn.Unflatten(dim=-1, unflattened_size=(n_latent_tokens, n_embd)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_size]
        returns: [B, n_latent_tokens, n_embd]
        """
        return self.projector(x)


class MLPProjector(nn.Module):
    """
    Multi-layer projector: input_size -> hidden layers -> n_latent_tokens * n_embd.
    """

    def __init__(
        self,
        input_size: int,
        n_latent_tokens: int,
        n_embd: int,
        n_layers: int,
        n_hidden: int,
    ):
        super().__init__()
        self.n_latent_tokens = n_latent_tokens
        self.n_embd = n_embd

        E = n_latent_tokens * n_embd
        sizes = [input_size] + [n_hidden] * (n_layers - 1) + [E]

        layers: list[nn.Module] = []
        for i in range(n_layers):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
        layers.append(nn.Unflatten(dim=-1, unflattened_size=(n_latent_tokens, n_embd)))

        self.projector = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_size]
        returns: [B, n_latent_tokens, n_embd]
        """
        return self.projector(x)


class PretrainedProjector(nn.Module):
    """
    Pretrained embedding lookup + linear projection to n_embd.
    Input is token IDs (not continuous values).

    The pretrained embedding can optionally be frozen.
    Output: one embedding per token, so n_latent_tokens = n_input_tokens.
    """

    def __init__(
        self,
        pretrained_path: str,
        n_embd: int,
        freeze: bool = False,
    ):
        super().__init__()
        weights = torch.load(pretrained_path, map_location="cpu")
        vocab_size, d_ext = weights.shape

        self.embed = nn.Embedding(vocab_size, d_ext)
        self.embed.weight.data.copy_(weights)
        self.embed.weight.requires_grad = not freeze

        self.linear = nn.Linear(d_ext, n_embd, bias=False)
        self.vocab_size = vocab_size
        self.d_ext = d_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, n_tokens] int — local token IDs
        returns: [B, n_tokens, n_embd]
        """
        h = self.embed(x)
        return self.linear(h)


# ═══════════════════════════════════════════════════════════════════════════════
#  MultiDomainEmbedding
# ═══════════════════════════════════════════════════════════════════════════════


class MultiDomainEmbedding(nn.Module):
    """
    Unified embedding layer for all domains.

    Architecture:
    - One global nn.Embedding for all "simple" categorical domains
      (including padding). Each domain occupies a contiguous slice
      of the global table, determined by domain_offsets.
    - Separate projector modules for "projected" domains (continuous,
      pretrained). These domains get placeholder rows (zeroed) in the
      global table; their real embeddings are injected via scatter_add.

    Parameters
    ----------
    config : DelphiConfig
        Full model config (needs config.domains, config.n_embd, config.token_dropout).
    domain_offsets : dict[int, int]
        {domain_int: global_offset}. Computed by build_domain_offsets.
    global_vocab_size : int
        Total rows in the global embedding table.
    domain_to_int : dict[str, int]
        {domain_name: domain_int}.
    """

    def __init__(
        self,
        config,
        domain_offsets: dict[int, int],
        global_vocab_size: int,
        domain_to_int: dict[str, int],
    ):
        super().__init__()

        self.n_embd = config.n_embd
        self.domain_configs = config.domains
        self.domain_offsets = domain_offsets
        self.domain_to_int = domain_to_int
        self.int_to_domain = {v: k for k, v in domain_to_int.items()}
        self.global_vocab_size = global_vocab_size

        # ── Global embedding table ────────────────────────────────────────
        self.global_embed = nn.Embedding(global_vocab_size, config.n_embd)

        # Zero out placeholder rows for projected/pretrained domains
        self._projected_domain_names: list[str] = []  # continuous: injected via continuous_data
        self._pretrained_domain_names: list[str] = []  # categorical pretrained: injected via domain_ids mask
        self._projected_offsets: dict[str, tuple[int, int]] = {}  # domain_name -> (offset, n_slots)
        self._init_projectors(config)

        # ── Token dropout ─────────────────────────────────────────────────
        self.token_drop = nn.Dropout(config.token_dropout)

        # ── Cache predicted domain info ───────────────────────────────────
        self._predicted_domains = [dname for dname, dcfg in config.domains.items() if dcfg.predict]

        # ── Cache vocab sizes (avoid reading YAML on every forward) ───────
        self._domain_vocab_sizes = self._compute_vocab_sizes(config.domains)

    def _compute_vocab_sizes(self, domain_configs) -> dict[str, int]:
        """Resolve and cache vocab size for every domain, once at init."""
        sizes = {}
        for dname, dcfg in domain_configs.items():
            if dname == "padding":
                sizes[dname] = 2
            elif dcfg.input_size is not None:
                sizes[dname] = dcfg.input_size
            else:
                tokenizer_path = Path(dcfg.path) / "tokenizer.yaml"
                with tokenizer_path.open() as f:
                    sizes[dname] = len(yaml.safe_load(f))
        return sizes

    def _init_projectors(self, config):
        """
        Build projector modules for non-simple domains and register
        which global embedding rows should stay zero.
        """
        self.projectors = nn.ModuleDict()
        n_embd = config.n_embd

        for dname, dcfg in config.domains.items():
            if dname == "padding":
                continue

            d_int = self.domain_to_int[dname]
            is_projected = False

            if dcfg.type == "continuous":
                is_projected = True
                n_latent = dcfg.n_latent_tokens or 1

                if dcfg.projector.lower() == "linear":
                    self.projectors[dname] = LinearProjector(
                        input_size=dcfg.input_size,
                        n_latent_tokens=n_latent,
                        n_embd=n_embd,
                    )
                elif dcfg.projector.lower() == "mlp":
                    self.projectors[dname] = MLPProjector(
                        input_size=dcfg.input_size,
                        n_latent_tokens=n_latent,
                        n_embd=n_embd,
                        n_layers=dcfg.n_layers,
                        n_hidden=dcfg.n_hidden,
                    )
                else:
                    raise ValueError(f"Unknown projector '{dcfg.projector}' for continuous domain '{dname}'")

            elif dcfg.projector.lower() == "pretrained":
                assert dcfg.pretrained_path is not None, (
                    f"Domain '{dname}' has projector='pretrained' but pretrained_path is not set."
                )
                projector = PretrainedProjector(
                    pretrained_path=dcfg.pretrained_path,
                    n_embd=n_embd,
                    freeze=dcfg.freeze,
                )
                self.projectors[dname] = projector
                self._pretrained_domain_names.append(dname)
                offset = self.domain_offsets[d_int]
                self._projected_offsets[dname] = (offset, projector.vocab_size)
                is_projected = False  # handled separately, not via continuous_data

            if is_projected:
                self._projected_domain_names.append(dname)
                offset = self.domain_offsets[d_int]
                n_slots = dcfg.n_latent_tokens or 1
                self._projected_offsets[dname] = (offset, n_slots)

        # Zero out placeholder rows so the global lookup gives zeros
        self._zero_placeholder_rows()

    @torch.no_grad()
    def _zero_placeholder_rows(self):
        """Set global embedding rows for projected domains to zero."""
        for _, (offset, n_slots) in self._projected_offsets.items():
            self.global_embed.weight[offset : offset + n_slots] = 0.0

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def predicted_domains(self) -> list[str]:
        return self._predicted_domains

    def _get_domain_weight(self, dname: str) -> torch.Tensor:
        """
        Get the embedding weight matrix for a simple categorical domain.
        This is a *view* into the global embedding table.
        """
        d_int = self.domain_to_int[dname]
        offset = self.domain_offsets[d_int]
        vocab_size = self._domain_vocab_sizes[dname]
        return self.global_embed.weight[offset : offset + vocab_size]

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, batch) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : DelphiBatch
            Must have: global_token_ids, continuous_data, continuous_positions.

        Returns
        -------
        emb : [B, T, n_embd]
        """
        # Step 1: global lookup (projected slots give zeros)
        emb = self.global_embed(batch.global_token_ids)  # [B, T, n_embd]

        # Step 2: inject continuous projected domain embeddings via scatter_add
        for dname in self._projected_domain_names:
            if dname not in batch.continuous_data:
                continue

            data = batch.continuous_data[dname]  # [B, dim] or [B, n_tokens]
            positions = batch.continuous_positions[dname]  # [B, n_latent]
            proj_emb = self.projectors[dname](data)  # [B, n_latent, n_embd]
            proj_emb = proj_emb.to(emb.dtype)

            B, n_latent, E = proj_emb.shape

            # Expand positions to [B, n_latent, n_embd] for scatter_add
            idx = positions.unsqueeze(-1).expand(B, n_latent, E)  # [B, n_latent, n_embd]
            emb.scatter_add_(1, idx, proj_emb)

        # Step 3: inject pretrained categorical domain embeddings
        for dname in self._pretrained_domain_names:
            domain_int = self.domain_to_int[dname]
            offset = self.domain_offsets[domain_int]
            mask = batch.domain_ids == domain_int  # [B, T]
            if not mask.any():
                continue
            local_ids = batch.global_token_ids[mask] - offset  # [N]
            h = self.projectors[dname].embed(local_ids)  # [N, d_ext]
            proj = self.projectors[dname].linear(h).to(emb.dtype)  # [N, n_embd]
            emb[mask] = proj

        return emb

    # ── Logits (tied weights) ─────────────────────────────────────────────

    def to_logits(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Project hidden states back to token logits using tied weights.
        Only for predicted (simple categorical) domains.

        Parameters
        ----------
        h : [B, T, n_embd]

        Returns
        -------
        {domain_name: [B, T, vocab_size]}
        """
        logits = {}
        for dname in self._predicted_domains:
            W = self._get_domain_weight(dname)  # [vocab_size, n_embd]
            logits[dname] = F.linear(h, W)  # [B, T, vocab_size]
        return logits

    # ── Add domain (between training runs) ────────────────────────────────

    def add_domain(
        self,
        domain_name: str,
        domain_config,
        domain_int: int,
        offset: int,
        vocab_size: int,
    ):
        """
        Extend the global embedding table to accommodate a new domain.
        Preserves existing weights.

        Parameters
        ----------
        domain_name : str
        domain_config : DomainConfig
        domain_int : int
            The integer ID for this domain.
        offset : int
            Where this domain starts in the new global table.
        vocab_size : int
            Number of new rows to add.
        """
        old_weight = self.global_embed.weight.data
        old_vocab, n_embd = old_weight.shape
        new_vocab = old_vocab + vocab_size

        # Create new embedding
        new_embed = nn.Embedding(new_vocab, n_embd)
        with torch.no_grad():
            # Copy old weights
            new_embed.weight[:old_vocab] = old_weight
            # Initialize new rows
            nn.init.normal_(new_embed.weight[old_vocab:], mean=0.0, std=0.02)

        self.global_embed = new_embed
        self.global_vocab_size = new_vocab

        # Update mappings
        self.domain_to_int[domain_name] = domain_int
        self.int_to_domain[domain_int] = domain_name
        self.domain_offsets[domain_int] = offset
        self.domain_configs[domain_name] = domain_config

        if domain_config.predict:
            self._predicted_domains.append(domain_name)

        # If it's a projected or pretrained domain, register accordingly
        if domain_config.type == "continuous":
            self._projected_domain_names.append(domain_name)
            n_slots = domain_config.n_latent_tokens or 1
            self._projected_offsets[domain_name] = (offset, n_slots)
            self._zero_placeholder_rows()
        elif domain_config.projector == "pretrained":
            self._pretrained_domain_names.append(domain_name)
            n_slots = vocab_size
            self._projected_offsets[domain_name] = (offset, n_slots)
            self._zero_placeholder_rows()

        logger.info(
            "Added domain '%s': domain_int=%d, offset=%d, vocab_size=%d, new global_vocab_size=%d",
            domain_name,
            domain_int,
            offset,
            vocab_size,
            new_vocab,
        )

    # ── Utilities ─────────────────────────────────────────────────────────

    def __repr__(self):
        simple = [d for d in self.domain_configs if d not in self._projected_domain_names and d != "padding"]
        return (
            f"MultiDomainEmbedding(\n"
            f"  global_embed: Embedding({self.global_vocab_size}, {self.n_embd})\n"
            f"  simple domains: {simple}\n"
            f"  projected domains: {self._projected_domain_names}\n"
            f"  predicted domains: {self._predicted_domains}\n"
            f")"
        )
