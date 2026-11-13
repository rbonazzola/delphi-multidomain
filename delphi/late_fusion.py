"""
DelphiLateFusion: frozen Delphi trunk + trainable HLA late-fusion head.

The trunk processes a token sequence (diseases, drugs, lifestyle, sex, genetic PCs,
etc. — no HLA) and produces per-domain logits. The head takes a precomputed static
HLA feature vector [B, d_hla] per subject and adds a contribution to each predicted
domain's logits (broadcast over the sequence length T).

Three HLA encoders are provided:
  MultiHotHLAEncoder  — multi-hot over 2-field (359-dim) or 1-field (138-dim) alleles
  PretrainedHLAEncoder — mean-pool pretrained (ESM2) per-allele embeddings → [B, 5120]
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════════
#  HLA Encoders  (CPU, used at dataset-build time, not inside the model forward)
# ═══════════════════════════════════════════════════════════════════════════════

class MultiHotHLAEncoder:
    """
    Encodes a subject's HLA alleles as a multi-hot vector.

    Parameters
    ----------
    metadata_path : path to hla_alleles/token_metadata.tsv
        Required columns: token_id, allele_1field (only for mode="1field").
    mode : "2field" → 359-dim; "1field" → 138-dim.
    """

    def __init__(self, metadata_path: Union[str, Path], mode: str = "2field"):
        assert mode in ("1field", "2field"), (
            f"mode must be '1field' or '2field', got {mode!r}"
        )
        self.mode = mode
        meta = pd.read_csv(metadata_path, sep="\t")

        if mode == "2field":
            self.n_features = int(meta["token_id"].max()) + 1   # 359
            self._token_to_idx: Dict[int, int] = {
                int(r.token_id): int(r.token_id) for _, r in meta.iterrows()
            }
        else:   # 1field
            unique_1f = sorted(meta["allele_1field"].unique())
            self._1f_to_idx = {a: i for i, a in enumerate(unique_1f)}
            self.n_features = len(unique_1f)   # 138
            self._token_to_idx = {
                int(r.token_id): self._1f_to_idx[r.allele_1field]
                for _, r in meta.iterrows()
            }

    def encode(self, token_ids_per_subject: list) -> torch.Tensor:
        """
        Parameters
        ----------
        token_ids_per_subject : list of length N.
            Each element is an iterable of int 2-field token IDs for that subject.

        Returns
        -------
        Tensor [N, n_features], float32.
        """
        N = len(token_ids_per_subject)
        out = torch.zeros(N, self.n_features)
        for i, token_ids in enumerate(token_ids_per_subject):
            for tid in token_ids:
                idx = self._token_to_idx.get(int(tid))
                if idx is not None:
                    out[i, idx] = 1.0
        return out


class PretrainedHLAEncoder:
    """
    Encodes a subject's HLA genotype by mean-pooling pretrained allele embeddings.

    Loads per-locus .pt files from embeddings_dir and assembles a global
    [n_alleles, d_embd] matrix indexed by global 2-field token IDs.
    Each subject's representation is the mean of their alleles' embeddings.

    Parameters
    ----------
    embeddings_dir : directory with files named <locus>.pt (e.g. hla_a.pt)
    metadata_path : path to hla_alleles/token_metadata.tsv
        Required columns: token_id, locus.
    """

    def __init__(
        self,
        embeddings_dir: Union[str, Path],
        metadata_path: Union[str, Path],
    ):
        embeddings_dir = Path(embeddings_dir)
        meta = pd.read_csv(metadata_path, sep="\t")

        first_locus = meta["locus"].iloc[0]
        d_embd = torch.load(
            embeddings_dir / f"{first_locus}.pt", map_location="cpu", weights_only=False
        ).shape[1]
        self.n_features = d_embd

        n_alleles = len(meta)
        global_emb = torch.zeros(n_alleles, d_embd)

        for locus in meta["locus"].unique():
            locus_emb = torch.load(
                embeddings_dir / f"{locus}.pt", map_location="cpu", weights_only=False
            )   # [n_locus_alleles, d_embd]
            # local_idx 0..n-1 corresponds to alleles sorted by global token_id within this locus
            locus_meta = meta[meta["locus"] == locus].sort_values("token_id")
            for local_idx, (_, row) in enumerate(locus_meta.iterrows()):
                global_emb[int(row.token_id)] = locus_emb[local_idx]

        self._global_emb = global_emb   # [n_alleles, d_embd], CPU

    def encode(self, token_ids_per_subject: list) -> torch.Tensor:
        """
        Parameters
        ----------
        token_ids_per_subject : list of length N.

        Returns
        -------
        Tensor [N, d_embd] — mean of pretrained embeddings; zero for subjects with no alleles.
        """
        N = len(token_ids_per_subject)
        out = torch.zeros(N, self.n_features)
        for i, token_ids in enumerate(token_ids_per_subject):
            if len(token_ids) > 0:
                idx = torch.tensor([int(t) for t in token_ids], dtype=torch.long)
                out[i] = self._global_emb[idx].mean(dim=0)
        return out


class LocusESM2HLAEncoder:
    """
    Per-locus ESM2 encoding for a low-rank-per-locus linear head (LocusLowRankHLAHead).

    encode() returns a light [N, n_alleles] matrix: each of a subject's alleles gets weight
    count / (number of the subject's alleles at that locus), so that, for each locus l,
    features[:, locus_l] @ embeddings[locus_l] is the mean ESM2 embedding of the subject's
    alleles at l (homozygotes count twice). The [n_alleles, d_embd] ESM2 table itself is
    kept in `embeddings` and applied inside the head, which avoids materialising
    n_loci x d_embd floats per subject.
    """

    def __init__(self, embeddings_dir: Union[str, Path], metadata_path: Union[str, Path]):
        pre = PretrainedHLAEncoder(embeddings_dir, metadata_path)
        meta = pd.read_csv(metadata_path, sep="\t")
        self.embeddings = pre._global_emb                       # [n_alleles, d_embd]
        self.n_features = self.embeddings.shape[0]               # input width of the head
        self.loci = sorted(meta["locus"].unique())
        self.locus_groups = {l: sorted(int(t) for t in meta.loc[meta["locus"] == l, "token_id"]) for l in self.loci}
        self._locus_of = {int(r.token_id): r.locus for r in meta.itertuples()}

    def encode(self, token_ids_per_subject: list) -> torch.Tensor:
        out = torch.zeros(len(token_ids_per_subject), self.n_features)
        for i, token_ids in enumerate(token_ids_per_subject):
            per_locus: Dict[str, list] = {}
            for t in token_ids:
                per_locus.setdefault(self._locus_of.get(int(t)), []).append(int(t))
            for locus, ts in per_locus.items():
                if locus is None:
                    continue
                for t in ts:
                    out[i, t] += 1.0 / len(ts)
        return out


class LocusLowRankHLAHead(nn.Module):
    """
    Linear head with a separate rank-`rank` map per HLA locus:

        contrib = W_out · concat_l( A_l · mean_ESM2_l ) + b

    A_l: d_embd → rank (no bias), one per locus, applied to the mean frozen ESM2 embedding of
    the subject's alleles at locus l; W_out: n_loci·rank → outputs. Linear overall. Input is
    LocusESM2HLAEncoder's [B, n_alleles] locus-normalised weights.
    """

    def __init__(self, embeddings: torch.Tensor, locus_groups: Dict[str, List[int]],
                 predicted_vocab_sizes: Dict[str, int], rank: int):
        super().__init__()
        self.domain_names = list(predicted_vocab_sizes.keys())
        self.vocab_sizes = list(predicted_vocab_sizes.values())
        self.register_buffer("embeddings", embeddings.float().clone(), persistent=False)
        self.loci = list(locus_groups)
        for l in self.loci:
            self.register_buffer(f"idx_{l}", torch.tensor(locus_groups[l], dtype=torch.long), persistent=False)
        d = embeddings.shape[1]
        self.proj = nn.ModuleDict({l: nn.Linear(d, rank, bias=False) for l in self.loci})
        self.out = nn.Linear(len(self.loci) * rank, sum(self.vocab_sizes))

    def forward(self, hla_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = []
        for l in self.loci:
            idx = getattr(self, f"idx_{l}")
            mean_emb = hla_features[:, idx] @ self.embeddings[idx]     # [B, d_embd]
            z.append(self.proj[l](mean_emb))                            # [B, rank]
        out = self.out(torch.cat(z, dim=-1))
        return dict(zip(self.domain_names, torch.split(out, self.vocab_sizes, dim=-1)))


class FactorizedHLAHead(nn.Module):
    """
    HLA head factorised as embedding × unembedding: contrib = U · (E · x).

    E: d_hla → d_model (no bias) puts the HLA genotype into the trunk's residual-stream space.
    U maps that space to the outputs:
      unembed given ("tied")  U is the trunk's own output matrix (its tied embedding rows for the
                              predicted domains, frozen). Since the trunk computes logits as
                              W · ln_f(h) with no bias, contrib = W · E·x is exactly adding E·x to
                              the trunk's final residual stream. Only E is trained.
      unembed None ("separate") U is a new learned Linear(d_model, outputs) with bias.
    """

    def __init__(self, d_hla: int, predicted_vocab_sizes: Dict[str, int], d_model: int,
                 unembed: Optional[torch.Tensor] = None):
        super().__init__()
        self.domain_names = list(predicted_vocab_sizes.keys())
        self.vocab_sizes = list(predicted_vocab_sizes.values())
        self.E = nn.Linear(d_hla, d_model, bias=False)
        if unembed is not None:
            assert unembed.shape == (sum(self.vocab_sizes), d_model), (tuple(unembed.shape), sum(self.vocab_sizes), d_model)
            self.register_buffer("U_tied", unembed.detach().clone(), persistent=False)
            self.U = None
        else:
            self.U = nn.Linear(d_model, sum(self.vocab_sizes))

    def forward(self, hla_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.E(hla_features)
        out = z @ self.U_tied.T if self.U is None else self.U(z)
        return dict(zip(self.domain_names, torch.split(out, self.vocab_sizes, dim=-1)))


class HLAHistoryBlock(nn.Module):
    """
    One correction block on the HLA stream z [B, T, d], reading the frozen trunk's states h [B, T, d]:
      z += Attn(q = LN(z_t), k/v = LN(h_j), age_j < age_t or j = t, real tokens only)
      z += MLP([LN(z_t), LN(h_t)])
    Both output projections start at zero, so a new block is the identity. Like the trunk's
    causal(mask_ties=True), same-day earlier tokens are not visible: otherwise the block can read
    codes recorded on the day of the target.
    """

    def __init__(self, d: int, n_head: int = 4, mlp_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln_q, self.ln_kv, self.ln_z, self.ln_h = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(2 * d, mlp_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_mult * d, d))
        for lin in (self.attn.out_proj, self.mlp[-1]):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def forward(self, z: torch.Tensor, h: torch.Tensor, ages: torch.Tensor) -> torch.Tensor:
        T = z.shape[1]
        allowed = (ages[:, None, :] < ages[:, :, None]) & (ages >= 0)[:, None, :]  # [B, T(q), T(k)]
        allowed = allowed | torch.eye(T, dtype=torch.bool, device=z.device)[None]   # never an empty row
        mask = ~allowed.repeat_interleave(self.attn.num_heads, dim=0)               # True = blocked
        a, _ = self.attn(self.ln_q(z), self.ln_kv(h), self.ln_kv(h), attn_mask=mask, need_weights=False)
        z = z + a
        return z + self.mlp(torch.cat([self.ln_z(z), self.ln_h(h)], dim=-1))


class HLASelfAttentionBlock(nn.Module):
    """
    Pre-norm transformer block on a single stream r [B, T, d]:
      r += SelfAttn(LN(r)), position t attends to j with age_j < age_t or j = t (real tokens only,
           as the trunk's causal(mask_ties=True))
      r += MLP(LN(r))
    Both output projections start at zero, so a new block is the identity.
    """

    def __init__(self, d: int, n_head: int = 4, mlp_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln_1, self.ln_2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_mult * d, d))
        for lin in (self.attn.out_proj, self.mlp[-1]):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def forward(self, r: torch.Tensor, ages: torch.Tensor) -> torch.Tensor:
        T = r.shape[1]
        allowed = (ages[:, None, :] < ages[:, :, None]) & (ages >= 0)[:, None, :]  # [B, T(q), T(k)]
        allowed = allowed | torch.eye(T, dtype=torch.bool, device=r.device)[None]
        mask = ~allowed.repeat_interleave(self.attn.num_heads, dim=0)
        x = self.ln_1(r)
        r = r + self.attn(x, x, x, attn_mask=mask, need_weights=False)[0]
        return r + self.mlp(self.ln_2(r))


class HLAAlleleAttention(nn.Module):
    """
    HLA head with the alleles as tokens (block_type "allele"):

      a_k      = E[:, allele_k]                          one token per carried allele (Σ_k a_k = E·x)
      a       += SelfAttn(LN(a)) over the subject's alleles;  a += MLP(LN(a))      (allele × allele)
      z_t      = Σ_k a_k + CrossAttn(q = LN(h_t), k/v = LN(a))                     (history × HLA)
      z_t     += MLP(LN(z_t))

    All output projections start at zero, so a fresh module gives z_t = E·x (model 2). Every path
    into z goes through the allele tokens: the trunk states only choose which alleles to read
    (attention weights sum to 1), so without HLA nothing can correct the trunk.
    """

    def __init__(self, E: nn.Linear, d: int, n_head: int = 4, mlp_mult: int = 4, dropout: float = 0.1,
                 max_alleles: int = 24):
        super().__init__()
        self.E, self.max_alleles = E, max_alleles
        self.ln_s1, self.ln_s2, self.ln_q, self.ln_kv, self.ln_z = (nn.LayerNorm(d) for _ in range(5))
        self.self_attn = nn.MultiheadAttention(d, n_head, dropout=dropout, batch_first=True)
        self.self_mlp = nn.Sequential(nn.Linear(d, mlp_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_mult * d, d))
        self.cross_attn = nn.MultiheadAttention(d, n_head, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_mult * d, d))
        for lin in (self.self_attn.out_proj, self.self_mlp[-1], self.cross_attn.out_proj, self.mlp[-1]):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def allele_tokens(self, x: torch.Tensor):
        """x [B, n_alleles] (0/1 multi-hot) -> tokens [B, K, d], valid [B, K]."""
        K = min(self.max_alleles, x.shape[1])
        w, idx = torch.topk(x, K, dim=1)                     # carried alleles first
        assert bool((x > 0).sum(1).max() <= K), "more carried alleles than max_alleles"
        valid = w > 0
        tok = self.E.weight.T[idx] * w[..., None]            # [B, K, d]
        return tok, valid

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        a, valid = self.allele_tokens(x)
        has = valid.any(1)
        pad = ~valid
        pad[:, 0] = pad[:, 0] & has                          # subjects without HLA: unmask slot 0 (avoids NaN) ...
        s = self.ln_s1(a)
        a = a + self.self_attn(s, s, s, key_padding_mask=pad, need_weights=False)[0] * valid[..., None]
        a = a + self.self_mlp(self.ln_s2(a)) * valid[..., None]
        kv = self.ln_kv(a)
        c = self.cross_attn(self.ln_q(h), kv, kv, key_padding_mask=pad, need_weights=False)[0]
        z = a.sum(1)[:, None, :] + c * has[:, None, None]    # ... and drop its output
        return z + self.mlp(self.ln_z(z)) * has[:, None, None]


class HLAResidualStream(nn.Module):
    """
    Late fusion + residual transformer corrections (stage 2 of the factorised head):

        z_t     = E·x + Σ blocks(z, h)_t      (E·x alone when the blocks are at their zero init)
        contrib = U'·z_t + b                   [B, T, outputs]

    E and U' are those of a stage-1 FactorizedHLAHead("separate"), so a fresh model reproduces it
    exactly; the blocks let the HLA contribution depend on the subject's history (trunk states).

    block_type "cross": HLAHistoryBlock — the HLA stream z (queries) reads the trunk states h.
    block_type "self":  one stream r = h + E·x, HLASelfAttentionBlock(s) on r, and
                        stream() returns r − h (= E·x at init), read by U'.
    block_type "allele": HLAAlleleAttention — self-attention between the subject's allele tokens,
                        then cross-attention from the trunk states to them (n_blocks must be 1).
    """

    def __init__(self, d_hla: int, predicted_vocab_sizes: Dict[str, int], d_model: int,
                 n_blocks: int = 1, n_head: int = 4, mlp_mult: int = 4, dropout: float = 0.1,
                 block_type: str = "cross"):
        super().__init__()
        assert block_type in ("cross", "self", "allele"), block_type
        self.block_type = block_type
        self.domain_names = list(predicted_vocab_sizes.keys())
        self.vocab_sizes = list(predicted_vocab_sizes.values())
        self.E = nn.Linear(d_hla, d_model, bias=False)
        self.U = nn.Linear(d_model, sum(self.vocab_sizes))
        if block_type == "allele":
            assert n_blocks == 1, "allele: one self+cross block"
            self.blocks = nn.ModuleList([HLAAlleleAttention(self.E, d_model, n_head, mlp_mult, dropout)])
        else:
            block = HLAHistoryBlock if block_type == "cross" else HLASelfAttentionBlock
            self.blocks = nn.ModuleList([block(d_model, n_head, mlp_mult, dropout) for _ in range(n_blocks)])

    def stream(self, hla_features: torch.Tensor, h: torch.Tensor, ages: torch.Tensor) -> torch.Tensor:
        if self.block_type == "allele":
            return self.blocks[0](hla_features, h)
        e = self.E(hla_features)[:, None, :]
        if self.block_type == "self":
            r = h + e
            for blk in self.blocks:
                r = blk(r, ages)
            return r - h
        z = e.expand(-1, h.shape[1], -1)
        for blk in self.blocks:
            z = blk(z, h, ages)
        return z

    def forward(self, hla_features: torch.Tensor, h: torch.Tensor, ages: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.U(self.stream(hla_features, h, ages))
        return dict(zip(self.domain_names, torch.split(out, self.vocab_sizes, dim=-1)))


def trunk_unembedding(trunk, domain_names: List[str]) -> torch.Tensor:
    """The trunk's output (tied embedding) matrix rows for domain_names, concatenated in that order."""
    # on CPU, like a freshly built head; the module is moved to the right device with the rest
    return torch.cat([trunk.embed._get_domain_weight(d).detach().cpu() for d in domain_names], dim=0)


def build_hla_head(encoder, predicted_vocab_sizes: Dict[str, int],
                   hidden_dims: Optional[List[int]] = None, rank: Optional[int] = None,
                   factorized: Optional[str] = None, trunk=None) -> nn.Module:
    """The head matching an encoder and options: FactorizedHLAHead if `factorized` is "tied" or
    "separate" (needs `trunk` for its width and, if tied, its output matrix), LocusLowRankHLAHead for
    LocusESM2HLAEncoder, else HLAHead."""
    if factorized not in (None, "none"):
        if factorized not in ("tied", "separate") or trunk is None or hidden_dims or rank:
            raise ValueError("factorized head: 'tied' or 'separate', needs the trunk, no hidden_dims/rank")
        d_model = trunk.embed._get_domain_weight(next(iter(predicted_vocab_sizes))).shape[1]
        unembed = trunk_unembedding(trunk, list(predicted_vocab_sizes)) if factorized == "tied" else None
        return FactorizedHLAHead(encoder.n_features, predicted_vocab_sizes, d_model, unembed)
    if isinstance(encoder, LocusESM2HLAEncoder):
        if not rank or hidden_dims:
            raise ValueError("esm2_locus needs --head_rank and no --head_hidden_dims")
        return LocusLowRankHLAHead(encoder.embeddings, encoder.locus_groups, predicted_vocab_sizes, rank)
    return HLAHead(encoder.n_features, predicted_vocab_sizes, hidden_dims=hidden_dims, rank=rank)


# ═══════════════════════════════════════════════════════════════════════════════
#  HLAHead
# ═══════════════════════════════════════════════════════════════════════════════

class HLAHead(nn.Module):
    """
    Maps HLA features [B, d_hla] to per-domain logit contributions.

    Parameters
    ----------
    d_hla : int
        Input dimension (n_features from the chosen encoder).
    predicted_vocab_sizes : dict {domain_name: vocab_size}
        Must be ordered the same as trunk.predicted_domains.
    hidden_dims : list[int] or None
        Hidden layer sizes for an MLP; None → single linear layer (no hidden layers).
    rank : int or None
        Low-rank linear head: d_hla → rank (no bias, no activation) → outputs, i.e. a linear
        map whose weight matrix has rank <= `rank`. None → full-rank linear layer. Only for
        linear heads (incompatible with hidden_dims).
    """

    def __init__(
        self,
        d_hla: int,
        predicted_vocab_sizes: Dict[str, int],
        hidden_dims: Optional[List[int]] = None,
        rank: Optional[int] = None,
    ):
        super().__init__()
        if rank and hidden_dims:
            raise ValueError("HLAHead: `rank` (low-rank linear head) and `hidden_dims` (MLP) are exclusive")
        self.domain_names = list(predicted_vocab_sizes.keys())
        self.vocab_sizes = list(predicted_vocab_sizes.values())
        total_out = sum(self.vocab_sizes)

        if hidden_dims:
            layers: list[nn.Module] = []
            in_dim = d_hla
            for h in hidden_dims:
                layers += [nn.Linear(in_dim, h), nn.ReLU()]
                in_dim = h
            layers.append(nn.Linear(in_dim, total_out))
            self.net: nn.Module = nn.Sequential(*layers)
        elif rank:
            self.net = nn.Sequential(nn.Linear(d_hla, rank, bias=False), nn.Linear(rank, total_out))
        else:
            self.net = nn.Linear(d_hla, total_out)

    def forward(self, hla_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        hla_features : [B, d_hla]

        Returns
        -------
        {domain_name: Tensor [B, vocab_size]}
        """
        out = self.net(hla_features)   # [B, total_out]
        splits = torch.split(out, self.vocab_sizes, dim=-1)
        return dict(zip(self.domain_names, splits))


# ═══════════════════════════════════════════════════════════════════════════════
#  DelphiLateFusion
# ═══════════════════════════════════════════════════════════════════════════════

class DelphiLateFusion(nn.Module):
    """
    Frozen Delphi trunk + trainable HLA late-fusion head.

    Forward pass:
      1. trunk(batch.delphi_batch) → trunk_logits {domain: [B, T, vocab_size]}   (no_grad)
      2. head(batch.hla_features)  → hla_contrib  {domain: [B, vocab_size]}
      3. merged[domain] = trunk_logits[domain] + hla_contrib[domain].unsqueeze(1)

    The contribution is identical for all sequence positions (HLA genotype is static).
    Only head parameters receive gradients.

    Exposes the same interface as Delphi for loss utilities and Trainer compatibility:
    domain_to_int, domain_offsets, predicted_domains, config,
    cross_entropy_loss, time_to_event_loss.
    """

    def __init__(self, trunk, head: HLAHead):
        super().__init__()
        self.trunk = trunk
        for p in self.trunk.parameters():
            p.requires_grad = False
        self.head = head

    # ── Delegation to trunk ───────────────────────────────────────────────────

    @property
    def config(self):
        return self.trunk.config

    @property
    def domain_to_int(self):
        return self.trunk.domain_to_int

    @property
    def domain_offsets(self):
        return self.trunk.domain_offsets

    @property
    def predicted_domains(self):
        return self.trunk.predicted_domains

    @property
    def device(self):
        return next(self.head.parameters()).device

    def cross_entropy_loss(self, logits, targets, agg=None, token_weights=None):
        return self.trunk.cross_entropy_loss(
            logits, targets, agg=agg, token_weights=token_weights
        )

    def time_to_event_loss(self, logits, time_to_next, t_min=1.0, agg=None, pos_weights=None):
        return self.trunk.time_to_event_loss(
            logits, time_to_next, t_min=t_min, agg=agg
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        batch,
        return_attention: bool = False,
        return_embeddings: bool = False,
    ):
        """
        Parameters
        ----------
        batch : LateFusionBatch

        Returns
        -------
        (logits_dict, attention) — same interface as Delphi.forward.
        """
        with torch.no_grad():
            trunk_logits, attention = self.trunk(
                batch.delphi_batch, return_attention=return_attention
            )

        hla_contrib = self.head(batch.hla_features)   # {domain: [B, vocab_size]}

        logits: Dict[str, torch.Tensor] = {}
        for domain, tlogits in trunk_logits.items():
            if domain in hla_contrib:
                # [B, vocab] → [B, 1, vocab], broadcast over T
                logits[domain] = tlogits + hla_contrib[domain].unsqueeze(1)
            else:
                logits[domain] = tlogits

        if return_embeddings:
            return logits, attention, None
        return logits, attention


class DelphiLateFusionResidual(DelphiLateFusion):
    """DelphiLateFusion whose head (HLAResidualStream) also reads the frozen trunk's final states,
    so the HLA contribution varies along the sequence: logits_t = trunk_t + head(hla, h, ages)_t."""

    def forward(self, batch, return_attention: bool = False, return_embeddings: bool = False):
        with torch.no_grad():
            trunk_logits, attention, h = self.trunk(batch.delphi_batch, return_attention=return_attention,
                                                    return_embeddings=True)
        contrib = self.head(batch.hla_features, h, batch.delphi_batch.ages)
        logits = {d: (t + contrib[d]) if d in contrib else t for d, t in trunk_logits.items()}
        if return_embeddings:
            return logits, attention, h
        return logits, attention
