"""
Synthetic tests for DelphiCrossAttention.
No CSV data is loaded — only tokenizer.yaml (for vocab sizes) and a hand-built
DelphiBatch.  Runs on CPU so it can execute without a GPU.
"""
import pytest
import torch
from data.dataset import DelphiBatch
from delphi.model import DelphiConfig, DomainConfig
from delphi.cross_attn_model import (
    DelphiCrossAttention,
    parse_cross_attention_scheme,
    CrossAttentionScheme,
    EncoderSpec,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

ARCH_STR = (
    "CrossAttention("
    "[hla_alleles,sex]:(h4d32l2),"
    "[diseases,lifestyle,sex]:(h4d32l3),"
    "xattn:(h4d32)"
    "):(h4d32l2)"
)
ATTN_SCHEME = "[hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)"

TOKEN_PATH = "data/transforms/tokens"

DOMAINS = {
    "padding":     DomainConfig(path=None),
    "diseases":    DomainConfig(path=f"{TOKEN_PATH}/diseases",    predict=True),
    "hla_alleles": DomainConfig(path=f"{TOKEN_PATH}/hla_alleles"),
    "lifestyle":   DomainConfig(path=f"{TOKEN_PATH}/lifestyle"),
    "sex":         DomainConfig(path=f"{TOKEN_PATH}/sex",         at_birth=True),
}

# Vocab sizes (from tokenizer.yaml — needed to build synthetic global IDs)
# diseases=1256, hla_alleles=359, lifestyle=9, sex=2, padding=2
VOCAB = {"padding": 2, "diseases": 1256, "hla_alleles": 359, "lifestyle": 9, "sex": 2}


@pytest.fixture(scope="module")
def config():
    return DelphiConfig(
        n_embd=32, n_layer=2, n_head=4,
        domains=DOMAINS,
        block_size=32,
        seed=0,
    )


@pytest.fixture(scope="module")
def model(config):
    return DelphiCrossAttention.from_scheme_string(ARCH_STR, ATTN_SCHEME, config)


def _make_batch(model, B=4, T=32, device="cpu"):
    """
    Build a synthetic DelphiBatch that mimics collate output.
    Token layout per subject (sorted by age):
      - 2 sex tokens       (age 0,   at_birth)
      - 4 hla_alleles      (age 0)
      - 6 disease tokens   (ages 3650–18250)
      - 3 lifestyle tokens (ages 5000–15000)
      - rest: padding
    """
    domain_to_int = model.domain_to_int
    domain_offsets = model.domain_offsets

    pad_id   = domain_to_int["padding"]
    dis_id   = domain_to_int["diseases"]
    hla_id   = domain_to_int["hla_alleles"]
    life_id  = domain_to_int["lifestyle"]
    sex_id   = domain_to_int["sex"]

    domain_ids      = torch.full((B, T), pad_id,  dtype=torch.long)
    global_token_ids = torch.zeros(B, T,           dtype=torch.long)
    ages             = torch.full((B, T), -10000., dtype=torch.float)

    for b in range(B):
        pos = 0
        def _fill(d_int, local_ids, age_vals):
            nonlocal pos
            offset = domain_offsets[d_int]
            for lid, age in zip(local_ids, age_vals):
                domain_ids[b, pos]       = d_int
                global_token_ids[b, pos] = offset + lid
                ages[b, pos]             = float(age)
                pos += 1

        _fill(sex_id,   [0, 1],                     [0, 0])
        _fill(hla_id,   [0, 1, 2, 3],               [0, 0, 0, 0])
        _fill(dis_id,   [0, 1, 2, 3, 4, 5],         [3650, 5000, 7300, 10000, 14600, 18250])
        _fill(life_id,  [0, 1, 2],                   [5000, 10000, 15000])
        # rest stays padding

    batch = DelphiBatch(
        global_token_ids=global_token_ids,
        domain_ids=domain_ids,
        ages=ages,
        subject_ids=torch.arange(B),
        continuous_data={},
        continuous_positions={},
        eval_mask=None,
    )
    return batch.to(device)


# ── Parser tests ──────────────────────────────────────────────────────────────

def test_parser_basic():
    s = parse_cross_attention_scheme(ARCH_STR)
    assert s.encoder_A.domains == ["hla_alleles", "sex"]
    assert s.encoder_A.n_layer == 2
    assert s.encoder_B.domains == ["diseases", "lifestyle", "sex"]
    assert s.encoder_B.n_layer == 3
    assert s.xattn_n_head == 4
    assert s.trunk_n_layer == 2


def test_parser_rejects_missing_xattn():
    with pytest.raises(ValueError, match="xattn"):
        parse_cross_attention_scheme(
            "CrossAttention([a]:(h4d32l1),[b]:(h4d32l1)):(h4d32l2)"
        )


def test_parser_rejects_wrong_n_encoders():
    with pytest.raises(ValueError, match="2 encoder"):
        parse_cross_attention_scheme(
            "CrossAttention([a]:(h4d32l1),xattn:(h4d32)):(h4d32l2)"
        )


def test_parser_rejects_mismatched_n_embd():
    with pytest.raises(ValueError, match="n_embd"):
        DelphiCrossAttention.from_scheme_string(
            "CrossAttention([hla_alleles,sex]:(h4d32l1),[diseases,lifestyle,sex]:(h4d64l1),xattn:(h4d32)):(h4d32l2)",
            ATTN_SCHEME,
            DelphiConfig(n_embd=32, domains=DOMAINS, block_size=32),
        )


# ── Model construction ────────────────────────────────────────────────────────

def test_model_builds(model):
    assert len(model.encoder_A) == 2
    assert len(model.encoder_B) == 3
    assert len(model.trunk)     == 2


def test_at_birth_resolved(model):
    assert "at_birth" not in model.attention_scheme
    assert "sex" in model.attention_scheme


def test_group_buffers(model):
    d2i = model.domain_to_int
    assert d2i["hla_alleles"] in model._group_A_ids.tolist()
    assert d2i["sex"]         in model._group_A_ids.tolist()
    assert d2i["diseases"]    in model._group_B_ids.tolist()
    assert d2i["sex"]         in model._group_B_ids.tolist()


# ── Forward pass ──────────────────────────────────────────────────────────────

def test_forward_shapes(model, config):
    batch = _make_batch(model, B=4, T=32)
    model.eval()
    with torch.no_grad():
        logits, att = model(batch)

    assert att is None
    assert "diseases" in logits
    assert logits["diseases"].shape == (4, 32, VOCAB["diseases"])


def test_no_nan_in_logits(model):
    batch = _make_batch(model, B=4, T=32)
    model.eval()
    with torch.no_grad():
        logits, _ = model(batch)
    for name, lg in logits.items():
        assert not torch.isnan(lg).any(), f"NaN in logits[{name!r}]"


def test_no_nan_in_masks(model):
    batch = _make_batch(model, B=2, T=32)
    mask_A, mask_B, mask_x, mask_trunk = model._build_masks(batch)
    for name, m in [("A", mask_A), ("B", mask_B), ("xattn", mask_x), ("trunk", mask_trunk)]:
        assert not torch.isnan(m.float()).any(), f"NaN in mask_{name}"


def test_xattn_mask_is_cross_group(model):
    """Cross-attention mask must be zero for within-group pairs."""
    batch = _make_batch(model, B=1, T=32)
    _, _, mask_x, _ = model._build_masks(batch)
    mask_x = mask_x.squeeze()   # [T, T]

    d2i = model.domain_to_int
    domain_ids = batch.domain_ids[0]   # [T]

    is_A = torch.isin(domain_ids, model._group_A_ids)
    is_B = torch.isin(domain_ids, model._group_B_ids)
    is_A_only = is_A & ~is_B
    is_B_only = is_B & ~is_A

    # A-only tokens must not attend to other A-only tokens
    for i in is_A_only.nonzero(as_tuple=True)[0]:
        for j in is_A_only.nonzero(as_tuple=True)[0]:
            if i != j:
                assert mask_x[i, j].item() == 0, \
                    f"A-only token {i} should not attend to A-only token {j} in cross-attn"

    # B-only tokens must not attend to other B-only tokens
    for i in is_B_only.nonzero(as_tuple=True)[0]:
        for j in is_B_only.nonzero(as_tuple=True)[0]:
            if i != j:
                assert mask_x[i, j].item() == 0, \
                    f"B-only token {i} should not attend to B-only token {j} in cross-attn"


def test_return_attention(model):
    batch = _make_batch(model, B=2, T=32)
    model.eval()
    with torch.no_grad():
        logits, att = model(batch, return_attention=True)
    # 1 cross-attn block + 2 trunk layers = 3 attention tensors
    assert att is not None
    assert att.shape[0] == 1 + len(model.trunk)


def test_return_embeddings(model):
    batch = _make_batch(model, B=2, T=32)
    model.eval()
    with torch.no_grad():
        logits, att, h = model(batch, return_embeddings=True)
    assert h.shape == (2, 32, 32)   # [B, T, n_embd]


# ── Loss functions ────────────────────────────────────────────────────────────

def test_cross_entropy_loss(model):
    batch = _make_batch(model, B=4, T=32)
    model.eval()
    with torch.no_grad():
        logits, _ = model(batch)

    lg = logits["diseases"]        # [B, T, V]
    B, T, V = lg.shape
    targets = torch.randint(0, V, (B * T,))
    loss = model.cross_entropy_loss(lg, targets.view(B, T))
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_backward_pass(model, config):
    """Gradients must flow through all stages."""
    batch  = _make_batch(model, B=2, T=32)
    model.train()
    logits, _ = model(batch)

    lg = logits["diseases"]
    B, T, V = lg.shape
    targets = torch.randint(0, V, (B, T))
    loss = model.cross_entropy_loss(lg, targets)
    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
