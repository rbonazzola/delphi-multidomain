"""
Synthetic tests for DelphiMultiStream.
No CSV data is loaded — only tokenizer.yaml (for vocab sizes) and a hand-built
DelphiBatch.  Runs on CPU so it can execute without a GPU.
"""
import pytest
import torch
from data.dataset import DelphiBatch
from delphi.model import DomainConfig
from delphi.multi_stream_model import (
    DelphiMultiStream,
    DelphiMultiStreamConfig,
    parse_multi_stream_scheme,
    MultiStreamScheme,
    EncoderSpec,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

ARCH_STR = (
    "MultiStream("
    "[hla_alleles,sex]:(h4d32l2),"
    "[diseases,lifestyle,sex]:(h4d32l3)"
    "):(h4d32l2)"
)
ARCH_STR_3STREAM = (
    "MultiStream("
    "[hla_alleles,sex]:(h4d32l2),"
    "[diseases,sex]:(h4d32l2),"
    "[lifestyle,sex]:(h4d32l1)"
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
    return DelphiMultiStreamConfig(
        domains=DOMAINS,
        block_size=32,
        seed=0,
    )


@pytest.fixture(scope="module")
def model(config):
    return DelphiMultiStream.from_scheme_string(ARCH_STR, ATTN_SCHEME, config)


@pytest.fixture(scope="module")
def model_3stream(config):
    return DelphiMultiStream.from_scheme_string(ARCH_STR_3STREAM, ATTN_SCHEME, config)


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
    s = parse_multi_stream_scheme(ARCH_STR)
    assert len(s.encoders) == 2
    assert s.encoders[0].domains == ["hla_alleles", "sex"]
    assert s.encoders[0].n_layer == 2
    assert s.encoders[1].domains == ["diseases", "lifestyle", "sex"]
    assert s.encoders[1].n_layer == 3
    assert s.trunk_n_layer == 2


def test_parser_supports_more_than_two_encoders():
    s = parse_multi_stream_scheme(ARCH_STR_3STREAM)
    assert len(s.encoders) == 3
    assert [e.domains for e in s.encoders] == [
        ["hla_alleles", "sex"], ["diseases", "sex"], ["lifestyle", "sex"],
    ]
    assert [e.n_layer for e in s.encoders] == [2, 2, 1]


def test_parser_rejects_wrong_n_encoders():
    with pytest.raises(ValueError, match="at least 2 encoder"):
        parse_multi_stream_scheme(
            "MultiStream([a]:(h4d32l1)):(h4d32l2)"
        )


def test_parser_rejects_mismatched_n_embd():
    with pytest.raises(ValueError, match="n_embd"):
        DelphiMultiStream.from_scheme_string(
            "MultiStream([hla_alleles,sex]:(h4d32l1),[diseases,lifestyle,sex]:(h4d64l1)):(h4d32l2)",
            ATTN_SCHEME,
            DelphiMultiStreamConfig(domains=DOMAINS, block_size=32),
        )


# ── Model construction ────────────────────────────────────────────────────────

def test_model_builds(model):
    assert model.n_streams == 2
    assert len(model.encoders[0]) == 2
    assert len(model.encoders[1]) == 3
    assert len(model.trunk)       == 2


def test_model_builds_3stream(model_3stream):
    assert model_3stream.n_streams == 3
    assert [len(enc) for enc in model_3stream.encoders] == [2, 2, 1]


def test_at_birth_resolved(model):
    assert "at_birth" not in model.attention_scheme
    assert "sex" in model.attention_scheme


def test_group_buffers(model):
    d2i = model.domain_to_int
    assert d2i["hla_alleles"] in model.group_ids(0).tolist()
    assert d2i["sex"]         in model.group_ids(0).tolist()
    assert d2i["diseases"]    in model.group_ids(1).tolist()
    assert d2i["sex"]         in model.group_ids(1).tolist()


# ── Forward pass ──────────────────────────────────────────────────────────────

def test_forward_shapes(model, config):
    batch = _make_batch(model, B=4, T=32)
    model.eval()
    with torch.no_grad():
        logits, att = model(batch)

    assert att is None
    assert "diseases" in logits
    assert logits["diseases"].shape == (4, 32, VOCAB["diseases"])


def test_forward_shapes_3stream(model_3stream):
    batch = _make_batch(model_3stream, B=4, T=32)
    model_3stream.eval()
    with torch.no_grad():
        logits, att = model_3stream(batch)
    assert logits["diseases"].shape == (4, 32, VOCAB["diseases"])


def test_no_nan_in_logits(model):
    batch = _make_batch(model, B=4, T=32)
    model.eval()
    with torch.no_grad():
        logits, _ = model(batch)
    for name, lg in logits.items():
        assert not torch.isnan(lg).any(), f"NaN in logits[{name!r}]"


def test_no_nan_in_logits_3stream(model_3stream):
    batch = _make_batch(model_3stream, B=4, T=32)
    model_3stream.eval()
    with torch.no_grad():
        logits, _ = model_3stream(batch)
    for name, lg in logits.items():
        assert not torch.isnan(lg).any(), f"NaN in logits[{name!r}] (3-stream)"


def test_no_nan_in_masks(model):
    batch = _make_batch(model, B=2, T=32)
    masks, mask_trunk = model._build_masks(batch)
    assert len(masks) == model.n_streams
    for i, m in enumerate(masks):
        assert not torch.isnan(m.float()).any(), f"NaN in mask_{i}"
    assert not torch.isnan(mask_trunk.float()).any(), "NaN in mask_trunk"


def test_return_attention(model):
    batch = _make_batch(model, B=2, T=32)
    model.eval()
    with torch.no_grad():
        logits, att = model(batch, return_attention=True)
    # one attention tensor per trunk layer
    assert att is not None
    assert att.shape[0] == len(model.trunk)


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


def test_backward_pass_3stream(model_3stream):
    """Gradients must flow through all stages with 3 encoder groups."""
    batch  = _make_batch(model_3stream, B=2, T=32)
    model_3stream.train()
    logits, _ = model_3stream(batch)

    lg = logits["diseases"]
    B, T, V = lg.shape
    targets = torch.randint(0, V, (B, T))
    loss = model_3stream.cross_entropy_loss(lg, targets)
    loss.backward()

    for name, p in model_3stream.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
