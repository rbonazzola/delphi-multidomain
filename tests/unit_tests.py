import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataclasses import dataclass, field
from typing import Optional
from delphi.multimodal import Modality, module_name

import logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")

from delphi.model.components import (
    DomainConfig,
    DelphiConfig,
    DomainEmbedding,
    DelphiEmbedding,
    CrossEntropyHead,
    CompetingExpHead,
    causal_attention_mask,
    target_mask,
    ties_adjusted_delta_t,
)


def test_domain_embedding_linear():
    cfg = DomainConfig(projector="linear", input_size=10)
    emb = DomainEmbedding(cfg, n_embed=16)
    x = torch.randn(2, 5, 10)              # (B, T, input_size)
    y = emb(x)
    assert y.shape == (2, 5, 16)


def test_domain_embedding_mlp():
    cfg = DomainConfig(projector="mlp", input_size=8, n_layers=2, n_hidden=12)
    emb = DomainEmbedding(cfg, n_embed=20)
    x = torch.randn(4, 3, 8)
    y = emb(x)
    assert y.shape == (4, 3, 20)


def test_domain_embedding_embed():
    cfg = DomainConfig(projector="embed", input_size=50)
    emb = DomainEmbedding(cfg, n_embed=32)
    x = torch.randint(0, 50, (2, 7))       # (B, T)
    y = emb(x)
    assert y.shape == (2, 7, 32)


def test_domain_embedding_pretrained(tmp_path):
    # fake pretrained table
    weights = torch.randn(100, 64)
    path = tmp_path / "weights.pt"
    torch.save(weights, path)

    cfg = DomainConfig(projector="pretrained", input_size=100, pretrained_path=str(path), freeze=True)
    emb = DomainEmbedding(cfg, n_embed=32)
    x = torch.randint(0, 100, (3, 4))
    y = emb(x)
    assert y.shape == (3, 4, 32)


def test_crossentropy_head():
    head = CrossEntropyHead(config=None)
    logits = torch.randn(2, 5, 10)     # (B, T, vocab)
    targets = torch.randint(0, 10, (2, 5))
    loss = head(logits, targets)
    assert loss.shape == (2, 5)


def test_competingexp_head():
    head = CompetingExpHead(n_input=8, zero_inflate=True, pi_head="linear")
    logits = torch.randn(3, 5, 8)
    delta_t = torch.randint(0, 10, (3, 5)).float()
    loss = head(logits, delta_t)
    assert loss.shape == (3, 5)

def test_attention_mask():
    B, T = 2, 4
    pad = torch.ones(B, T)
    mask = causal_attention_mask(pad)
    assert mask.shape == (B, 1, T, T)


def test_target_mask():
    x1 = torch.tensor([[1, 2, 0]])
    mask = target_mask(x1, ignore_tokens=[0])
    assert mask.dtype == torch.bool


def test_ties_adjusted_delta_t():
    t0 = torch.tensor([[1., 2., 3.]])
    t1 = torch.tensor([[2., 3., 4.]])
    attn_mask = torch.ones(1, 1, 3, 3)
    delta = ties_adjusted_delta_t(t0, t1, attn_mask, mask_ties=False)
    assert delta.shape == (1, 3)
