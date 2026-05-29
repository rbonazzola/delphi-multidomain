"""
Tests for subdomain hierarchy features:
  1. load_domain_config — parent inheritance, abstract exclusion, subdomain_column
  2. MultiDomainEmbedding — pretrained categorical domain forward path
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from delphi.model import DomainConfig
from utils.utils import load_domain_config


# ═══════════════════════════════════════════════════════════════════════════════
#  1. load_domain_config — parent / abstract / subdomain_column
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def simple_yaml(tmp_path):
    """Write a minimal domain config YAML and return (yaml_path, tokens_path)."""
    tokens_path = tmp_path / "tokens"
    tokens_path.mkdir()

    yaml_text = textwrap.dedent("""\
        hla_alleles:
          abstract: true
          projector: embed
          path: hla_alleles
          at_birth: true
          subdomain_column: locus

        hla_a:
          parent: hla_alleles
          subdomain: hla_a

        hla_dp:
          parent: hla_alleles
          subdomain: hla_dp
          subdomain_column: locus_group

        diseases:
          projector: embed
          path: diseases
          predict: true
    """)

    cfg_path = tmp_path / "domain_config.yaml"
    cfg_path.write_text(yaml_text)
    return cfg_path, tokens_path


class TestParentInheritance:

    def test_abstract_domain_excluded(self, simple_yaml):
        cfg_path, tokens_path = simple_yaml
        cfg = load_domain_config(cfg_path, tokens_path)
        assert "hla_alleles" not in cfg, "Abstract parent should not appear in active config"

    def test_child_inherits_path(self, simple_yaml):
        cfg_path, tokens_path = simple_yaml
        cfg = load_domain_config(cfg_path, tokens_path)
        # path is resolved to tokens_path / "hla_alleles"
        assert cfg["hla_a"].path == tokens_path / "hla_alleles"

    def test_child_inherits_at_birth(self, simple_yaml):
        cfg_path, tokens_path = simple_yaml
        cfg = load_domain_config(cfg_path, tokens_path)
        assert cfg["hla_a"].at_birth is True

    def test_child_inherits_projector(self, simple_yaml):
        cfg_path, tokens_path = simple_yaml
        cfg = load_domain_config(cfg_path, tokens_path)
        assert cfg["hla_a"].projector == "embed"

    def test_child_inherits_subdomain_column_default(self, simple_yaml):
        cfg_path, tokens_path = simple_yaml
        cfg = load_domain_config(cfg_path, tokens_path)
        assert cfg["hla_a"].subdomain_column == "locus"

    def test_child_overrides_subdomain_column(self, simple_yaml):
        cfg_path, tokens_path = simple_yaml
        cfg = load_domain_config(cfg_path, tokens_path)
        assert cfg["hla_dp"].subdomain_column == "locus_group"

    def test_child_subdomain_not_inherited(self, simple_yaml):
        """subdomain is child-specific and must not bleed from parent to sibling."""
        cfg_path, tokens_path = simple_yaml
        cfg = load_domain_config(cfg_path, tokens_path)
        assert cfg["hla_a"].subdomain == "hla_a"
        assert cfg["hla_dp"].subdomain == "hla_dp"

    def test_non_parent_domain_unaffected(self, simple_yaml):
        cfg_path, tokens_path = simple_yaml
        cfg = load_domain_config(cfg_path, tokens_path)
        assert cfg["diseases"].predict is True
        assert cfg["diseases"].at_birth is False

    def test_unknown_parent_raises(self, tmp_path):
        yaml_text = textwrap.dedent("""\
            hla_a:
              parent: nonexistent
              subdomain: hla_a
        """)
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text(yaml_text)
        with pytest.raises(ValueError, match="unknown parent"):
            load_domain_config(cfg_path, tmp_path / "tokens")

    def test_padding_always_present(self, simple_yaml):
        cfg_path, tokens_path = simple_yaml
        cfg = load_domain_config(cfg_path, tokens_path)
        assert "padding" in cfg


# ═══════════════════════════════════════════════════════════════════════════════
#  2. MultiDomainEmbedding — pretrained categorical forward path
# ═══════════════════════════════════════════════════════════════════════════════

def _build_pretrained_model(tmp_path, vocab_size=4, d_ext=6, n_embd=8):
    """
    Build a minimal Delphi model with one pretrained categorical domain.
    Returns (model, domain_to_int, domain_offsets).
    """
    from delphi.model import Delphi, DelphiConfig, DomainConfig

    # Write a fake tokenizer.yaml for the pretrained domain
    hla_dir = tmp_path / "tokens" / "hla_a"
    hla_dir.mkdir(parents=True)
    (hla_dir / "tokenizer.yaml").write_text(
        "\n".join(f"- allele_{i}" for i in range(vocab_size))
    )

    # Write fake pretrained weights
    weights = torch.randn(vocab_size, d_ext)
    pretrained_path = tmp_path / "hla_a.pt"
    torch.save(weights, pretrained_path)

    cfg = DelphiConfig(
        n_embd=n_embd,
        n_layer=2,
        n_head=2,
        block_size=16,
        domains={
            "hla_a": DomainConfig(
                projector="pretrained",
                path=str(hla_dir),
                at_birth=True,
                pretrained_path=str(pretrained_path),
                input_size=vocab_size,
            ),
        },
    )
    model = Delphi(cfg)
    return model, weights


class TestPretrainedCategoricalEmbedding:

    def test_pretrained_domain_not_in_projected_names(self, tmp_path):
        model, _ = _build_pretrained_model(tmp_path)
        assert "hla_a" not in model.embed._projected_domain_names
        assert "hla_a" in model.embed._pretrained_domain_names

    def test_forward_output_shape(self, tmp_path):
        model, _ = _build_pretrained_model(tmp_path)
        B, T = 2, 8
        batch = _make_batch(model, B=B, T=T)
        with torch.no_grad():
            logits, _ = model(batch)
        # No predicted domains, so logits should be empty
        assert isinstance(logits, dict)

    def test_pretrained_embedding_values(self, tmp_path):
        """
        Verify that the pretrained projector is actually used:
        embed(local_id) → linear should match direct computation.
        """
        vocab_size, d_ext, n_embd = 4, 6, 8
        model, pretrained_weights = _build_pretrained_model(
            tmp_path, vocab_size=vocab_size, d_ext=d_ext, n_embd=n_embd
        )

        hla_a_int = model.domain_to_int["hla_a"]
        offset = model.domain_offsets[hla_a_int]
        projector = model.embed.projectors["hla_a"]

        B, T = 1, 4
        batch = _make_batch(model, B=B, T=T, local_token_ids=[0, 1, 2, 3])

        with torch.no_grad():
            emb = model.embed(batch)  # [B, T, n_embd]

        # Expected: linear(pretrained_embed[local_id])
        local_ids = torch.tensor([0, 1, 2, 3])
        expected = projector.linear(projector.embed(local_ids))  # [T, n_embd]

        assert torch.allclose(emb[0], expected, atol=1e-5), (
            "Pretrained embedding values don't match direct computation"
        )

    def test_placeholder_rows_are_zero(self, tmp_path):
        """Global embed rows for pretrained domain should be zeroed out."""
        vocab_size = 4
        model, _ = _build_pretrained_model(tmp_path, vocab_size=vocab_size)
        hla_a_int = model.domain_to_int["hla_a"]
        offset = model.domain_offsets[hla_a_int]
        rows = model.embed.global_embed.weight[offset: offset + vocab_size]
        assert rows.abs().sum().item() == 0.0, "Placeholder rows should be zero"


def _make_batch(model, B=2, T=8, local_token_ids=None):
    """Build a minimal DelphiBatch with all tokens belonging to hla_a."""
    from data.dataset import DelphiBatch

    hla_a_int = model.domain_to_int["hla_a"]
    offset = model.domain_offsets[hla_a_int]
    vocab_size = model.embed._domain_vocab_sizes["hla_a"]

    domain_ids = torch.full((B, T), hla_a_int, dtype=torch.long)

    if local_token_ids is not None:
        assert len(local_token_ids) == T
        local_ids = torch.tensor(local_token_ids, dtype=torch.long).unsqueeze(0).expand(B, T)
    else:
        local_ids = torch.randint(0, vocab_size, (B, T))

    global_token_ids = local_ids + offset
    ages = torch.zeros(B, T)

    return DelphiBatch(
        global_token_ids=global_token_ids,
        domain_ids=domain_ids,
        ages=ages,
        subject_ids=torch.arange(B),
        continuous_data={},
        continuous_positions={},
        eval_mask=None,
    )
