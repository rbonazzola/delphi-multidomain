"""
Verify that the subdomain-hierarchy branch produces bit-identical forward-pass
outputs and CE losses for the BASE case (diseases, death, lifestyle, sex — no
HLA, no pretrained embeddings) compared to what the refactor branch would have
produced.

Strategy:
  1. Replicate the OLD load_domain_config logic (from refactor) for the base config.
  2. Build a model with a fixed seed using the old-style config.
  3. Build a second model identically using the NEW load_domain_config.
  4. Assert that the two configs are field-for-field identical.
  5. Assert that the two models produce bit-identical logits and loss on a shared
     fake batch.

Run from project root:
    pytest tests/test_base_vs_refactor.py -v
"""

import sys
from pathlib import Path
from copy import deepcopy

import torch
import yaml
import pytest

DELPHI_DIR = Path(__file__).resolve().parent.parent
if DELPHI_DIR not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

from delphi.model import Delphi, DelphiConfig, DomainConfig
from data.dataset import DelphiBatch
from utils.utils import load_domain_config

ROOT_TOKENS = DELPHI_DIR / "data" / "transforms" / "tokens"
BASE_DOMAINS = ["diseases", "death", "lifestyle", "sex"]
WITH_PCS_DOMAINS = ["diseases", "death", "lifestyle", "sex", "genetic_pcs"]
BASE_CONFIG_YAML = DELPHI_DIR / "config" / "domain_config_default.yaml"


# ── replica of the OLD load_domain_config (refactor branch) ──────────────────

def _normalize_domain_cfg_old(cfg):
    """Minimal version of the old _normalize_domain_cfg."""
    for dname, dcfg in cfg.items():
        if dcfg.dropout_rate == 0.0:
            dcfg.dropout_mode = None
    return cfg


def load_domain_config_old(cfg_path, tokens_path):
    """Replicates the refactor-branch load_domain_config exactly."""
    raw = yaml.safe_load(Path(cfg_path).read_text())
    cfg = {}
    for domain, params in raw.items():
        if domain == "padding":
            continue
        p = dict(params)
        if "path" in p:
            p["path"] = tokens_path / p["path"]
        # old DomainConfig didn't have parent/abstract/subdomain_column,
        # but DomainConfig now has them with defaults — passing via **p is fine
        # as long as the YAML doesn't set them (the default config doesn't).
        cfg[domain] = DomainConfig(**p)
    cfg["padding"] = DomainConfig(projector="embed")
    return _normalize_domain_cfg_old(cfg)


# ── helpers ───────────────────────────────────────────────────────────────────

def build_model(domain_cfg: dict, domains: list[str], seed: int = 42,
                n_embd: int = 32, n_layer: int = 2, n_head: int = 4,
                block_size: int = 32) -> Delphi:
    filtered = {k: v for k, v in domain_cfg.items() if k in domains or k == "padding"}
    cfg = DelphiConfig(
        n_embd=n_embd, n_layer=n_layer, n_head=n_head,
        domains=filtered,
        attention_scheme="all:causal(mask_ties=True)",
        block_size=block_size,
        seed=seed,
    )
    torch.manual_seed(seed)
    return Delphi(cfg)


def make_fake_batch_with_pcs(model, B=4, T=64, seed=7) -> DelphiBatch:
    """Fake batch that also includes genetic_pcs continuous data."""
    torch.manual_seed(seed)
    disease_int   = model.domain_to_int["diseases"]
    disease_off   = model.domain_offsets[disease_int]
    disease_vocab = model.embed._domain_vocab_sizes["diseases"]
    death_int     = model.domain_to_int["death"]
    death_off     = model.domain_offsets[death_int]
    pcs_int       = model.domain_to_int["genetic_pcs"]
    pcs_off       = model.domain_offsets[pcs_int]
    pcs_n_latent  = model.config.domains["genetic_pcs"].n_latent_tokens  # 5
    padding_int   = model.domain_to_int["padding"]
    padding_off   = model.domain_offsets[padding_int]

    G = torch.full((B, T), padding_off, dtype=torch.long)
    D = torch.full((B, T), padding_int, dtype=torch.long)
    A = torch.full((B, T), -10000.0,   dtype=torch.float)

    for b in range(B):
        pos = 0
        # genetic_pcs: n_latent_tokens slots at age 0
        for i in range(pcs_n_latent):
            G[b, pos] = pcs_off + i
            D[b, pos] = pcs_int
            A[b, pos] = 0.0
            pos += 1
        # disease events
        n_dis = min(8, T - pos - 2)
        local_dis = torch.randint(0, disease_vocab, (n_dis,))
        G[b, pos:pos + n_dis] = disease_off + local_dis
        D[b, pos:pos + n_dis] = disease_int
        A[b, pos:pos + n_dis] = torch.arange(1, n_dis + 1, dtype=torch.float) * 200
        pos += n_dis
        # death
        G[b, pos] = death_off
        D[b, pos] = death_int
        A[b, pos] = A[b, pos - 1] + 300
        pos += 1
        # rest = padding (already filled)

    pcs_input_size = model.config.domains["genetic_pcs"].input_size  # 40
    pcs_data = torch.randn(B, pcs_input_size)
    pcs_positions = torch.stack([
        torch.arange(pcs_n_latent) + (b * 0)  # latent slots are at fixed positions per subject
        for b in range(B)
    ])
    # positions where pcs tokens are (same for all subjects in our fake batch: 0..4)
    pcs_positions = torch.zeros(B, pcs_n_latent, dtype=torch.long)
    for b in range(B):
        pcs_positions[b] = torch.where(D[b] == pcs_int)[0][:pcs_n_latent]

    return DelphiBatch(
        global_token_ids=G,
        domain_ids=D,
        ages=A,
        subject_ids=torch.arange(B),
        continuous_data={"genetic_pcs": pcs_data},
        continuous_positions={"genetic_pcs": pcs_positions},
        eval_mask=None,
    )


def make_fake_batch(model, B=4, T=16, seed=7) -> DelphiBatch:
    torch.manual_seed(seed)
    disease_int   = model.domain_to_int["diseases"]
    disease_off   = model.domain_offsets[disease_int]
    disease_vocab = model.embed._domain_vocab_sizes["diseases"]
    death_int     = model.domain_to_int["death"]
    death_off     = model.domain_offsets[death_int]
    padding_int   = model.domain_to_int["padding"]
    padding_off   = model.domain_offsets[padding_int]

    G = torch.zeros(B, T, dtype=torch.long)
    D = torch.zeros(B, T, dtype=torch.long)
    A = torch.zeros(B, T, dtype=torch.float)

    for b in range(B):
        n_dis = 8
        local_dis = torch.randint(0, disease_vocab, (n_dis,))
        G[b, :n_dis] = disease_off + local_dis
        D[b, :n_dis] = disease_int
        A[b, :n_dis] = torch.arange(n_dis, dtype=torch.float) * 100

        G[b, n_dis] = death_off
        D[b, n_dis] = death_int
        A[b, n_dis] = A[b, n_dis - 1] + 200

        G[b, n_dis + 1:] = padding_off
        D[b, n_dis + 1:] = padding_int
        A[b, n_dis + 1:] = -10000.0

    return DelphiBatch(
        global_token_ids=G,
        domain_ids=D,
        ages=A,
        subject_ids=torch.arange(B),
        continuous_data={},
        continuous_positions={},
        eval_mask=None,
    )


def run_loss(model: Delphi, batch: DelphiBatch) -> torch.Tensor:
    """Compute CE loss the same way trainer.shared_step does."""
    model.eval()
    with torch.no_grad():
        logits_dict, _ = model(batch, return_attention=True)

    logits_cat = torch.cat([logits_dict[d] for d in model.predicted_domains], dim=-1)
    logits_cat = logits_cat[:, :-1, :]

    target_global_ids = batch.global_token_ids[:, 1:]
    target_domain_ids = batch.domain_ids[:, 1:]
    predicted_ints = torch.tensor([model.domain_to_int[d] for d in model.predicted_domains])
    predict_mask = torch.isin(target_domain_ids, predicted_ints)

    f_logits     = logits_cat[predict_mask]
    f_global_ids = target_global_ids[predict_mask]
    f_domain_ids = target_domain_ids[predict_mask]

    # NEW remapping (current branch)
    f_targets = f_global_ids.clone()
    cum = 0
    for dname in model.predicted_domains:
        d_int = model.domain_to_int[dname]
        mask = f_domain_ids == d_int
        f_targets[mask] = f_global_ids[mask] - model.domain_offsets[d_int] + cum
        cum += logits_dict[dname].shape[-1]

    return model.cross_entropy_loss(f_logits, f_targets)


# ── tests ─────────────────────────────────────────────────────────────────────

class TestBaseVsRefactor:

    @pytest.mark.parametrize("domains,label", [
        (BASE_DOMAINS,      "base"),
        (WITH_PCS_DOMAINS,  "with_genetic_pcs"),
    ])
    def test_domain_configs_identical(self, domains, label):
        """Old and new load_domain_config produce the same DomainConfig objects."""
        old_cfg = load_domain_config_old(BASE_CONFIG_YAML, ROOT_TOKENS)
        new_cfg = load_domain_config(BASE_CONFIG_YAML, ROOT_TOKENS)

        for dname in domains + ["padding"]:
            old = old_cfg[dname]
            new = new_cfg[dname]
            for field in (
                "projector", "type", "path", "predict", "at_birth",
                "n_latent_tokens", "input_size", "pretrained_path", "freeze",
                "age_jitter", "subdomain", "group", "dropout_mode", "dropout_rate",
            ):
                assert getattr(old, field) == getattr(new, field), (
                    f"[{label}] Field '{field}' differs for domain '{dname}': "
                    f"old={getattr(old, field)!r}, new={getattr(new, field)!r}"
                )

    @pytest.mark.parametrize("domains,label", [
        (BASE_DOMAINS,      "base"),
        (WITH_PCS_DOMAINS,  "with_genetic_pcs"),
    ])
    def test_domain_order_preserved(self, domains, label):
        """Domain insertion order must be preserved."""
        old_cfg = load_domain_config_old(BASE_CONFIG_YAML, ROOT_TOKENS)
        new_cfg = load_domain_config(BASE_CONFIG_YAML, ROOT_TOKENS)

        old_order = [k for k in old_cfg if k in domains or k == "padding"]
        new_order = [k for k in new_cfg if k in domains or k == "padding"]
        assert old_order == new_order, (
            f"[{label}] Domain order changed!\n  old: {old_order}\n  new: {new_order}"
        )

    @pytest.mark.parametrize("domains,label", [
        (BASE_DOMAINS,      "base"),
        (WITH_PCS_DOMAINS,  "with_genetic_pcs"),
    ])
    def test_model_architecture_identical(self, domains, label):
        """Same domain_to_int, domain_offsets, and parameter count."""
        old_cfg = load_domain_config_old(BASE_CONFIG_YAML, ROOT_TOKENS)
        new_cfg = load_domain_config(BASE_CONFIG_YAML, ROOT_TOKENS)

        m_old = build_model(old_cfg, domains, seed=42)
        m_new = build_model(new_cfg, domains, seed=42)

        assert m_old.domain_to_int  == m_new.domain_to_int,  f"[{label}] domain_to_int differs"
        assert m_old.domain_offsets == m_new.domain_offsets,  f"[{label}] domain_offsets differ"
        assert (
            sum(p.numel() for p in m_old.parameters()) ==
            sum(p.numel() for p in m_new.parameters())
        ), f"[{label}] parameter count differs"

    def test_forward_pass_bit_identical_base(self):
        """With the same seed, both models produce identical logits (base domains)."""
        old_cfg = load_domain_config_old(BASE_CONFIG_YAML, ROOT_TOKENS)
        new_cfg = load_domain_config(BASE_CONFIG_YAML, ROOT_TOKENS)
        m_old = build_model(old_cfg, BASE_DOMAINS, seed=42)
        m_new = build_model(new_cfg, BASE_DOMAINS, seed=42)

        for (n1, p1), (n2, p2) in zip(m_old.named_parameters(), m_new.named_parameters()):
            assert torch.equal(p1, p2), f"Parameter '{n1}' differs"

        batch = make_fake_batch(m_old)
        m_old.eval(); m_new.eval()
        with torch.no_grad():
            logits_old, _ = m_old(batch, return_attention=True)
            logits_new, _ = m_new(batch, return_attention=True)

        for dname in m_old.predicted_domains:
            assert torch.equal(logits_old[dname], logits_new[dname]), \
                f"Logits differ for domain '{dname}'"

    def test_forward_pass_bit_identical_with_pcs(self):
        """With the same seed, both models produce identical logits (with genetic_pcs)."""
        old_cfg = load_domain_config_old(BASE_CONFIG_YAML, ROOT_TOKENS)
        new_cfg = load_domain_config(BASE_CONFIG_YAML, ROOT_TOKENS)
        m_old = build_model(old_cfg, WITH_PCS_DOMAINS, seed=42)
        m_new = build_model(new_cfg, WITH_PCS_DOMAINS, seed=42)

        for (n1, p1), (n2, p2) in zip(m_old.named_parameters(), m_new.named_parameters()):
            assert torch.equal(p1, p2), f"Parameter '{n1}' differs"

        batch = make_fake_batch_with_pcs(m_old)
        m_old.eval(); m_new.eval()
        with torch.no_grad():
            logits_old, _ = m_old(batch, return_attention=True)
            logits_new, _ = m_new(batch, return_attention=True)

        for dname in m_old.predicted_domains:
            assert torch.equal(logits_old[dname], logits_new[dname]), \
                f"Logits differ for domain '{dname}'"

    def test_ce_loss_bit_identical_base(self):
        """CE loss bit-identical, base domains."""
        old_cfg = load_domain_config_old(BASE_CONFIG_YAML, ROOT_TOKENS)
        new_cfg = load_domain_config(BASE_CONFIG_YAML, ROOT_TOKENS)
        m_old = build_model(old_cfg, BASE_DOMAINS, seed=42)
        m_new = build_model(new_cfg, BASE_DOMAINS, seed=42)
        batch = make_fake_batch(m_old)
        loss_old = run_loss(m_old, batch)
        loss_new = run_loss(m_new, batch)
        assert torch.equal(loss_old, loss_new), \
            f"CE loss differs! old={loss_old:.8f}, new={loss_new:.8f}"

    def test_ce_loss_bit_identical_with_pcs(self):
        """CE loss bit-identical, with genetic_pcs (n_embd=120, n_layer=12, n_head=12)."""
        old_cfg = load_domain_config_old(BASE_CONFIG_YAML, ROOT_TOKENS)
        new_cfg = load_domain_config(BASE_CONFIG_YAML, ROOT_TOKENS)
        m_old = build_model(old_cfg, WITH_PCS_DOMAINS, seed=42,
                            n_embd=120, n_layer=12, n_head=12, block_size=64)
        m_new = build_model(new_cfg, WITH_PCS_DOMAINS, seed=42,
                            n_embd=120, n_layer=12, n_head=12, block_size=64)
        batch = make_fake_batch_with_pcs(m_old, T=64)
        loss_old = run_loss(m_old, batch)
        loss_new = run_loss(m_new, batch)
        assert torch.equal(loss_old, loss_new), \
            f"CE loss differs! old={loss_old:.8f}, new={loss_new:.8f}"
