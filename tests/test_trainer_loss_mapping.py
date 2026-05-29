"""
Tests that the f_targets remapping in Trainer.shared_step is correct.

Key invariant: with `domain_config_default.yaml`, death comes right after
diseases, so death_offset == vocab_diseases. In that layout the remapping
is an identity, so old code and new code produce **identical** CE losses.

With `domain_config_per_hla_locus.yaml`, sex precedes death, giving
death_offset == vocab_diseases + vocab_sex (= 1258 instead of 1256). The
old code passed those global IDs directly as logits_cat targets, causing
  IndexError: Target 1258 is out of bounds for input with size 1257.
The new code remaps them to the correct logits_cat index.

Run from project root:
    pytest tests/test_trainer_loss_mapping.py -v
"""

import sys, os
from pathlib import Path

import torch
import torch.nn.functional as F
import pytest

DELPHI_DIR = Path(__file__).resolve().parent.parent
if DELPHI_DIR not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

from delphi.model import Delphi, DelphiConfig, DomainConfig
from data.dataset import DelphiBatch
from utils.utils import load_domain_config

ROOT = DELPHI_DIR / "data" / "transforms"


# ── helpers ──────────────────────────────────────────────────────────────────

def make_model(config_yaml: str, domains: list[str], n_embd=32, n_layer=1, seed=0) -> Delphi:
    domain_cfg_all = load_domain_config(DELPHI_DIR / config_yaml, ROOT / "tokens")
    domain_cfg = {k: v for k, v in domain_cfg_all.items() if k in domains or k == "padding"}
    cfg = DelphiConfig(
        n_embd=n_embd, n_layer=n_layer, n_head=4,
        domains=domain_cfg,
        attention_scheme="all:causal(mask_ties=True)",
        block_size=32,
        seed=seed,
    )
    torch.manual_seed(seed)
    return Delphi(cfg)


def remap_targets(f_global_ids, f_domain_ids, model, logits_dict):
    """Replicate the current f_targets remapping from trainer.shared_step."""
    f_targets = f_global_ids.clone()
    cum = 0
    for dname in model.predicted_domains:
        d_int = model.domain_to_int[dname]
        mask = f_domain_ids == d_int
        f_targets[mask] = f_global_ids[mask] - model.domain_offsets[d_int] + cum
        cum += logits_dict[dname].shape[-1]
    return f_targets


def make_fake_batch(model, B=4, T=16, seed=1) -> DelphiBatch:
    """Create a minimal fake batch with global token IDs drawn from each domain."""
    torch.manual_seed(seed)
    domain_to_int = model.domain_to_int
    domain_offsets = model.domain_offsets
    block_size = T

    global_ids = torch.zeros(B, T, dtype=torch.long)
    domain_ids_t = torch.zeros(B, T, dtype=torch.long)
    ages = torch.zeros(B, T, dtype=torch.float)

    padding_int = domain_to_int["padding"]
    padding_offset = domain_offsets[padding_int]
    # Fill first half with diseases, rest with death; pad the remainder
    disease_int = domain_to_int["diseases"]
    disease_offset = domain_offsets[disease_int]
    disease_vocab = model.embed._domain_vocab_sizes["diseases"]

    death_int = domain_to_int["death"]
    death_offset = domain_offsets[death_int]

    for b in range(B):
        # positions 0..7 = disease tokens
        n_disease = 8
        n_death = 2
        local_disease_ids = torch.randint(0, disease_vocab, (n_disease,))
        global_ids[b, :n_disease] = disease_offset + local_disease_ids
        domain_ids_t[b, :n_disease] = disease_int
        ages[b, :n_disease] = torch.arange(n_disease, dtype=torch.float) * 100

        # positions 8..9 = death tokens
        global_ids[b, n_disease:n_disease + n_death] = death_offset  # only 1 death token (id=0)
        domain_ids_t[b, n_disease:n_disease + n_death] = death_int
        ages[b, n_disease:n_disease + n_death] = ages[b, n_disease - 1] + 200

        # rest = padding
        global_ids[b, n_disease + n_death:] = padding_offset
        domain_ids_t[b, n_disease + n_death:] = padding_int
        ages[b, n_disease + n_death:] = -10000.0

    return DelphiBatch(
        global_token_ids=global_ids,
        domain_ids=domain_ids_t,
        ages=ages,
        subject_ids=torch.arange(B),
        continuous_data={},
        continuous_positions={},
        eval_mask=None,
    )


def compute_ce_loss_both_ways(model, batch):
    """
    Compute CE loss using both the old way (f_global_ids directly) and the new
    remapped way (f_targets). Return (loss_old, loss_new, targets_old, targets_new).
    """
    model.eval()
    with torch.no_grad():
        logits_dict, _ = model(batch, return_attention=True)

    logits_cat = torch.cat([logits_dict[d] for d in model.predicted_domains], dim=-1)
    logits_cat = logits_cat[:, :-1, :]  # shift

    target_global_ids = batch.global_token_ids[:, 1:]
    target_domain_ids = batch.domain_ids[:, 1:]

    predicted_ints = torch.tensor([model.domain_to_int[d] for d in model.predicted_domains])
    predict_mask = torch.isin(target_domain_ids, predicted_ints)

    f_logits = logits_cat[predict_mask]
    f_global_ids = target_global_ids[predict_mask]
    f_domain_ids = target_domain_ids[predict_mask]

    f_targets = remap_targets(f_global_ids, f_domain_ids, model, logits_dict)

    loss_old = model.cross_entropy_loss(f_logits, f_global_ids)
    loss_new = model.cross_entropy_loss(f_logits, f_targets)

    return loss_old, loss_new, f_global_ids, f_targets


# ── tests ────────────────────────────────────────────────────────────────────

class TestFTargetsRemapping:

    def test_default_config_targets_unchanged(self):
        """With domain_config_default.yaml (death right after diseases),
        f_targets must equal f_global_ids — the remapping is an identity."""
        model = make_model(
            "config/domain_config_default.yaml",
            domains=["diseases", "death", "lifestyle", "sex"],
        )
        batch = make_fake_batch(model)
        _, _, f_global_ids, f_targets = compute_ce_loss_both_ways(model, batch)
        assert torch.equal(f_targets, f_global_ids), (
            "With the default config (death offset == vocab_diseases), "
            "f_targets must equal f_global_ids but they differ:\n"
            f"  max |diff| = {(f_targets - f_global_ids).abs().max().item()}"
        )

    def test_default_config_loss_unchanged(self):
        """CE loss must be identical before and after the remapping for the default config."""
        model = make_model(
            "config/domain_config_default.yaml",
            domains=["diseases", "death", "lifestyle", "sex"],
        )
        batch = make_fake_batch(model)
        loss_old, loss_new, _, _ = compute_ce_loss_both_ways(model, batch)
        assert torch.isclose(loss_old, loss_new), (
            f"Loss changed with default config: old={loss_old:.6f}, new={loss_new:.6f}"
        )

    def test_hla_config_death_targets_remapped(self):
        """With domain_config_per_hla_locus.yaml, sex precedes death so
        death_offset = vocab_diseases + vocab_sex = 1258.  The old code would
        pass target=1258 to a logits tensor of size 1257 (IndexError). The new
        code remaps it to 1256, the correct index."""
        model = make_model(
            "config/domain_config_per_hla_locus.yaml",
            domains=["diseases", "sex", "death", "lifestyle"],
        )
        death_int = model.domain_to_int["death"]
        disease_int = model.domain_to_int["diseases"]
        death_offset = model.domain_offsets[death_int]
        disease_vocab = model.embed._domain_vocab_sizes["diseases"]

        # Death offset must be > vocab_diseases (sex sits between them)
        assert death_offset > disease_vocab, (
            f"Expected death_offset ({death_offset}) > vocab_diseases ({disease_vocab})"
        )

        batch = make_fake_batch(model)

        model.eval()
        with torch.no_grad():
            logits_dict, _ = model(batch, return_attention=True)

        logits_cat = torch.cat([logits_dict[d] for d in model.predicted_domains], dim=-1)
        logits_cat = logits_cat[:, :-1, :]

        target_global_ids = batch.global_token_ids[:, 1:]
        target_domain_ids = batch.domain_ids[:, 1:]
        predicted_ints = torch.tensor([model.domain_to_int[d] for d in model.predicted_domains])
        predict_mask = torch.isin(target_domain_ids, predicted_ints)

        f_logits = logits_cat[predict_mask]
        f_global_ids = target_global_ids[predict_mask]
        f_domain_ids = target_domain_ids[predict_mask]
        f_targets = remap_targets(f_global_ids, f_domain_ids, model, logits_dict)

        V_total = f_logits.shape[-1]
        death_mask = f_domain_ids == death_int

        # Old targets for death tokens exceed logits size → would have caused IndexError
        assert (f_global_ids[death_mask] >= V_total).any(), (
            "Expected at least some death global IDs to be out of bounds (old bug)"
        )

        # New targets are all valid
        assert (f_targets[death_mask] < V_total).all(), (
            "Remapped death targets must all be < V_total"
        )
        assert (f_targets[death_mask] >= 0).all()

    def test_hla_config_loss_computable(self):
        """CE loss must not raise IndexError with the remapped targets."""
        model = make_model(
            "config/domain_config_per_hla_locus.yaml",
            domains=["diseases", "sex", "death", "lifestyle"],
        )
        batch = make_fake_batch(model)

        model.eval()
        with torch.no_grad():
            logits_dict, _ = model(batch, return_attention=True)

        logits_cat = torch.cat([logits_dict[d] for d in model.predicted_domains], dim=-1)
        logits_cat = logits_cat[:, :-1, :]
        target_global_ids = batch.global_token_ids[:, 1:]
        target_domain_ids = batch.domain_ids[:, 1:]
        predicted_ints = torch.tensor([model.domain_to_int[d] for d in model.predicted_domains])
        predict_mask = torch.isin(target_domain_ids, predicted_ints)
        f_logits = logits_cat[predict_mask]
        f_global_ids = target_global_ids[predict_mask]
        f_domain_ids = target_domain_ids[predict_mask]
        f_targets = remap_targets(f_global_ids, f_domain_ids, model, logits_dict)

        loss_new = model.cross_entropy_loss(f_logits, f_targets)
        assert loss_new.isfinite(), f"CE loss should be finite, got {loss_new}"

    def test_disease_targets_always_identity(self):
        """Disease global IDs == local IDs (offset=0 in all configs)
        so disease f_targets must always equal f_global_ids."""
        for config_yaml, domains in [
            ("config/domain_config_default.yaml",     ["diseases", "death", "lifestyle", "sex"]),
            ("config/domain_config_per_hla_locus.yaml", ["diseases", "sex", "death", "lifestyle"]),
        ]:
            model = make_model(config_yaml, domains)
            batch = make_fake_batch(model)

            model.eval()
            with torch.no_grad():
                logits_dict, _ = model(batch, return_attention=True)

            target_global_ids = batch.global_token_ids[:, 1:]
            target_domain_ids = batch.domain_ids[:, 1:]
            predicted_ints = torch.tensor([model.domain_to_int[d] for d in model.predicted_domains])
            predict_mask = torch.isin(target_domain_ids, predicted_ints)
            f_global_ids = target_global_ids[predict_mask]
            f_domain_ids = target_domain_ids[predict_mask]
            f_targets = remap_targets(f_global_ids, f_domain_ids, model, logits_dict)

            disease_int = model.domain_to_int["diseases"]
            target_domain_ids = batch.domain_ids[:, 1:]
            predicted_ints = torch.tensor([model.domain_to_int[d] for d in model.predicted_domains])
            predict_mask = torch.isin(target_domain_ids, predicted_ints)
            f_domain_ids_flat = target_domain_ids[predict_mask]
            disease_mask = f_domain_ids_flat == disease_int

            # diseases offset must be 0 in both configs
            assert model.domain_offsets[disease_int] == 0, \
                f"diseases offset should be 0, got {model.domain_offsets[disease_int]}"

            # targets for disease tokens must be unchanged
            assert torch.equal(f_targets[disease_mask], f_global_ids[disease_mask]), \
                f"Disease targets changed in {config_yaml}"
