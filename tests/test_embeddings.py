"""
Test for MultiDomainEmbedding v2.
Extends the dataset/collate test: loads real data, builds a batch,
and runs it through the new embedding layer.

Run from project root:
    python test_embedding.py
"""

# %%
import os, sys
from pathlib import Path
import yaml
from pprint import pprint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

DELPHI_DIR = Path(__file__).resolve().parent
if DELPHI_DIR not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

from delphi.model import DomainConfig, DelphiConfig
from delphi.embedding import MultiDomainEmbedding
from utils.cv_utils import get_data_partitions
from utils.utils import load_domain_config

from data.dataset import (
    DelphiDataset,
    DelphiCollateFn,
    DelphiBatch,
    AgeSampler,
    create_friendly_view,
)

root_path = DELPHI_DIR / "data" / "transforms"

# ═══════════════════════════════════════════════════════════════════════════════
#  1. Config
# ═══════════════════════════════════════════════════════════════════════════════

args = edict(
    domains="hla_alleles,diseases,death,lifestyle,sex,padding,genetic_pcs",
    domain_config_yaml="config/domain_config_default.yaml",
    test_fold=1,
    block_size=128,
    batch_size=128,
    n_embd=120,
    n_layer=12,
    n_head=6,
    no_event_token_rate=2.0,
    no_event_insertion_mode="random",
    token_dropout=0.1,
    seed=42,
)

domains = args.domains.split(",")
domain_config_yaml = DELPHI_DIR / args.domain_config_yaml
default_cfg_per_domain = load_domain_config(domain_config_yaml, root_path / "tokens")
domain_cfg = {k: v for k, v in default_cfg_per_domain.items() if k in domains}

print("Domains:")
pprint(list(domain_cfg.keys()))

# ═══════════════════════════════════════════════════════════════════════════════
#  2. Shared metadata
# ═══════════════════════════════════════════════════════════════════════════════

def build_domain_to_int(domain_names):
    ordered = [d for d in domain_names if d != "padding"] + ["padding"]
    return {name: i for i, name in enumerate(ordered)}


def resolve_vocab_size(cfg):
    if cfg.input_size is not None:
        return cfg.input_size
    tokenizer_path = Path(cfg.path) / "tokenizer.yaml"
    with open(tokenizer_path, "r") as f:
        return len(yaml.safe_load(f))


def build_domain_offsets(domain_to_int, domain_cfg):
    offsets = {}
    running = 0
    for dname in domain_to_int:
        d_int = domain_to_int[dname]
        cfg = domain_cfg.get(dname)

        if dname == "padding":
            offsets[d_int] = running
            running += 2
        elif cfg is None:
            offsets[d_int] = running
        elif cfg.type == "categorical" and cfg.projector in ("embed", "Embed"):
            offsets[d_int] = running
            running += resolve_vocab_size(cfg)
        else:
            offsets[d_int] = running
            n_slots = getattr(cfg, "n_latent_tokens", 1) or 1
            running += n_slots

    return offsets, running


def get_continuous_domains(domain_cfg):
    return {
        dname: cfg.n_latent_tokens or 1
        for dname, cfg in domain_cfg.items()
        if cfg.type == "continuous"
    }


domain_to_int = build_domain_to_int(list(domain_cfg.keys()))
int_to_domain = {v: k for k, v in domain_to_int.items()}
continuous_domains = get_continuous_domains(domain_cfg)
domain_offsets, global_vocab_size = build_domain_offsets(domain_to_int, domain_cfg)

print(f"\ndomain_to_int: {domain_to_int}")
print(f"domain_offsets: {domain_offsets}")
print(f"global_vocab_size: {global_vocab_size}")
print(f"continuous_domains: {continuous_domains}")

# ═══════════════════════════════════════════════════════════════════════════════
#  3. Dataset + DataLoader (same as test_refactor.py)
# ═══════════════════════════════════════════════════════════════════════════════

train_ids, val_ids, test_ids = get_data_partitions(
    "./data/transforms/subject_lists", fold=args.test_fold
)


dataset_kwargs = dict(
    root=root_path,
    domains_cfg=domain_cfg,
    domain_to_int=domain_to_int,
    block_size=args.block_size,
    exclusions=[],
    required_domains=["diseases"],
    no_event_token_rate=args.no_event_token_rate,
    no_event_insertion_mode=args.no_event_insertion_mode,
    continuous_domains=continuous_domains,
    age_domains=["diseases", "death"],
)

train_dataset = DelphiDataset(subjects=train_ids, **dataset_kwargs)

age_sampler = AgeSampler(
    insertion_mode=args.no_event_insertion_mode,
    token_rate=args.no_event_token_rate,
    seed=args.seed,
)

collate = DelphiCollateFn(
    age_sampler=age_sampler,
    block_size=args.block_size,
    domain_to_int=domain_to_int,
    domain_offsets=domain_offsets,
    padding_domain_id=domain_to_int["padding"],
    no_event_token_id=1,
    continuous_domains=continuous_domains,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    collate_fn=collate,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  4. Build MultiDomainEmbedding
# ═══════════════════════════════════════════════════════════════════════════════

device = "cuda" if torch.cuda.is_available() else "cpu"

# We need a DelphiConfig-like object with .domains, .n_embd, .token_dropout
# We can use the real DelphiConfig or a simple namespace
config = edict(
    domains=domain_cfg,
    n_embd=args.n_embd,
    token_dropout=args.token_dropout,
)

embed = MultiDomainEmbedding(
    config=config,
    domain_offsets=domain_offsets,
    global_vocab_size=global_vocab_size,
    domain_to_int=domain_to_int,
).to(device)

print(f"\n{embed}")
print(f"Global embedding weight shape: {embed.global_embed.weight.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
#  5. Get a batch and run forward
# ═══════════════════════════════════════════════════════════════════════════════

# %%
batch: DelphiBatch = next(iter(train_loader))
batch = batch.to(device)

print(f"\nBatch: {batch}")

# ── Forward ──
emb = embed(batch)
print(f"\nEmbedding output shape: {emb.shape}")
assert emb.shape == (args.batch_size, args.block_size, args.n_embd), \
    f"Expected {(args.batch_size, args.block_size, args.n_embd)}, got {emb.shape}"

# ── Check no NaNs ──
assert not torch.isnan(emb).any(), "NaN in embedding output!"
print("No NaNs in embedding output ✓")

# ── Check that projected slots are NOT zero (scatter_add injected real values) ──
for cd_name in continuous_domains:
    cd_int = domain_to_int[cd_name]
    cd_mask = batch.domain_ids == cd_int  # [B, T]
    if cd_mask.any():
        cd_embs = emb[cd_mask]  # [N, n_embd]
        nonzero_ratio = (cd_embs.abs() > 1e-8).float().mean().item()
        print(f"Continuous domain '{cd_name}': nonzero ratio = {nonzero_ratio:.3f}")
        assert nonzero_ratio > 0.5, \
            f"Projected embeddings for '{cd_name}' look too sparse — scatter_add may have failed"
    else:
        print(f"Continuous domain '{cd_name}': not present in this batch")

# ── Check that padding slots (token=0) have proper embedding ──
pad_int = domain_to_int["padding"]
is_real_padding = (batch.domain_ids == pad_int) & (batch.global_token_ids == domain_offsets[pad_int])
if is_real_padding.any():
    pad_embs = emb[is_real_padding]
    print(f"Padding embeddings shape: {pad_embs.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
#  6. Test to_logits
# ═══════════════════════════════════════════════════════════════════════════════

# %%
# Simulate a hidden state (same shape as embedding output)
h = torch.randn_like(emb)

logits = embed.to_logits(h)
print(f"\nto_logits output domains: {list(logits.keys())}")
print(f"predicted_domains: {embed.predicted_domains}")

for dname, logit_tensor in logits.items():
    print(f"  {dname}: {logit_tensor.shape}")
    assert logit_tensor.shape[0] == args.batch_size
    assert logit_tensor.shape[1] == args.block_size
    assert not torch.isnan(logit_tensor).any(), f"NaN in logits for {dname}"

print("to_logits shapes and values OK ✓")

# ═══════════════════════════════════════════════════════════════════════════════
#  7. Test gradient flow
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- Gradient flow test ---")

# Zero grads
embed.zero_grad()

# Forward
emb = embed(batch)
loss = emb.sum()
loss.backward()

# Check global_embed got gradients
assert embed.global_embed.weight.grad is not None, "No gradient on global_embed!"
grad_nonzero = (embed.global_embed.weight.grad.abs() > 0).sum().item()
print(f"Global embed: {grad_nonzero}/{embed.global_vocab_size} rows have nonzero grad")

# Check projectors got gradients
for dname, proj in embed.projectors.items():
    has_grad = False
    for pname, param in proj.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    status = "✓" if has_grad else "✗ (no grad!)"
    print(f"Projector '{dname}': {status}")

# ═══════════════════════════════════════════════════════════════════════════════
#  8. Test with num_workers > 0
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- num_workers=2 test ---")
loader_mp = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    collate_fn=collate,
)

batch_mp = next(iter(loader_mp)).to(device)
emb_mp = embed(batch_mp)
print(f"OK — embedding shape: {emb_mp.shape}")
assert not torch.isnan(emb_mp).any()

print("\n✓ All embedding tests passed!")

from tqdm import tqdm

for batch in tqdm(loader_mp):
    batch = batch.to(device)
    emb = embed(batch)
    pass

# %%
import importlib
import delphi.model_v2

delphi.model_v2 = importlib.reload(delphi.model_v2)

Delphi = delphi.model_v2.Delphi
DelphiConfig = delphi.model_v2.DelphiConfig

# %%
args = edict(
    n_layer=24,n_head=12,n_embd=240,
    domains="hla_alleles,diseases,death,lifestyle,sex,padding,genetic_pcs",
    domain_config_yaml="config/domain_config_default.yaml",
    test_fold=1,
    block_size=128,
    batch_size=128,
    no_event_token_rate=4.0,
    no_event_insertion_mode="random",
    seed=42,
)

domains = args.domains.split(",")
domain_config_yaml = DELPHI_DIR / args.domain_config_yaml
default_cfg_per_domain = load_domain_config(domain_config_yaml, root_path / "tokens")
domain_cfg = {k: v for k, v in default_cfg_per_domain.items() if k in domains}

delphi_config = DelphiConfig(domains=domain_cfg)
model = Delphi(delphi_config)
model = model.to('cuda')

for dname, cfg in delphi_config.domains.items():
    pprint(f"{dname}: type={cfg.type!r}, projector={cfg.projector!r}, input_size={cfg.input_size}, path={cfg.path}")
# %%
batch = next(iter(train_loader))  # todavía en CPU

print(f"global_token_ids: min={batch.global_token_ids.min()}, max={batch.global_token_ids.max()}")
print(f"global_vocab_size: {embed.global_vocab_size}")
assert batch.global_token_ids.max() < embed.global_vocab_size, "Token ID out of range!"
assert batch.global_token_ids.min() >= 0, "Negative token ID!"

# %%

from tqdm import tqdm

model.to("cuda")

for batch in tqdm(train_loader):
    batch = batch.to('cuda')
    logits, att = model(batch, return_attention=True)
    
# %%
import time

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

sync(); t0 = time.perf_counter()
emb = model.embed(batch)
sync(); print(f"embed:        {time.perf_counter()-t0:.4f}s")

sync(); t0 = time.perf_counter()
age_emb = model.transformer.age_embedding(batch.ages.T)
if age_emb.ndim == 2:
    age_emb = age_emb.unsqueeze(1)
age_emb = age_emb.transpose(0, 1)
sync(); print(f"age_encoding: {time.perf_counter()-t0:.4f}s")

sync(); t0 = time.perf_counter()
h = emb + age_emb
h = model.transformer.drop(h)
sync(); print(f"add+drop:     {time.perf_counter()-t0:.4f}s")

sync(); t0 = time.perf_counter()
attn_mask = model.build_attn_mask(batch)
sync(); print(f"attn_mask:    {time.perf_counter()-t0:.4f}s")

for i, block in enumerate(model.transformer.h):
    sync(); t0 = time.perf_counter()
    h, _ = block(h, attn_mask=attn_mask[i])
    sync(); print(f"block {i:2d}:     {time.perf_counter()-t0:.4f}s")

sync(); t0 = time.perf_counter()
h = model.transformer.ln_f(h)
sync(); print(f"ln_f:         {time.perf_counter()-t0:.4f}s")

sync(); t0 = time.perf_counter()
logits = model.embed.to_logits(h)
sync(); print(f"to_logits:    {time.perf_counter()-t0:.4f}s")