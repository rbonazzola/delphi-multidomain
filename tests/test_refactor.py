"""
Smoke test for the refactored dataset + collate pipeline.
Uses real data — run from the project root (DELPHI_DIR).

Usage:
    python test_refactor.py
"""

# %%
import os, sys
from pathlib import Path
import yaml
from pprint import pprint

import torch
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

DELPHI_DIR = Path(__file__).resolve().parent
if DELPHI_DIR not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

from delphi.model import DomainConfig
from utils.cv_utils import get_data_partitions
from utils.utils import load_domain_config

from data.dataset import (
    DelphiDataset,
    DelphiCollateFn,
    DelphiBatch,
    AgeSampler,
    create_friendly_view,
    color_by_domain,
)

root_path = DELPHI_DIR / "data" / "transforms"

# ═══════════════════════════════════════════════════════════════════════════════
#  1. Load domain config (same as before)
# ═══════════════════════════════════════════════════════════════════════════════

# %%
args = edict(
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

print("Domain configs:")
pprint(domain_cfg)

# ═══════════════════════════════════════════════════════════════════════════════
#  2. Shared metadata (single source of truth)
# ═══════════════════════════════════════════════════════════════════════════════

# %%
def build_domain_to_int(domain_names):
    ordered = [d for d in domain_names if d != "padding"] + ["padding"]
    return {name: i for i, name in enumerate(ordered)}


def resolve_vocab_size(cfg):
    """
    Get vocab size for a domain. If input_size is None,
    read it from the tokenizer.yaml file.
    """
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
            running += 2  # PADDING_TOKEN=0, NO_EVENT_TOKEN=1
        elif cfg is None:
            offsets[d_int] = running
        elif cfg.type == "categorical" and cfg.projector in ("embed", "Embed"):
            offsets[d_int] = running
            running += resolve_vocab_size(cfg)
        else:
            # projected domain: placeholder slots
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
#  3. Subject splits
# ═══════════════════════════════════════════════════════════════════════════════

# %%
train_ids, val_ids, test_ids = get_data_partitions(
    "./data/transforms/subject_lists", fold=args.test_fold
)

train_ids = train_ids[:10000]
val_ids   = val_ids[:1000]
test_ids  = test_ids[:1000]

print(f"\nTrain: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

# ═══════════════════════════════════════════════════════════════════════════════
#  4. Dataset (CPU only, no .to(device))
# ═══════════════════════════════════════════════════════════════════════════════

# %%
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
print(f"\nDataset length: {len(train_dataset)}")

# ── Quick sanity check: single item ──
item = train_dataset[0]
print(f"\n__getitem__ keys: {list(item.keys())}")
print(f"  domain_ids shape:      {item['domain_ids'].shape}")
print(f"  local_token_ids shape: {item['local_token_ids'].shape}")
print(f"  ages shape:            {item['ages'].shape}")
print(f"  real_count:            {item['real_count']}")
print(f"  max_age:               {item['max_age']:.1f}")
print(f"  subject_id:            {item['subject_id']}")
for cd_name, cd_tensor in item["continuous"].items():
    print(f"  continuous[{cd_name}] shape: {cd_tensor.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
#  5. Collate function
# ═══════════════════════════════════════════════════════════════════════════════

# %%
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

# ═══════════════════════════════════════════════════════════════════════════════
#  6. DataLoader (with workers!)
# ═══════════════════════════════════════════════════════════════════════════════

# %%
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,  # start with 0 to debug, then try 4
    pin_memory=True,
    collate_fn=collate,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  7. Iterate and check
# ═══════════════════════════════════════════════════════════════════════════════

# %%
print("\n--- First batch ---")
batch: DelphiBatch = next(iter(train_loader))

print(f"Type: {type(batch)}")
print(f"Batch: {batch}")
print(f"global_token_ids: {batch.global_token_ids.shape} {batch.global_token_ids.dtype}")
print(f"domain_ids:       {batch.domain_ids.shape} {batch.domain_ids.dtype}")
print(f"ages:             {batch.ages.shape} {batch.ages.dtype}")
print(f"subject_ids:      {batch.subject_ids.shape} {batch.subject_ids.dtype}")
for k, v in batch.continuous_data.items():
    print(f"continuous_data[{k}]:      {v.shape} {v.dtype}")
for k, v in batch.continuous_positions.items():
    print(f"continuous_positions[{k}]: {v.shape} {v.dtype}")

# ── Check no NaNs ──
assert not torch.isnan(batch.ages).any(), "NaN in ages!"
assert not torch.isnan(batch.global_token_ids.float()).any(), "NaN in token ids!"

# ── Check padding ages are -10000 ──
is_padding = (batch.domain_ids == domain_to_int["padding"]) & (batch.global_token_ids == domain_offsets[domain_to_int["padding"]])
if is_padding.any():
    padding_ages = batch.ages[is_padding]
    print(f"\nPadding token ages (should be -10000): min={padding_ages.min():.1f}, max={padding_ages.max():.1f}")

# ── Check no-event tokens exist ──
no_event_global_id = domain_offsets[domain_to_int["padding"]] + 1
is_no_event = batch.global_token_ids == no_event_global_id
print(f"No-event tokens in batch: {is_no_event.sum().item()}")
if is_no_event.any():
    ne_ages = batch.ages[is_no_event]
    print(f"  age range: [{ne_ages.min():.1f}, {ne_ages.max():.1f}]")

# ── Check sort order ──
for b in range(min(3, batch.batch_size)):
    ages_b = batch.ages[b]
    # ages should be non-decreasing (padding at -10000 is at the start)
    diffs = ages_b[1:] - ages_b[:-1]
    violations = (diffs < -0.01).sum().item()  # small tolerance for domain_scale
    print(f"  Subject {b}: sort violations = {violations}")

# ═══════════════════════════════════════════════════════════════════════════════
#  8. Friendly view
# ═══════════════════════════════════════════════════════════════════════════════

# %%
print("\n--- Friendly view (first 2 subjects) ---")

# Take a small batch for display
small_batch = DelphiBatch(
    global_token_ids=batch.global_token_ids[:2],
    domain_ids=batch.domain_ids[:2],
    ages=batch.ages[:2],
    subject_ids=batch.subject_ids[:2],
    continuous_data={k: v[:2] for k, v in batch.continuous_data.items()},
    continuous_positions={k: v[:2] for k, v in batch.continuous_positions.items()},
)

df = create_friendly_view(
    small_batch,
    int_to_domain_name=int_to_domain,
    tokenizers=train_dataset.tokenizers,
    domain_offsets=domain_offsets,
)
display(df.head(128))
print(df.to_string(max_rows=40))

# ═══════════════════════════════════════════════════════════════════════════════
#  9. Test .to(device)
# ═══════════════════════════════════════════════════════════════════════════════

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_gpu = batch.to(device)
print(f"\n--- .to('{device}') ---")
print(f"global_token_ids device: {batch_gpu.global_token_ids.device}")
print(f"ages device:             {batch_gpu.ages.device}")

# ═══════════════════════════════════════════════════════════════════════════════
#  10. Test with num_workers > 0
# ═══════════════════════════════════════════════════════════════════════════════

# %%
print("\n--- Testing with num_workers=4 ---")
loader_mp = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate,
)

batch_mp: DelphiBatch = next(iter(loader_mp))
print(f"OK — got DelphiBatch with shape {batch_mp.global_token_ids.shape}")

# %%
print("\n✓ All checks passed!")

from tqdm import tqdm

for batch in tqdm(loader_mp):
    pass


# %%
batch.continuous_positions['genetic_pcs']