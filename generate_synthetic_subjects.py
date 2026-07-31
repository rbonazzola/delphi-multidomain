# %%
"""
Generates a folder of synthetic subjects using a trained Delphi checkpoint's
`.generate()` method, in the same tokens.csv/tokenizer.yaml layout as
data/transforms/tokens/<domain>/ (see CLAUDE.md).

Conditioning: real sex, real genetic PCs, and each subject's real lifestyle
tokens (with their real ages) are taken from N real subjects pooled across
all of the given MLflow run's splits. None of these three are `predict:
true` domains, so the model can't generate them on its own -- they're copied
from real subjects instead. Everything else (diseases, death, and the age
each occurs at) is sampled by the model (see Delphi.generate's docstring).
Synthetic subjects get new sequential IDs, decoupled from the real
subject_ids used for conditioning.

genetic_pcs itself is not written to the output -- it's only used as
conditioning input, never generated.

Caveats (age-dependent calibration; see run c7cd0d84e11548e7bd381d75004e4957
validation): well-calibrated timing up to ~age 70; beyond that, accumulated
comorbidity inflates late-life disease/death rates above real population
rates (~3x for absolute prevalence, though relative prevalence ranking
across diseases stays well correlated with real data, ~0.90-0.91).

Usage
-----
python generate_synthetic_subjects.py \\
    --run_id c7cd0d84e11548e7bd381d75004e4957 \\
    --n_subjects 1000 \\
    --output_dir data/transforms/tokens/synthetic
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

def _find_delphi_dir() -> Path:
    """Repo root, found via the `delphi` package rather than this script's own
    location -- this file may be copied/run from anywhere (e.g. the repo root
    on a cluster), not necessarily from data/transforms/."""
    import delphi
    return Path(delphi.__file__).resolve().parent.parent


DELPHI_DIR = _find_delphi_dir()


def load_model_and_domains(run_id: str, mlruns_root: Path):
    from delphi.model import Delphi, DelphiConfig
    from utils.mlflow_utils import load_run_params, parse_domains_param
    from utils.ckpt_utils import strip_compiled_prefix

    params = load_run_params(run_id)
    domain_cfg = parse_domains_param(params["domains"], run_id=run_id)

    tokens_root = DELPHI_DIR / "data" / "transforms" / "tokens"
    for dname, dcfg in domain_cfg.items():
        if dname in ("padding", "no_event") or not getattr(dcfg, "path", None):
            continue
        dcfg.path = str(tokens_root / Path(str(dcfg.path)).name)

    # Locate the run's experiment dir and best checkpoint by scanning mlruns_root,
    # since meta.yaml's artifact_uri may point at a cluster-only path.
    matches = list(mlruns_root.glob(f"*/{run_id}"))
    if not matches:
        raise FileNotFoundError(f"Run {run_id!r} not found under {mlruns_root}")
    run_dir = matches[0]
    ckpt_path = run_dir / "artifacts" / "checkpoints" / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metadata = ckpt["metadata"]

    attn_scheme = params.get("attention_scheme", ["all:causal(mask_ties=True)"])
    n_layer = int(params.get("n_layer", 12))
    if isinstance(attn_scheme, str):
        attn_scheme = [attn_scheme]
    if len(attn_scheme) == 1:
        attn_scheme = n_layer * attn_scheme

    cfg = DelphiConfig(
        n_embd=int(params["n_embd"]), n_layer=n_layer, n_head=int(params["n_head"]),
        domains=domain_cfg, attention_scheme=attn_scheme,
        block_size=int(params["block_size"]), seed=int(params["seed"]),
    )
    model = Delphi(cfg)
    model.load_state_dict(strip_compiled_prefix(ckpt["state_dict"]), strict=False)
    model.eval()

    return model, domain_cfg, params, metadata, tokens_root


def build_conditioning_batch(model, domain_cfg, subject_ids):
    from data.dataset import DelphiDataset, DelphiCollateFn, AgeSampler, DelphiBatch

    n_latent = domain_cfg["genetic_pcs"].n_latent_tokens
    continuous_domains = {"genetic_pcs": n_latent}

    # Only load the domains this function actually reads (sex, genetic_pcs,
    # lifestyle): TokenDomain._load_tokens() pd.read_csv()'s a domain's
    # tokens.csv in full (the whole cohort) before filtering to subject_ids,
    # so loading unused domains like diseases/drugs/hla_alleles here costs
    # the same regardless of --n_subjects and dominates startup time for no
    # benefit -- diseases/death were only ever used to seed the no-event
    # AgeSampler below, whose output is entirely discarded.
    conditioning_domains = {"sex", "genetic_pcs"}
    if "lifestyle" in domain_cfg:
        conditioning_domains.add("lifestyle")
    conditioning_cfg = {k: v for k, v in domain_cfg.items() if k in conditioning_domains}

    ds = DelphiDataset(
        root="data/transforms", domains_cfg=conditioning_cfg, domain_to_int=model.domain_to_int,
        block_size=model.block_size, subjects=subject_ids, required_domains=["sex"],
        continuous_domains=continuous_domains, age_domains=[],
    )
    subjects_loaded = ds._subject_ids.tolist()
    B = len(subjects_loaded)

    age_sampler = AgeSampler(insertion_mode="random", token_rate=2.0, seed=0)
    collate = DelphiCollateFn(
        age_sampler=age_sampler, block_size=model.block_size, domain_to_int=model.domain_to_int,
        domain_offsets=model.domain_offsets, padding_domain_id=model.domain_to_int["padding"],
        no_event_domain_id=model.domain_to_int["no_event"], continuous_domains=continuous_domains,
        training=False,
    )
    items = [ds[i] for i in range(B)]
    full_batch = collate(items)

    sex_int = model.domain_to_int["sex"]
    pad_int = model.domain_to_int["padding"]
    has_lifestyle = "lifestyle" in model.domain_to_int

    # lifestyle is not a `predict: true` domain, so the model can never generate
    # it on its own -- it must be copied from real subjects, same as sex/PCs.
    # Each subject can have a different number of real lifestyle events, so we
    # pad every subject's lifestyle slots out to the batch max (unused slots
    # get the dataset's standard padding convention: domain=padding, age=-10000).
    if has_lifestyle:
        lifestyle_int = model.domain_to_int["lifestyle"]
        lifestyle_mask = full_batch.domain_ids == lifestyle_int
        max_lifestyle = int(lifestyle_mask.sum(dim=1).max().item()) if B else 0
    else:
        lifestyle_mask = None
        max_lifestyle = 0

    lifestyle_start = 1 + n_latent
    cond_len = lifestyle_start + max_lifestyle
    cond_global = torch.zeros(B, cond_len, dtype=torch.long)
    cond_domain = torch.full((B, cond_len), pad_int, dtype=torch.long)
    cond_age = torch.full((B, cond_len), DelphiCollateFn.PADDING_AGE)

    cond_domain[:, 0] = sex_int
    cond_age[:, 0] = 0.0
    cond_domain[:, 1:lifestyle_start] = model.domain_to_int["genetic_pcs"]
    cond_age[:, 1:lifestyle_start] = 0.0

    sex_mask = full_batch.domain_ids == sex_int
    for b in range(B):
        idx = sex_mask[b].nonzero(as_tuple=True)[0][0]
        cond_global[b, 0] = full_batch.global_token_ids[b, idx]

        if has_lifestyle:
            lf_idx = lifestyle_mask[b].nonzero(as_tuple=True)[0]
            n_lf = lf_idx.numel()
            if n_lf:
                end = lifestyle_start + n_lf
                cond_global[b, lifestyle_start:end] = full_batch.global_token_ids[b, lf_idx]
                cond_domain[b, lifestyle_start:end] = lifestyle_int
                cond_age[b, lifestyle_start:end] = full_batch.ages[b, lf_idx]

    # Real per-subject PC vectors, loaded natively by DelphiDataset/DelphiCollateFn
    # (genetic_pcs/tokens.csv is now in the standard subject_id,token_id,value format).
    pcs_values = full_batch.continuous_data["genetic_pcs"]
    continuous_positions = torch.arange(1, 1 + n_latent).unsqueeze(0).expand(B, n_latent).clone()

    return DelphiBatch(
        cond_global, cond_domain, cond_age, torch.tensor(subjects_loaded),
        {"genetic_pcs": pcs_values}, {"genetic_pcs": continuous_positions},
    ), subjects_loaded, cond_len


def generate_chunked(model, cond_batch, cond_len, max_new_tokens, max_age_days,
                      termination_domain, chunk_size, device):
    from data.dataset import DelphiBatch

    model = model.to(device)
    B = cond_batch.batch_size
    pad_int = model.domain_to_int["padding"]
    target_T = cond_len + max_new_tokens

    domain_chunks, global_chunks, age_chunks = [], [], []
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        sub_cond = DelphiBatch(
            cond_batch.global_token_ids[start:end], cond_batch.domain_ids[start:end],
            cond_batch.ages[start:end], cond_batch.subject_ids[start:end],
            {k: v[start:end] for k, v in cond_batch.continuous_data.items()},
            {k: v[start:end] for k, v in cond_batch.continuous_positions.items()},
        )
        sub_out = model.generate(
            sub_cond, max_new_tokens=max_new_tokens, max_age=max_age_days,
            termination_domain=termination_domain,
        )
        d, g, a = sub_out.domain_ids.cpu(), sub_out.global_token_ids.cpu(), sub_out.ages.cpu()
        pad_n = target_T - d.shape[1]
        if pad_n > 0:
            nb = d.shape[0]
            d = torch.cat([d, torch.full((nb, pad_n), pad_int, dtype=d.dtype)], dim=1)
            g = torch.cat([g, torch.zeros(nb, pad_n, dtype=g.dtype)], dim=1)
            a = torch.cat([a, torch.full((nb, pad_n), -10000.0, dtype=a.dtype)], dim=1)
        domain_chunks.append(d)
        global_chunks.append(g)
        age_chunks.append(a)
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  chunk {start}-{end} done (T={sub_out.domain_ids.shape[1]})")

    return (
        torch.cat(domain_chunks, dim=0),
        torch.cat(global_chunks, dim=0),
        torch.cat(age_chunks, dim=0),
    )


def write_output(model, domain_ids, global_token_ids, ages, output_dir, tokens_root, synth_id_offset):
    output_dir = Path(output_dir)
    pad_int = model.domain_to_int["padding"]
    B = domain_ids.shape[0]
    synth_ids = torch.arange(synth_id_offset, synth_id_offset + B)

    output_domains = ("sex", "diseases", "death")
    if "lifestyle" in model.domain_to_int:
        output_domains = ("sex", "lifestyle", "diseases", "death")

    for domain in output_domains:
        d_int = model.domain_to_int[domain]
        d_off = model.domain_offsets[d_int]
        mask = domain_ids == d_int

        rows = []
        for b in range(B):
            local_ids = (global_token_ids[b][mask[b]] - d_off).tolist()
            row_ages = ages[b][mask[b]].tolist()
            rows.extend(
                {"subject_id": int(synth_ids[b]), "age": age, "token_id": tid}
                for age, tid in zip(row_ages, local_ids)
            )
        df = pd.DataFrame(rows, columns=["subject_id", "age", "token_id"])

        dst = output_dir / domain
        dst.mkdir(parents=True, exist_ok=True)
        df.to_csv(dst / "tokens.csv", index=False)
        import shutil
        shutil.copyfile(tokens_root / domain / "tokenizer.yaml", dst / "tokenizer.yaml")
        print(f"{domain:10s} rows={len(df):7d}  subjects={df['subject_id'].nunique() if len(df) else 0}")

    return synth_ids.tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_id", required=True, help="MLflow run ID to load the model from")
    p.add_argument("--n_subjects", type=int, default=1000)
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--max_age_years", type=float, default=85.0)
    p.add_argument("--termination_domain", default="death")
    p.add_argument("--chunk_size", type=int, default=150, help="Subjects per generate() call (GPU memory)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mlruns_root", default="train_scripts/mlruns")
    p.add_argument("--device", default=None, help="Defaults to cuda if available, else cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synth_id_offset", type=int, default=9_000_000,
                    help="Synthetic subject_ids start here, to stay clear of real UKB IDs")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, domain_cfg, params, metadata, tokens_root = load_model_and_domains(
        args.run_id, DELPHI_DIR / args.mlruns_root
    )
    print(f"Model: epoch={metadata['epoch']} val_loss={metadata['val_loss']:.4f}")

    subject_ids = (
        list(metadata["train_ids"]) + list(metadata["valid_ids"]) + list(metadata["test_ids"])
    )[: args.n_subjects]

    cond_batch, subjects_loaded, cond_len = build_conditioning_batch(
        model, domain_cfg, subject_ids
    )
    print(f"Conditioning on {len(subjects_loaded)} real subjects' sex + genetic_pcs "
          f"(pooled across train+val+test)")

    torch.manual_seed(args.seed)
    domain_ids, global_token_ids, ages = generate_chunked(
        model, cond_batch, cond_len, args.max_new_tokens, args.max_age_years * 365.25,
        args.termination_domain, args.chunk_size, device,
    )

    synth_ids = write_output(
        model, domain_ids, global_token_ids, ages, args.output_dir, tokens_root,
        args.synth_id_offset,
    )

    meta_out = {
        "source_run_id": args.run_id,
        "source_epoch": metadata["epoch"],
        "source_val_loss": metadata["val_loss"],
        "conditioning_split": "all",
        "n_subjects_requested": args.n_subjects,
        "n_subjects_generated": len(subjects_loaded),
        "max_new_tokens": args.max_new_tokens,
        "max_age_years": args.max_age_years,
        "termination_domain": args.termination_domain,
        "seed": args.seed,
        "synth_subject_id_range": [synth_ids[0], synth_ids[-1]],
        "conditioning_real_subject_ids": subjects_loaded,
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "generation_metadata.json", "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"\nWrote synthetic subjects to {args.output_dir}")


if __name__ == "__main__":
    main()
