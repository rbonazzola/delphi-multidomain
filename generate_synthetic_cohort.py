# %%
"""
Generate synthetic cohorts from a set of CV-fold models (DelphiMultiStream).

Each fold model generates one synthetic "twin" for every real subject of the full
cohort (train + val + test), so N fold models give N synthetic replicates of the
cohort, written to <output_dir>/fold{k}/. A synthetic subject keeps the same
synthetic_id across folds, so replicates can be compared subject by subject.
Note that most conditioning subjects are in each model's training set;
synthetic_to_real.csv records which split each real subject was in for that model.

Conditioning (copied from the real subject):
  - every non-predicted domain of the model: sex, genetic PCs, HLA tokens, lifestyle, ...
  - at_birth domains (sex, PCs, HLA) form the fixed prefix at age 0;
  - the others (lifestyle) are inserted into the timeline at their real ages
    (see delphi/generation.py::generate_conditioned).
Generated (sampled autoregressively): all predicted domains -- diseases, death, and
no_event if the run trained with `no_event.predict=True` (no_event tokens are kept as
context during sampling but not written out).

By default each synthetic subject is forced to survive until their last real lifestyle
token (real UKB participants were alive at assessment), and is censored at the real
subject's age at --date_cutoff (same follow-up window as the real cohort); both can be
switched off.

Output: each <output_dir>/fold{k}/ has the same layout as data/transforms, so it can be
used directly as `train_multi_stream.py --data_root`: tokens/<domain path>/ (per-locus HLA
subdomains are written back to tokens/hla_alleles/ with their original token ids), each
with tokens.csv + tokenizer.yaml (+ token_metadata files if present); subject_lists/
(the real CV folds mapped to synthetic ids); year_and_month_of_birth.txt; plus:
  - synthetic_to_real.csv: synthetic_id, real_subject_id, real_subject_split
  - generation_metadata.json

Usage (DelphiMultiStream runs trained with --dcfg no_event.predict=True):

    python generate_synthetic_cohort.py \\
        --experiment_name multistream_hla_esm2raw_totaldepth12_nodrb345 \\
        --dcfg no_event.predict=True \\
        --output_dir data/synthetic/hla1trunk11_nodrb345

Smoke test: add `--n_subjects 64`.

Sharded over several GPUs (single run), then merged:

    sbatch --array=1-8 --gres=gpu:1 ... --wrap="python generate_synthetic_cohort.py \
        --run_ids <RUN_ID> --dcfg no_event.predict=True --n_shards 8 --output_dir <DIR>"
    sbatch --dependency=afterok:<ARRAY_JOB_ID> ... --wrap="python generate_synthetic_cohort.py \
        --merge --output_dir <DIR>"

"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DELPHI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DELPHI_DIR))

from data.dataset import AgeSampler, DelphiBatch, DelphiCollateFn, DelphiDataset
from delphi.generation import PADDING_AGE, generate_conditioned
from utils.ckpt_utils import strip_compiled_prefix
from utils.mlflow_utils import get_checkpoint_path, setup_mlflow
from utils.utils import apply_domain_overrides, load_domain_config

TOKENS_ROOT = DELPHI_DIR / "data" / "transforms" / "tokens"
DAYS_PER_YEAR = 365.25


# ── Model loading ──────────────────────────────────────────────────────────────

def load_run(run_id: str, dcfg_overrides: list[str], device: str):
    """Rebuild a DelphiMultiStream (or Delphi) model from an MLflow run + its best checkpoint."""
    import mlflow

    client = mlflow.tracking.MlflowClient()
    params = client.get_run(run_id).data.params

    ckpt = torch.load(get_checkpoint_path(run_id), map_location="cpu", weights_only=False)
    metadata = ckpt["metadata"]

    # The `domains` param is truncated by MLflow on ESM2 runs -> rebuild from the logged yaml.
    yaml_artifacts = [a.path for a in client.list_artifacts(run_id) if a.path.endswith(".yaml")]
    assert len(yaml_artifacts) == 1, f"{run_id}: expected one domain config yaml artifact, got {yaml_artifacts}"
    yaml_path = DELPHI_DIR / "config" / Path(yaml_artifacts[0]).name
    assert yaml_path.exists(), f"expected local config at {yaml_path}"
    all_cfg = load_domain_config(yaml_path, TOKENS_ROOT)

    if "arch" in params:
        import re
        from delphi.multi_stream_model import DelphiMultiStream, DelphiMultiStreamConfig

        names = {n.strip() for g in re.findall(r"\[([^\]]+)\]", params["arch"]) for n in g.split(",")}
        domain_cfg = {k: v for k, v in all_cfg.items() if k in names or k in ("padding", "no_event")}
        if dcfg_overrides:
            apply_domain_overrides(domain_cfg, dcfg_overrides)
        config = DelphiMultiStreamConfig(
            domains=domain_cfg,
            token_dropout=float(params.get("token_dropout", 0.1)),
            block_size=int(params.get("block_size", 128)),
            seed=int(params.get("seed", 142)),
        )
        model = DelphiMultiStream.from_scheme_string(
            arch_str=params["arch"], attention_scheme=params["attention_scheme"], config=config,
        )
    else:
        raise NotImplementedError(
            "Single-trunk Delphi runs: use generate_synthetic_subjects.py, or extend load_run()."
        )

    missing, unexpected = model.load_state_dict(strip_compiled_prefix(ckpt["state_dict"]), strict=False)
    assert not missing and not unexpected, f"{run_id}: state_dict mismatch missing={missing} unexpected={unexpected}"
    model = model.to(device).eval()
    return model, domain_cfg, params, metadata


# ── Conditioning ───────────────────────────────────────────────────────────────

def conditioning_domains(domain_cfg) -> dict:
    return {
        k: v for k, v in domain_cfg.items()
        if k not in ("padding", "no_event") and not v.predict
    }


def build_conditioning_dataset(model, domain_cfg, subject_ids):
    cond_cfg = conditioning_domains(domain_cfg)
    continuous = {k: v.n_latent_tokens or 1 for k, v in cond_cfg.items() if v.type == "continuous"}
    ds = DelphiDataset(
        root=str(DELPHI_DIR / "data" / "transforms"), domains_cfg=cond_cfg,
        domain_to_int=model.domain_to_int, block_size=128, subjects=list(subject_ids),
        required_domains=["sex"], continuous_domains=continuous, age_domains=[],
    )
    collate = DelphiCollateFn(
        age_sampler=AgeSampler(insertion_mode="random", token_rate=1e6, seed=0),
        block_size=128, domain_to_int=model.domain_to_int, domain_offsets=model.domain_offsets,
        padding_domain_id=model.domain_to_int["padding"],
        no_event_domain_id=model.domain_to_int["no_event"],
        continuous_domains=continuous, training=False,
    )
    return ds, collate, continuous


def split_prefix_and_pending(batch: DelphiBatch, model, domain_cfg, continuous):
    """
    From a collated conditioning batch, build:
      - prefix: at_birth tokens only, right-aligned (padding in front), continuous
        positions recomputed for the new layout;
      - pending: the remaining real tokens (e.g. lifestyle), left-aligned, sorted by
        age, +inf-padded.
    """
    pad_int, ne_int = model.domain_to_int["padding"], model.domain_to_int["no_event"]
    at_birth_ints = torch.tensor(
        [model.domain_to_int[k] for k, v in conditioning_domains(domain_cfg).items() if v.at_birth]
    )
    real = (batch.domain_ids != pad_int) & (batch.domain_ids != ne_int)
    is_prefix = real & torch.isin(batch.domain_ids, at_birth_ints)
    is_pending = real & ~is_prefix

    B = batch.domain_ids.shape[0]
    # +1: column 0 is always padding, where subjects without a continuous domain (e.g. no
    # genetic PCs) get their projector output scattered -- same as DelphiCollateFn.
    Tp = int(is_prefix.sum(1).max()) + 1
    Lp = max(int(is_pending.sum(1).max()), 1)

    g = torch.zeros(B, Tp, dtype=torch.long)
    d = torch.full((B, Tp), pad_int, dtype=torch.long)
    a = torch.full((B, Tp), PADDING_AGE)
    pg = torch.zeros(B, Lp, dtype=torch.long)
    pd_ = torch.full((B, Lp), pad_int, dtype=torch.long)
    pa = torch.full((B, Lp), float("inf"))
    for b in range(B):
        idx = is_prefix[b].nonzero(as_tuple=True)[0]
        n = idx.numel()
        g[b, Tp - n:] = batch.global_token_ids[b, idx]
        d[b, Tp - n:] = batch.domain_ids[b, idx]
        a[b, Tp - n:] = batch.ages[b, idx]

        idx = is_pending[b].nonzero(as_tuple=True)[0]
        idx = idx[torch.argsort(batch.ages[b, idx], stable=True)]
        n = idx.numel()
        pg[b, :n] = batch.global_token_ids[b, idx]
        pd_[b, :n] = batch.domain_ids[b, idx]
        pa[b, :n] = batch.ages[b, idx]

    positions = {}
    for name, n_latent in continuous.items():
        m = d == model.domain_to_int[name]
        pos = torch.zeros(B, n_latent, dtype=torch.long)
        for b in range(B):
            found = m[b].nonzero(as_tuple=True)[0][:n_latent]
            pos[b, :found.numel()] = found
        positions[name] = pos

    prefix = DelphiBatch(
        global_token_ids=g, domain_ids=d, ages=a, subject_ids=batch.subject_ids,
        continuous_data=batch.continuous_data, continuous_positions=positions, eval_mask=None,
    )
    return prefix, pg, pd_, pa


def real_cutoff_ages(subject_ids, birth_dates_file, date_cutoff) -> np.ndarray:
    """Age (days) of each subject at date_cutoff; +inf when the birth date is unknown."""
    births = pd.read_csv(birth_dates_file, sep="\t").set_index("eid").reindex(subject_ids)
    out = np.full(len(subject_ids), np.inf)
    ok = births["year"].notna().values
    born = pd.to_datetime({"year": births["year"][ok].astype(int), "month": births["month"][ok].astype(int), "day": 1})
    out[ok] = (pd.Timestamp(date_cutoff) - born).dt.days.values
    return out


# ── Output ─────────────────────────────────────────────────────────────────────

def sequences_to_rows(out: DelphiBatch, model, domain_cfg, ds, synth_ids) -> dict:
    """Flatten generated sequences into {domain_path: DataFrame(subject_id, age, token_id)}."""
    int_to_domain = {v: k for k, v in model.domain_to_int.items()}
    d, g, a = out.domain_ids.cpu().numpy(), out.global_token_ids.cpu().numpy(), out.ages.cpu().numpy()
    b_idx, t_idx = np.nonzero(a >= 0)
    frames = {}
    for d_int in np.unique(d[b_idx, t_idx]):
        dname = int_to_domain[int(d_int)]
        cfg = domain_cfg[dname]
        if dname in ("padding", "no_event") or cfg.type == "continuous":
            continue
        sel = d[b_idx, t_idx] == d_int
        bb, tt = b_idx[sel], t_idx[sel]
        local = g[bb, tt] - model.domain_offsets[int(d_int)]
        new_to_old = getattr(ds.domains.get(dname), "_new_to_old", None) if dname in ds.domains else None
        if new_to_old:
            local = np.vectorize(new_to_old.__getitem__)(local)
        path = Path(str(cfg.path)).name
        frames.setdefault(path, []).append(pd.DataFrame({
            "subject_id": np.asarray(synth_ids)[bb],
            "age": np.rint(a[bb, tt]).astype(np.int64),
            "token_id": local.astype(np.int64),
        }))
    return {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}


def write_output(output_dir: Path, frames: dict, pcs_df: pd.DataFrame | None, domain_paths: set):
    for path in sorted(domain_paths):
        dst = output_dir / "tokens" / path
        dst.mkdir(parents=True, exist_ok=True)
        if path in frames:
            df = frames[path].sort_values(["subject_id", "age", "token_id"], kind="stable")
        elif path == "genetic_pcs" and pcs_df is not None:
            df = pcs_df
        else:
            df = pd.DataFrame(columns=["subject_id", "age", "token_id"])
        df.to_csv(dst / "tokens.csv", index=False)
        for f in ("tokenizer.yaml", "token_metadata.csv", "token_metadata.tsv"):
            if (TOKENS_ROOT / path / f).exists():
                shutil.copyfile(TOKENS_ROOT / path / f, dst / f)
        print(f"  {path:14s} rows={len(df):9d}  subjects={df['subject_id'].nunique() if len(df) else 0}")


def write_data_root_extras(fold_dir: Path, mapping: pd.DataFrame, birth_dates_file: str | None):
    """
    Make <fold_dir> usable as a training --data_root (same layout as data/transforms):
      - subject_lists/subset{i}of10.csv: the real CV folds mapped to synthetic ids, so a
        given --test_fold selects the twins of exactly the same real subjects;
      - year_and_month_of_birth.txt: the real subjects' birth dates under synthetic ids,
        so --date_cutoff/--birth_dates_file behave as for the real cohort.
    """
    real2synth = dict(zip(mapping["real_subject_id"], mapping["synthetic_id"]))
    real_lists = DELPHI_DIR / "data" / "transforms" / "subject_lists"
    out_lists = fold_dir / "subject_lists"
    out_lists.mkdir(parents=True, exist_ok=True)
    for f in sorted(real_lists.glob("subset*of10.csv")):
        ids = pd.read_csv(f, header=None)[0]
        ids.map(real2synth).dropna().astype(np.int64).to_csv(out_lists / f.name, index=False, header=False)
    if birth_dates_file:
        births = pd.read_csv(DELPHI_DIR / birth_dates_file, sep="\t", dtype={"month": str, "year_month": str})
        births = births[births["eid"].isin(real2synth)].copy()
        births["eid"] = births["eid"].map(real2synth)
        births.sort_values("eid").to_csv(fold_dir / "year_and_month_of_birth.txt", sep="\t", index=False)
    print(f"  wrote {out_lists} and birth dates for {len(mapping)} synthetic subjects")


# ── Main ───────────────────────────────────────────────────────────────────────

def resolve_run_ids(args) -> list[str]:
    if args.run_ids:
        return args.run_ids
    import mlflow
    runs = mlflow.search_runs(experiment_names=[args.experiment_name], filter_string="attributes.status = 'FINISHED'")
    runs = runs.sort_values("params.test_fold")
    print(runs[["run_id", "params.test_fold", "tags.mlflow.runName"]].to_string(index=False))
    return runs["run_id"].tolist()


def generate_for_run(model, domain_cfg, ds, collate, continuous, sids, synth_ids_all, max_age_all, args, device):
    """Generate one synthetic twin per conditioning subject with one model. Returns {path: DataFrame}."""
    import time
    frames: dict[str, list] = {}
    t0 = time.time()
    for start in range(0, len(sids), args.chunk_size):
        idx = range(start, min(start + args.chunk_size, len(sids)))
        batch = collate([ds[i] for i in idx])
        prefix, pg, pdom, pa = split_prefix_and_pending(batch, model, domain_cfg, continuous)
        block_until = None
        if not args.no_survival_conditioning:
            block_until = torch.where(torch.isinf(pa), torch.zeros_like(pa), pa).max(dim=1).values
        out = generate_conditioned(
            model, prefix, pg, pdom, pa,
            max_new_tokens=args.max_new_tokens,
            max_age=torch.as_tensor(max_age_all[idx.start:idx.stop], dtype=torch.float32),
            termination_domain="death",
            block_termination_until=block_until,
        )
        synth_ids = synth_ids_all[idx.start:idx.stop]
        for path, df in sequences_to_rows(out, model, domain_cfg, ds, synth_ids).items():
            frames.setdefault(path, []).append(df)
        n_steps = out.domain_ids.shape[1] - prefix.domain_ids.shape[1]
        mem = f", peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if device == "cuda" else ""
        print(f"    chunk {start}-{idx.stop}/{len(sids)}: {n_steps} steps, {time.time() - t0:.0f}s elapsed{mem}", flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()
    return {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}


def merge_shards(fold_dir: Path):
    """Concatenate <fold_dir>/shards/shard*/ into <fold_dir>/ (tokens, mapping, metadata)."""
    shard_dirs = sorted((fold_dir / "shards").glob("shard*"))
    assert shard_dirs, f"no shards under {fold_dir / 'shards'}"
    metas = [json.loads((d / "generation_metadata.json").read_text()) for d in shard_dirs]
    n_shards = metas[0]["n_shards"]
    got = sorted(m["shard_index"] for m in metas)
    assert got == list(range(n_shards)), f"expected shards 0..{n_shards - 1}, found {got}"
    assert len({m["run_id"] for m in metas}) == 1, "shards come from different runs"

    domain_paths = sorted({p.name for d in shard_dirs for p in (d / "tokens").iterdir() if p.is_dir()})
    for path in domain_paths:
        df = pd.concat([pd.read_csv(d / "tokens" / path / "tokens.csv") for d in shard_dirs
                        if (d / "tokens" / path / "tokens.csv").exists()],
                       ignore_index=True)
        sort_cols = ["subject_id", "token_id"] if "value" in df.columns else ["subject_id", "age", "token_id"]
        dst = fold_dir / "tokens" / path
        dst.mkdir(parents=True, exist_ok=True)
        df.sort_values(sort_cols, kind="stable").to_csv(dst / "tokens.csv", index=False)
        for f in (shard_dirs[0] / "tokens" / path).iterdir():
            if f.name != "tokens.csv":
                shutil.copyfile(f, dst / f.name)
        print(f"  {path:14s} rows={len(df):9d}  subjects={df['subject_id'].nunique() if len(df) else 0}")

    mapping = pd.concat([pd.read_csv(d / "synthetic_to_real.csv") for d in shard_dirs], ignore_index=True)
    assert mapping["synthetic_id"].is_unique and mapping["real_subject_id"].is_unique
    mapping.sort_values("synthetic_id").to_csv(fold_dir / "synthetic_to_real.csv", index=False)
    write_data_root_extras(fold_dir, mapping, metas[0].get("birth_dates_file"))

    meta = {k: v for k, v in metas[0].items() if k not in ("shard_index", "seed", "synth_id_range")}
    meta.update(n_subjects=int(len(mapping)), seeds=[m["seed"] for m in metas],
                synth_id_range=[int(mapping["synthetic_id"].min()), int(mapping["synthetic_id"].max())])
    (fold_dir / "generation_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Merged {n_shards} shards -> {fold_dir} ({len(mapping)} subjects)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--experiment_name", help="Use every FINISHED run of this MLflow experiment (one per fold)")
    src.add_argument("--run_ids", nargs="+", help="Explicit run ids (one per fold)")
    p.add_argument("--dcfg", nargs="+", default=[],
                   help="Domain overrides used at training time (not logged by train_multi_stream.py), "
                        "e.g. no_event.predict=True")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_subjects", type=int, default=None, help="Default: the whole cohort")
    p.add_argument("--n_shards", type=int, default=1,
                   help="Split the cohort into this many shards (one GPU job each; single run only). "
                        "Shards are written to <output_dir>/fold{k}/shards/ -- combine with --merge.")
    p.add_argument("--shard_index", type=int, default=None,
                   help="0-based shard to generate. Default: $SLURM_ARRAY_TASK_ID - 1 (array --array=1-N)")
    p.add_argument("--merge", action="store_true",
                   help="Merge <output_dir>/fold*/shards/ into <output_dir>/fold*/ and exit (no GPU needed)")
    p.add_argument("--chunk_size", type=int, default=512, help="Subjects per generation batch (GPU memory)")
    p.add_argument("--max_new_tokens", type=int, default=256, help="Max steps per subject (incl. inserted lifestyle)")
    p.add_argument("--max_age_years", type=float, default=85.0)
    p.add_argument("--no_censor_at_real_cutoff", action="store_true",
                   help="Don't stop each subject at the real subject's age at --date_cutoff (only at --max_age_years)")
    p.add_argument("--date_cutoff", default="2020-01-01")
    p.add_argument("--birth_dates_file", default="data/datasets/year_and_month_of_birth.txt")
    p.add_argument("--no_survival_conditioning", action="store_true",
                   help="Allow death before the subject's last real lifestyle token")
    p.add_argument("--synth_id_offset", type=int, default=9_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    if args.merge:
        fold_dirs = sorted(d for d in output_dir.glob("fold*") if (d / "shards").is_dir())
        assert fold_dirs, f"no fold*/shards/ under {output_dir}"
        for fold_dir in fold_dirs:
            merge_shards(fold_dir)
        return
    if not (args.experiment_name or args.run_ids):
        p.error("one of --experiment_name / --run_ids is required (unless --merge)")

    sharded = args.n_shards > 1
    if sharded and args.shard_index is None:
        assert "SLURM_ARRAY_TASK_ID" in os.environ, "--n_shards > 1 needs --shard_index or a SLURM array task"
        args.shard_index = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
    shard_index = args.shard_index or 0
    assert 0 <= shard_index < args.n_shards, f"shard_index {shard_index} out of range for n_shards={args.n_shards}"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_mlflow()

    run_ids = resolve_run_ids(args)
    assert run_ids, "no runs found"
    assert not sharded or len(run_ids) == 1, "--n_shards > 1 supports a single run (pass one --run_ids)"

    ds = collate = continuous = sids = None
    ref_layout = None
    for fold_i, run_id in enumerate(run_ids, start=1):
        seed = args.seed + 1000 * fold_i + shard_index
        torch.manual_seed(seed)
        model, domain_cfg, params, metadata = load_run(run_id, args.dcfg, device)
        layout = (model.domain_to_int, model.domain_offsets)
        assert ref_layout is None or layout == ref_layout, "runs have different vocab layouts"
        fold = params.get("test_fold", fold_i)
        print(f"\n=== fold {fold}  run {run_id}  epoch={metadata['epoch']}  val_loss={metadata['val_loss']:.4f}"
              + (f"  shard {shard_index + 1}/{args.n_shards}" if sharded else ""))
        print(f"    predicted: {list(model.predicted_domains)}")
        print(f"    conditioning: {list(conditioning_domains(domain_cfg))}")

        if ds is None:
            # Whole cohort (or this shard of it), built once and shared by every fold model.
            ref_layout = layout
            all_ids = sorted(set(metadata["train_ids"]) | set(metadata["valid_ids"]) | set(metadata["test_ids"]))
            if args.n_subjects:
                all_ids = all_ids[: args.n_subjects]
            # Synthetic ids come from the position in the full sorted cohort, so they are
            # identical across shards and folds.
            global_synth_id = {r: args.synth_id_offset + i for i, r in enumerate(all_ids)}
            shard_ids = np.array_split(np.asarray(all_ids), args.n_shards)[shard_index].tolist()
            ds, collate, continuous = build_conditioning_dataset(model, domain_cfg, shard_ids)
            sids = ds._subject_ids.tolist()
            synth_ids_all = np.array([global_synth_id[s] for s in sids])
            print(f"    conditioning subjects: {len(sids)} (of {len(all_ids)} in the cohort)")
            max_age_all = np.full(len(sids), args.max_age_years * DAYS_PER_YEAR)
            if not args.no_censor_at_real_cutoff:
                max_age_all = np.minimum(max_age_all, real_cutoff_ages(sids, args.birth_dates_file, args.date_cutoff))
            real2synth = dict(zip(sids, synth_ids_all.tolist()))
            domain_paths = {Path(str(c.path)).name for d, c in domain_cfg.items()
                            if d not in ("padding", "no_event") and c.path}
            pcs_df = None
            if "genetic_pcs" in domain_cfg:
                pcs_df = pd.read_csv(TOKENS_ROOT / Path(str(domain_cfg["genetic_pcs"].path)).name / "tokens.csv")
                pcs_df = pcs_df[pcs_df["subject_id"].isin(real2synth)].copy()
                pcs_df["subject_id"] = pcs_df["subject_id"].map(real2synth)
                pcs_df = pcs_df.sort_values(["subject_id", "token_id"])

        frames = generate_for_run(model, domain_cfg, ds, collate, continuous, sids, synth_ids_all,
                                  max_age_all, args, device)

        fold_dir = output_dir / f"fold{fold}"
        out_dir = fold_dir / "shards" / f"shard{shard_index:03d}" if sharded else fold_dir
        print(f"  writing {out_dir}")
        write_output(out_dir, frames, pcs_df, domain_paths)

        split = {s: "train" for s in metadata["train_ids"]}
        split.update({s: "val" for s in metadata["valid_ids"]})
        split.update({s: "test" for s in metadata["test_ids"]})
        pd.DataFrame({
            "synthetic_id": synth_ids_all,
            "real_subject_id": sids,
            "real_subject_split": [split.get(s, "none") for s in sids],
        }).to_csv(out_dir / "synthetic_to_real.csv", index=False)
        if not sharded:
            write_data_root_extras(out_dir, pd.read_csv(out_dir / "synthetic_to_real.csv"), args.birth_dates_file)
        with open(out_dir / "generation_metadata.json", "w") as f:
            json.dump({
                "run_id": run_id,
                "test_fold": fold,
                "experiment_name": args.experiment_name,
                "dcfg": args.dcfg,
                "n_subjects": len(sids),
                "n_shards": args.n_shards,
                "shard_index": shard_index,
                "max_new_tokens": args.max_new_tokens,
                "max_age_years": args.max_age_years,
                "censor_at_real_cutoff": not args.no_censor_at_real_cutoff,
                "date_cutoff": args.date_cutoff,
                "birth_dates_file": args.birth_dates_file,
                "survival_conditioning": not args.no_survival_conditioning,
                "seed": seed,
                "synth_id_range": [int(synth_ids_all.min()), int(synth_ids_all.max())] if len(sids) else None,
            }, f, indent=2)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
