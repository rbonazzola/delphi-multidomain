"""
Utilities for reconstructing a trained Delphi model (and optional dataloaders)
from an MLflow run ID.
"""
from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import DataLoader
from data.dataset import FlexibleDataLoader, BatchSizeScheduler, DataModule

DELPHI_DIR = Path(__file__).resolve().parent.parent

AUTO_BLOCK_SIZE = 512  # used when a run was trained with block_size="auto"

_SPLIT_TO_METADATA_KEY = {
    "train": "train_ids",
    "val":   "valid_ids",
    "test":  "test_ids",
}


def reconstruct_from_run(
    run_id: str,
    split: Union[str, list[str]] = "test",
    block_size: int | None = None,
    batch_size: int = 512,
    num_workers: int = 4,
    date_cutoff: str | None = None,
    birth_dates_file: str | None = None,
    device: str | None = None,
    tokens_path: Union[str, Path, None] = None,
    domain_config_yaml: Union[str, Path, None] = None,
    subjects: Union[str, Path, list, None] = None,
) -> tuple:
    """
    Reconstruct a trained Delphi model and dataloaders from an MLflow run.

    Parameters
    ----------
    run_id : MLflow run ID.
    split  : Which data split(s) to load — "train", "val", or "test", or a list
             of those strings. Always returns a dict keyed by split name.
    block_size : Override the stored block size. Required when the run used
                 block_size="auto" (falls back to AUTO_BLOCK_SIZE with a warning).
    batch_size, num_workers : DataLoader settings.
    date_cutoff : ISO date "YYYY-MM-DD" forwarded to DelphiDataset for eval_mask.
    birth_dates_file : Path to TSV (eid, year, month) used with date_cutoff.
    device : Target device; defaults to $DEVICE env var or auto-detect.
    tokens_path : If given, rewrite every domain's data path to
                  `tokens_path/<domain_name>` instead of wherever the run's
                  own config points -- e.g. to evaluate on a different cohort
                  (synthetic subjects, a held-out dataset) sharing the same
                  vocabulary/tokenizer.yaml layout.
    subjects : If given, use these subject_ids instead of the run's own
               stored split (checkpoint metadata's train_ids/valid_ids/
               test_ids) -- a list of ints, or a path to an ids file (read via
               utils.utils.read_ids). Applied to every split in `split`.

    Returns
    -------
    model        : Delphi — loaded, on device, in eval mode.
    loaders      : dict[str, DataLoader] — one entry per requested split.
    run_params   : dict — raw MLflow params.
    """
    from delphi.model import Delphi, DelphiConfig
    from data.dataset import DelphiDataset, DelphiCollateFn, AgeSampler
    from utils.mlflow_utils import get_checkpoint_path, load_run_params, parse_domains_param
    from utils.ckpt_utils import strip_compiled_prefix

    if device is None:
        device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    splits: list[str] = [split] if isinstance(split, str) else list(split)
    for s in splits:
        if s not in _SPLIT_TO_METADATA_KEY:
            raise ValueError(f"split must be one of {list(_SPLIT_TO_METADATA_KEY)}, got {s!r}")

    params = load_run_params(run_id)

    ckpt_path = get_checkpoint_path(run_id)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    metadata = ckpt.get("metadata", {})

    if subjects is not None:
        if isinstance(subjects, (str, Path)):
            from utils.utils import read_ids
            subject_override = sorted(read_ids(subjects))
        else:
            subject_override = list(subjects)
    else:
        subject_override = None
        for s in splits:
            key = _SPLIT_TO_METADATA_KEY[s]
            if metadata.get(key) is None:
                raise ValueError(f"Checkpoint metadata missing '{key}' (needed for split={s!r}).")

    if domain_config_yaml is not None:
        from utils.utils import load_domain_config
        tokens_root = Path(tokens_path).resolve() if tokens_path else DELPHI_DIR / "data" / "transforms" / "tokens"
        domain_cfg = load_domain_config(domain_config_yaml, tokens_path=tokens_root)
    else:
        domain_cfg = parse_domains_param(params["domains"], run_id=run_id)

    if tokens_path is not None:
        # Must be absolute: DelphiDataset only uses a domain's `path` as-is when
        # it's absolute, otherwise it re-joins it under root/"tokens"/... -- a
        # relative tokens_path here would silently double that prefix.
        tokens_root = Path(tokens_path).resolve()
        for dname, dcfg in domain_cfg.items():
            if dname in ("padding", "no_event") or not getattr(dcfg, "path", None):
                continue
            dcfg.path = str(tokens_root / Path(str(dcfg.path)).name)

    stored_block_size = params.get("block_size", "96")
    if block_size is not None:
        bs = block_size
    elif stored_block_size == "auto":
        logging.warning(
            "Run was trained with block_size='auto'; using AUTO_BLOCK_SIZE=%d "
            "as the fixed block size.", AUTO_BLOCK_SIZE
        )
        bs = AUTO_BLOCK_SIZE
    else:
        bs = int(stored_block_size)

    attn_scheme = params.get("attention_scheme", ["all:causal(mask_ties=True)"])
    n_layer = int(params.get("n_layer", 12))
    if isinstance(attn_scheme, str):
        attn_scheme = [attn_scheme]
    if len(attn_scheme) == 1:
        attn_scheme = n_layer * attn_scheme

    delphi_config = DelphiConfig(
        n_embd=int(params.get("n_embd", 120)),
        n_layer=n_layer,
        n_head=int(params.get("n_head", 6)),
        domains=domain_cfg,
        attention_scheme=attn_scheme,
        block_size=bs,
        token_dropout=float(params.get("token_dropout", 0.1)),
        no_event_token_rate=float(params.get("no_event_token_rate", 2.0)),
        no_event_token_insertion_mode=params.get("no_event_token_insertion_mode", "random"),
        seed=int(params.get("seed", 42)),
    )

    model = Delphi(delphi_config)
    state_dict = strip_compiled_prefix(ckpt["state_dict"])
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    logging.info("Model loaded: %d parameters", sum(p.numel() for p in model.parameters()))

    continuous_domains = {
        dname: cfg.n_latent_tokens or 1
        for dname, cfg in domain_cfg.items()
        if cfg.type == "continuous"
    }

    root_path = DELPHI_DIR / "data" / "transforms"

    age_sampler = AgeSampler(
        insertion_mode=delphi_config.no_event_token_insertion_mode,
        token_rate=delphi_config.no_event_token_rate,
        seed=delphi_config.seed,
    )

    collate = DelphiCollateFn(
        age_sampler=age_sampler,
        block_size=bs,
        domain_to_int=model.domain_to_int,
        domain_offsets=model.domain_offsets,
        padding_domain_id=model.domain_to_int["padding"],
        no_event_domain_id=model.domain_to_int["no_event"],
        continuous_domains=continuous_domains,
        domain_dropout={},
        training=False,
    )

    loaders: dict[str, DataLoader] = {}
    for s in splits:
        split_subjects = subject_override if subject_override is not None else metadata[_SPLIT_TO_METADATA_KEY[s]]
        dataset = DelphiDataset(
            root=str(root_path),
            domains_cfg=domain_cfg,
            domain_to_int=model.domain_to_int,
            block_size=bs,
            subjects=split_subjects,
            exclusions=[],
            required_domains=["diseases"],
            no_event_token_rate=delphi_config.no_event_token_rate,
            no_event_insertion_mode=delphi_config.no_event_token_insertion_mode,
            continuous_domains=continuous_domains,
            age_domains=["diseases", "death"],
            date_cutoff=date_cutoff,
            birth_dates_file=birth_dates_file,
        )
        loaders[s] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(s == "train"),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate,
        )

    return model, loaders, params


def reconstruct_model(run_id: str):
    """
    Lightweight model-only reconstruction from an MLflow run.
    Infers config from the state dict; applies legacy weight migrations.

    Returns: model, test_ids, ckpt_path, params
    """
    from delphi.model import Delphi, DelphiConfig
    from utils.mlflow_utils import load_run_params, load_checkpoint, parse_domains_param
    from utils.ckpt_utils import (
        infer_delphi_config_from_state_dict,
        migrate_legacy_state_dict,
        migrate_domain_embed_to_global_embed,
        strip_compiled_prefix,
    )

    params = load_run_params(run_id)
    attn_scheme = params["attention_scheme"]

    ckpt, ckpt_path = load_checkpoint(run_id)
    weights = strip_compiled_prefix(ckpt["state_dict"])
    weights = migrate_legacy_state_dict(weights)
    test_ids = ckpt["metadata"]["test_ids"]

    cfg = infer_delphi_config_from_state_dict(weights)
    domain_cfg = parse_domains_param(params["domains"], run_id=run_id)

    tokens_dir = DELPHI_DIR / "data" / "transforms" / "tokens"
    for dname, dcfg in domain_cfg.items():
        if dname in ("padding", "no_event") or not getattr(dcfg, "path", None):
            continue
        dcfg.path = str(tokens_dir / Path(str(dcfg.path)).name)

    block_size = int(params.get("block_size", cfg.get("block_size", 64)))
    delphi_cfg = DelphiConfig(
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        token_dropout=0.1,
        domains=domain_cfg,
        attention_scheme=attn_scheme,
        block_size=block_size,
    )

    model = Delphi(delphi_cfg)
    weights = migrate_domain_embed_to_global_embed(weights, model)
    model.load_state_dict(weights, strict=True)
    model.to("cpu")
    model.eval()

    return model, test_ids, ckpt_path, params


def config_from_runid(runid: str):
    """
    Reconstruct model + all three dataloaders from a run for training resumption.

    Returns: model, dataloaders, optim_config,
             optimizer_state, scheduler_state,
             start_epoch, logged_params, previous_run_name
    """
    import mlflow
    from delphi.model import Delphi, DelphiConfig
    from delphi.optim import OptimConfig
    from data.dataset import DelphiDataset, DelphiCollateFn, AgeSampler
    from utils.mlflow_utils import get_checkpoint_path, parse_domains_param

    VAL_BATCH_SIZE = 256

    runinfo = mlflow.get_run(runid)

    batch_size = int(runinfo.data.params.pop("batch_size", 16))
    test_fold  = runinfo.data.params.pop("test_fold", 0)
    runinfo.data.params.pop("learning_rate", None)
    runinfo.data.params.pop("ema_alpha", None)

    bs_schedule_str = runinfo.data.params.pop("batch_size_schedule", None)
    bs_scheduler = (
        BatchSizeScheduler.from_string(bs_schedule_str)
        if bs_schedule_str and bs_schedule_str != "None"
        else None
    )

    try:
        runinfo.data.params["attention_scheme"] = ast.literal_eval(
            runinfo.data.params["attention_scheme"]
        )
    except (ValueError, SyntaxError):
        runinfo.data.params["attention_scheme"] = [runinfo.data.params["attention_scheme"]]

    runinfo.data.params["domains"] = parse_domains_param(runinfo.data.params["domains"], run_id=runid)

    for param, value in runinfo.data.params.items():
        if "drop" in param and not param.endswith("_mode"):
            runinfo.data.params[param] = float(value)
        if param in {"n_embd", "n_head", "n_layer", "block_size"}:
            runinfo.data.params[param] = int(value)
        if param in {"seed"}:
            runinfo.data.params[param] = int(value)
        if param in {"no_event_token_rate"}:
            runinfo.data.params[param] = float(value)
        if param in {"zero_inflate", "bias"}:
            runinfo.data.params[param] = runinfo.data.params[param] == "True"

    ckpt_path = get_checkpoint_path(runid)
    logging.info("Loading checkpoint: %s", ckpt_path)

    optim_config_raw = runinfo.data.params.pop("optim_config")
    if isinstance(optim_config_raw, str):
        m = re.match(r"OptimConfig\((.*)\)$", optim_config_raw, re.DOTALL)
        if m:
            optim_kwargs = dict(re.findall(r"(\w+)=([^,)]+)", m.group(1)))
            optim_kwargs = {k: ast.literal_eval(v) for k, v in optim_kwargs.items()}
            optim_config = OptimConfig(**optim_kwargs)
        else:
            raise ValueError(f"Cannot parse optim_config string: {optim_config_raw!r}")
    else:
        optim_config = optim_config_raw

    valid_keys = {f.name for f in dc_fields(DelphiConfig)}
    delphi_params = {k: v for k, v in runinfo.data.params.items() if k in valid_keys}
    delphi_cfg = DelphiConfig(**delphi_params)
    model = Delphi(delphi_cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    start_epoch = ckpt.get("metadata", {}).get("epoch", 0) + 1

    from utils.ckpt_utils import strip_compiled_prefix
    state_dict = strip_compiled_prefix(ckpt["state_dict"])
    model.load_state_dict(state_dict, strict=False)

    root_path = DELPHI_DIR / "data" / "transforms"
    continuous_domains = {
        dname: cfg.n_latent_tokens or 1
        for dname, cfg in delphi_cfg.domains.items()
        if cfg.type == "continuous"
    }

    _ds_kwargs = dict(
        root=str(root_path),
        domains_cfg=delphi_cfg.domains,
        domain_to_int=model.domain_to_int,
        block_size=delphi_cfg.block_size,
        exclusions=[],
        required_domains=["diseases"],
        no_event_token_rate=delphi_cfg.no_event_token_rate,
        no_event_insertion_mode=delphi_cfg.no_event_token_insertion_mode,
        continuous_domains=continuous_domains,
        age_domains=["diseases", "death"],
    )

    train_dataset = DelphiDataset(subjects=ckpt["metadata"]["train_ids"], **_ds_kwargs)  # type: ignore[arg-type]
    valid_dataset = DelphiDataset(subjects=ckpt["metadata"]["valid_ids"], **_ds_kwargs)  # type: ignore[arg-type]
    test_dataset  = DelphiDataset(subjects=ckpt["metadata"]["test_ids"],  **_ds_kwargs)  # type: ignore[arg-type]

    age_sampler = AgeSampler(
        insertion_mode=delphi_cfg.no_event_token_insertion_mode,
        token_rate=delphi_cfg.no_event_token_rate,
        seed=delphi_cfg.seed,
    )

    collate = DelphiCollateFn(
        age_sampler=age_sampler,
        block_size=delphi_cfg.block_size,
        domain_to_int=model.domain_to_int,
        domain_offsets=model.domain_offsets,
        padding_domain_id=model.domain_to_int["padding"],
        no_event_domain_id=model.domain_to_int["no_event"],
        continuous_domains=continuous_domains,
    )

    train_batch_size = (
        bs_scheduler.step(start_epoch).batch_size if bs_scheduler is not None else batch_size
    )
    dataloaders = DataModule(
        FlexibleDataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,  pin_memory=True, collate_fn=collate),
        DataLoader(valid_dataset,         batch_size=VAL_BATCH_SIZE,   shuffle=False, pin_memory=True, collate_fn=collate),
        DataLoader(test_dataset,          batch_size=VAL_BATCH_SIZE,   shuffle=False, pin_memory=True, collate_fn=collate),
    )

    previous_run_name = runinfo.data.tags.get("mlflow.runName", None)
    logged_params = {
        "test_fold": test_fold,
        "batch_size": batch_size,
        "batch_size_scheduler": bs_scheduler,
    }

    optimizer_state = ckpt.get("optimizer_state", None)
    scheduler_state = ckpt.get("scheduler_state", None)

    return (
        model, dataloaders, optim_config,
        optimizer_state, scheduler_state,
        start_epoch, logged_params, previous_run_name,
    )


def config_from_runid_multistream(runid: str, no_event_token_rate: float | None = None,
                                   no_event_token_insertion_mode: str | None = None,
                                   birth_dates_file: str | None = None):
    """
    Reconstruct a DelphiMultiStream model + all three dataloaders from a run for
    training resumption. Mirrors config_from_runid, adapted for the two-stream
    MultiStream architecture built by train_multi_stream.py, which logs a
    sparser param set than train.py:

    - No 'domain_list' param is logged, and the 'domains' param routinely
      exceeds MLflow's 6000-char param limit on ESM2 runs (truncated,
      unparseable) — domains are instead reconstructed from the run's own
      logged domain_config_yaml artifact, filtered to the names embedded in
      the 'arch' param's bracketed stream groups (same approach used in
      auc/compute_aucs_from_run.py's load_run() for the identical issue).
    - 'no_event_token_rate', 'no_event_token_insertion_mode', 'date_cutoff',
      and 'birth_dates_file' are logged explicitly as of the fix that added
      them to train_multi_stream.py's logged_params (previously they weren't
      logged at all, unlike train.py, where they ride along "for free" via
      DelphiConfig fields that Trainer auto-logs). For runs predating that
      fix, these params won't be present — pass them explicitly as arguments
      in that case; the caller-supplied value is used only as a fallback when
      the corresponding MLflow param is missing, the logged value always wins
      when both are available.

    Returns: model, dataloaders, optim_config,
             optimizer_state, scheduler_state,
             start_epoch, logged_params, previous_run_name
    """
    import mlflow
    from delphi.multi_stream_model import DelphiMultiStream, DelphiMultiStreamConfig
    from delphi.optim import OptimConfig
    from data.dataset import DelphiDataset, DelphiCollateFn, AgeSampler
    from utils.mlflow_utils import get_checkpoint_path
    from utils.utils import load_domain_config
    from utils.ckpt_utils import strip_compiled_prefix

    VAL_BATCH_SIZE = 256

    runinfo = mlflow.get_run(runid)
    params = dict(runinfo.data.params)

    batch_size = int(params.pop("batch_size", 32))
    test_fold  = params.pop("test_fold", 0)
    params.pop("learning_rate", None)
    params.pop("ema_alpha", None)

    # Prefer the value logged on the run itself (present for runs submitted after the
    # logged_params fix); fall back to the caller-supplied argument for older runs that
    # predate it. Raise a clear error rather than silently defaulting if neither is
    # available — a wrong no_event_token_rate silently corrupts the resumed dataset.
    logged_no_event_rate = params.pop("no_event_token_rate", None)
    no_event_token_rate = float(logged_no_event_rate) if logged_no_event_rate is not None else no_event_token_rate
    if no_event_token_rate is None:
        raise ValueError(
            f"Run {runid} predates logging 'no_event_token_rate' and none was passed "
            "explicitly — pass no_event_token_rate=<value> matching the original submission."
        )

    logged_insertion_mode = params.pop("no_event_token_insertion_mode", None)
    no_event_token_insertion_mode = logged_insertion_mode or no_event_token_insertion_mode or "random"

    logged_birth_dates_file = params.pop("birth_dates_file", None)
    birth_dates_file = logged_birth_dates_file or birth_dates_file
    params.pop("date_cutoff", None)  # authoritative copy comes from checkpoint metadata below

    bs_schedule_str = params.pop("batch_size_schedule", None)
    bs_scheduler = (
        BatchSizeScheduler.from_string(bs_schedule_str)
        if bs_schedule_str and bs_schedule_str != "None"
        else None
    )

    arch_str = params["arch"]
    try:
        attention_scheme = ast.literal_eval(params["attention_scheme"])
    except (ValueError, SyntaxError):
        attention_scheme = params["attention_scheme"]

    ckpt_path = get_checkpoint_path(runid)
    logging.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    start_epoch = ckpt.get("metadata", {}).get("epoch", 0) + 1
    date_cutoff = ckpt.get("metadata", {}).get("date_cutoff", None)

    optim_config_raw = params.pop("optim_config")
    if isinstance(optim_config_raw, str):
        m = re.match(r"OptimConfig\((.*)\)$", optim_config_raw, re.DOTALL)
        if m:
            optim_kwargs = dict(re.findall(r"(\w+)=([^,)]+)", m.group(1)))
            optim_kwargs = {k: ast.literal_eval(v) for k, v in optim_kwargs.items()}
            optim_config = OptimConfig(**optim_kwargs)
        else:
            raise ValueError(f"Cannot parse optim_config string: {optim_config_raw!r}")
    else:
        optim_config = optim_config_raw

    # ── domains: reconstruct from the run's own domain_config_yaml artifact,
    #    filtered to the domain names embedded in the arch string.
    client = mlflow.tracking.MlflowClient()
    yaml_artifacts = [a.path for a in client.list_artifacts(runid) if a.path.endswith(".yaml")]
    if len(yaml_artifacts) != 1:
        raise ValueError(f"Expected exactly one domain config yaml artifact for run {runid}, got {yaml_artifacts}")
    yaml_path = DELPHI_DIR / "config" / Path(yaml_artifacts[0]).name
    if not yaml_path.exists():
        raise FileNotFoundError(f"Expected local domain config at {yaml_path} (from run {runid}'s logged artifact)")

    domain_names = sorted(set(
        name.strip()
        for group in re.findall(r"\[([^\]]+)\]", arch_str)
        for name in group.split(",")
    ))
    root_path = DELPHI_DIR / "data" / "transforms"
    default_cfg_per_domain = load_domain_config(yaml_path, root_path / "tokens")
    domain_cfg = {
        k: v for k, v in default_cfg_per_domain.items()
        if k in domain_names or k in ("padding", "no_event")
    }
    missing = [d for d in domain_names if d not in default_cfg_per_domain]
    if missing:
        raise ValueError(f"Domains in arch string not found in {yaml_path}: {missing}")

    delphi_config = DelphiMultiStreamConfig(
        domains=domain_cfg,
        token_dropout=float(params.get("token_dropout", 0.1)),
        block_size=128,
        seed=int(params.get("seed", 142)),
    )

    model = DelphiMultiStream.from_scheme_string(
        arch_str=arch_str,
        attention_scheme=attention_scheme,
        config=delphi_config,
    )
    state_dict = strip_compiled_prefix(ckpt["state_dict"])
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Checkpoint state_dict mismatch resuming run {runid}: "
            f"missing={missing_keys} unexpected={unexpected_keys}"
        )

    continuous_domains = {
        dname: cfg.n_latent_tokens or 1
        for dname, cfg in domain_cfg.items()
        if cfg.type == "continuous"
    }

    _ds_kwargs = dict(
        root=root_path,
        domains_cfg=domain_cfg,
        domain_to_int=model.domain_to_int,
        block_size=128,
        exclusions=[],
        required_domains=["diseases"],
        no_event_token_rate=no_event_token_rate,
        no_event_insertion_mode=no_event_token_insertion_mode,
        continuous_domains=continuous_domains,
        age_domains=["diseases", "death"],
        date_cutoff=date_cutoff,
        birth_dates_file=birth_dates_file,
    )

    train_dataset = DelphiDataset(subjects=ckpt["metadata"]["train_ids"], **_ds_kwargs)  # type: ignore[arg-type]
    valid_dataset = DelphiDataset(subjects=ckpt["metadata"]["valid_ids"], **_ds_kwargs)  # type: ignore[arg-type]
    test_dataset  = DelphiDataset(subjects=ckpt["metadata"]["test_ids"],  **_ds_kwargs)  # type: ignore[arg-type]

    age_sampler = AgeSampler(
        insertion_mode=no_event_token_insertion_mode,
        token_rate=no_event_token_rate,
        seed=delphi_config.seed,
    )

    domain_dropout = {
        model.domain_to_int[dname]: (cfg.dropout_mode, cfg.dropout_rate, cfg.token_dropout_rate, cfg.parent or dname)
        for dname, cfg in domain_cfg.items()
        if cfg.dropout_mode is not None and cfg.dropout_rate > 0 and dname in model.domain_to_int
    }

    collate = DelphiCollateFn(
        age_sampler=age_sampler,
        block_size=128,
        domain_to_int=model.domain_to_int,
        domain_offsets=model.domain_offsets,
        padding_domain_id=model.domain_to_int["padding"],
        no_event_domain_id=model.domain_to_int["no_event"],
        continuous_domains=continuous_domains,
        domain_dropout=domain_dropout,
    )

    train_batch_size = (
        bs_scheduler.step(start_epoch).batch_size if bs_scheduler is not None else batch_size
    )
    dataloaders = DataModule(
        FlexibleDataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,  pin_memory=True, collate_fn=collate),
        DataLoader(valid_dataset,         batch_size=VAL_BATCH_SIZE,   shuffle=False, pin_memory=True, collate_fn=collate),
        DataLoader(test_dataset,          batch_size=VAL_BATCH_SIZE,   shuffle=False, pin_memory=True, collate_fn=collate),
    )

    previous_run_name = runinfo.data.tags.get("mlflow.runName", None)
    logged_params = {
        "arch": arch_str,
        "attention_scheme": attention_scheme,
        "test_fold": test_fold,
        "batch_size": batch_size,
        "batch_size_scheduler": bs_scheduler,
    }

    optimizer_state = ckpt.get("optimizer_state", None)
    scheduler_state = ckpt.get("scheduler_state", None)

    return (
        model, dataloaders, optim_config,
        optimizer_state, scheduler_state,
        start_epoch, logged_params, previous_run_name,
    )
