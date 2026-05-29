# %%
import logging
import os
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Any, TypeVar, cast

import mlflow
import pandas as pd
import torch
import torch._inductor.config as _inductor_config
import yaml
from torch.utils.data import DataLoader

from auc.aucs import evaluate_aucs
from data.dataset import (
    AgeSampler,
    BatchSizeScheduler,
    DataModule,
    DelphiCollateFn,
    DelphiDataset,
    FlexibleDataLoader,
)
from delphi.model import (
    Delphi,
    DelphiConfig,
)
from delphi.optim import OptimConfig, configure_optimizers
from utils import (  # cache upper bound when block_size="auto"
    AUTO_BLOCK_SIZE,
    apply_domain_overrides,
    load_domain_config,
    setup_mlflow,
)
from utils.cv_utils import get_data_partitions
from utils.trainer import (
    MLFlowLogger,
    Trainer,
    clone_run_to_new_experiment,
)

_inductor_config.fx_graph_cache = (
    True  # persist compiled Triton kernels across runs (set TORCHINDUCTOR_CACHE_DIR to a non-/tmp path)
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DELPHI_DIR = Path(__file__).resolve().parent

setup_mlflow()


class _MLflowWarningFilter(logging.Filter):
    _SUPPRESS = ("malformed experiment", "malformed run")

    def filter(self, record):
        if record.levelno == logging.WARNING:
            msg = record.getMessage().lower()
            if any(kw in msg for kw in self._SUPPRESS):
                return False
        return True


_mlflow_filter = _MLflowWarningFilter()
for _handler in logging.root.handlers:
    _handler.addFilter(_mlflow_filter)

root_path = DELPHI_DIR / "data" / "transforms"
ATTENTION_SCHEMES: dict[str, dict[str, str]] = yaml.safe_load(
    (DELPHI_DIR / "config" / "attention_schemes.yaml").read_text()
)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

USE_TQDM = sys.stdout.isatty()


def print_config_rich(delphi_config, args, overrides: list[str] | None = None) -> None:
    """Print a Rich-formatted config summary and exit cleanly (used by --dryrun)."""
    import shutil
    from dataclasses import asdict as _asdict

    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    term_width = max(shutil.get_terminal_size(fallback=(120, 40)).columns, 120)
    console = Console(width=term_width)

    # Which (domain, field) pairs were explicitly overridden
    overridden: set[tuple[str, str]] = set()
    for ov in overrides or []:
        if "=" in ov and "." in ov.split("=", 1)[0]:
            domain, field = ov.split("=", 1)[0].split(".", 1)
            overridden.add((domain, field))

    domains = _asdict(delphi_config).get("domains", {})

    # ── Domain table ──────────────────────────────────────────────────────
    cols = ["predict", "at_birth", "projector", "type", "dropout_mode", "dropout_rate", "age_jitter", "freeze"]

    col_labels = {
        "predict": "predict",
        "at_birth": "at_birth",
        "projector": "projector",
        "type": "type",
        "dropout_mode": "drop_mode",
        "dropout_rate": "drop_rate",
        "age_jitter": "jitter",
        "freeze": "freeze",
    }

    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan", show_edge=False, expand=False)
    table.add_column("domain", style="bold white", no_wrap=True)
    for c in cols:
        table.add_column(col_labels[c], justify="center", no_wrap=True)

    def _cell(domain_name: str, field: str, value) -> Text:
        override_style = (domain_name, field) in overridden
        if value is True:
            return Text("✓", style="bold yellow" if override_style else "green")
        if value is False:
            return Text("✗", style="bold yellow" if override_style else "dim")
        if value is None:
            return Text("—", style="bold yellow" if override_style else "dim")
        return Text(str(value), style="bold yellow" if override_style else "")

    for name, dc in domains.items():
        if name == "padding":
            continue
        cells = []
        for c in cols:
            val = dc.get(c)
            if c == "age_jitter" and val:
                lo = (dc.get("age_jitter_min") or -20 * 365.25) / 365.25
                hi = (dc.get("age_jitter_max") or 40 * 365.25) / 365.25
                style = "bold yellow" if (name, c) in overridden else "green"
                cells.append(Text(f"Unif({lo:+.0f}y,{hi:+.0f}y)", style=style))
            else:
                cells.append(_cell(name, c, val))
        table.add_row(name, *cells)

    console.print(Panel(table, title="[bold]Domain configuration[/bold]", border_style="blue"))

    # ── Model params ──────────────────────────────────────────────────────
    attn = args.attention_scheme
    attn_str = attn[0] if isinstance(attn, list) and len(attn) == 1 else str(attn)
    attn_str = attn_str.replace("[", r"\[")  # escape Rich markup
    model_text = (
        f"n_layer=[cyan]{args.n_layer}[/]  n_head=[cyan]{args.n_head}[/]  n_embd=[cyan]{args.n_embd}[/]  "
        f"no_event_token_rate=[cyan]{args.no_event_token_rate}[/]\n"
        f"attention_scheme: [cyan]{attn_str}[/]"
    )
    console.print(Panel(model_text, title="[bold]Model[/bold]", border_style="blue"))

    # ── Training params ───────────────────────────────────────────────────
    lr_str = str(args.lr) if args.lr is not None else "1e-4 (default)"
    min_lr_str = str(args.min_lr) if args.min_lr is not None else "lr/10 (default)"
    bs_str = str(args.batch_size)
    if args.batch_size_schedule:
        bs_str += f"  schedule=[cyan]{args.batch_size_schedule}[/]"
    train_text = (
        f"max_epochs=[cyan]{args.max_epochs}[/]  min_epochs=[cyan]{args.min_epochs}[/]  patience=[cyan]{args.patience}[/]\n"
        f"lr=[cyan]{lr_str}[/]  min_lr=[cyan]{min_lr_str}[/]  schedule=[cyan]{args.schedule}[/]  warmup_iters=[cyan]{args.warmup_iters}[/]\n"
        f"batch_size=[cyan]{bs_str}[/]\n"
        f"test_fold=[cyan]{args.test_fold}[/]  seed=[cyan]{args.seed}[/]"
    )
    console.print(Panel(train_text, title="[bold]Training[/bold]", border_style="blue"))

    if overridden and overrides:
        console.print(f"[yellow]Overrides:[/yellow] {', '.join(overrides)}")


def parse_attention_scheme(attention_scheme: list[str] | str, as_list=True) -> list[str] | str:
    if isinstance(attention_scheme, str):
        if attention_scheme in ATTENTION_SCHEMES:
            attention_scheme = ATTENTION_SCHEMES[attention_scheme]["scheme"]
        if as_list:
            attention_scheme = [attention_scheme]
        return attention_scheme
    elif isinstance(attention_scheme, list):
        return [cast(str, parse_attention_scheme(scheme, as_list=False)) for scheme in attention_scheme]


# ——————————————— CLI ———————————————————————————————————————————————————————————————


def get_cli_args():

    import argparse

    class _AppendFlat(argparse.Action):
        """Accumulate multiple values per flag invocation into a flat list."""

        def __call__(self, parser, namespace, values, option_string=None):
            current = getattr(namespace, self.dest) or []
            setattr(namespace, self.dest, current + list(values or []))

    def _parse_kv_list(pairs, flag):
        result = {}
        for item in pairs:
            if "=" not in item:
                raise argparse.ArgumentTypeError(f"Invalid {flag} value {item!r}: expected 'KEY=VALUE'")
            k, _, v = item.partition("=")
            result[k.strip()] = v.strip()
        return result

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--domain_config_yaml",
        "--domain-config-yaml",
        dest="domain_config_yaml",
        default="config/domain_config_default.yaml",
    )
    parser.add_argument(
        "--domain_config",
        "--override_domain_config",
        "--domain-config",
        "--override-domain-config",
        "--dcfg",
        dest="domain_config_overrides",
        nargs="+",
        action=_AppendFlat,
        default=[],
        metavar="DOMAIN.FIELD=VALUE",
        help="Override domain config fields, e.g. --dcfg diseases.predict=True hla.dropout_rate=0.1",
    )
    parser.add_argument("--domains", default="diseases,death,cv_drugs,ns_drugs,lifestyle,hla_alleles,sex")
    parser.add_argument(
        "--attention_scheme",
        "--attention-scheme",
        dest="attention_scheme",
        default="[hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,death,padding]:causal(mask_ties=True)",
        nargs="+",
    )
    parser.add_argument("--n_layer", "--n-layer", dest="n_layer", default=12, type=int)
    parser.add_argument("--n_head", "--n-head", dest="n_head", default=6, type=int)
    parser.add_argument("--n_embd", "--n-embd", dest="n_embd", default=120, type=int)
    parser.add_argument(
        "--block_size",
        "--block-size",
        "--blocksize",
        dest="block_size",
        default="auto",
        type=lambda v: v if v == "auto" else int(v),
        help="Max tokens per subject in the cache. 'auto' (default) uses a "
        "generous upper bound; the effective per-batch length is always "
        "trimmed to the longest sequence in that batch.",
    )
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", default=32, type=int)
    parser.add_argument(
        "--batch_size_schedule",
        "--batch-size-schedule",
        dest="batch_size_schedule",
        default=None,
        type=BatchSizeScheduler.from_string,
        help=(
            "Batch size schedule. Format: comma-separated stages, each "
            "'n_epochs:batch_size' or 'n_epochs:batch_sizexgrad_accum'. "
            "Use '*' as n_epochs for the last (open-ended) stage. "
            "Overrides --batch_size. "
            "Examples: "
            "'20:32,20:128,*:256' (no accumulation); "
            "'20:32,20:128,10:256,*:256x4' (last stage: effective batch=1024)."
        ),
    )
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", default=4, type=int)
    parser.add_argument("--token_dropout", "--token-dropout", dest="token_dropout", default=0.1, type=float)
    parser.add_argument("--no-compile", "--no_compile", dest="no_compile", default=False, action="store_true")
    parser.add_argument(
        "--learning_rate",
        "--learning-rate",
        "--lr",
        dest="lr",
        default=None,
        type=float,
        help="Peak learning rate. For fresh runs defaults to 1e-4. "
        "On resume, if provided, overrides the stored LR and resets the scheduler.",
    )

    # ── Optimizer / LR schedule ───────────────────────────────────────────────
    parser.add_argument(
        "--min_lr",
        "--min-lr",
        dest="min_lr",
        default=None,
        type=float,
        help="Minimum LR at end of cosine decay (default: lr/10)",
    )
    parser.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", default=1e-1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument(
        "--grad_clip",
        "--grad-clip",
        dest="grad_clip",
        default=1.0,
        type=float,
        help="Gradient clipping norm (0 = disabled)",
    )
    parser.add_argument("--schedule", default="cosine", choices=["cosine", "constant"])
    parser.add_argument("--warmup_iters", "--warmup-iters", dest="warmup_iters", default=2000, type=int)
    parser.add_argument("--lr_decay_iters", "--lr-decay-iters", dest="lr_decay_iters", default=10000, type=int)
    parser.add_argument("--test_fold", "--test-fold", dest="test_fold", default=1, type=int)
    parser.add_argument("--subjects", default=None, type=str)
    parser.add_argument(
        "--date_cutoff",
        "--date-cutoff",
        dest="date_cutoff",
        default=None,
        type=str,
        help="ISO date (YYYY-MM-DD). Tokens after this date are marked via eval_mask.",
    )
    parser.add_argument(
        "--birth_dates_file",
        "--birth-dates-file",
        dest="birth_dates_file",
        default=None,
        type=str,
        help="Path to TSV with columns eid, year, month (used with --date_cutoff).",
    )
    parser.add_argument("--seed", default=142, type=int)
    parser.add_argument("--max_epochs", "--max-epochs", dest="max_epochs", default=1000, type=int)
    parser.add_argument("--min_epochs", "--min-epochs", dest="min_epochs", default=0, type=int)
    parser.add_argument("--patience", default=20, type=int)
    parser.add_argument(
        "--compute_aucs", "--compute-aucs", "--aucs", "--auc", dest="compute_aucs", default=False, action="store_true"
    )
    parser.add_argument(
        "--eval_batch_size",
        "--eval-batch-size",
        dest="eval_batch_size",
        default=512,
        type=int,
        help="Batch size for validation and AUC evaluation (default: 512)",
    )
    parser.add_argument(
        "--log_loss_per_disease",
        "--log-loss-per-disease",
        dest="log_loss_per_disease",
        default=False,
        action="store_true",
        help="Log per-disease CE loss breakdown as a CSV artifact each validation epoch",
    )
    parser.add_argument(
        "--baseline_incidence_path",
        "--baseline-incidence-path",
        dest="baseline_incidence_path",
        default=None,
        type=str,
        help="Path to age-sex stratified disease incidence parquet "
        "(from auc/compute_disease_incidence.py). Enables relative CCE logging.",
    )
    parser.add_argument(
        "--checkpoint_every",
        "--checkpoint-every",
        dest="checkpoint_every",
        default=None,
        type=int,
        help="Save a periodic checkpoint every N epochs (in addition to best-model checkpoints)",
    )

    parser.add_argument(
        "--no_event_token_rate", "--no-event-token-rate", dest="no_event_token_rate", default=2, type=float
    )
    parser.add_argument(
        "--no_event_token_insertion_mode",
        "--no-event-token-insertion-mode",
        dest="no_event_token_insertion_mode",
        default="random",
        type=str,
    )

    parser.add_argument("--no-warnings", "--no_warnings", dest="no_warnings", default=False, action="store_true")
    parser.add_argument(
        "--use_amp",
        "--use-amp",
        "--amp",
        dest="use_amp",
        default=False,
        action="store_true",
        help="Enable mixed precision training (float16)",
    )

    parser.add_argument(
        "--experiment_name", "--experiment-name", "--exp_name", "--exp-name", "-x", dest="experiment_name", default=None
    )
    parser.add_argument(
        "--run_name", "--run-name", "--run_name_prefix", "--run-name-prefix", dest="run_name_prefix", default=None
    )
    parser.add_argument("--run_name_suffix", "--run-name-suffix", dest="run_name_suffix", default="")
    parser.add_argument(
        "--param",
        dest="extra_params",
        nargs="+",
        action=_AppendFlat,
        default=[],
        metavar="KEY=VALUE",
        help="Extra MLflow params logged alongside training params, e.g. --param setup=baseline note=v2",
    )
    parser.add_argument(
        "--tag",
        dest="extra_tags",
        nargs="+",
        action=_AppendFlat,
        default=[],
        metavar="KEY=VALUE",
        help="Extra MLflow tags, e.g. --tag env=cluster status=production",
    )
    parser.add_argument(
        "--resume_run_id",
        "--resume-run-id",
        "--resume_runid",
        "--resume-runid",
        dest="resume_run_id",
        type=str,
        default=None,
        help="Resume training from the latest checkpoint of this MLflow run",
    )

    parser.add_argument(
        "--interactive",
        "-i",
        dest="interactive",
        action="store_true",
        default=False,
        help=(
            "Enable interactive mode. When combined with --resume_from_previous, "
            "presents a numbered menu to select the source experiment and run "
            "instead of requiring --resume_run_id explicitly. Also lets you choose "
            "a different target experiment for the resumed run."
        ),
    )
    parser.add_argument(
        "--resume_from_previous",
        "--resume-from-previous",
        "--resume_from_run",
        "--resume-from-run",
        "--resume_from_runid",
        "--resume-from-runid",
        "--resume_from_run_id",
        "--resume-from-run-id",
        "-r",
        dest="resume_from_previous",
        action="store_true",
        default=False,
        help=(
            "Resume training from the latest checkpoint of a previous run. "
            "Requires either --resume_run_id <RUN_ID> or --interactive (-i) "
            "to select the run interactively."
        ),
    )

    parser.add_argument(
        "--dryrun", "--dry-run", "--dry_run", "--dry", dest="dry_run", action="store_true", default=False
    )

    parser.add_argument(
        "--no_rich",
        "--no-rich",
        dest="no_rich",
        action="store_true",
        default=False,
        help="Disable rich display (use tqdm instead, e.g. for cluster log files)",
    )

    args = parser.parse_args()

    if args.experiment_name is None and not args.resume_run_id and not args.resume_from_previous and not args.dry_run:
        parser.error("--experiment_name / -x is required unless --resume_run_id or --resume_from_previous is set.")

    prefix = args.run_name_prefix or ""
    raw_run_name = (prefix + args.run_name_suffix) or None
    if raw_run_name:
        try:
            args.run_name = raw_run_name.format_map(vars(args))
        except (KeyError, ValueError):
            args.run_name = raw_run_name
    else:
        args.run_name = None

    if args.extra_params:
        args.extra_params = _parse_kv_list(args.extra_params, "--param")
    if args.extra_tags:
        args.extra_tags = _parse_kv_list(args.extra_tags, "--tag")

    return args


def _interactive_select_run() -> tuple[str, str]:
    """Prompt the user to pick an experiment and a run.

    Returns (run_id, target_experiment_name) where the target experiment
    is either the original one or a different one chosen by the user.
    """
    experiments = mlflow.search_experiments(order_by=["last_update_time DESC"])
    if not experiments:
        raise RuntimeError("No MLflow experiments found.")

    print("\nAvailable experiments:")
    for i, exp in enumerate(experiments):
        print(f"  [{i}] {exp.name}")
    idx = int(input("Select experiment [0]: ").strip() or "0")
    source_experiment = experiments[idx]

    runs_df = mlflow.search_runs(
        experiment_ids=[source_experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=20,
    )
    assert isinstance(runs_df, pd.DataFrame)
    if runs_df.empty:
        raise RuntimeError(f"No runs found in experiment '{source_experiment.name}'.")

    cols = ["run_id", "tags.mlflow.runName", "start_time", "status"]
    cols = [c for c in cols if c in runs_df.columns]
    print(f"\nRuns in '{source_experiment.name}' (most recent first):")
    for row_i, row in runs_df[cols].iterrows():
        name = row.get("tags.mlflow.runName", "")
        print(f"  [{row_i}] {row['run_id'][:8]}…  {name:<30}  {row['status']}  {row['start_time']}")
    run_idx = int(input("Select run [0]: ").strip() or "0")
    run_id = runs_df.iloc[run_idx]["run_id"]

    answer = input(f"\nKeep original experiment '{source_experiment.name}'? [Y/n]: ").strip().lower()
    if answer in ("", "y", "yes"):
        target_experiment = source_experiment.name
    else:
        print("\nAvailable experiments:")
        for i, exp in enumerate(experiments):
            print(f"  [{i}] {exp.name}")
        target_idx = int(input("Select target experiment [0]: ").strip() or "0")
        target_experiment = experiments[target_idx].name

    return run_id, target_experiment


# ——————————————— Data loading ——————————————————————————————————————————————————————


_ModelT = TypeVar("_ModelT", bound=torch.nn.Module)


def _compile_model(model: _ModelT, disable: bool = False) -> _ModelT:
    """Wrap torch.compile with cache-aware logging.

    Logs an uncertain message immediately, then a definitive one (hit/miss)
    on each graph compilation during the first forward pass.
    """
    if disable:
        return model
    logging.info("torch.compile enabled — will compile or load from cache on first forward pass")
    try:
        from torch._inductor import codecache
        from torch._inductor.utils import counters as _ic  # type: ignore[attr-defined]

        _orig = codecache.FxGraphCache.load

        def _logged_load(compile_fx_fn, gm, example_inputs, fx_kwargs):
            h_before = _ic["inductor"]["fxgraph_cache_hit"]
            m_before = _ic["inductor"]["fxgraph_cache_miss"]
            result = _orig(compile_fx_fn, gm, example_inputs, fx_kwargs)
            if _ic["inductor"]["fxgraph_cache_hit"] > h_before:
                logging.info("torch.compile: graph loaded from cache")
            elif _ic["inductor"]["fxgraph_cache_miss"] > m_before:
                logging.info("torch.compile: graph compiled (cache miss)")
            return result

        codecache.FxGraphCache.load = staticmethod(_logged_load)  # type: ignore[method-assign]
    except Exception:
        pass
    return cast(_ModelT, torch.compile(model))


def get_continuous_domains(domain_cfg):
    return {dname: cfg.n_latent_tokens or 1 for dname, cfg in domain_cfg.items() if cfg.type == "continuous"}


def get_dataloaders(
    domain_cfg,
    model,
    test_fold,
    block_size,
    batch_size,
    eval_batch_size=None,
    num_workers=4,
    no_event_token_rate=2.0,
    no_event_insertion_mode="random",
    seed=42,
    subjects_include_list=None,
):
    train_ids, val_ids, test_ids = get_data_partitions("./data/transforms/subject_lists", fold=test_fold)

    if subjects_include_list is not None:
        subject_ids = pd.read_csv(subjects_include_list, header=None)[0].tolist()
        train_ids = list(set(train_ids) & set(subject_ids))
        val_ids = list(set(val_ids) & set(subject_ids))
        test_ids = list(set(test_ids) & set(subject_ids))

    continuous_domains = get_continuous_domains(domain_cfg)

    # Dataset always needs a fixed integer cache size; collate receives the
    # original value ("auto" or int) to decide per-batch trimming behaviour.
    cache_block_size = AUTO_BLOCK_SIZE if block_size == "auto" else block_size

    dataset_kwargs: dict[str, Any] = dict(
        root=root_path,
        domains_cfg=domain_cfg,
        domain_to_int=model.domain_to_int,
        block_size=cache_block_size,
        exclusions=[],
        required_domains=["diseases"],
        no_event_token_rate=no_event_token_rate,
        no_event_insertion_mode=no_event_insertion_mode,
        continuous_domains=continuous_domains,
        age_domains=["diseases", "death"],
        date_cutoff=args.date_cutoff,
        birth_dates_file=args.birth_dates_file,
    )

    train_dataset = DelphiDataset(subjects=train_ids, **dataset_kwargs)
    valid_dataset = DelphiDataset(subjects=val_ids, **dataset_kwargs)
    test_dataset = DelphiDataset(subjects=test_ids, **dataset_kwargs)

    # Collate function (shared by all loaders)
    age_sampler = AgeSampler(
        insertion_mode=no_event_insertion_mode,
        token_rate=no_event_token_rate,
        seed=seed,
    )

    domain_dropout = {
        model.domain_to_int[dname]: (cfg.dropout_mode, cfg.dropout_rate)
        for dname, cfg in domain_cfg.items()
        if cfg.dropout_mode is not None and cfg.dropout_rate > 0
    }

    age_jitter = {
        model.domain_to_int[dname]: (cfg.age_jitter_min, cfg.age_jitter_max)
        for dname, cfg in domain_cfg.items()
        if cfg.age_jitter and dname in model.domain_to_int
    }

    collate_kwargs: dict[str, Any] = dict(
        age_sampler=age_sampler,
        block_size=block_size,  # "auto" or int
        domain_to_int=model.domain_to_int,
        domain_offsets=model.domain_offsets,
        padding_domain_id=model.domain_to_int["padding"],
        no_event_token_id=1,
        continuous_domains=continuous_domains,
        domain_dropout=domain_dropout,
    )
    train_collate = DelphiCollateFn(**collate_kwargs, age_jitter=age_jitter, training=True)
    eval_collate = DelphiCollateFn(**collate_kwargs, training=False)

    train_kwargs: dict[str, Any] = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    eval_kwargs: dict[str, Any] = dict(
        batch_size=eval_batch_size or batch_size, num_workers=num_workers, pin_memory=True
    )

    train_loader = FlexibleDataLoader(train_dataset, shuffle=True, collate_fn=train_collate, **train_kwargs)
    valid_loader = FlexibleDataLoader(valid_dataset, shuffle=False, collate_fn=eval_collate, **eval_kwargs)
    test_loader = FlexibleDataLoader(test_dataset, shuffle=False, collate_fn=eval_collate, **eval_kwargs)

    return DataModule(train_loader, valid_loader, test_loader)


# ——————————————————————————————————————————————————————————————————————————————————

if __name__ == "__main__":
    args = get_cli_args()

    if args.resume_from_previous and args.interactive:
        args.resume_run_id, args.experiment_name = _interactive_select_run()
    elif args.resume_from_previous and not args.resume_run_id:
        raise ValueError("--resume_from_previous requires --interactive or --resume_run_id.")

    cache_block_size = AUTO_BLOCK_SIZE if args.block_size == "auto" else args.block_size
    logging.info("block_size=%s  cache_block_size=%d", args.block_size, cache_block_size)

    bs_scheduler = None
    start_epoch = 0

    if train_from_scratch := not args.resume_run_id:
        _raw = args.attention_scheme
        if isinstance(_raw, list):
            _raw = _raw[0] if len(set(_raw)) == 1 else None
        attention_scheme_alias = _raw if _raw in ATTENTION_SCHEMES else None
        args.attention_scheme = parse_attention_scheme(args.attention_scheme)
        domains = [d for d in args.domains.split(",") if d != "padding"]
        domain_config_yaml = DELPHI_DIR / args.domain_config_yaml
        default_cfg_per_domain = load_domain_config(domain_config_yaml, root_path / "tokens")

        # Expand group aliases (e.g. "core" → ["diseases", "death", "lifestyle", "sex"])
        group_to_domains: dict[str, list[str]] = {}
        for dname, dcfg in default_cfg_per_domain.items():
            if dname == "padding" or dcfg.group is None:
                continue
            group_to_domains.setdefault(dcfg.group, []).append(dname)
        expanded: list[str] = []
        for d in domains:
            expanded.extend(group_to_domains[d] if d in group_to_domains and d not in default_cfg_per_domain else [d])
        domains = list(dict.fromkeys(expanded))  # deduplicate, preserve order

        domain_cfg = {k: v for k, v in default_cfg_per_domain.items() if k in domains or k == "padding"}

        if args.domain_config_overrides:
            apply_domain_overrides(domain_cfg, args.domain_config_overrides)

        assert all(k in default_cfg_per_domain for k in domains), (
            f"{[k for k in domains if k not in default_cfg_per_domain]}"
        )
        assert len(args.attention_scheme) in {1, args.n_layer}, (
            f"len of --attention_scheme should be 1 or n_layer (={args.n_layer})"
        )

        # Pass as-is: single string → same scheme for all layers,
        # list of n_layer strings → per-layer schemes. The model expands internally.
        attention_scheme = args.attention_scheme[0] if len(args.attention_scheme) == 1 else args.attention_scheme

        # ── Model ─────────────────────────────────────────────────────────
        delphi_config = DelphiConfig(
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            domains=domain_cfg,
            attention_scheme=attention_scheme,
            token_dropout=args.token_dropout,
            # block_size=cache_block_size,
            block_size=128,
            no_event_token_rate=args.no_event_token_rate,
            no_event_token_insertion_mode=args.no_event_token_insertion_mode,
            seed=args.seed,
        )

        logging.info("Domain config: %s", Path(domain_config_yaml).relative_to(Path.cwd()))
        print_config_rich(delphi_config, args, overrides=args.domain_config_overrides)

        if args.dry_run:
            sys.exit(0)
        model = Delphi(delphi_config).to(DEVICE)
        model = _compile_model(model, disable=args.no_compile)

        # ── Data ──────────────────────────────────────────────────────────
        bs_scheduler = args.batch_size_schedule
        initial_batch_size = bs_scheduler.step(0).batch_size if bs_scheduler is not None else args.batch_size
        dataloaders = get_dataloaders(
            domain_cfg,
            model=model,
            test_fold=args.test_fold,
            block_size=args.block_size,
            batch_size=initial_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            no_event_token_rate=args.no_event_token_rate,
            no_event_insertion_mode=args.no_event_token_insertion_mode,
            seed=args.seed,
            subjects_include_list=args.subjects,
        )

        # ── Optimizer ─────────────────────────────────────────────────────
        lr = args.lr if args.lr is not None else 1e-4
        optim_config = OptimConfig(
            learning_rate=lr,
            min_lr=args.min_lr if args.min_lr is not None else lr / 10,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            grad_clip=args.grad_clip,
            schedule=args.schedule,
            warmup_iters=args.warmup_iters,
            lr_decay_iters=args.lr_decay_iters,
        )
        logging.info("Optimizer configuration: \n%s", pformat(asdict(optim_config), sort_dicts=False))
        optimizer, scheduler = configure_optimizers(model=model, cfg=optim_config, device_type=DEVICE)

        logger = MLFlowLogger(experiment_name=args.experiment_name, run_name=args.run_name)

        mlflow.log_artifact(domain_config_yaml)

        logged_params = {
            "test_fold": args.test_fold,
            "batch_size": args.batch_size,
            "batch_size_schedule": args.batch_size_schedule,
            "learning_rate": lr,
            "seed": args.seed,
            "optim_config": optim_config,
            "max_epochs": args.max_epochs,
            "min_epochs": args.min_epochs,
            "patience": args.patience,
            "attention_scheme_alias": attention_scheme_alias,
            "domain_list": ",".join(domains),
        }

    else:
        ################################ FROM PREVIOUS RUN ################################
        from utils.run_loader import config_from_runid

        (
            model,
            dataloaders,
            optim_config,
            optimizer_state,
            scheduler_state,
            start_epoch,
            logged_params,
            previous_run_name,
        ) = config_from_runid(args.resume_run_id)

        bs_scheduler = logged_params.pop("batch_size_scheduler", None)
        if args.batch_size_schedule is not None:
            bs_scheduler = args.batch_size_schedule
            new_bs = bs_scheduler.step(start_epoch).batch_size
            dataloaders.train.set_batch_size(new_bs)
            logging.info(
                "batch_size_schedule overridden from CLI: %s (batch_size at epoch %d: %d)",
                bs_scheduler,
                start_epoch,
                new_bs,
            )
        elif bs_scheduler is not None:
            logging.info("batch_size_schedule restored from run params: %s", bs_scheduler)
        else:
            logging.info("No batch_size_schedule — using fixed batch_size=%d", logged_params.get("batch_size", "?"))

        if args.lr is not None:
            optim_config.learning_rate = args.lr
            optim_config.min_lr = args.min_lr if args.min_lr is not None else args.lr / 10

        model = model.to(DEVICE)
        model = _compile_model(model, disable=args.no_compile)
        optimizer, scheduler = configure_optimizers(model=model, cfg=optim_config, device_type=DEVICE)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

        if args.lr is not None:
            # Keep position in the cosine schedule (last_epoch) but rescale to new peak LR.
            # scheduler.base_lrs was restored from checkpoint; override it and recompute current LR.
            old_base_lr = scheduler.base_lrs[0]
            scheduler.base_lrs = [args.lr] * len(scheduler.base_lrs)
            current_lrs: list[float] = scheduler.get_lr()  # type: ignore[assignment]
            for pg, new_lr in zip(optimizer.param_groups, current_lrs, strict=True):
                pg["lr"] = new_lr
            logging.info(
                "Learning rate peak changed: %.2e → %.2e; current LR at schedule step %d: %.2e (min_lr=%.2e)",
                old_base_lr,
                args.lr,
                scheduler.last_epoch,
                current_lrs[0],
                optim_config.min_lr,
            )

        if args.experiment_name is None:
            run_info = mlflow.get_run(args.resume_run_id)
            args.experiment_name = mlflow.get_experiment(run_info.info.experiment_id).name

        new_run_id = clone_run_to_new_experiment(args.resume_run_id, args.experiment_name)
        logger = MLFlowLogger(experiment_name=args.experiment_name, run_name=previous_run_name, autostart=False)
        logger.start(resume_run_id=new_run_id)

        print(f"Resuming from MLflow run {args.resume_run_id} ...")

    # —————————————————————————————————————————————————————————————————————————————————————

    # ── Run metadata ──────────────────────────────────────────────────────────────────
    logger.log_run_metadata(cwd=DELPHI_DIR)
    if args.extra_params:
        logger.log_params(args.extra_params)
    if args.extra_tags:
        logger.log_tags(args.extra_tags)

    n_params = sum(p.numel() for p in model.parameters())
    logged_params["n_params"] = n_params

    dataloaders.log_info()
    logged_params["n_train"] = dataloaders.n_train
    logged_params["n_val"] = dataloaders.n_val
    logged_params["n_test"] = dataloaders.n_test

    if args.no_warnings:
        warnings.filterwarnings("ignore")

    trainer = Trainer(
        model,
        dataloaders,
        optimizer,
        scheduler,
        log_loss_per_disease=args.log_loss_per_disease,
        baseline_incidence_path=args.baseline_incidence_path,
        checkpoint_every=args.checkpoint_every,
        logger=logger,
        mlflow_params=logged_params,
        use_tqdm=args.no_rich,
        use_amp=args.use_amp,
        use_rich=not args.no_rich,
        optim_config=optim_config,
        batch_size_scheduler=bs_scheduler,
        start_epoch=start_epoch,
    )

    trainer.train(max_epochs=args.max_epochs, min_epochs=args.min_epochs, patience=args.patience)

    if args.compute_aucs:
        import copy

        model.eval()

        # evaluate_aucs requires fixed T across all batches; swap collate to use
        # block_size=128 instead of "auto" so torch.cat on embeddings doesn't fail.
        auc_collate = copy.copy(dataloaders[2]._collate_fn)
        auc_collate.block_size = 128
        test_loader = DataLoader(
            dataloaders[2].dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=dataloaders[2]._num_workers,
            pin_memory=True,
            collate_fn=auc_collate,
        )
        del dataloaders

        assert logger.active_run is not None
        auc_df = evaluate_aucs(
            model,
            test_loader,
            block_size=cache_block_size,
            run_id=logger.active_run.info.run_id,
            n_jobs=8,
            logger=logger,
        )
        logging.info("AUCs:\n%s", pformat(auc_df, sort_dicts=False))

        diseases_domain = "diseases"
        domain_path = model.config.domains.get(diseases_domain, None)
        if domain_path is not None and domain_path.path is not None:
            metadata_path = Path(domain_path.path) / "token_metadata.tsv"
            if metadata_path.exists():
                metadata = pd.read_csv(metadata_path, sep="\t")
                _d = auc_df[auc_df["domain"] == diseases_domain].copy()
                _d["_w"] = _d["n_case"] + _d["n_ctrl"]
                # age bins 40-70 only
                _d4070 = _d[(_d["age_start"] >= 40) & (_d["age_end"] <= 70)]
                # per (token_id, age_bin): weighted mean across sexes
                _by_age = (
                    _d4070.groupby(["token_id", "age_start"])
                    .apply(lambda g: (g["auc_delong"] * g["_w"]).sum() / g["_w"].sum(), include_groups=False)  # type: ignore[call-overload]
                    .rename("auc")
                    .reset_index()
                )
                pivoted = _by_age.pivot(index="token_id", columns="age_start", values="auc")
                pivoted.columns = [f"{int(c)}-{int(c) + 5}" for c in pivoted.columns]
                # overall weighted mean (40-70, both sexes)
                pivoted["mean_auc"] = _d4070.groupby("token_id").apply(
                    lambda g: (g["auc_delong"] * g["_w"]).sum() / g["_w"].sum(), include_groups=False
                )  # type: ignore[call-overload]
                age_cols = [c for c in pivoted.columns if c != "mean_auc"]
                top50 = (
                    metadata.merge(pivoted.reset_index(), on="token_id", how="inner")
                    .sort_values("n_subjects", ascending=False)
                    .head(50)[["name", "n_subjects", "mean_auc", *age_cols]]
                )
                logging.info("Top 50 diseases by case count:\n%s", top50.to_string(index=False))
