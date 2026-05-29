# %%
import os
import sys
from pathlib import Path
import yaml
from pprint import pformat
import warnings
import pandas as pd
import torch
from torch.utils.data import DataLoader
import mlflow
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
if (DELPHI_DIR := Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

from data.dataset import DelphiDataset, DelphiCollateFn, AgeSampler, FlexibleDataLoader, BatchSizeScheduler, DataModule
from delphi.optim import OptimConfig, configure_optimizers
from delphi.model import DelphiConfig
from delphi.cross_attn_model import DelphiCrossAttention, parse_cross_attention_scheme
from utils.trainer import MLFlowLogger, Trainer
from utils.cv_utils import get_data_partitions
from utils import load_domain_config, apply_domain_overrides, setup_mlflow, AUTO_BLOCK_SIZE

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
ATTENTION_SCHEMES = yaml.safe_load((DELPHI_DIR / "config" / "attention_schemes.yaml").read_text())

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

USE_TQDM = sys.stdout.isatty()


def _resolve_attention_scheme(s: str) -> str:
    """Resolve a named alias from attention_schemes.yaml, or return the string as-is."""
    if s in ATTENTION_SCHEMES:
        scheme = ATTENTION_SCHEMES[s]["scheme"]
        # Named schemes may be lists (per-layer); flatten to single string if uniform.
        if isinstance(scheme, list):
            if len(set(scheme)) == 1:
                return scheme[0]
            raise ValueError(
                f"Attention scheme alias '{s}' defines a per-layer list, "
                f"which is not supported in DelphiCrossAttention. "
                f"Provide a single policy string instead."
            )
        return scheme
    return s


def print_config_rich(arch_str, attention_scheme, delphi_config, args, overrides=None):
    from dataclasses import asdict as _asdict
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    import shutil

    term_width = max(shutil.get_terminal_size(fallback=(120, 40)).columns, 120)
    console = Console(width=term_width)

    overridden: set[tuple[str, str]] = set()
    for ov in (overrides or []):
        if "=" in ov and "." in ov.split("=", 1)[0]:
            domain, field = ov.split("=", 1)[0].split(".", 1)
            overridden.add((domain, field))

    domains = _asdict(delphi_config).get("domains", {})
    cols = ["predict", "at_birth", "projector", "type", "dropout_mode", "dropout_rate", "age_jitter", "freeze"]
    col_labels = {
        "predict": "predict", "at_birth": "at_birth", "projector": "projector",
        "type": "type", "dropout_mode": "drop_mode", "dropout_rate": "drop_rate",
        "age_jitter": "jitter", "freeze": "freeze",
    }

    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan",
                  show_edge=False, expand=False)
    table.add_column("domain", style="bold white", no_wrap=True)
    for c in cols:
        table.add_column(col_labels[c], justify="center", no_wrap=True)

    def _cell(domain_name, field, value):
        override_style = (domain_name, field) in overridden
        if value is True:  return Text("✓", style="bold yellow" if override_style else "green")
        if value is False: return Text("✗", style="bold yellow" if override_style else "dim")
        if value is None:  return Text("—", style="bold yellow" if override_style else "dim")
        return Text(str(value), style="bold yellow" if override_style else "")

    for name, dc in domains.items():
        if name == "padding":
            continue
        cells = []
        for c in cols:
            val = dc.get(c)
            if c == "age_jitter" and val:
                lo = (dc.get("age_jitter_min") or -20 * 365.25) / 365.25
                hi = (dc.get("age_jitter_max") or  40 * 365.25) / 365.25
                style = "bold yellow" if (name, c) in overridden else "green"
                cells.append(Text(f"Unif({lo:+.0f}y,{hi:+.0f}y)", style=style))
            else:
                cells.append(_cell(name, c, val))
        table.add_row(name, *cells)

    console.print(Panel(table, title="[bold]Domain configuration[/bold]", border_style="blue"))

    scheme = parse_cross_attention_scheme(arch_str)
    attn_escaped = attention_scheme.replace("[", r"\[")
    arch_text = (
        f"encoder_A: domains=[cyan]{scheme.encoder_A.domains}[/]  "
        f"h={scheme.encoder_A.n_head} d={scheme.encoder_A.n_embd} l={scheme.encoder_A.n_layer}\n"
        f"encoder_B: domains=[cyan]{scheme.encoder_B.domains}[/]  "
        f"h={scheme.encoder_B.n_head} d={scheme.encoder_B.n_embd} l={scheme.encoder_B.n_layer}\n"
        f"xattn:     h={scheme.xattn_n_head} d={scheme.xattn_n_embd}\n"
        f"trunk:     h={scheme.trunk_n_head} d={scheme.trunk_n_embd} l={scheme.trunk_n_layer}\n"
        f"attention: [cyan]{attn_escaped}[/]\n"
        f"no_event_token_rate=[cyan]{args.no_event_token_rate}[/]"
    )
    console.print(Panel(arch_text, title="[bold]Model[/bold]", border_style="blue"))

    lr_str     = str(args.lr) if args.lr is not None else "1e-4 (default)"
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

    if overridden:
        console.print(f"[yellow]Overrides:[/yellow] {', '.join(overrides)}")


# ── Data loading ──────────────────────────────────────────────────────────────

def get_continuous_domains(domain_cfg):
    return {
        dname: cfg.n_latent_tokens or 1
        for dname, cfg in domain_cfg.items()
        if cfg.type == "continuous"
    }


def get_dataloaders(domain_cfg, model, args):
    train_ids, val_ids, test_ids = get_data_partitions(
        "./data/transforms/subject_lists", fold=args.test_fold
    )

    if args.subjects is not None:
        subject_ids = pd.read_csv(args.subjects, header=None)[0].tolist()
        train_ids = list(set(train_ids) & set(subject_ids))
        val_ids   = list(set(val_ids)   & set(subject_ids))
        test_ids  = list(set(test_ids)  & set(subject_ids))

    continuous_domains = get_continuous_domains(domain_cfg)
    cache_block_size = AUTO_BLOCK_SIZE if args.block_size == "auto" else args.block_size

    dataset_kwargs = dict(
        root=root_path,
        domains_cfg=domain_cfg,
        domain_to_int=model.domain_to_int,
        block_size=cache_block_size,
        exclusions=[],
        required_domains=["diseases"],
        no_event_token_rate=args.no_event_token_rate,
        no_event_insertion_mode=args.no_event_token_insertion_mode,
        continuous_domains=continuous_domains,
        age_domains=["diseases", "death"],
        date_cutoff=args.date_cutoff,
        birth_dates_file=args.birth_dates_file,
    )

    train_dataset = DelphiDataset(subjects=train_ids, **dataset_kwargs)
    valid_dataset = DelphiDataset(subjects=val_ids,   **dataset_kwargs)
    test_dataset  = DelphiDataset(subjects=test_ids,  **dataset_kwargs)

    age_sampler = AgeSampler(
        insertion_mode=args.no_event_token_insertion_mode,
        token_rate=args.no_event_token_rate,
        seed=args.seed,
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
    collate_kwargs = dict(
        age_sampler=age_sampler,
        block_size=args.block_size,
        domain_to_int=model.domain_to_int,
        domain_offsets=model.domain_offsets,
        padding_domain_id=model.domain_to_int["padding"],
        no_event_token_id=1,
        continuous_domains=continuous_domains,
        domain_dropout=domain_dropout,
    )
    train_collate = DelphiCollateFn(**collate_kwargs, age_jitter=age_jitter, training=True)
    eval_collate  = DelphiCollateFn(**collate_kwargs, training=False)

    bs          = args.batch_size
    eval_bs     = args.eval_batch_size or bs
    num_workers = args.num_workers

    train_loader = FlexibleDataLoader(train_dataset, shuffle=True,  collate_fn=train_collate, batch_size=bs,      num_workers=num_workers, pin_memory=True)
    valid_loader = FlexibleDataLoader(valid_dataset, shuffle=False, collate_fn=eval_collate,  batch_size=eval_bs, num_workers=num_workers, pin_memory=True)
    test_loader  = FlexibleDataLoader(test_dataset,  shuffle=False, collate_fn=eval_collate,  batch_size=eval_bs, num_workers=num_workers, pin_memory=True)

    return DataModule(train_loader, valid_loader, test_loader)


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_cli_args():
    import argparse

    class _AppendFlat(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            current = getattr(namespace, self.dest) or []
            setattr(namespace, self.dest, current + list(values))

    def _parse_kv_list(pairs, flag):
        result = {}
        for item in pairs:
            if "=" not in item:
                raise argparse.ArgumentTypeError(f"Invalid {flag} value {item!r}: expected 'KEY=VALUE'")
            k, _, v = item.partition("=")
            result[k.strip()] = v.strip()
        return result

    parser = argparse.ArgumentParser(description="Train DelphiCrossAttention model.")

    # ── Architecture ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--arch", required=True,
        help=(
            "CrossAttention architecture string. Example: "
            "'CrossAttention([hla_alleles,sex]:(h24d240l3),[diseases,lifestyle,sex]:(h24d240l6),xattn:(h24d240)):(h24d240l6)'"
        ),
    )
    parser.add_argument(
        "--attention_scheme", "--attention-scheme", dest="attention_scheme",
        default="all:causal(mask_ties=True)",
        help=(
            "Global attention policy (same syntax as Delphi). "
            "Applied to all stages, restricted by group membership per stage. "
            "Example: '[hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)'. "
            "Named aliases from attention_schemes.yaml are also accepted."
        ),
    )

    # ── Domains ───────────────────────────────────────────────────────────────
    parser.add_argument("--domain_config_yaml", "--domain-config-yaml", dest="domain_config_yaml",
                        default="config/domain_config_default.yaml")
    parser.add_argument("--domain_config", "--dcfg", dest="domain_config_overrides",
                        nargs="+", action=_AppendFlat, default=[], metavar="DOMAIN.FIELD=VALUE")
    parser.add_argument("--domains", default="diseases,death,cv_drugs,ns_drugs,lifestyle,hla_alleles,sex")

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--block_size", "--block-size", dest="block_size", default="auto",
                        type=lambda v: v if v == "auto" else int(v))
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", default=32, type=int)
    parser.add_argument("--batch_size_schedule", "--batch-size-schedule", dest="batch_size_schedule",
                        default=None, type=BatchSizeScheduler.from_string)
    parser.add_argument("--eval_batch_size", "--eval-batch-size", dest="eval_batch_size",
                        default=512, type=int)
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", default=4, type=int)
    parser.add_argument("--no_event_token_rate", "--no-event-token-rate",
                        dest="no_event_token_rate", default=2, type=float)
    parser.add_argument("--no_event_token_insertion_mode", "--no-event-token-insertion-mode",
                        dest="no_event_token_insertion_mode", default="random", type=str)
    parser.add_argument("--subjects", default=None, type=str)
    parser.add_argument("--date_cutoff", "--date-cutoff", dest="date_cutoff", default=None)
    parser.add_argument("--birth_dates_file", "--birth-dates-file", dest="birth_dates_file", default=None)
    parser.add_argument("--test_fold", "--test-fold", dest="test_fold", default=1, type=int)

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument("--token_dropout", "--token-dropout", dest="token_dropout", default=0.1, type=float)
    parser.add_argument("--no-compile", "--no_compile", dest="no_compile", default=False, action="store_true")
    parser.add_argument("--seed", default=142, type=int)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    parser.add_argument("--lr", "--learning_rate", "--learning-rate", dest="lr", default=None, type=float)
    parser.add_argument("--min_lr", "--min-lr", dest="min_lr", default=None, type=float)
    parser.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", default=1e-1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--grad_clip", "--grad-clip", dest="grad_clip", default=1.0, type=float)
    parser.add_argument("--schedule", default="cosine", choices=["cosine", "constant"])
    parser.add_argument("--warmup_iters", "--warmup-iters", dest="warmup_iters", default=2000, type=int)
    parser.add_argument("--lr_decay_iters", "--lr-decay-iters", dest="lr_decay_iters", default=10000, type=int)

    # ── Training loop ─────────────────────────────────────────────────────────
    parser.add_argument("--max_epochs", "--max-epochs", dest="max_epochs", default=1000, type=int)
    parser.add_argument("--min_epochs", "--min-epochs", dest="min_epochs", default=0, type=int)
    parser.add_argument("--patience", default=20, type=int)
    parser.add_argument("--use_amp", "--amp", dest="use_amp", default=False, action="store_true")
    parser.add_argument("--checkpoint_every", "--checkpoint-every", dest="checkpoint_every",
                        default=None, type=int)

    # ── Evaluation ────────────────────────────────────────────────────────────
    parser.add_argument("--compute_aucs", "--auc", dest="compute_aucs", default=False, action="store_true")
    parser.add_argument("--log_loss_per_disease", "--log-loss-per-disease",
                        dest="log_loss_per_disease", default=False, action="store_true")
    parser.add_argument("--baseline_incidence_path", "--baseline-incidence-path",
                        dest="baseline_incidence_path", default=None)

    # ── MLflow ────────────────────────────────────────────────────────────────
    parser.add_argument("--experiment_name", "-x", dest="experiment_name", default=None)
    parser.add_argument("--run_name", "--run-name", dest="run_name_prefix", default=None)
    parser.add_argument("--run_name_suffix", "--run-name-suffix", dest="run_name_suffix", default="")
    parser.add_argument("--param", dest="extra_params", nargs="+", action=_AppendFlat,
                        default=[], metavar="KEY=VALUE")
    parser.add_argument("--tag", dest="extra_tags", nargs="+", action=_AppendFlat,
                        default=[], metavar="KEY=VALUE")

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--no-warnings", "--no_warnings", dest="no_warnings",
                        default=False, action="store_true")
    parser.add_argument("--no_rich", "--no-rich", dest="no_rich", default=False, action="store_true")
    parser.add_argument("--dryrun", "--dry-run", dest="dry_run", default=False, action="store_true")

    args = parser.parse_args()

    if args.experiment_name is None and not args.dry_run:
        parser.error("--experiment_name / -x is required (or --dryrun).")

    prefix = args.run_name_prefix or ""
    args.run_name = (prefix + args.run_name_suffix) or None

    if args.extra_params:
        args.extra_params = _parse_kv_list(args.extra_params, "--param")
    if args.extra_tags:
        args.extra_tags = _parse_kv_list(args.extra_tags, "--tag")

    return args


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    args = get_cli_args()

    # ── Resolve attention scheme alias ────────────────────────────────────────
    attention_scheme = _resolve_attention_scheme(args.attention_scheme)

    # ── Parse arch string to extract n_embd ──────────────────────────────────
    arch_scheme = parse_cross_attention_scheme(args.arch)
    n_embd = arch_scheme.trunk_n_embd   # same across all stages (validated in model __init__)

    # ── Domain config ─────────────────────────────────────────────────────────
    domains = [d for d in args.domains.split(",") if d != "padding"]
    domain_config_yaml = DELPHI_DIR / args.domain_config_yaml
    default_cfg_per_domain = load_domain_config(domain_config_yaml, root_path / "tokens")
    domain_cfg = {k: v for k, v in default_cfg_per_domain.items() if k in domains or k == "padding"}

    missing = [k for k in domains if k not in default_cfg_per_domain]
    if missing:
        raise ValueError(f"Domains not found in config: {missing}")

    if args.domain_config_overrides:
        apply_domain_overrides(domain_cfg, args.domain_config_overrides)

    # ── DelphiConfig (used for embedding + metadata) ──────────────────────────
    delphi_config = DelphiConfig(
        n_embd=n_embd,
        n_layer=arch_scheme.trunk_n_layer,   # stored for reference; not used by the model
        n_head=arch_scheme.trunk_n_head,
        domains=domain_cfg,
        attention_scheme=attention_scheme,
        token_dropout=args.token_dropout,
        block_size=128,
        no_event_token_rate=args.no_event_token_rate,
        no_event_token_insertion_mode=args.no_event_token_insertion_mode,
        seed=args.seed,
    )

    logging.info("Domain config: %s", Path(domain_config_yaml).relative_to(Path.cwd()))
    print_config_rich(args.arch, attention_scheme, delphi_config, args,
                      overrides=args.domain_config_overrides)

    if args.dry_run:
        sys.exit(0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DelphiCrossAttention.from_scheme_string(
        arch_str=args.arch,
        attention_scheme=attention_scheme,
        config=delphi_config,
    ).to(DEVICE)

    if not args.no_compile:
        logging.info("Compiling model with torch.compile (first batch will be slower)...")
    model = torch.compile(model, disable=args.no_compile)

    # ── Data ──────────────────────────────────────────────────────────────────
    bs_scheduler = args.batch_size_schedule
    if bs_scheduler is not None:
        args.batch_size = bs_scheduler.step(0).batch_size

    dataloaders = get_dataloaders(domain_cfg, model=model, args=args)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr = args.lr if args.lr is not None else 1e-4
    optim_config = OptimConfig(
        learning_rate  = lr,
        min_lr         = args.min_lr if args.min_lr is not None else lr / 10,
        weight_decay   = args.weight_decay,
        beta1          = args.beta1,
        beta2          = args.beta2,
        grad_clip      = args.grad_clip,
        schedule       = args.schedule,
        warmup_iters   = args.warmup_iters,
        lr_decay_iters = args.lr_decay_iters,
    )
    optimizer, scheduler = configure_optimizers(model=model, cfg=optim_config, device_type=DEVICE)

    # ── MLflow ────────────────────────────────────────────────────────────────
    logger = MLFlowLogger(experiment_name=args.experiment_name, run_name=args.run_name)
    mlflow.log_artifact(domain_config_yaml)

    logged_params = {
        "arch":                       args.arch,
        "attention_scheme":           attention_scheme,
        "test_fold":                  args.test_fold,
        "batch_size":                 args.batch_size,
        "batch_size_schedule":        args.batch_size_schedule,
        "learning_rate":              lr,
        "seed":                       args.seed,
        "optim_config":               optim_config,
        "max_epochs":                 args.max_epochs,
        "min_epochs":                 args.min_epochs,
        "patience":                   args.patience,
    }

    logger.log_run_metadata(cwd=DELPHI_DIR)
    if args.extra_params:
        logger.log_params(args.extra_params)
    if args.extra_tags:
        logger.log_tags(args.extra_tags)

    n_params = sum(p.numel() for p in model.parameters())
    logged_params["n_params"] = n_params

    dataloaders.log_info()
    logged_params["n_train"] = dataloaders.n_train
    logged_params["n_val"]   = dataloaders.n_val
    logged_params["n_test"]  = dataloaders.n_test

    if args.no_warnings:
        warnings.filterwarnings("ignore")

    # ── Training ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model, dataloaders,
        optimizer, scheduler,
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
    )

    trainer.train(max_epochs=args.max_epochs, min_epochs=args.min_epochs, patience=args.patience)

    # ── AUC evaluation (on best model, restored by Trainer) ───────────────────
    if args.compute_aucs:
        import copy
        from auc.aucs import evaluate_aucs

        model.eval()
        cache_block_size = AUTO_BLOCK_SIZE if args.block_size == "auto" else args.block_size
        
        TRAIN, VAL, TEST = 0, 1, 2
        
        auc_collate = copy.copy(dataloaders[2]._collate_fn)
        auc_collate.block_size = 128
        test_loader = DataLoader(
            dataloaders[TEST].dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=dataloaders[2]._num_workers,
            pin_memory=True,
            collate_fn=auc_collate,
        )
        del dataloaders

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
        if diseases_domain in model.config.domains:
            metadata_path = Path(model.config.domains[diseases_domain].path) / "token_metadata.tsv"
            if metadata_path.exists():
                metadata = pd.read_csv(metadata_path, sep="\t")
                _d = auc_df[auc_df["domain"] == diseases_domain].copy()
                _d["_w"] = _d["n_case"] + _d["n_ctrl"]
                _d4070 = _d[(_d["age_start"] >= 40) & (_d["age_end"] <= 70)]
                _by_age = (
                    _d4070.groupby(["token_id", "age_start"])
                    .apply(lambda g: (g["auc_delong"] * g["_w"]).sum() / g["_w"].sum(), include_groups=False)
                    .rename("auc")
                    .reset_index()
                )
                pivoted = _by_age.pivot(index="token_id", columns="age_start", values="auc")
                pivoted.columns = [f"{int(c)}-{int(c)+5}" for c in pivoted.columns]
                pivoted["mean_auc"] = (
                    _d4070.groupby("token_id")
                    .apply(lambda g: (g["auc_delong"] * g["_w"]).sum() / g["_w"].sum(), include_groups=False)
                )
                age_cols = [c for c in pivoted.columns if c != "mean_auc"]
                top50 = (
                    metadata
                    .merge(pivoted.reset_index(), on="token_id", how="inner")
                    .sort_values("n_subjects", ascending=False)
                    .head(50)
                    [["name", "n_subjects", "mean_auc"] + age_cols]
                )
                logging.info("Top 50 diseases by case count:\n%s", top50.to_string(index=False))
