"""
EXPERIMENTAL: AUCs for any checkpoint, Delphi or DelphiMultiStream.

Unifies auc/compute_aucs.py (Delphi) and auc/compute_aucs_multistream.py (MultiStream): the
only thing that differs between the two model types is how the model is rebuilt; data,
collate and evaluate_aucs are the same.

Model reconstruction, in order of preference:
  1. ckpt["model_spec"] (newer checkpoints, any model class): neither the domain config
     yaml nor the MLflow params are needed. With --ckpt, the run doesn't even have to be
     in the local mlruns.
  2. Older checkpoint of a MultiStream run (has an `arch` param): arch/attention_scheme
     from the run params, domains from the yaml logged in its artifacts (resolving
     `extends:` against the local repo's config/ yamls), or from --domain_config_yaml.
  3. Older checkpoint of a plain Delphi: from the run params, as in utils/run_loader.py
     (or --domain_config_yaml if the `domains` param is truncated).
Data params (no_event_token_rate, ...) come from the checkpoint's metadata["data_config"],
or else from the run params. Token paths come from <data_root>/tokens.

Usage:
    python auc/compute_aucs_unified.py --runid <id>                      # the run's test split
    python auc/compute_aucs_unified.py --runid <id> --split both
    python auc/compute_aucs_unified.py --runid <id> --ckpt /path/best_model.pt \\
        --data_root /path/to/fold --output aucs.csv                      # from the .pt alone
    python auc/compute_aucs_unified.py --runid <id> --subjects ids.txt \\
        --data_root data/transforms                                       # new subjects
    ... --diseases config/disease_list/doi.yaml                            # disease subset
    ... --max_subjects 500                                                 # quick test
"""
import argparse
import copy
import logging
import os
import re
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DELPHI_DIR = HERE if (HERE / "utils").is_dir() else HERE.parent
if str(DELPHI_DIR) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

import mlflow
import torch
import yaml
from torch.utils.data import DataLoader

from auc.aucs import evaluate_aucs
from data.dataset import DelphiDataset, DelphiCollateFn, AgeSampler
from utils.ckpt_utils import load_model_from_checkpoint, strip_compiled_prefix, _domain_configs_from_spec
from utils.mlflow_utils import setup_mlflow
from utils.run_loader import AUTO_BLOCK_SIZE, _build_model_from_params, _model_config
from utils.utils import load_domain_config, apply_domain_overrides

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
SPECIAL_DOMAINS = ("padding", "no_event")
SPLIT_KEYS = {"test": "test_ids", "val": "valid_ids"}

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True


# ── helpers ──────────────────────────────────────────────────────────────────

def read_subject_ids(path):
    """Integer ids, one per line (or the first column of a csv). Skips non-numeric headers."""
    ids = []
    for line in Path(path).read_text().splitlines():
        field = line.strip().split(",")[0].strip()
        if not field:
            continue
        try:
            ids.append(int(field))
        except ValueError:
            continue  # header
    if not ids:
        raise SystemExit(f"Could not read any integer ids from {path}")
    return ids


def find_checkpoint(run, name="best_model.pt"):
    """Look for the checkpoint directly in the local mlruns (independent of the logged artifact_uri)."""
    tracking = re.sub(r"^file:(//)?", "", mlflow.get_tracking_uri())
    p = Path(tracking) / run.info.experiment_id / run.info.run_id / "artifacts" / "checkpoints" / name
    if p.exists():
        return p
    return Path(mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path=f"checkpoints/{name}"))


def arch_domains(arch_str):
    """Domains listed in the [..] of each arch stream, in order, without repeats."""
    names = []
    for group in re.findall(r"\[([^\]]*)\]", arch_str):
        for d in group.split(","):
            d = d.strip()
            if d and d not in names:
                names.append(d)
    return names


def resolve_domain_yaml(client, run_id):
    """The run's domain config yaml, placed in config/ (so its `extends:` chain resolves)."""
    yamls = [a.path for a in client.list_artifacts(run_id) if a.path.endswith(".yaml")]
    if len(yamls) != 1:
        raise SystemExit(f"Expected exactly 1 domain config yaml among the run artifacts, found: {yamls}")
    name = Path(yamls[0]).name
    local = DELPHI_DIR / "config" / name
    if not local.exists():
        src = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=yamls[0])
        shutil.copy(src, local)
        print(f"Copied {name} from the run artifacts to {local}")
    return local


def spec_for_data_root(spec, data_root, dcfg):
    """Copy of the model_spec with token paths pointing to <data_root>/tokens (the spec
    stores the training ones) and the --dcfg overrides applied."""
    spec = copy.deepcopy(spec)
    domains = spec["config"]["domains"]
    for d in domains.values():
        if d.get("path") is not None:
            d["path"] = str(data_root / "tokens" / Path(d["path"]).name)
    if dcfg:
        cfgs = _domain_configs_from_spec(domains, DELPHI_DIR)
        apply_domain_overrides(cfgs, dcfg)
        spec["config"]["domains"] = {k: asdict(v) for k, v in cfgs.items()}
    return spec


def _norm_token_name(name):
    """'G03 Meningitis due to...' and 'g03_(meningitis_due_to...)' -> 'g03meningitisdueto...'"""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def read_disease_tokens(path, model, domain_cfg):
    """(domain, token_id) for the names listed in `path`, looked up in each predicted
    domain's tokenizer.yaml. Accepts a yaml list (like config/disease_list/*.yaml) or one
    name per line; names are matched ignoring case and punctuation."""
    text = Path(path).read_text()
    try:
        names = yaml.safe_load(text)
    except yaml.YAMLError:
        names = None
    if not isinstance(names, list):
        names = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]

    index = {}
    for dname in model.predicted_domains:
        cfg = domain_cfg.get(dname)
        if getattr(cfg, "path", None) is None:
            continue  # no_event: no tokenizer
        vocab = yaml.safe_load((Path(cfg.path) / "tokenizer.yaml").read_text())
        for tid, tname in enumerate(vocab):
            index.setdefault(_norm_token_name(tname), (dname, tid))

    missing = [n for n in names if _norm_token_name(n) not in index]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(names)} names in {path} are not in the tokenizer of "
                         f"{list(model.predicted_domains)}: {missing[:10]}")
    return list(dict.fromkeys(index[_norm_token_name(n)] for n in names))


# ── model reconstruction ─────────────────────────────────────────────────────

def _block_size(args, params, spec):
    if args.block_size:
        return args.block_size
    if spec is not None:
        return int(spec["config"]["block_size"])
    stored = params.get("block_size", "96")
    if stored == "auto":
        logging.warning("Run was trained with block_size='auto'; using AUTO_BLOCK_SIZE=%d", AUTO_BLOCK_SIZE)
        return AUTO_BLOCK_SIZE
    return int(stored)


def build_model(args, client, params, ckpt, data_root, block_size):
    """(model, domain_cfg) for any model class. See the module docstring."""
    spec = ckpt.get("model_spec")
    if spec is not None and args.domain_config_yaml is None:
        if args.domains:
            print("WARNING: ignoring --domains: domains come from the checkpoint's model_spec")
        ckpt = {**ckpt, "model_spec": spec_for_data_root(spec, data_root, args.dcfg)}
        print(f"Rebuilding {spec['model_class']} from the checkpoint's model_spec")
        model = load_model_from_checkpoint(ckpt, block_size=block_size)
        return model, _model_config(model).domains

    if not params:
        raise SystemExit("The checkpoint has no model_spec and the run is not in MLflow: "
                         "there is nowhere to get the architecture and domains from.")

    if "arch" in params:
        # Older MultiStream: domains from the yaml, arch from the run params
        from delphi.multi_stream_model import DelphiMultiStream, DelphiMultiStreamConfig

        arch_str = params["arch"]
        requested = args.domains.split(",") if args.domains else arch_domains(arch_str)
        domain_names = [d for d in requested if d not in SPECIAL_DOMAINS]
        yaml_path = Path(args.domain_config_yaml) if args.domain_config_yaml else resolve_domain_yaml(client, args.runid)
        default_cfg = load_domain_config(yaml_path, data_root / "tokens")
        domain_cfg = {k: v for k, v in default_cfg.items() if k in domain_names or k in SPECIAL_DOMAINS}
        not_found = [d for d in domain_names if d not in domain_cfg]
        if not_found:
            raise SystemExit(f"Domains {not_found} are not in {yaml_path.name} (nor in its extends chain)")
        if args.dcfg:
            apply_domain_overrides(domain_cfg, args.dcfg)
        config = DelphiMultiStreamConfig(
            domains=domain_cfg,
            token_dropout=float(params.get("token_dropout", 0.1)),
            block_size=block_size,
            seed=int(params.get("seed", 142)),
        )
        print(f"Rebuilding DelphiMultiStream from the run params: {arch_str}")
        model = DelphiMultiStream.from_scheme_string(
            arch_str=arch_str, attention_scheme=params["attention_scheme"], config=config)
        missing, unexpected = model.load_state_dict(strip_compiled_prefix(ckpt["state_dict"]), strict=False)
        assert not missing and not unexpected, f"state_dict mismatch: missing={missing} unexpected={unexpected}"
        return model, domain_cfg

    # Older plain Delphi: same as utils/run_loader.py
    print("Rebuilding Delphi from the run params")
    model, domain_cfg = _build_model_from_params(
        args.runid, params, ckpt, block_size, args.domain_config_yaml, data_root / "tokens")
    if args.dcfg:
        print("WARNING: ignoring --dcfg for Delphi checkpoints without a model_spec")
    return model, domain_cfg


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runid", required=True)
    ap.add_argument("--ckpt", default=None, help="Path to a .pt file (default: the run's best_model.pt)")
    ap.add_argument("--data_root", default=None,
                    help="Directory that CONTAINS tokens/. Default: the training data_root, "
                         "or data/transforms if --subjects is given")
    ap.add_argument("--subjects", default=None,
                    help="File with subject ids (one per line, or a csv with the id in the first column). "
                         "If given, evaluate on these subjects instead of the checkpoint's stored ids.")
    ap.add_argument("--split", choices=["test", "val", "both"], default="test",
                    help="Only without --subjects: the checkpoint's test_ids / valid_ids, or both")
    ap.add_argument("--diseases", default=None,
                    help="File with the diseases to evaluate (a yaml list, e.g. config/disease_list/doi.yaml, "
                         "or one name per line). Default: every predicted token")
    ap.add_argument("--score", choices=["logit", "softmax"], default="logit",
                    help="AUC score: raw logit (default) or softmax over the domain vocabulary. "
                         "See auc/aucs.py:evaluate_aucs")
    ap.add_argument("--output", default=None,
                    help="Output CSV. Default: aucs.csv / aucs_val.csv (with a _softmax suffix if --score softmax). "
                         "With --split both the defaults are always used.")
    ap.add_argument("--log_mlflow", action="store_true",
                    help="Also log the CSV as a run artifact (the run must be in MLflow)")
    ap.add_argument("--block_size", type=int, default=None, help="Default: the training block size")
    ap.add_argument("--dcfg", nargs="+", default=[], help="Domain config overrides, DOMAIN.FIELD=VALUE")
    ap.add_argument("--domain_config_yaml", default=None,
                    help="Only for checkpoints without a model_spec: local domain config yaml instead of the run's")
    ap.add_argument("--domains", default=None,
                    help="Only for MultiStream without a model_spec: comma-separated list "
                         "(default: the domains in the run's arch)")
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--max_subjects", type=int, default=None, help="Only the first N subjects (debugging)")
    args = ap.parse_args()

    # resolve user-given paths before changing directory
    output_arg = Path(args.output).resolve() if args.output else None
    data_root_arg = Path(args.data_root).resolve() if args.data_root else None
    ckpt_arg = Path(args.ckpt).resolve() if args.ckpt else None
    subjects_arg = Path(args.subjects).resolve() if args.subjects else None
    diseases_arg = Path(args.diseases).resolve() if args.diseases else None
    if args.domain_config_yaml:
        args.domain_config_yaml = str(Path(args.domain_config_yaml).resolve())
    launch_dir = Path.cwd()  # default CSVs go where the script was launched from
    # relative paths in yamls/specs (embeddings/hla/..., config/...) resolve from the repo
    os.chdir(DELPHI_DIR)

    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    try:
        run = client.get_run(args.runid)
        params = run.data.params
    except Exception as e:
        if ckpt_arg is None:
            raise SystemExit(f"Run {args.runid} not found in {mlflow.get_tracking_uri()} ({e}).\n"
                             f"If you only have the .pt file, pass it with --ckpt.")
        if args.log_mlflow:
            raise SystemExit("--log_mlflow requires the run to be in MLflow")
        print(f"WARNING: run {args.runid} is not in {mlflow.get_tracking_uri()}; using the checkpoint alone")
        run, params = None, {}

    # ── checkpoint ──
    ckpt_path = ckpt_arg or find_checkpoint(run)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metadata = ckpt.get("metadata", {})
    print(f"Checkpoint epoch={metadata.get('epoch')} val_loss={metadata.get('val_loss')}")
    # data params: from the checkpoint if present, otherwise from MLflow
    data_config = {**params, **(metadata.get("data_config") or {})}

    # ── data root ──
    # With --subjects (new ids) we evaluate on our own data: default data/transforms.
    # Without --subjects we use the checkpoint's ids, which live in the training data_root.
    default_root = "data/transforms" if subjects_arg else data_config.get("data_root", "data/transforms")
    data_root = data_root_arg or (DELPHI_DIR / default_root)
    if not (data_root / "tokens").is_dir():
        raise SystemExit(
            f"{data_root / 'tokens'} does not exist. --data_root is the directory that CONTAINS tokens/.\n"
            f"(The run was trained with data_root={data_config.get('data_root')!r}.)")

    # ── model ──
    block_size = _block_size(args, params, ckpt.get("model_spec"))
    model, domain_cfg = build_model(args, client, params, ckpt, data_root, block_size)
    model = model.to(DEVICE).eval()
    # DelphiDataset uses `path` as-is only when absolute (otherwise it re-joins it under root/tokens)
    for dname, dcfg in domain_cfg.items():
        if dname not in SPECIAL_DOMAINS and getattr(dcfg, "path", None):
            dcfg.path = str(data_root / "tokens" / Path(str(dcfg.path)).name)
            if not Path(dcfg.path).exists():
                print(f"WARNING: path for '{dname}' does not exist: {dcfg.path}")
    print(f"Domains: {list(domain_cfg)}")
    print(f"{type(model).__name__} loaded: {sum(p.numel() for p in model.parameters())} params, "
          f"block_size={block_size}")

    tokens = None
    if diseases_arg:
        tokens = read_disease_tokens(diseases_arg, model, domain_cfg)
        print(f"Evaluating {len(tokens)} tokens from {diseases_arg.name}")

    # ── data ──
    # DelphiConfig carries the no-event settings; DelphiMultiStreamConfig does not
    model_cfg = _model_config(model)
    no_event_rate = float(data_config.get("no_event_token_rate", getattr(model_cfg, "no_event_token_rate", 5.0)))
    no_event_mode = data_config.get("no_event_token_insertion_mode",
                                    getattr(model_cfg, "no_event_token_insertion_mode", "random"))
    continuous_domains = {d: c.n_latent_tokens or 1 for d, c in domain_cfg.items() if c.type == "continuous"}
    collate = DelphiCollateFn(
        age_sampler=AgeSampler(insertion_mode=no_event_mode, token_rate=no_event_rate, seed=model_cfg.seed),
        block_size=block_size,
        domain_to_int=model.domain_to_int,
        domain_offsets=model.domain_offsets,
        padding_domain_id=model.domain_to_int["padding"],
        no_event_domain_id=model.domain_to_int["no_event"],
        continuous_domains=continuous_domains,
        domain_dropout={},
        training=False,
    )

    if subjects_arg:
        if args.split != "test":
            print("WARNING: ignoring --split since --subjects was given")
        jobs = [(f"{subjects_arg.name}", read_subject_ids(subjects_arg), "aucs.csv")]
    else:
        splits = ["val", "test"] if args.split == "both" else [args.split]
        jobs = [(s, list(metadata[SPLIT_KEYS[s]]), "aucs.csv" if s == "test" else "aucs_val.csv") for s in splits]
    if args.score == "softmax":
        jobs = [(label, ids, out.replace(".csv", "_softmax.csv")) for label, ids, out in jobs]
    jobs = [(label, ids, launch_dir / out) for label, ids, out in jobs]
    if output_arg and len(jobs) == 1:
        jobs = [(jobs[0][0], jobs[0][1], output_arg)]

    logger = None
    if args.log_mlflow:
        from utils.trainer import MLFlowLogger
        logger = MLFlowLogger(experiment_name=client.get_experiment(run.info.experiment_id).name,
                              run_name=None, autostart=False)
        logger.start(resume_run_id=args.runid)
    try:
        for label, ids, output in jobs:
            if args.max_subjects:
                ids = ids[:args.max_subjects]
            dataset = DelphiDataset(
                subjects=ids,
                root=data_root,
                domains_cfg=domain_cfg,
                domain_to_int=model.domain_to_int,
                block_size=block_size,
                exclusions=[],
                required_domains=["diseases"],
                no_event_token_rate=no_event_rate,
                no_event_insertion_mode=no_event_mode,
                continuous_domains=continuous_domains,
                age_domains=["diseases", "death"],
            )
            loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=(DEVICE == "cuda"),
                                collate_fn=collate)
            print(f"\n=== AUCs: {label} ({len(ids)} subjects) ===")
            auc_df = evaluate_aucs(model, loader, block_size=block_size, run_id=args.runid,
                                   n_jobs=args.n_jobs, logger=logger, output_file=Path(output).name,
                                   score_transform=args.score, tokens=tokens)
            auc_df.to_csv(output, index=False)
            print(f"Done: {auc_df.shape} -> {output}")
    finally:
        if logger is not None:
            logger.end()


if __name__ == "__main__":
    main()
