"""
One-off backfill: add "model_spec" (and metadata["data_config"]) to the checkpoints
of runs saved before checkpoints stored them (see utils/ckpt_utils.py:model_spec).
After this, every run loads through the same self-contained path in
utils/run_loader.py:reconstruct_from_run, without parsing MLflow params.

For each run, the construction config is recovered from, in order:
  1. param        -- the MLflow 'domains' param parses in full (not truncated)
  2. artifact     -- the resolved config/domain_config_*.yaml artifact (train.py runs
                     since 2026-06-04)
  3. partial      -- the 'domains' param is truncated: every domain dict that survived
                     the cut is taken from it as-is; the rest come from the run's domain
                     config yaml, WITHOUT the --dcfg overrides the run was launched with
                     (lost with the truncation). These runs are only written with
                     --accept-partial, after supplying the lost overrides via --rules.

Every recovered config is verified by building the model and loading the checkpoint's
state_dict into it with strict=True (catches a wrong domain set, vocab size, projector
or architecture). It can NOT catch a wrong `predict` flag or dropout setting, which
leave no trace in the weights -- hence the explicit --rules for partial runs.

Dry-run by default: prints/writes a report and touches nothing. With --apply, each
checkpoint file is rewritten atomically (temp file + rename) with only "model_spec"
and metadata["data_config"] added; weights, optimizer state and the rest of the
metadata are left untouched. Checkpoints that already have a model_spec are skipped,
so it is safe to re-run.

Also writes the other two copies of the spec that Trainer writes for new runs (see
utils/ckpt_utils.py:MODEL_SPEC_ARTIFACT) -- both for runs whose spec was just recovered
and for runs whose checkpoint already had one:
  - the config/model_spec.json artifact, if missing (never overwritten);
  - the per-domain `domains.<name>` MLflow params. Only missing keys are logged; an
    existing key with a different value is reported as a conflict, never overwritten
    (MLflow params are immutable).

Rules file (--rules): TSV with header `pattern<TAB>overrides`. `pattern` is a regex
searched in the run name; `overrides` is a space-separated list of
`domain.field=value` (applied only to domains recovered from the yaml, never to ones
parsed from the run's own params) and/or `data.key=value` (data_config entries, e.g.
`data.no_event_token_rate=5` for MultiStream runs that predate logging it). All
matching rules are applied, in file order. Example:

    pattern	overrides
    _noevent	no_event.predict=True
    ^screen_fold	data.no_event_token_rate=5

Usage:
    python -m utils.backfill_model_spec --experiments ablation-hla-esm2 --report report.tsv
    python -m utils.backfill_model_spec --all --rules rules.tsv --accept-partial --apply
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import datetime
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field, fields as dc_fields
from pathlib import Path

import mlflow
import torch
import yaml

DELPHI_DIR = Path(__file__).resolve().parent.parent
if str(DELPHI_DIR) not in sys.path:
    sys.path.insert(0, str(DELPHI_DIR))

from delphi.model import Delphi, DelphiConfig, DomainConfig
from delphi.multi_stream_model import DelphiMultiStream, DelphiMultiStreamConfig, parse_multi_stream_scheme
from utils.ckpt_utils import MODEL_SPEC_ARTIFACT, domain_params, model_spec, normalize_state_dict, strip_compiled_prefix
from utils.mlflow_utils import _extract_domain_dict, get_checkpoint_path, setup_mlflow
from utils.run_loader import AUTO_BLOCK_SIZE
from utils.utils import load_domain_config

log = logging.getLogger("backfill_model_spec")

_DOMAIN_FIELDS = {f.name for f in dc_fields(DomainConfig)}
_DATA_KEYS_FROM_PARAMS = (
    "no_event_token_rate", "no_event_token_insertion_mode",
    "date_cutoff", "birth_dates_file", "test_fold", "subjects", "data_root",
)


class RecoveryError(Exception):
    pass


@dataclass
class Recovery:
    model: torch.nn.Module
    data_config: dict
    source: str                                  # "param" | "artifact" | "partial"
    notes: list[str] = field(default_factory=list)
    spec_paths: dict = field(default_factory=dict)


# ── Rules ─────────────────────────────────────────────────────────────────────

def load_rules(path: str | None) -> list[tuple[re.Pattern, list[str]]]:
    if path is None:
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [(re.compile(r["pattern"]), r["overrides"].split()) for r in rows]


def overrides_for(run_name: str, rules) -> tuple[list[str], dict]:
    """Split the overrides of every rule matching run_name into (domain overrides, data_config dict)."""
    domain_ovs, data = [], {}
    for pattern, ovs in rules:
        if not pattern.search(run_name or ""):
            continue
        for ov in ovs:
            lhs, value = ov.split("=", 1)
            if lhs.startswith("data."):
                key = lhs[len("data."):]
                data[key] = _parse_param(key, value)
            else:
                domain_ovs.append(ov)
    return domain_ovs, data


# ── Domain config recovery ────────────────────────────────────────────────────

# Directory where token folders with the same vocabulary can be found when a run's own
# data isn't on this machine (--vocab_fallback), e.g. a synthetic cohort trained on the
# cluster: the model is built from these tokenizers, the spec keeps the run's own path.
VOCAB_FALLBACK: Path | None = None
_REPO_ANCHORS = ("data", "transforms", "embeddings")


def _relocate(path) -> str | None:
    """Map a path from wherever the run was trained to this repo: find the longest
    suffix of it that exists under DELPHI_DIR (e.g. /gpfs/.../delphi-refactor/data/
    transforms/tokens/diseases, or ../data/transforms/tokens/diseases when training
    was launched from a subdirectory -> <repo>/data/transforms/tokens/diseases)."""
    if path is None:
        return None
    p = Path(str(path))
    if p.is_absolute() and p.exists():
        return str(p)
    parts = [part for part in p.parts if part not in ("/", "..", ".")]
    for i in range(len(parts)):
        candidate = DELPHI_DIR.joinpath(*parts[i:])
        if candidate.exists():
            return str(candidate)
    raise RecoveryError(f"domain path {path} not found under {DELPHI_DIR}")


def _repo_relative(path) -> str:
    """The run's own path, as it would sit under a repo checkout: from the first
    repo-level directory on (data/, transforms/, ...), even if not present here."""
    parts = [part for part in Path(str(path)).parts if part not in ("/", "..", ".")]
    for i, part in enumerate(parts):
        if part in _REPO_ANCHORS:
            return str(Path(*parts[i:]))
    return str(path)


def _domain_from_dict(d: dict, spec_paths: dict, name: str) -> DomainConfig:
    """DomainConfig with a path usable on this machine. If the run's data isn't here
    but --vocab_fallback has a folder with the same name, build from that one and
    record the run's own (repo-relative) path in spec_paths[name] for the spec."""
    d = {k: v for k, v in d.items() if k in _DOMAIN_FIELDS}
    try:
        d["path"] = _relocate(d.get("path"))
    except RecoveryError:
        fallback = VOCAB_FALLBACK / Path(str(d["path"])).name if VOCAB_FALLBACK else None
        if fallback is None or not fallback.exists():
            raise
        spec_paths[name] = _repo_relative(d["path"])
        d["path"] = str(fallback)
    return DomainConfig(**d)


def _clean(domains_str: str) -> str:
    return re.sub(r"PosixPath\(([^)]+)\)", r"\1", domains_str)


def _run_yaml(artifacts_dir: Path) -> Path:
    """The domain config yaml the run was launched with (artifact root). Prefer the
    same-named file in this repo's config/, so `extends:` chains resolve."""
    found = sorted(artifacts_dir.glob("*.yaml"))
    if len(found) != 1:
        raise RecoveryError(f"expected exactly one domain config yaml artifact, got {[p.name for p in found]}")
    local = DELPHI_DIR / "config" / found[0].name
    return local if local.exists() else found[0]


def _resolved_config_artifact(artifacts_dir: Path) -> dict | None:
    """train.py's resolved domain config (config/domain_config_*.yaml), if logged."""
    found = sorted((artifacts_dir / "config").glob("domain_config_*.yaml"))
    # unsafe_load: written by our own train.py via yaml.dump of dataclasses.asdict,
    # which leaves `path` as a PosixPath object
    return yaml.unsafe_load(found[-1].read_text()) if found else None


def recover_domains(run, artifacts_dir: Path, state_dict, domain_rules, spec_paths: dict) -> tuple[dict, str, list[str]]:
    params = run.data.params
    domains_str = params["domains"]

    try:
        raw = ast.literal_eval(_clean(domains_str))
        return {k: _domain_from_dict(v, spec_paths, k) for k, v in raw.items()}, "param", []
    except (ValueError, SyntaxError):
        pass

    raw = _resolved_config_artifact(artifacts_dir)
    if raw is not None:
        return {k: _domain_from_dict(v, spec_paths, k) for k, v in raw.items()}, "artifact", []

    # Partial: domains whose dict survived the truncation, verbatim ...
    domains = {}
    for name in re.findall(r"'(\w+)':\s*\{", domains_str):
        try:
            domains[name] = _domain_from_dict(_extract_domain_dict(domains_str, name), spec_paths, name)
        except ValueError:
            pass  # the dict cut in half by the truncation: recovered from the yaml below

    # ... the rest from the run's yaml (post-parent-inheritance, pre --dcfg)
    yaml_path = _run_yaml(artifacts_dir)
    from_yaml = load_domain_config(yaml_path, DELPHI_DIR / "data" / "transforms" / "tokens")
    if "domain_list" in params:
        wanted = [d.strip() for d in params["domain_list"].split(",")]
    elif "arch" in params:
        wanted = sorted({n.strip() for g in re.findall(r"\[([^\]]+)\]", params["arch"]) for n in g.split(",")})
    else:
        # Neither is logged: take every yaml domain not already recovered, but a projected
        # domain only if its projector is in the checkpoint. Embed-type extras that weren't
        # really used surface as a global_embed size mismatch in the strict load.
        wanted = [
            name for name, cfg in from_yaml.items()
            if cfg.projector == "embed" or any(k.startswith(f"embed.projectors.{name}.") for k in state_dict)
        ]
    wanted = [n for n in wanted if n not in ("padding", "no_event")] + ["padding", "no_event"]

    missing = [n for n in wanted if n not in domains]
    unknown = [n for n in missing if n not in from_yaml]
    if unknown:
        raise RecoveryError(f"domains {unknown} neither in the truncated param nor in {yaml_path.name}")
    for n in missing:
        domains[n] = from_yaml[n]
        # The yaml was resolved against the real tokens here; a run trained on another
        # cohort (--data_root) read every domain from <data_root>/tokens instead.
        data_root = params.get("data_root")
        if data_root and data_root != "data/transforms" and domains[n].path is not None:
            spec_paths[n] = str(Path(data_root) / "tokens" / Path(str(domains[n].path)).name)

    # Per-domain dropout params logged by train.py (only when dropout was on)
    for n in missing:
        if f"{n}_dropout_mode" in params:
            domains[n].dropout_mode = params[f"{n}_dropout_mode"]
            domains[n].dropout_rate = float(params[f"{n}_dropout_rate"])
            if f"{n}_token_dropout_rate" in params:
                domains[n].token_dropout_rate = float(params[f"{n}_token_dropout_rate"])

    applied = []
    for ov in domain_rules:
        dname, fld = ov.split("=", 1)[0].split(".", 1)
        if dname not in missing:
            continue  # never override what the run's own params recorded
        if fld not in _DOMAIN_FIELDS:
            raise RecoveryError(f"rule override {ov!r}: unknown DomainConfig field {fld!r}")
        setattr(domains[dname], fld, yaml.safe_load(ov.split("=", 1)[1]))
        applied.append(ov)

    notes = [f"from yaml {yaml_path.name}: {','.join(missing)}"]
    if applied:
        notes.append(f"rules: {' '.join(applied)}")
    return domains, "partial", notes


# ── Model recovery ────────────────────────────────────────────────────────────

def _parse_param(name: str, value: str):
    """MLflow stores every param as str: recover its Python value."""
    if value == "None":
        return None
    if name == "block_size" and value == "auto":
        return AUTO_BLOCK_SIZE
    parsed = yaml.safe_load(value)
    # YAML turns "2020-01-01" into a datetime.date; dates are passed around as strings
    return value if isinstance(parsed, (datetime.date, datetime.datetime)) else parsed


def _config_kwargs(params: dict, config_cls) -> dict:
    """Scalar config fields (everything but domains / attention_scheme) from MLflow params."""
    return {
        f.name: _parse_param(f.name, params[f.name])
        for f in dc_fields(config_cls)
        if f.name in params and f.name not in ("domains", "attention_scheme")
    }


def recover_run(run, artifacts_dir: Path, state_dict, rules) -> Recovery:
    """Rebuild and verify the run's model. Recovery.spec_paths maps domains whose data
    isn't on this machine to the path to record in the spec instead of the local one."""
    params = run.data.params
    if "trunk_run_id" in params:
        raise RecoveryError("late-fusion head run: not a Delphi/DelphiMultiStream checkpoint")
    if "domains" not in params:
        raise RecoveryError("no 'domains' param")

    domain_rules, data_rules = overrides_for(run.info.run_name, rules)
    spec_paths: dict = {}
    domains, source, notes = recover_domains(run, artifacts_dir, state_dict, domain_rules, spec_paths)

    if "arch" in params:
        config = DelphiMultiStreamConfig(domains=domains, **_config_kwargs(params, DelphiMultiStreamConfig))
        model = DelphiMultiStream(config, parse_multi_stream_scheme(params["arch"]), params["attention_scheme"])
    else:
        attn = params.get("attention_scheme", "all:causal(mask_ties=True)")
        try:
            attn = ast.literal_eval(attn)
        except (ValueError, SyntaxError):
            pass
        model = Delphi(DelphiConfig(domains=domains, attention_scheme=attn, **_config_kwargs(params, DelphiConfig)))

    try:
        missing, unexpected = model.load_state_dict(normalize_state_dict(state_dict, model), strict=False)
    except RuntimeError as e:  # shape mismatch, e.g. a different domain set -> vocab size
        raise RecoveryError(f"state_dict mismatch ({source}): {str(e).splitlines()[-1].strip()[:200]}") from e
    if missing or unexpected:
        raise RecoveryError(
            f"state_dict mismatch ({source}): missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}"
        )

    data_config = {"data_root": "data/transforms"}
    source_script = run.data.tags.get("mlflow.source.name", "")
    data_config["required_domains"] = ["sex"] if source_script.endswith("train_synthetic.py") else ["diseases"]
    for k in _DATA_KEYS_FROM_PARAMS:
        if k in params and params[k] != "None":
            data_config[k] = _parse_param(k, params[k])
    data_config |= data_rules
    if "arch" in params and "no_event_token_rate" not in data_config:
        # DelphiMultiStreamConfig doesn't carry it, and the loader's fallback would be wrong
        raise RecoveryError("MultiStream run predates logging no_event_token_rate: supply it via a data.* rule")

    if spec_paths:
        notes.append(f"run's data not here, built with the real vocab; spec paths: {sorted(set(str(Path(p).parent) for p in spec_paths.values()))}")
    return Recovery(model, data_config, source, notes, spec_paths)


# ── Checkpoint rewriting ──────────────────────────────────────────────────────

def checkpoint_files(run_id: str) -> list[Path]:
    """Real checkpoint files of the run (best_model.pt symlinks point to one of them)."""
    ckpt_dir = get_checkpoint_path(run_id).parent
    return sorted(p for p in ckpt_dir.glob("*.pt") if not p.is_symlink())


def rewrite_checkpoint(path: Path, spec: dict, data_config: dict, model) -> bool:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("model_spec") is not None:
        return False
    model.load_state_dict(normalize_state_dict(ckpt["state_dict"], model), strict=True)

    ckpt["model_spec"] = spec
    metadata = ckpt.setdefault("metadata", {}) or {}
    ckpt["metadata"] = metadata
    metadata.setdefault("data_config", data_config)
    if metadata.get("date_cutoff") is None and data_config.get("date_cutoff"):
        metadata["date_cutoff"] = data_config["date_cutoff"]

    tmp = path.with_name(f".{path.name}.backfill_tmp")
    torch.save(ckpt, tmp)
    torch.load(tmp, map_location="cpu", weights_only=False)  # readable before replacing
    shutil.copystat(path, tmp)
    os.replace(tmp, path)
    return True


def write_spec_json(artifacts_dir: Path, spec: dict, apply: bool) -> str:
    """Write the run's MODEL_SPEC_ARTIFACT if missing. Written straight into the
    (relocated) artifacts dir, like the checkpoints, rather than via MLflow's logged
    artifact_uri, which may point to another filesystem. Returns exists/written/missing."""
    path = artifacts_dir / MODEL_SPEC_ARTIFACT
    if path.exists():
        return "exists"
    if not apply:
        return "missing"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.backfill_tmp")
    tmp.write_text(json.dumps(spec, indent=2))
    os.replace(tmp, path)
    return "written"


def log_domain_params(client, run, spec: dict, apply: bool) -> tuple[int, list[str]]:
    """Log the run's missing `domains.*` params, derived from spec. Returns (number of
    params missing -- i.e. logged, with apply --, keys whose logged value differs)."""
    from mlflow.entities import Param

    logged = run.data.params
    missing = {k: v for k, v in domain_params(spec).items() if k not in logged}
    conflicts = [k for k, v in domain_params(spec).items() if k in logged and logged[k] != v]
    if apply and missing:
        params = [Param(k, v) for k, v in missing.items()]
        for i in range(0, len(params), 100):  # MLflow's per-batch limit
            client.log_batch(run.info.run_id, params=params[i:i + 100])
    return len(missing), conflicts


# ── Driver ────────────────────────────────────────────────────────────────────

def iter_runs(client, args):
    if args.run_ids:
        for rid in args.run_ids:
            yield client.get_run(rid)
        return
    if args.all:
        exp_ids = [e.experiment_id for e in client.search_experiments()]
    else:
        exp_ids = []
        for e in args.experiments:
            exp = client.get_experiment_by_name(e) or client.get_experiment(e)
            exp_ids.append(exp.experiment_id)
    for eid in exp_ids:
        try:
            runs = client.search_runs(experiment_ids=[eid], max_results=50000)
        except Exception as e:  # malformed experiment in the store (bad metric files, missing meta.yaml, ...)
            log.warning("Skipping experiment %s, can't list its runs: %s", eid, str(e)[:200])
            continue
        yield from runs


def _add_domain_params(row, client, run, spec, apply, artifacts_dir):
    row["spec_json"] = write_spec_json(artifacts_dir, spec, apply)
    row["n_params"], conflicts = log_domain_params(client, run, spec, apply)
    if conflicts:
        row["notes"] = "; ".join(filter(None, [row["notes"], f"param conflicts: {','.join(conflicts)}"]))


def process_run(run, args, rules, client) -> dict:
    row = {
        "run_id": run.info.run_id, "experiment_id": run.info.experiment_id,
        "run_name": run.info.run_name or "", "model_class": "", "source": "",
        "status": "", "n_written": 0, "spec_json": "", "n_params": 0, "notes": "",
    }
    try:
        files = checkpoint_files(run.info.run_id)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as e:
        row["status"] = "no_checkpoints"
        row["notes"] = str(e)[:200]
        return row
    if not files:
        row["status"] = "no_checkpoints"
        return row

    try:
        best = get_checkpoint_path(run.info.run_id).resolve()
        ckpt = torch.load(best, map_location="cpu", weights_only=False)
        if ckpt.get("model_spec") is not None:
            row["status"] = "has_spec"
            row["model_class"] = ckpt["model_spec"]["model_class"]
            _add_domain_params(row, client, run, ckpt["model_spec"], args.apply, best.parent.parent)
            return row
        rec = recover_run(run, best.parent.parent, strip_compiled_prefix(ckpt["state_dict"]), rules)
    except RecoveryError as e:
        row["status"] = "error"
        row["notes"] = str(e)
        return row
    except Exception as e:  # unexpected: report it, keep going with the other runs
        row["status"] = "error"
        row["notes"] = f"{type(e).__name__}: {e}"[:300]
        return row
    del ckpt

    spec = model_spec(rec.model)
    for dname, path in rec.spec_paths.items():
        spec["config"]["domains"][dname]["path"] = path
    row.update(model_class=spec["model_class"], source=rec.source, notes="; ".join(rec.notes))
    if rec.source == "partial" and not args.accept_partial:
        row["status"] = "partial_needs_accept"
        return row
    if not args.apply:
        row["status"] = "ok_dry_run"
        _add_domain_params(row, client, run, spec, apply=False, artifacts_dir=best.parent.parent)
        return row

    row["n_written"] = sum(rewrite_checkpoint(p, spec, rec.data_config, rec.model) for p in files)
    row["status"] = "written"
    _add_domain_params(row, client, run, spec, apply=True, artifacts_dir=best.parent.parent)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--experiments", nargs="+", help="Experiment names or IDs")
    target.add_argument("--run_ids", nargs="+")
    target.add_argument("--all", action="store_true", help="Every run in every experiment")
    ap.add_argument("--rules", default=None, help="TSV of run-name regex -> overrides (see module docstring)")
    ap.add_argument("--accept-partial", dest="accept_partial", action="store_true",
                    help="Also write runs whose config was partly recovered from the yaml")
    ap.add_argument("--apply", action="store_true",
                    help="Rewrite checkpoints, write model_spec.json and log domains.* params (default: dry run)")
    ap.add_argument("--report", default="backfill_model_spec_report.tsv")
    ap.add_argument("--vocab_fallback", default=None,
                    help="Token folders with the same vocabulary, used to build runs whose own data "
                         "isn't on this machine (e.g. data/transforms/tokens for a synthetic cohort); "
                         "the spec still records the run's own paths")
    args = ap.parse_args()
    global VOCAB_FALLBACK
    VOCAB_FALLBACK = (DELPHI_DIR / args.vocab_fallback) if args.vocab_fallback else None

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("mlflow").setLevel(logging.CRITICAL)  # malformed-experiment tracebacks from the store
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    rules = load_rules(args.rules)

    rows = []
    for run in iter_runs(client, args):
        row = process_run(run, args, rules, client)
        log.info("%s %-22s %-9s json:%-7s +%d params %s %s", row["run_id"][:8], row["status"], row["source"],
                 row["spec_json"], row["n_params"], row["run_name"], row["notes"])
        rows.append(row)

    with open(args.report, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["run_id"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    print(f"\n{'APPLIED' if args.apply else 'DRY RUN'} -- {len(rows)} runs, report: {args.report}")
    for (status, source), n in sorted(Counter((r["status"], r["source"]) for r in rows).items()):
        print(f"  {status:22s} {source or '-':9s} {n}")


if __name__ == "__main__":
    main()
