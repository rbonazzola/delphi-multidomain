from dataclasses import fields as dc_fields
from pathlib import Path

import pandas as pd
import yaml

from delphi.model import DomainConfig

_DOMAIN_CONFIG_FIELDS = {f.name for f in dc_fields(DomainConfig)}


def _normalize_domain_cfg(cfg: dict) -> dict:
    """Apply consistency rules to a domain config dict in-place.

    Current rules:
    - dropout_rate == 0 → dropout_mode = None
    """
    for dc in cfg.values():
        if dc.dropout_rate == 0:
            dc.dropout_mode = None
    return cfg


def _load_raw_yaml(cfg_path: Path) -> dict:
    """Load a domain config YAML, recursively resolving ``extends:`` inheritance.

    If the YAML contains ``extends: other.yaml``, the base file is loaded first
    and the current file is deep-merged on top of it (field-level within each
    domain, so child fields win without discarding unmentioned base fields).
    Paths in ``extends`` are resolved relative to the file that declares them.
    Chaining (A extends B extends C) is supported.
    """
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    extends = raw.pop("extends", None)
    if extends is None:
        return raw

    base = _load_raw_yaml(cfg_path.parent / extends)

    # Start from base, then overlay child fields domain by domain
    merged = {k: dict(v) if v is not None else {} for k, v in base.items()}
    for k, v in raw.items():
        child_fields = dict(v) if v is not None else {}
        if k in merged:
            merged[k].update(child_fields)
        else:
            merged[k] = child_fields
    return merged


def load_domain_config(cfg_path, tokens_path):
    raw = _load_raw_yaml(Path(cfg_path))

    # First pass: collect raw dicts (excluding padding)
    raw_configs = {k: dict(v) for k, v in raw.items() if k != "padding" and v is not None}

    # Second pass: resolve parent inheritance
    for domain, params in raw_configs.items():
        parent_name = params.get("parent")
        if parent_name is None:
            continue
        if parent_name not in raw_configs:
            raise ValueError(
                f"Domain '{domain}' references unknown parent '{parent_name}'. Available domains: {sorted(raw_configs)}"
            )
        parent_params = {
            k: v
            for k, v in raw_configs[parent_name].items()
            if k not in ("parent", "subdomain", "subdomain_column", "predict", "group", "abstract")
        }
        raw_configs[domain] = {**parent_params, **params}

    # Third pass: build DomainConfig objects, skipping abstract domains
    cfg = {}
    for domain, params in raw_configs.items():
        p = dict(params)
        if p.pop("abstract", False):
            continue
        if "path" in p:
            p["path"] = tokens_path / p["path"]
        cfg[domain] = DomainConfig(**p)
    cfg["padding"] = DomainConfig(projector="embed")
    return _normalize_domain_cfg(cfg)


def apply_domain_overrides(domain_cfg: dict, overrides: list[str]) -> dict:
    """Apply dot-notation overrides to a loaded domain config dict.

    Each override must be a string of the form ``domain.field=value``.
    Values are parsed with ``yaml.safe_load`` so Python types are inferred
    correctly: ``True``/``False`` → bool, integers → int, floats → float,
    ``null`` → None, plain strings stay as str.

    Raises ``ValueError`` for unknown domains or unknown DomainConfig fields.
    """
    for override in overrides:
        if "=" not in override or "." not in override.split("=", 1)[0]:
            raise ValueError(f"Invalid override {override!r}: expected 'domain.field=value'")
        lhs, value_str = override.split("=", 1)
        domain, field = lhs.split(".", 1)

        if domain not in domain_cfg:
            raise ValueError(f"Domain {domain!r} not in config. Available: {sorted(domain_cfg)}")
        if field not in _DOMAIN_CONFIG_FIELDS:
            raise ValueError(
                f"Field {field!r} is not a valid DomainConfig field. Valid fields: {sorted(_DOMAIN_CONFIG_FIELDS)}"
            )

        value = yaml.safe_load(value_str)
        setattr(domain_cfg[domain], field, value)

    return _normalize_domain_cfg(domain_cfg)


def read_ids(path, type=int):
    """
    Read UK Biobank IDs from a CSV file.
    Handles files with or without header; uses first column only.
    """
    s = pd.read_csv(path, dtype=str, comment="#").iloc[:, 0]
    return set(s.str.strip().str.replace(r"\.0$", "", regex=True).dropna().astype(type).tolist())
