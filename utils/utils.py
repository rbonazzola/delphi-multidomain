import re
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

import pandas as pd
import yaml

DELPHI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DELPHI_DIR))

from delphi.model import DomainConfig

_DOMAIN_CONFIG_FIELDS = {f.name for f in dc_fields(DomainConfig)}


_ARCH_LETTER_TO_KEY = {"h": "n_head", "d": "n_embd", "l": "n_layer"}
_ARCH_TOKEN_RE = re.compile(r"([hdl])\s*(\d+)", re.IGNORECASE)
_ARCH_FULL_RE = re.compile(r"\s*(?:[hdl]\s*\d+\s*)+", re.IGNORECASE)


def parse_arch_str(s: str, strict: bool = False) -> dict:
    """Parse an architecture string such as ``'h12d120l12'``.

    Returns ``{'n_head': 12, 'n_embd': 120, 'n_layer': 12}``.

    Letters can be in any order and in upper or lower case (``'L12H12D120'`` is the
    same as ``'h12d120l12'``); spaces are allowed. A repeated letter is an error.

    - ``strict=False`` (MultiStream stages): each letter is optional; missing keys are
      left out of the result.
    - ``strict=True`` (``train.py --arch``): the string must contain only ``h``, ``d``
      and ``l`` tokens, each exactly once.
    """
    expected = "'h<n_head>d<n_embd>l<n_layer>' (any order), e.g. 'h12d120l12'"
    if strict and not _ARCH_FULL_RE.fullmatch(s):
        raise ValueError(f"Invalid architecture string {s!r}: expected {expected}.")
    out = {}
    for letter, value in _ARCH_TOKEN_RE.findall(s):
        key = _ARCH_LETTER_TO_KEY[letter.lower()]
        if key in out:
            raise ValueError(f"Invalid architecture string {s!r}: '{letter.lower()}' appears more than once.")
        out[key] = int(value)
    if strict:
        missing = [l for l, k in _ARCH_LETTER_TO_KEY.items() if k not in out]
        if missing:
            raise ValueError(f"Invalid architecture string {s!r}: missing {', '.join(missing)}; expected {expected}.")
    return out


def format_arch_str(n_head: int, n_embd: int, n_layer: int) -> str:
    """Inverse of :func:`parse_arch_str`: ``(12, 120, 12)`` → ``'h12d120l12'``."""
    return f"h{n_head}d{n_embd}l{n_layer}"


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

    # First pass: collect raw dicts (excluding padding/no_event reserved pseudo-domains)
    raw_configs = {
        k: dict(v) for k, v in raw.items()
        if k not in ("padding", "no_event") and v is not None
    }

    # Second pass: resolve parent inheritance
    for domain, params in raw_configs.items():
        parent_name = params.get("parent")
        if parent_name is None:
            continue
        if parent_name not in raw_configs:
            raise ValueError(
                f"Domain '{domain}' references unknown parent '{parent_name}'. "
                f"Available domains: {sorted(raw_configs)}"
            )
        parent_params = {
            k: v for k, v in raw_configs[parent_name].items()
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

    # padding is pure filler for unused sequence slots: never predictable.
    cfg["padding"] = DomainConfig(projector="embed")

    # no_event is the "an interval passed with no event" token inserted by
    # AgeSampler at collate time. It's a distinct reserved domain, so
    # `predict` can be toggled for it independently of the (always
    # unpredictable) padding filler.
    no_event_raw = raw.get("no_event") or {}
    cfg["no_event"] = DomainConfig(projector="embed", predict=bool(no_event_raw.get("predict", False)))

    return _normalize_domain_cfg(cfg)


def apply_domain_overrides(domain_cfg: dict, overrides: list[str]) -> dict:
    """Apply dot-notation overrides to a loaded domain config dict.

    Each override must be a string of the form ``domain.field=value``.
    Values are parsed with ``yaml.safe_load`` so Python types are inferred
    correctly: ``True``/``False`` → bool, integers → int, floats → float,
    ``null`` → None, plain strings stay as str.

    Raises ``ValueError`` for unknown domains or unknown DomainConfig fields.

    A single element may hold several whitespace-separated overrides. sarray_params passes a
    TSV cell like "no_event.predict=True diseases.dropout_rate=0.0" as ONE argument; without
    this split, everything after the first "=" became the value of the first override, and the
    other overrides were silently dropped.
    """
    overrides = [tok for override in overrides for tok in override.split()]
    for override in overrides:
        if "=" not in override or "." not in override.split("=", 1)[0]:
            raise ValueError(
                f"Invalid override {override!r}: expected 'domain.field=value'"
            )
        lhs, value_str = override.split("=", 1)
        domain, field = lhs.split(".", 1)

        if domain not in domain_cfg:
            raise ValueError(
                f"Domain {domain!r} not in config. Available: {sorted(domain_cfg)}"
            )
        if field not in _DOMAIN_CONFIG_FIELDS:
            raise ValueError(
                f"Field {field!r} is not a valid DomainConfig field. "
                f"Valid fields: {sorted(_DOMAIN_CONFIG_FIELDS)}"
            )

        value = yaml.safe_load(value_str)
        setattr(domain_cfg[domain], field, value)

    return _normalize_domain_cfg(domain_cfg)


def read_ids(path, type=int):
    """
    Read UK Biobank IDs from a CSV file.
    Handles files with or without header; uses first column only.
    """
    s = pd.read_csv(path, dtype=str, comment="#", header=None).iloc[:, 0]
    # A non-numeric first line is a header (e.g. "eid"); a numeric one is already an ID.
    if not s.empty and not s.iloc[0].strip().replace(".0", "").isdigit():
        s = s.iloc[1:]
    return set(
        s.str.strip()
         .str.replace(r"\.0$", "", regex=True)
         .dropna()
         .astype(type)
         .tolist()
    )


def read_disease_ids(path, metadata_path=None) -> list:
    """
    Read disease identifiers from a file (one per line; blank lines and '#'
    comments — whole-line or trailing — are ignored). Each entry is either an
    integer token_id, or an icd_code (e.g. "E11") matched case-insensitively
    against metadata_path (the diseases domain's token_metadata.tsv).

    Used to restrict fine-tuning loss to a disease subset (see finetune.py).
    """
    lines = Path(path).read_text().splitlines()
    stripped = [l.split("#", 1)[0].strip() for l in lines]
    entries = [l for l in stripped if l]

    meta = None
    token_ids = []
    for entry in entries:
        if entry.isdigit():
            token_ids.append(int(entry))
            continue
        if meta is None:
            if metadata_path is None:
                raise ValueError(
                    f"Non-numeric disease identifier {entry!r} in {path} requires "
                    "metadata_path for icd_code lookup"
                )
            meta = pd.read_csv(metadata_path, sep="\t")
        match = meta[meta["icd_code"].str.lower() == entry.lower()]
        if match.empty:
            raise ValueError(f"No disease found with icd_code={entry!r} (from {path}) in {metadata_path}")
        token_ids.extend(match["token_id"].tolist())
    return sorted(set(token_ids))


