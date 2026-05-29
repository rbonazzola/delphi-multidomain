from pathlib import Path

import pandas as pd
import yaml

from delphi.model import DomainConfig


def load_domain_config(cfg_path, tokens_path):
    raw = yaml.safe_load(Path(cfg_path).read_text())
    cfg = {}
    for domain, params in raw.items():
        if domain == "padding":
            continue
        p = dict(params)
        if "path" in p:
            p["path"] = tokens_path / p["path"]
        cfg[domain] = DomainConfig(**p)
    cfg["padding"] = DomainConfig(projector="embed")
    return cfg


def read_ids(path, type=int):
    """
    Read UK Biobank IDs from a CSV file.
    Handles files with or without header; uses first column only.
    """
    s = pd.read_csv(path, dtype=str, comment="#").iloc[:, 0]
    return set(s.str.strip().str.replace(r"\.0$", "", regex=True).dropna().astype(type).tolist())
