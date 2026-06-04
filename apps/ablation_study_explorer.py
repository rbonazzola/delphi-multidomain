"""
Streamlit app: compare AUC across experiment configurations.

Groups runs by configuration (all params except test_fold and seed),
aggregates over folds and seeds, and shows AUC for a selected disease.

Run from repo root:
    streamlit run apps/experiment_comparator.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def _resolve_mlruns_dir() -> Path:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
    parsed = urlparse(uri)
    path = parsed.path if parsed.scheme in ("file", "") else uri
    return Path(path).resolve()

MLRUNS_DIR = _resolve_mlruns_dir()
DEFAULT_EXP_ID = "6614694254594957006"  # hla-experiments-merged

DOMAIN_PARAMS = [
    "n_embd", "n_head", "n_layer", "block_size",
    "batch_size_schedule", "learning_rate", "no_event_token_rate",
    "token_dropout", "attention_scheme", "no_repeat",
]
IGNORE_PARAMS = {"test_fold", "seed", "domains", "n_params", "n_train", "n_val", "n_test",
                 "optim_config", "hostname", "batch_size", "ema_alpha", "attn_pdrop",
                 "embd_pdrop", "resid_pdrop", "dropout", "bias", "max_epochs",
                 "min_epochs", "patience", "no_event_token_insertion_mode",
                 "attention_scheme_alias"}

TOKENIZER_PATH = Path(__file__).resolve().parents[1] / "data" / "transforms" / "tokens" / "diseases" / "tokenizer.yaml"
ATTENTION_SCHEMES_PATH = Path(__file__).resolve().parents[1] / "config" / "attention_schemes.yaml"

AUTOIMMUNE_ICD = [
    "E05", "E03", "D86", "K90", "L40", "E14", "M07", "M45",
    "G35", "E10", "H20", "M05", "K51", "M32", "L80", "J45", "H16",
]

DISPLAY_DEFAULTS = {
    "n_layer": "12",
    "n_head": "12",
    "n_embd": "120",
    "block_size": "128",
}

_HLA_LOCUS_RE = re.compile(r"^hla_(?!alleles)[a-z][a-z0-9]*\.")


def _display_params(vary: list[str], row: "pd.Series") -> dict:
    """Build display-ready {param: value} dict for a config row.

    - Skips params matching DISPLAY_DEFAULTS.
    - Collapses per-HLA-locus columns (hla_a.*, hla_b.*, ...) into hla_encoding=ESM2.
    """
    result = {}
    hla_projectors = set()
    for p in vary:
        if _HLA_LOCUS_RE.match(p):
            if p.endswith(".projector"):
                hla_projectors.add(str(row.get(p, "")))
            continue
        val = str(row.get(p, ""))
        if DISPLAY_DEFAULTS.get(p) == val:
            continue
        result[p] = val
    if hla_projectors:
        if "pretrained" in hla_projectors:
            result["hla_encoding"] = "ESM2"
        else:
            result["hla_encoding"] = ", ".join(sorted(hla_projectors))
    return result

REFERENCE_PARAMS = {
    "n_layer": "12",
    "n_head": "12",
    "n_embd": "120",
    "genetic_pcs.n_latent_tokens": "5",
    "no_event_token_rate": "1",
    "learning_rate": 3e-4,   # numeric for tolerance comparison
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_prefix(col: str) -> str:
    return re.sub(r"^(params|metrics|tags)\.", "", col)


def _sig_figs(x, n=3):
    import math
    if x == 0:
        return 0.0
    d = math.ceil(math.log10(abs(x)))
    return round(x, n - d)


@st.cache_resource(show_spinner=False)
def _load_attention_schemes() -> dict:
    import yaml
    if not ATTENTION_SCHEMES_PATH.exists():
        return {}
    return yaml.safe_load(ATTENTION_SCHEMES_PATH.read_text())


def _split_scheme_rules(scheme: str) -> list[str]:
    rules, current, depth = [], [], 0
    for ch in scheme:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            rules.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        rules.append("".join(current).strip())
    return rules


def _normalize_scheme(scheme: str) -> str:
    """Normalize for order-independent alias matching.

    - Bidirectional rules: sort domains, strip padding (they are the distinguishing feature).
    - Causal rules with explicit domain lists: collapse to `all:<type>` since they always
      mean "everything attends causally" regardless of which domains are listed.
    """
    normalized = []
    for rule in _split_scheme_rules(scheme):
        m = re.match(r"\[([^\]]+)\]:(.*)", rule)
        if m:
            rule_type = m.group(2).strip()
            if "bidirectional" in rule_type:
                domains = sorted(d.strip() for d in m.group(1).split(",") if d.strip() != "padding")
                normalized.append(f"[{','.join(domains)}]:{rule_type}")
            else:
                normalized.append(f"all:{rule_type}")
        else:
            normalized.append(rule)
    return ",".join(normalized)


def _extract_scheme_str(scheme: str) -> str:
    """Unwrap Python list repr stored by MLflow (per-layer list → single scheme string)."""
    import ast
    s = scheme.strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and parsed:
            unique = list(dict.fromkeys(parsed))
            return unique[0] if len(unique) == 1 else s
    except (ValueError, SyntaxError):
        pass
    return s


def _scheme_to_alias(scheme: str, stored_alias: str | None = None) -> str:
    """Return a human-readable name for a scheme string."""
    if stored_alias and str(stored_alias) not in {"(missing)", "None", "nan", ""}:
        return str(stored_alias)
    clean = _extract_scheme_str(str(scheme))
    norm = _normalize_scheme(clean)
    for alias, entry in _load_attention_schemes().items():
        if _normalize_scheme(entry.get("scheme", "")) == norm:
            return alias
    return clean


@st.cache_resource(show_spinner="Loading runs from MLflow…")
def load_runs(exp_ids: tuple[str, ...]) -> pd.DataFrame:
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    df = mlflow.search_runs(experiment_ids=list(exp_ids), filter_string="status = 'FINISHED'")
    # Use last logged metrics.learning_rate (peak after warmup), rounded to 3 sig figs
    if "metrics.learning_rate" in df.columns:
        df["params.learning_rate"] = df["metrics.learning_rate"].apply(
            lambda x: _sig_figs(x, n=1) if pd.notna(x) else x
        )
    df.columns = [_strip_prefix(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


@st.cache_data(show_spinner=False)
def load_disease_names() -> dict[int, str]:
    import yaml
    if not TOKENIZER_PATH.exists():
        return {}
    with open(TOKENIZER_PATH) as f:
        tokens = yaml.safe_load(f)
    return {i: name for i, name in enumerate(tokens)}


@st.cache_data(show_spinner=False)
def load_autoimmune_token_ids() -> dict[int, str]:
    """Return {token_id: icd3} for the autoimmune disease panel, using the tokenizer.yaml names."""
    import yaml
    if not TOKENIZER_PATH.exists():
        return {}
    with open(TOKENIZER_PATH) as f:
        tokens = yaml.safe_load(f)
    pattern = re.compile(r"^(" + "|".join(re.escape(c) for c in AUTOIMMUNE_ICD) + r")\b")
    return {
        i: name[:3]
        for i, name in enumerate(tokens)
        if pattern.match(name)
    }


def _wavg(df: pd.DataFrame, val_col: str = "auc_delong", weight_col: str = "n_case") -> float:
    total = df[weight_col].sum()
    return float((df[val_col] * df[weight_col]).sum() / total) if total > 0 else float("nan")


def compute_autoimmune_scores(
    auc_all: pd.DataFrame,
    filtered: pd.DataFrame,
    key_to_short: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (summary_df, per_disease_df).

    summary_df  : (config, fold, sex) × [score_unweighted, score_weighted]
    per_disease_df : (config, sex, icd3) mean auroc across folds
    """
    token_to_icd3 = load_autoimmune_token_ids()
    if not token_to_icd3 or auc_all.empty:
        return pd.DataFrame(), pd.DataFrame()

    run_to_config = filtered.set_index("run_id")["config_key"].to_dict()
    auc = auc_all[auc_all["token_id"].isin(token_to_icd3)].copy()
    auc["icd3"] = auc["token_id"].map(token_to_icd3)
    auc["config_key"] = auc["run_id"].map(run_to_config)
    auc["config_label"] = auc["config_key"].map(key_to_short)
    auc = auc.dropna(subset=["config_key"])

    has_sex = "sex" in auc.columns
    sex_values = list(auc["sex"].unique()) if has_sex else [None]

    def _per_disease(df: pd.DataFrame, sex_val) -> pd.DataFrame:
        sub = df if sex_val is None else df[df["sex"] == sex_val]
        result = (
            sub.groupby(["config_label", "test_fold", "icd3"])
            .apply(_wavg)
            .reset_index(name="auroc")
        )
        result["sex"] = "combined" if sex_val is None else sex_val
        return result

    def _weighted(df: pd.DataFrame, sex_val) -> pd.DataFrame:
        sub = df if sex_val is None else df[df["sex"] == sex_val]
        result = (
            sub.groupby(["config_label", "test_fold"])
            .apply(_wavg)
            .reset_index(name="score_weighted")
        )
        result["sex"] = "combined" if sex_val is None else sex_val
        return result

    per_disease_parts = [_per_disease(auc, s) for s in sex_values]
    if has_sex:
        per_disease_parts.append(_per_disease(auc, None))
    per_disease = pd.concat(per_disease_parts, ignore_index=True)

    unweighted = (
        per_disease.groupby(["config_label", "test_fold", "sex"])["auroc"]
        .mean()
        .reset_index(name="score_unweighted")
    )

    weighted_parts = [_weighted(auc, s) for s in sex_values]
    if has_sex:
        weighted_parts.append(_weighted(auc, None))
    weighted = pd.concat(weighted_parts, ignore_index=True)

    summary = unweighted.merge(weighted, on=["config_label", "test_fold", "sex"])

    fold_mean = (
        summary.groupby(["config_label", "sex"])[["score_unweighted", "score_weighted"]]
        .mean()
        .reset_index()
    )
    fold_mean["test_fold"] = "mean"
    summary_full = pd.concat([summary, fold_mean], ignore_index=True)

    per_disease_mean = (
        per_disease.groupby(["config_label", "sex", "icd3"])["auroc"]
        .mean()
        .reset_index()
    )

    return summary_full, per_disease, per_disease_mean


def _parse_no_repeat(domains_str: str) -> str:
    matches = re.findall(r"'(\w+)':\s*\{[^}]*'no_repeat':\s*(True|False)", domains_str or "")
    present = [k for k, v in matches if v == "True"]
    if not matches:
        return "absent"
    return ",".join(present) if present else "none"


def _parse_domain_names(domains_str: str) -> list[str]:
    # Matches both dict-style {'domain': {...}} and DomainConfig-style {'domain': DomainConfig(...)}
    return re.findall(r"'(\w+)':\s*(?:\{|DomainConfig\()", domains_str or "")


def _parse_domain_field(domains_str: str, domain: str, field: str) -> str:
    s = domains_str or ""
    idx = s.find(f"'{domain}'")
    if idx < 0:
        return "—"
    # Search within ~800 chars after the domain key to stay within its block
    block = s[idx: idx + 800]
    m = re.search(rf"'{re.escape(field)}':\s*([^,}}\]]+)", block)
    return m.group(1).strip() if m else "—"


def build_domain_diff_df(configs: pd.DataFrame, key_to_short: dict[str, str]) -> pd.DataFrame | None:
    """
    Return a pivoted DataFrame with rows=(domain, field) that differ across configs,
    columns = config short labels. Returns None if no differences found.
    """
    if "domains" not in configs.columns:
        return None

    FIELDS = ["predict", "dropout_rate", "projector", "n_latent_tokens", "freeze", "at_birth"]

    per_config: dict[str, str] = {}
    all_domains: set[str] = set()
    for _, row in configs.drop_duplicates("config_key").iterrows():
        ds = row.get("domains", "") or ""
        per_config[row["config_key"]] = ds
        all_domains.update(_parse_domain_names(ds))

    if not all_domains:
        return None

    records = []
    for domain in sorted(all_domains):
        for field in FIELDS:
            vals = {ck: _parse_domain_field(ds, domain, field) for ck, ds in per_config.items()}
            if len(set(vals.values())) <= 1:
                continue
            records.append(
                {"domain": domain, "field": field}
                | {key_to_short.get(ck, ck[:6]): v for ck, v in vals.items()}
            )

    if not records:
        return None

    return pd.DataFrame(records).set_index(["domain", "field"])


def build_config_df(runs: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = runs.copy()

    # Replace raw attention_scheme strings with aliases before building config_key
    if "attention_scheme" in df.columns:
        alias_col = df["attention_scheme_alias"] if "attention_scheme_alias" in df.columns else pd.Series(dtype=str, index=df.index)
        df["attention_scheme"] = df.apply(
            lambda row: _scheme_to_alias(str(row["attention_scheme"]), row.get("attention_scheme_alias")),
            axis=1,
        )

    if "domains" in df.columns:
        df["no_repeat"] = df["domains"].fillna("").map(_parse_no_repeat)

        # Extract per-domain fields that vary across runs as individual columns
        DOMAIN_FIELDS = ["n_latent_tokens", "projector", "dropout_rate", "freeze"]
        domain_field_cols = []
        for domains_str in df["domains"].fillna(""):
            for domain in _parse_domain_names(domains_str):
                for field in DOMAIN_FIELDS:
                    col = f"{domain}.{field}"
                    if col not in domain_field_cols:
                        domain_field_cols.append(col)

        for col in domain_field_cols:
            domain, field = col.split(".", 1)
            df[col] = df["domains"].fillna("").map(
                lambda s, d=domain, f=field: _parse_domain_field(s, d, f)
            )

        # Only keep those that actually vary
        varying_domain_cols = [c for c in domain_field_cols if df[c].nunique() > 1]
    else:
        varying_domain_cols = []

    config_cols = [c for c in DOMAIN_PARAMS if c in df.columns] + varying_domain_cols
    df[config_cols] = df[config_cols].fillna("(missing)")
    df["config_key"] = df[config_cols].astype(str).apply(
        lambda row: "|".join(f"{c}={row[c]}" for c in config_cols), axis=1
    )
    return df, config_cols


def filter_pairs_vs_ref(pairs: list[dict], ref_ck: str | None) -> list[dict]:
    """If ref_ck is set, keep only pairs where one side is the reference (always as config_b)."""
    if not ref_ck:
        return pairs
    result = []
    for p in pairs:
        if p["config_key_a"] == ref_ck:
            # Swap so reference is always B (ΔAUC = other - ref)
            result.append({**p,
                "config_key_a": p["config_key_b"], "config_key_b": p["config_key_a"],
                "label_a": p["label_b"],           "label_b": p["label_a"],
                "val_a":   p["val_b"],              "val_b":   p["val_a"],
                "mean_auc_a": p["mean_auc_b"],      "mean_auc_b": p["mean_auc_a"],
            })
        elif p["config_key_b"] == ref_ck:
            result.append(p)
    return result


def find_reference_config_key(configs: pd.DataFrame, ref_params: dict) -> str | None:
    """Return the config_key whose params best match ref_params, or None."""
    for _, row in configs.drop_duplicates("config_key").iterrows():
        match = True
        for param, ref_val in ref_params.items():
            if param not in configs.columns:
                continue
            actual = row.get(param, None)
            if actual is None or str(actual) == "(missing)":
                match = False
                break
            if isinstance(ref_val, float):
                try:
                    if abs(float(actual) - ref_val) > ref_val * 0.01:
                        match = False
                        break
                except (ValueError, TypeError):
                    match = False
                    break
            else:
                if str(actual) != str(ref_val):
                    match = False
                    break
        if match:
            return row["config_key"]
    return None


def varying_params(df: pd.DataFrame, config_cols: list[str]) -> list[str]:
    return [c for c in config_cols if df[c].astype(str).nunique() > 1]



def load_auc_for_run(artifact_uri: str) -> pd.DataFrame | None:
    path = Path(urlparse(str(artifact_uri)).path) / "aucs" / "aucs.csv"
    if not path.exists():
        rel = re.sub(r".*mlruns/?", "", str(artifact_uri))
        path = MLRUNS_DIR / rel / "aucs" / "aucs.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner="Loading AUC artifacts…", ttl=300)
def load_all_aucs(exp_ids: tuple[str, ...], run_ids: tuple[str, ...]) -> pd.DataFrame:
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    runs = mlflow.search_runs(experiment_ids=list(exp_ids), filter_string="status = 'FINISHED'")
    runs.columns = [_strip_prefix(c) for c in runs.columns]
    runs = runs.loc[:, ~runs.columns.duplicated(keep="first")]
    runs = runs[runs["run_id"].isin(run_ids)]

    pieces = []
    for _, row in runs.iterrows():
        auc = load_auc_for_run(row.get("artifact_uri", ""))
        if auc is not None and not auc.empty:
            auc["run_id"] = row["run_id"]
            auc["test_fold"] = row.get("test_fold")
            auc["seed"] = row.get("seed")
            pieces.append(auc)

    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def find_clean_pairs(
    config_params: pd.DataFrame,
    vary: list[str],
    key_to_short: dict[str, str],
    n_diffs: int = 1,
) -> list[dict]:
    """Return config pairs differing by exactly n_diffs parameters."""
    pairs = []
    idx = list(config_params.index)
    for ii, i in enumerate(idx):
        for j in idx[ii + 1:]:
            diffs = [p for p in vary if config_params.loc[i, p] != config_params.loc[j, p]]
            if n_diffs == 0:
                if not (1 <= len(diffs) <= 2):
                    continue
            elif len(diffs) != n_diffs:
                continue
            ck_a = config_params.loc[i, "config_key"]
            ck_b = config_params.loc[j, "config_key"]
            pairs.append({
                "params": diffs,
                "param": " + ".join(diffs),
                "val_a": " / ".join(str(config_params.loc[i, p]) for p in diffs),
                "val_b": " / ".join(str(config_params.loc[j, p]) for p in diffs),
                "label_a": key_to_short.get(ck_a, ck_a[:8]),
                "label_b": key_to_short.get(ck_b, ck_b[:8]),
                "config_key_a": ck_a,
                "config_key_b": ck_b,
                "mean_auc_a": config_params.loc[i, "mean_auc"],
                "mean_auc_b": config_params.loc[j, "mean_auc"],
            })
    return pairs


# ---------------------------------------------------------------------------
# Tab helpers
# ---------------------------------------------------------------------------

def render_by_disease_tab(
    auc_all: pd.DataFrame,
    filtered: pd.DataFrame,
    vary: list[str],
    key_to_short: dict[str, str],
    disease_names: dict[int, str],
    *,
    show_legend: bool = False,
    n_diffs: int = 1,
    ref_ck: str | None = None,
) -> None:
    # Disease selector
    if disease_names:
        disease_options = [f"{tid}: {name}" for tid, name in sorted(disease_names.items())]
        selected_disease_str = st.selectbox("Disease", disease_options)
        token_id = int(selected_disease_str.split(":")[0])
    else:
        token_id = st.number_input("Disease token_id", min_value=0, value=0, step=1)

    sex_options = ["both", "male", "female"]
    selected_sex = st.selectbox("Sex", sex_options)

    # Setup selector
    all_labels = sorted(key_to_short.values())
    setup_mode = st.radio("Setups to show", ["All", "Select"], horizontal=True, key="disease_setup_mode")
    if setup_mode == "Select":
        cols = st.columns(len(all_labels))
        selected_labels = [
            label for label, col in zip(all_labels, cols)
            if col.checkbox(label, value=False, key=f"setup_cb_{label}")
        ]
        if vary:
            legend_rows = []
            for k, short in key_to_short.items():
                rows_k = filtered[filtered["config_key"] == k]
                row_data = rows_k.iloc[0] if not rows_k.empty else pd.Series(dtype=str)
                legend_rows.append({"label": short} | _display_params(vary, row_data))
            st.dataframe(pd.DataFrame(legend_rows).set_index("label"), use_container_width=True)
    else:
        selected_labels = all_labels

    st.markdown("---")

    if auc_all.empty:
        st.warning("No AUC artifacts found for the filtered runs.")
        return

    if "token_id" not in auc_all.columns:
        st.warning("AUC CSV does not have a `token_id` column.")
        return

    auc_dis = auc_all[auc_all["token_id"] == token_id].copy()
    if selected_sex != "both" and "sex" in auc_dis.columns:
        auc_dis = auc_dis[auc_dis["sex"] == selected_sex]

    if auc_dis.empty:
        st.warning(f"No AUC data found for token_id={token_id}.")
        return

    run_to_config = filtered.set_index("run_id")["config_key"].to_dict()
    auc_dis["config_key"] = auc_dis["run_id"].map(run_to_config)

    group_cols = ["config_key"]
    if "sex" in auc_dis.columns:
        group_cols.append("sex")
    if "age_start" in auc_dis.columns:
        group_cols += ["age_start", "age_end"]

    agg = (
        auc_dis.groupby(group_cols, dropna=False)["auc_delong"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    agg["se"] = agg["std"] / np.sqrt(agg["n"])
    agg["config_label"] = agg["config_key"].map(key_to_short)
    agg = agg[agg["config_label"].isin(selected_labels)]

    if agg.empty:
        st.warning("No setups selected.")
        return

    if "age_start" in agg.columns:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        st.subheader("AUC by age stratum")
        sex_vals = list(agg["sex"].unique()) if "sex" in agg.columns else ["all"]
        n_sex = len(sex_vals)

        _colors = px.colors.qualitative.Plotly
        _symbols = ["circle", "square", "diamond", "triangle-up", "triangle-down",
                    "cross", "x", "star", "hexagon", "pentagon"]

        all_labels_ordered = sorted(agg["config_label"].unique())
        style_map = {
            lbl: {"color": _colors[i % len(_colors)], "symbol": _symbols[i % len(_symbols)]}
            for i, lbl in enumerate(all_labels_ordered)
        }

        ref_label = key_to_short.get(ref_ck) if ref_ck else None

        # Build per-label config description for hover
        label_to_config_str: dict[str, str] = {}
        for ck, lbl in key_to_short.items():
            rows = filtered[filtered["config_key"] == ck]
            if rows.empty or not vary:
                label_to_config_str[lbl] = lbl
                continue
            row = rows.iloc[0]
            parts = []
            label_to_config_str[lbl] = "<br>".join(
                f"{p}: {v}" for p, v in _display_params(vary, row).items()
            )

        fig = make_subplots(rows=1, cols=n_sex, subplot_titles=sex_vals, shared_yaxes=True)

        for col_idx, sx in enumerate(sex_vals, start=1):
            sub = agg[agg["sex"] == sx] if "sex" in agg.columns else agg
            sorted_labels = sorted(sub["config_label"].unique(),
                                   key=lambda l: (l == ref_label, l))
            show_legend = col_idx == 1

            for label in sorted_labels:
                grp = sub[sub["config_label"] == label].sort_values("age_start")
                is_ref = label == ref_label
                color = "black" if is_ref else style_map[label]["color"]
                symbol = "star" if is_ref else style_map[label]["symbol"]
                lw = 2.5 if is_ref else 1.8

                x = grp["age_start"].tolist()
                y = grp["mean"].tolist()
                y_upper = (grp["mean"] + grp["se"]).tolist()
                y_lower = (grp["mean"] - grp["se"]).tolist()

                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode="lines+markers",
                    name=label + (" [ref]" if is_ref else ""),
                    line=dict(color=color, width=lw),
                    marker=dict(symbol=symbol, size=8 if is_ref else 6),
                    legendgroup=label,
                    showlegend=show_legend,
                    hovertemplate=f"<b>{label}</b><br>{label_to_config_str.get(label, '')}<br>Age: %{{x}}<br>AUROC: %{{y:.4f}}<extra></extra>",
                ), row=1, col=col_idx)

                fig.add_trace(go.Scatter(
                    x=x + x[::-1],
                    y=y_upper + y_lower[::-1],
                    fill="toself",
                    fillcolor=color,
                    opacity=0.15,
                    line=dict(width=0),
                    legendgroup=label,
                    showlegend=False,
                    hoverinfo="skip",
                ), row=1, col=col_idx)

            # Dashed line at 0.5
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                          line_width=0.8, row=1, col=col_idx)

        fig.update_layout(
            title=disease_names.get(token_id, f"token {token_id}"),
            height=550,
            yaxis_title="Mean AUROC ± SE",
            xaxis_title="Age start",
            legend=dict(orientation="v", x=1.02, y=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader("AUROC per configuration")
        st.dataframe(agg, use_container_width=True)

    if show_legend and vary:
        with st.expander("Configuration legend"):
            legend_rows = []
            for k, short in key_to_short.items():
                rows_k = filtered[filtered["config_key"] == k]
                row_data = rows_k.iloc[0] if not rows_k.empty else pd.Series(dtype=str)
                legend_rows.append({"label": short} | _display_params(vary, row_data))
            st.dataframe(pd.DataFrame(legend_rows).set_index("label"), use_container_width=True)

    # Summary table
    st.subheader("Summary table")
    display_cols = ["config_label"] + (["sex"] if "sex" in agg.columns else []) + \
                   (["age_start", "age_end"] if "age_start" in agg.columns else []) + \
                   ["mean", "se", "n"]
    st.dataframe(agg[display_cols].round(4), use_container_width=True)

    # Per-fold detail
    with st.expander("Per-fold detail"):
        detail = auc_dis.copy()
        detail["config_label"] = detail["config_key"].map(key_to_short)
        show_cols = ["config_label", "test_fold", "seed"] + \
                    ([c for c in ["sex", "age_start", "age_end"] if c in detail.columns]) + \
                    ["auc_delong"]
        show_cols = [c for c in show_cols if c in detail.columns]
        st.dataframe(detail[show_cols].sort_values(["config_label", "test_fold"]).round(4),
                     use_container_width=True)

    # -----------------------------------------------------------------------
    # Linear regression: which hyperparams explain AUC differences?
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Feature importance — linear regression")
    st.caption(
        "Mean AUROC per config (aggregated over folds, seeds, sex, and age strata) "
        "is regressed on the hyperparameters that vary across configurations. "
        "Coefficients show the estimated effect of each parameter value relative to "
        "an automatically chosen reference level."
    )

    try:
        config_auc = (
            auc_dis.groupby("config_key", dropna=False)["auc_delong"]
            .mean()
            .reset_index()
            .rename(columns={"auc_delong": "mean_auc"})
        )

        if len(config_auc) < 2:
            st.info("Need at least 2 configurations to fit a regression.")
        elif not vary:
            st.info("No varying parameters — nothing to regress on.")
        else:
            config_params = (
                filtered[["config_key"] + vary]
                .drop_duplicates("config_key")
                .merge(config_auc, on="config_key")
            )

            clean_pairs = filter_pairs_vs_ref(
                find_clean_pairs(config_params, vary, key_to_short, n_diffs=n_diffs), ref_ck
            )

            if not clean_pairs:
                st.info(
                    "No pairs of configurations differ by exactly one parameter. "
                    "The regression cannot cleanly isolate individual effects."
                )
            else:
                st.caption(f"Found **{len(clean_pairs)}** clean pair(s) differing by exactly one parameter.")

                pair_rows = []
                for pair in clean_pairs:
                    pair_rows.append({
                        "param": pair["param"],
                        "value_a": pair["val_a"],
                        "value_b": pair["val_b"],
                        "config_a": pair["label_a"],
                        "config_b": pair["label_b"],
                        "delta_auc": pair["mean_auc_a"] - pair["mean_auc_b"],
                        "auc_a": pair["mean_auc_a"],
                        "auc_b": pair["mean_auc_b"],
                    })
                pairs_df = pd.DataFrame(pair_rows)

                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, max(3, len(pairs_df) * 0.5)))
                labels = [
                    f"{r['param']}: {r['value_a']} vs {r['value_b']}\n({r['config_a']} vs {r['config_b']})"
                    for _, r in pairs_df.iterrows()
                ]
                colors = ["steelblue" if d >= 0 else "tomato" for d in pairs_df["delta_auc"]]
                ax.barh(labels, pairs_df["delta_auc"], color=colors)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("ΔAUROC (A − B)")
                ax.set_title("Effect of single-parameter changes on mean AUROC")
                plt.tight_layout()
                st.pyplot(fig)

                st.dataframe(pairs_df.round(5), use_container_width=True)

    except ImportError:
        st.warning("Install scikit-learn to enable the regression analysis (`uv add scikit-learn`).")


def render_auc_vs_cases_tab(
    auc_all: pd.DataFrame,
    filtered: pd.DataFrame,
    vary: list[str],
    key_to_short: dict[str, str],
    disease_names: dict[int, str],
    n_diffs: int = 1,
    ref_ck: str | None = None,
) -> None:
    st.subheader("ΔAUC vs number of cases per disease")
    st.caption(
        "Each point is a disease. X = total cases summed across test folds, "
        "Y = ΔAUC between two configurations that differ by exactly one parameter."
    )

    if auc_all.empty:
        st.warning("No AUC artifacts found for the filtered runs.")
        return

    if not vary:
        st.info("All configurations share the same hyperparameters — no pairs to compare.")
        return

    # Build config_params with mean_auc across all diseases/strata
    run_to_config = filtered.set_index("run_id")["config_key"].to_dict()
    auc_keyed = auc_all.copy()
    auc_keyed["config_key"] = auc_keyed["run_id"].map(run_to_config)

    config_auc_global = (
        auc_keyed.groupby("config_key", dropna=False)["auc_delong"]
        .mean()
        .reset_index()
        .rename(columns={"auc_delong": "mean_auc"})
    )
    config_params = (
        filtered[["config_key"] + vary]
        .drop_duplicates("config_key")
        .merge(config_auc_global, on="config_key", how="left")
    )

    clean_pairs = filter_pairs_vs_ref(
        find_clean_pairs(config_params, vary, key_to_short), ref_ck
    )

    if not clean_pairs:
        st.info("No pairs of configurations differ by exactly one parameter.")
        return

    # Controls
    col_left, col_mid, col_right = st.columns([3, 1, 1])
    with col_left:
        pair_labels = [
            f"{p['param']}: {p['val_a']} vs {p['val_b']}  ({p['label_a']} vs {p['label_b']})"
            for p in clean_pairs
        ]
        selected_idx = st.selectbox(
            "Pair to compare",
            range(len(clean_pairs)),
            format_func=lambda i: pair_labels[i],
        )
    with col_mid:
        agg_mode = st.radio(
            "Aggregate by",
            ["Combined", "By sex", "By sex + age bracket"],
        )
    with col_right:
        plot_type = st.radio(
            "Plot type",
            ["Scatter", "Density map", "Scatter + density"],
        )
        show_smooth = st.checkbox("Smooth curve", value=True)

    min_cases = st.slider("Minimum number of cases", min_value=1, max_value=500, value=10, step=1)

    pair = clean_pairs[selected_idx]
    ck_a, ck_b = pair["config_key_a"], pair["config_key_b"]

    run_ids_a = set(filtered.loc[filtered["config_key"] == ck_a, "run_id"])
    run_ids_b = set(filtered.loc[filtered["config_key"] == ck_b, "run_id"])

    auc_a = auc_keyed[auc_keyed["run_id"].isin(run_ids_a)].copy()
    auc_b = auc_keyed[auc_keyed["run_id"].isin(run_ids_b)].copy()

    if auc_a.empty or auc_b.empty:
        st.warning("AUC data missing for one of the selected configurations.")
        return

    # Determine stratification columns
    base_group = ["token_id"]
    has_sex = "sex" in auc_all.columns
    has_age = "age_start" in auc_all.columns

    if agg_mode in ("By sex", "By sex + age bracket") and has_sex:
        base_group.append("sex")
    if agg_mode == "By sex + age bracket" and has_age:
        base_group += ["age_start", "age_end"]

    # n_case: deduplicate at finest granularity (removes duplicate seeds),
    # then sum over whatever strata are not in base_group.
    finest_dedup = ["token_id"]
    if "sex" in auc_a.columns:
        finest_dedup.append("sex")
    if "age_start" in auc_a.columns:
        finest_dedup += ["age_start", "age_end"]
    if "test_fold" in auc_a.columns:
        finest_dedup.append("test_fold")
    n_case_df = (
        auc_a.drop_duplicates(subset=finest_dedup)
        .groupby(base_group, as_index=False)["n_case"]
        .sum()
    )

    # Mean AUC per (disease, stratum) for each config
    mean_a = (
        auc_a.groupby(base_group, as_index=False)["auc_delong"]
        .mean()
        .rename(columns={"auc_delong": "auc_a"})
    )
    mean_b = (
        auc_b.groupby(base_group, as_index=False)["auc_delong"]
        .mean()
        .rename(columns={"auc_delong": "auc_b"})
    )

    scatter_df = (
        mean_a
        .merge(mean_b, on=base_group)
        .merge(n_case_df, on=base_group)
    )
    scatter_df["delta_auc"] = scatter_df["auc_a"] - scatter_df["auc_b"]
    scatter_df["disease_name"] = scatter_df["token_id"].map(disease_names).fillna(
        scatter_df["token_id"].astype(str)
    )

    scatter_df = scatter_df[scatter_df["n_case"] >= min_cases].copy()

    if scatter_df.empty:
        st.warning("No rows with n_case > 0 after merging configs.")
        return

    # Color column
    color_col = None
    if agg_mode == "By sex" and "sex" in scatter_df.columns:
        color_col = "sex"
    elif agg_mode == "By sex + age bracket" and "age_start" in scatter_df.columns:
        scatter_df["stratum"] = (
            scatter_df["sex"] + " "
            + scatter_df["age_start"].astype(int).astype(str)
            + "–"
            + scatter_df["age_end"].astype(int).astype(str)
        )
        color_col = "stratum"

    import math
    import plotly.graph_objects as go

    # Log-transform x so all plot types (scatter, heatmap, contour) bin consistently.
    scatter_df["log_n_case"] = np.log10(scatter_df["n_case"])

    log_min = math.floor(scatter_df["log_n_case"].min())
    log_max = math.ceil(scatter_df["log_n_case"].max())
    tick_vals = list(range(log_min, log_max + 1))
    tick_texts = [
        f"{10**v:,}" if 10**v < 10_000 else f"{10**v // 1000}K"
        for v in tick_vals
    ]

    y_label = f"ΔAUC  ({pair['label_a']} − {pair['label_b']})"
    plot_title = (
        f"ΔAUC vs cases — {pair['param']}: "
        f"{pair['val_a']} vs {pair['val_b']}"
    )
    hover_cols: dict = {
        "token_id": True,
        "n_case": True,
        "log_n_case": False,   # hide raw log column from hover
        "auc_a": ":.4f",
        "auc_b": ":.4f",
    }
    if color_col and color_col not in ("sex",):
        hover_cols[color_col] = False

    if plot_type == "Scatter":
        fig = px.scatter(
            scatter_df,
            x="log_n_case", y="delta_auc",
            color=color_col,
            hover_name="disease_name",
            hover_data=hover_cols,
            labels={"log_n_case": "Number of cases", "delta_auc": y_label},
            title=plot_title,
        )
        fig.update_traces(marker=dict(size=8, opacity=0.5))

    elif plot_type == "Density map":
        fig = px.density_heatmap(
            scatter_df,
            x="log_n_case", y="delta_auc",
            nbinsx=20, nbinsy=60,
            labels={"log_n_case": "Number of cases", "delta_auc": y_label},
            title=plot_title,
            color_continuous_scale="Viridis",
        )

    else:  # Scatter + density
        fig = px.scatter(
            scatter_df,
            x="log_n_case", y="delta_auc",
            color=color_col,
            hover_name="disease_name",
            hover_data=hover_cols,
            labels={"log_n_case": "Number of cases", "delta_auc": y_label},
            title=plot_title,
        )
        fig.update_traces(marker=dict(size=8, opacity=0.45))

        # Density contour uses the same log-transformed x → bins align with scatter
        fig.add_trace(go.Histogram2dContour(
            x=scatter_df["log_n_case"],
            y=scatter_df["delta_auc"],
            colorscale="Greys",
            reversescale=True,
            showscale=False,
            line=dict(width=1),
            opacity=0.5,
            name="density",
            showlegend=False,
            hoverinfo="skip",
            ncontours=12,
        ))
        # Push density behind scatter traces
        fig.data = fig.data[-1:] + fig.data[:-1]

    if show_smooth:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
            def _smooth(x, y, frac=0.4):
                if len(x) < 5:
                    return x, y
                out = _lowess(y, x, frac=frac, return_sorted=True)
                return out[:, 0], out[:, 1]
        except ImportError:
            def _smooth(x, y, frac=0.4):
                n_bins = max(5, len(x) // 5)
                bins = np.linspace(x.min(), x.max(), n_bins + 1)
                idx = np.digitize(x, bins) - 1
                xs, ys = [], []
                for i in range(n_bins):
                    mask = idx == i
                    if mask.sum() >= 2:
                        xs.append(x[mask].mean())
                        ys.append(y[mask].mean())
                return np.array(xs), np.array(ys)

        if color_col and color_col in scatter_df.columns:
            # One smooth curve per color group, matching its color
            palette = px.colors.qualitative.Plotly
            for gi, grp in enumerate(sorted(scatter_df[color_col].unique())):
                sub = scatter_df[scatter_df[color_col] == grp]
                xs, ys = _smooth(sub["log_n_case"].values, sub["delta_auc"].values)
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="lines",
                    name=f"{grp} (smooth)",
                    line=dict(width=2.5, color=palette[gi % len(palette)]),
                    showlegend=True,
                ))
        else:
            xs, ys = _smooth(scatter_df["log_n_case"].values, scatter_df["delta_auc"].values)
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="lines",
                name="smooth",
                line=dict(width=2.5, color="black"),
                showlegend=False,
            ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        height=550,
        xaxis=dict(
            title="Number of cases (summed across folds)",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_texts,
        ),
        yaxis_title=y_label,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Data table"):
        display_cols = base_group + ["disease_name", "n_case", "auc_a", "auc_b", "delta_auc"]
        if color_col == "stratum":
            display_cols.append("stratum")
        display_cols = [c for c in display_cols if c in scatter_df.columns]
        st.dataframe(
            scatter_df[display_cols].sort_values("n_case", ascending=False).round(4),
            use_container_width=True,
        )


def render_consensus_tab(
    auc_all: pd.DataFrame,
    filtered: pd.DataFrame,
    vary: list[str],
    key_to_short: dict[str, str],
    disease_names: dict[int, str],
    n_diffs: int = 1,
    ref_ck: str | None = None,
) -> None:
    st.subheader("Consensus effect sizes across diseases")
    st.caption(
        "For each config pair, ΔAUC is computed per disease (mean over sex × age bins, "
        "after the global bin filter). Consensus metrics aggregate those per-disease values:\n\n"
        "- **Weighted mean ΔAUC** — weighted average of per-disease ΔAUC, with weights = total n_case "
        "for that disease (more prevalent diseases contribute more).\n"
        "- **Median ΔAUC** — median across diseases; robust to outliers.\n"
        "- **% improved** — fraction of diseases where ΔAUC > 0 (config A beats B).\n"
        "- **Wilcoxon p** — two-sided Wilcoxon signed-rank test on the per-disease ΔAUC values "
        "against H₀: median = 0. A small p indicates the shift is unlikely by chance.\n\n"
        "All metrics are computed on diseases passing the **Min cases per bin** and "
        "**Min cases (per disease)** filters."
    )

    if auc_all.empty:
        st.warning("No AUC artifacts found for the filtered runs.")
        return
    if not vary:
        st.info("All configurations share the same hyperparameters — no pairs to compare.")
        return

    min_cases = st.slider(
        "Min cases per disease", min_value=1, max_value=500, value=20, step=5,
        key="consensus_mincases",
    )

    run_to_config = filtered.set_index("run_id")["config_key"].to_dict()
    auc_keyed = auc_all.copy()
    auc_keyed["config_key"] = auc_keyed["run_id"].map(run_to_config)

    config_auc_global = (
        auc_keyed.groupby("config_key", dropna=False)["auc_delong"]
        .mean().reset_index().rename(columns={"auc_delong": "mean_auc"})
    )
    config_params = (
        filtered[["config_key"] + vary]
        .drop_duplicates("config_key")
        .merge(config_auc_global, on="config_key", how="left")
    )
    clean_pairs = filter_pairs_vs_ref(
        find_clean_pairs(config_params, vary, key_to_short, n_diffs=n_diffs), ref_ck
    )

    if not clean_pairs:
        st.info("No qualifying config pairs found.")
        return

    from scipy.stats import wilcoxon

    rows = []
    for pair in clean_pairs:
        ck_a, ck_b = pair["config_key_a"], pair["config_key_b"]
        run_ids_a = set(filtered.loc[filtered["config_key"] == ck_a, "run_id"])
        run_ids_b = set(filtered.loc[filtered["config_key"] == ck_b, "run_id"])

        auc_a = auc_keyed[auc_keyed["run_id"].isin(run_ids_a)]
        auc_b = auc_keyed[auc_keyed["run_id"].isin(run_ids_b)]
        if auc_a.empty or auc_b.empty:
            continue

        mean_a = auc_a.groupby("token_id", as_index=False)["auc_delong"].mean().rename(columns={"auc_delong": "auc_a"})
        mean_b = auc_b.groupby("token_id", as_index=False)["auc_delong"].mean().rename(columns={"auc_delong": "auc_b"})

        # n_case per disease (deduplicated across folds/seeds)
        dedup = ["token_id"]
        if "test_fold" in auc_a.columns:
            dedup.append("test_fold")
        n_case_df = (
            auc_a.drop_duplicates(subset=dedup)
            .groupby("token_id", as_index=False)["n_case"].sum()
        )

        merged = mean_a.merge(mean_b, on="token_id").merge(n_case_df, on="token_id")
        merged = merged[merged["n_case"] >= min_cases]
        if merged.empty:
            continue

        merged["delta"] = merged["auc_a"] - merged["auc_b"]

        w_mean = (merged["delta"] * merged["n_case"]).sum() / merged["n_case"].sum()
        median = merged["delta"].median()
        pct_improved = (merged["delta"] > 0).mean() * 100

        try:
            _, pval = wilcoxon(merged["delta"].values)
        except Exception:
            pval = float("nan")

        rows.append({
            "pair": f"{pair['label_a']} vs {pair['label_b']}",
            "params": pair["param"],
            f"{pair['label_a']}": pair["val_a"],
            f"{pair['label_b']}": pair["val_b"],
            "n_diseases": len(merged),
            "weighted_mean_ΔAUC": round(w_mean, 5),
            "median_ΔAUC": round(median, 5),
            "pct_improved": round(pct_improved, 1),
            "wilcoxon_p": round(pval, 4) if not pd.isna(pval) else float("nan"),
        })

    if not rows:
        st.warning("No data available for any pair after filtering.")
        return

    df = pd.DataFrame(rows).set_index("pair")
    st.dataframe(df, use_container_width=True)


def render_effect_by_disease_tab(
    auc_all: pd.DataFrame,
    filtered: pd.DataFrame,
    vary: list[str],
    key_to_short: dict[str, str],
    disease_names: dict[int, str],
    n_diffs: int = 1,
    ref_ck: str | None = None,
) -> None:
    st.subheader("Effect size by disease")
    st.caption(
        "Diseases ranked by ΔAUC for a selected config pair. "
        "Useful for spotting which disease categories benefit (or suffer) most from a hyperparameter change."
    )

    if auc_all.empty:
        st.warning("No AUC artifacts found for the filtered runs.")
        return
    if not vary:
        st.info("All configurations share the same hyperparameters — no pairs to compare.")
        return

    run_to_config = filtered.set_index("run_id")["config_key"].to_dict()
    auc_keyed = auc_all.copy()
    auc_keyed["config_key"] = auc_keyed["run_id"].map(run_to_config)

    config_auc_global = (
        auc_keyed.groupby("config_key", dropna=False)["auc_delong"]
        .mean()
        .reset_index()
        .rename(columns={"auc_delong": "mean_auc"})
    )
    config_params = (
        filtered[["config_key"] + vary]
        .drop_duplicates("config_key")
        .merge(config_auc_global, on="config_key", how="left")
    )

    clean_pairs = filter_pairs_vs_ref(
        find_clean_pairs(config_params, vary, key_to_short), ref_ck
    )
    if not clean_pairs:
        st.info("No pairs of configurations differ by exactly one parameter.")
        return

    # Controls
    col_pair, col_sex, col_age = st.columns([3, 1, 2])
    with col_pair:
        pair_labels = [
            f"{p['param']}: {p['val_a']} vs {p['val_b']}  ({p['label_a']} vs {p['label_b']})"
            for p in clean_pairs
        ]
        selected_idx = st.selectbox(
            "Pair to compare",
            range(len(clean_pairs)),
            format_func=lambda i: pair_labels[i],
            key="ranked_pair",
        )
    with col_sex:
        sex_filter = st.selectbox("Sex", ["combined", "male", "female"], key="ranked_sex")
    with col_age:
        has_age = "age_start" in auc_all.columns
        age_options = ["all ages"]
        if has_age:
            age_brackets = sorted(auc_all["age_start"].dropna().unique().astype(int))
            age_options += [f"{a}–{a+5}" for a in age_brackets]
        age_filter = st.selectbox("Age bracket", age_options, key="ranked_age")

    col_cases, col_n = st.columns([2, 1])
    with col_cases:
        min_cases = st.slider("Min cases (per disease)", min_value=1, max_value=500, value=20, step=5, key="ranked_mincases")
    with col_n:
        top_n = st.number_input("Show top N diseases (0 = all)", min_value=0, value=0, step=10, key="ranked_topn")

    pair = clean_pairs[selected_idx]
    ck_a, ck_b = pair["config_key_a"], pair["config_key_b"]

    run_ids_a = set(filtered.loc[filtered["config_key"] == ck_a, "run_id"])
    run_ids_b = set(filtered.loc[filtered["config_key"] == ck_b, "run_id"])

    auc_a = auc_keyed[auc_keyed["run_id"].isin(run_ids_a)].copy()
    auc_b = auc_keyed[auc_keyed["run_id"].isin(run_ids_b)].copy()

    if auc_a.empty or auc_b.empty:
        st.warning("AUC data missing for one of the selected configurations.")
        return

    # Apply filters
    if sex_filter != "combined" and "sex" in auc_a.columns:
        auc_a = auc_a[auc_a["sex"] == sex_filter]
        auc_b = auc_b[auc_b["sex"] == sex_filter]

    if age_filter != "all ages" and has_age:
        age_start = int(age_filter.split("–")[0])
        auc_a = auc_a[auc_a["age_start"] == age_start]
        auc_b = auc_b[auc_b["age_start"] == age_start]

    mean_a = auc_a.groupby("token_id", as_index=False)["auc_delong"].mean().rename(columns={"auc_delong": "auc_a"})
    mean_b = auc_b.groupby("token_id", as_index=False)["auc_delong"].mean().rename(columns={"auc_delong": "auc_b"})

    # n_case: deduplicate across seeds/folds before summing
    dedup_cols = ["token_id"]
    if "sex" in auc_a.columns and sex_filter == "combined":
        dedup_cols.append("sex")
    if has_age and age_filter == "all ages":
        dedup_cols += ["age_start", "age_end"]
    if "test_fold" in auc_a.columns:
        dedup_cols.append("test_fold")
    n_case_df = (
        auc_a.drop_duplicates(subset=dedup_cols)
        .groupby("token_id", as_index=False)["n_case"]
        .sum()
    )

    ranked = mean_a.merge(mean_b, on="token_id").merge(n_case_df[["token_id", "n_case"]], on="token_id")
    ranked["delta_auc"] = ranked["auc_a"] - ranked["auc_b"]
    ranked["disease_name"] = ranked["token_id"].map(disease_names).fillna(ranked["token_id"].astype(str))
    ranked = ranked[ranked["n_case"] >= min_cases].sort_values("delta_auc")

    if ranked.empty:
        st.warning("No diseases pass the minimum cases filter.")
        return

    if top_n > 0:
        bottom = ranked.head(top_n)
        top = ranked.tail(top_n)
        ranked_plot = pd.concat([bottom, top]).drop_duplicates("token_id")
    else:
        ranked_plot = ranked

    import plotly.graph_objects as go

    colors = ["tomato" if d < 0 else "steelblue" for d in ranked_plot["delta_auc"]]
    fig = go.Figure(go.Bar(
        x=ranked_plot["delta_auc"],
        y=ranked_plot["disease_name"],
        orientation="h",
        marker_color=colors,
        customdata=ranked_plot[["token_id", "n_case", "auc_a", "auc_b"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "token_id: %{customdata[0]}<br>"
            "n_case: %{customdata[1]}<br>"
            f"{pair['label_a']} AUC: %{{customdata[2]:.4f}}<br>"
            f"{pair['label_b']} AUC: %{{customdata[3]:.4f}}<br>"
            "ΔAUC: %{x:.4f}<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.6)
    fig.update_layout(
        height=max(400, len(ranked_plot) * 22 + 80),
        xaxis_title=f"ΔAUC  ({pair['label_a']} − {pair['label_b']})",
        yaxis=dict(automargin=True, tickfont=dict(size=11)),
        margin=dict(l=0, r=20, t=40, b=40),
        title=f"{pair['param']}: {pair['val_a']} vs {pair['val_b']}",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Data table"):
        st.dataframe(
            ranked_plot[["disease_name", "token_id", "n_case", "auc_a", "auc_b", "delta_auc"]]
            .sort_values("delta_auc")
            .round(4),
            use_container_width=True,
        )


def render_autoimmune_tab(
    auc_all: pd.DataFrame,
    filtered: pd.DataFrame,
    key_to_short: dict[str, str],
    disease_names: dict[int, str],
    ref_ck: str | None = None,
) -> None:
    st.subheader("Autoimmune disease panel")

    if auc_all.empty:
        st.warning("No AUC artifacts found for the filtered runs.")
        return

    token_to_icd3 = load_autoimmune_token_ids()
    if not token_to_icd3:
        st.warning("No autoimmune disease tokens found in the tokenizer.")
        return

    if not auc_all["token_id"].isin(token_to_icd3).any():
        st.warning("None of the 17 autoimmune diseases found in the AUC data.")
        return

    summary, per_disease_fold, per_disease_mean = compute_autoimmune_scores(auc_all, filtered, key_to_short)
    if summary.empty:
        st.warning("Could not compute scores.")
        return

    sex_options = sorted(summary["sex"].unique())
    pd_sub_all = per_disease_fold.copy()
    configs_available = sorted(pd_sub_all["config_label"].unique())

    col_sex, col_ref, col_folds = st.columns([2, 3, 2])
    with col_sex:
        selected_sex = st.radio("Sex", sex_options, horizontal=True, key="autoimmune_sex")
    with col_ref:
        ref_options = ["(none)"] + configs_available
        default_ref = key_to_short.get(ref_ck) if ref_ck and key_to_short.get(ref_ck) in configs_available else "(none)"
        selected_ref = st.radio("Reference config", ref_options, horizontal=True,
                                index=ref_options.index(default_ref), key="autoimmune_ref")
        ref_label = selected_ref if selected_ref != "(none)" else None
    with col_folds:
        show_folds = st.radio("Folds", ["Mean only", "All folds"], horizontal=True, key="autoimmune_folds") == "All folds"

    # ── Best config ───────────────────────────────────────────────────────────
    fold_means = summary[(summary["sex"] == selected_sex) & (summary["test_fold"] == "mean")]
    if not fold_means.empty:
        best_row = fold_means.loc[fold_means["score_unweighted"].idxmax()]
        best_label = best_row["config_label"]
        best_score = best_row["score_unweighted"]
        st.info(
            f"**Best config**: {best_label} — mean AUROC {best_score:.4f} "
            f"(unweighted mean over {len(AUTOIMMUNE_ICD)} diseases, averaged over folds)"
        )

    # ── Build disease labels ──────────────────────────────────────────────────
    icd3_to_name: dict[str, str] = {}
    for tid, name in disease_names.items():
        code = name[:3]
        if code in AUTOIMMUNE_ICD and code not in icd3_to_name:
            icd3_to_name[code] = name

    # Keep AUTOIMMUNE_ICD order, only diseases present in the data
    pd_sub = per_disease_fold[per_disease_fold["sex"] == selected_sex].copy()
    diseases_present = [d for d in AUTOIMMUNE_ICD if d in pd_sub["icd3"].unique()]
    configs = sorted(pd_sub["config_label"].unique())
    all_folds = sorted(pd_sub["test_fold"].dropna().unique())

    import plotly.graph_objects as go

    # ── Build matrix rows (diseases with NaN separators) ─────────────────────
    y_labels: list[str] = []
    y_diseases: list[str | None] = []
    for i, d in enumerate(diseases_present):
        if i > 0:
            y_labels.append("")
            y_diseases.append(None)
        y_labels.append(icd3_to_name.get(d, d))
        y_diseases.append(d)

    # ── Build matrix columns (config × fold + mean, with NaN separators) ─────
    x_labels: list[str] = []
    x_keys: list[tuple | None] = []  # (config, fold_or_"mean") or None for separator
    for ci, config in enumerate(configs):
        if ci > 0:
            x_labels.append("")
            x_keys.append(None)
        if show_folds:
            for fold in all_folds:
                x_labels.append(f"{config} f{fold}")
                x_keys.append((config, fold))
        x_labels.append(f"{config} mean")
        x_keys.append((config, "mean"))

    # ── Fill matrix ───────────────────────────────────────────────────────────
    matrix = np.full((len(y_labels), len(x_labels)), np.nan)
    for ri, d in enumerate(y_diseases):
        if d is None:
            continue
        d_data = pd_sub[pd_sub["icd3"] == d]
        for ci, key in enumerate(x_keys):
            if key is None:
                continue
            config, fold = key
            cfg_data = d_data[d_data["config_label"] == config]
            if fold == "mean":
                val = cfg_data["auroc"].mean()
            else:
                row = cfg_data[cfg_data["test_fold"] == fold]
                val = row["auroc"].iloc[0] if not row.empty else np.nan
            matrix[ri, ci] = val

    # ── Apply ΔAUROC vs fold-mean of reference ───────────────────────────────
    if ref_label and ref_label in configs:
        # One reference value per disease: mean across folds of the reference config
        ref_mean_by_disease = (
            pd_sub[pd_sub["config_label"] == ref_label]
            .groupby("icd3")["auroc"].mean()
        )
        ref_vec = np.array([
            ref_mean_by_disease.get(d, np.nan) if d is not None else np.nan
            for d in y_diseases
        ])  # shape (n_rows,)
        plot_matrix = matrix - ref_vec[:, np.newaxis]
        zmax = float(np.nanmax(np.abs(plot_matrix)))
        zmin, zmid = -zmax, 0
        colorbar_title = f"ΔAUROC vs mean({ref_label})"
    else:
        plot_matrix = matrix
        flat = plot_matrix[~np.isnan(plot_matrix)]
        zmid = float(np.median(flat)) if flat.size else 0.5
        zmin, zmax = float(flat.min()) if flat.size else 0, float(flat.max()) if flat.size else 1
        colorbar_title = "AUROC"

    text_matrix = [
        [f"{v:.3f}" if not np.isnan(v) else "" for v in row]
        for row in plot_matrix
    ]

    plot_matrix_T = plot_matrix.T
    text_matrix_T = list(map(list, zip(*text_matrix)))

    fig = go.Figure(go.Heatmap(
        z=plot_matrix_T,
        x=y_labels,
        y=x_labels,
        colorscale="RdBu",
        zmid=zmid,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title=colorbar_title),
        text=text_matrix_T,
        texttemplate="%{text}",
        hovertemplate="<b>%{x}</b><br>%{y}<br>" + colorbar_title + ": %{z:.4f}<extra></extra>",
    ))
    n_cols = len(diseases_present) + (len(diseases_present) - 1)
    fig.update_layout(
        height=max(400, len(x_labels) * 28 + 120),
        xaxis=dict(tickangle=-45, side="bottom"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=0, t=40, b=120),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Summary scores table ──────────────────────────────────────────────────
    with st.expander("Summary scores (per fold)"):
        sub = (
            summary[summary["sex"] == selected_sex]
            .drop(columns="sex")
            .sort_values(["config_label", "test_fold"])
            .round(4)
        )
        st.dataframe(sub, use_container_width=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Experiment Comparator", layout="wide")
    st.title("Experiment Comparator — AUC by disease")

    # --- Sidebar: experiment selector
    with st.sidebar:
        st.header("Experiments")
        mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
        all_exps = mlflow.search_experiments()
        exp_options = {f"{e.name} ({e.experiment_id})": e.experiment_id for e in all_exps}
        default_keys = [k for k, v in exp_options.items() if v == DEFAULT_EXP_ID]
        selected_keys = st.multiselect(
            "Select experiments",
            options=list(exp_options.keys()),
            default=default_keys,
        )
        if st.button("Reload runs"):
            st.cache_resource.clear()

    if not selected_keys:
        st.info("Select at least one experiment in the sidebar.")
        return

    exp_ids = tuple(exp_options[k] for k in selected_keys)
    runs = load_runs(exp_ids)
    if runs.empty:
        st.warning("No FINISHED runs found for the selected experiments.")
        return

    runs, config_cols = build_config_df(runs)
    vary = varying_params(runs, config_cols)

    # --- Sidebar: filter on varying params
    with st.sidebar:
        st.header("Filter configurations")
        filters: dict[str, list[str]] = {}
        for p in vary:
            options = sorted(runs[p].astype(str).unique())
            sel = st.multiselect(p, options, default=options, key=f"filter_{p}")
            filters[p] = sel

    mask = pd.Series(True, index=runs.index)
    for p, selected in filters.items():
        mask &= runs[p].astype(str).isin(selected)
    filtered = runs[mask].copy()

    if filtered.empty:
        st.warning("No runs match the selected filters.")
        return

    config_keys = sorted(filtered["config_key"].unique())
    n_configs = len(config_keys)

    # --- Load AUC data for all filtered runs (shared across tabs)
    run_ids = tuple(filtered["run_id"].tolist())
    auc_all = load_all_aucs(exp_ids, run_ids)

    # Build key_to_short from whichever config_keys appear in the AUC data
    if not auc_all.empty:
        run_to_config = filtered.set_index("run_id")["config_key"].to_dict()
        auc_all = auc_all.copy()
        auc_all["config_key"] = auc_all["run_id"].map(run_to_config)
        unique_keys = sorted(auc_all["config_key"].dropna().unique())
    else:
        unique_keys = config_keys
    key_to_short = {k: f"C{i+1}" for i, k in enumerate(unique_keys)}

    disease_names = load_disease_names()

    # --- Detect reference config
    ref_ck = find_reference_config_key(filtered, REFERENCE_PARAMS)
    ref_label = key_to_short.get(ref_ck, None) if ref_ck else None

    # --- Sidebar: AUC filter + comparison mode + reference
    with st.sidebar:
        st.markdown("---")
        st.header("AUC filters")
        min_cases_per_bin = st.slider(
            "Min cases per bin",
            min_value=0, max_value=200, value=10, step=5,
            help="Bins (disease × sex × age bracket) with fewer cases are excluded before averaging AUC.",
        )
        st.markdown("---")
        st.header("Comparison mode")
        n_diffs = st.radio(
            "Parameters varying between pairs",
            options=[1, 2, 0],
            format_func=lambda n: {0: "Up to 2 parameters", 1: "Exactly 1 parameter", 2: "Exactly 2 parameters"}[n],
        )
        st.markdown("---")
        st.header("Reference setup")
        label_options = ["(none)"] + sorted(key_to_short.values())
        default_ref_label = ref_label if ref_label else "(none)"
        chosen_ref_label = st.selectbox(
            "Reference config",
            options=label_options,
            index=label_options.index(default_ref_label),
            help="Auto-detected from REFERENCE_PARAMS. Used as baseline in comparison tabs.",
        )
        ref_ck = (
            next((k for k, v in key_to_short.items() if v == chosen_ref_label), None)
            if chosen_ref_label != "(none)" else None
        )

    if min_cases_per_bin > 0 and not auc_all.empty and "n_case" in auc_all.columns:
        auc_all = auc_all[auc_all["n_case"] >= min_cases_per_bin].copy()

    # --- Sidebar: config comparison (params + domain diffs)
    with st.sidebar:
        st.markdown("---")
        st.header("Configurations")
        st.caption(f"**{n_configs}** config(s), **{len(filtered)}** runs")

        if vary:
            with st.expander("Hyperparameter differences", expanded=True):
                raw = (
                    filtered[["config_key"] + vary]
                    .drop_duplicates("config_key")
                    .assign(label=lambda df: df["config_key"].map(key_to_short))
                    .set_index("label")
                )
                summary_rows = []
                for lbl, row_data in raw.iterrows():
                    summary_rows.append({"label": lbl} | _display_params(vary, row_data))
                summary = pd.DataFrame(summary_rows).set_index("label")
                st.dataframe(summary, use_container_width=True)

            domain_diff = build_domain_diff_df(filtered, key_to_short)
            if domain_diff is not None:
                with st.expander("Domain config differences", expanded=True):
                    st.dataframe(domain_diff, use_container_width=True)
        else:
            st.info("All visible runs share the same configuration.")

    # --- Tabs
    tab_disease, tab_scatter, tab_ranked, tab_consensus, tab_autoimmune = st.tabs(
        ["By disease", "AUC vs cases", "Effect by disease", "Consensus", "Autoimmune panel"]
    )

    with tab_disease:
        render_by_disease_tab(auc_all, filtered, vary, key_to_short, disease_names,
                              n_diffs=n_diffs, ref_ck=ref_ck)

    with tab_scatter:
        render_auc_vs_cases_tab(auc_all, filtered, vary, key_to_short, disease_names,
                                n_diffs=n_diffs, ref_ck=ref_ck)

    with tab_ranked:
        render_effect_by_disease_tab(auc_all, filtered, vary, key_to_short, disease_names,
                                     n_diffs=n_diffs, ref_ck=ref_ck)

    with tab_consensus:
        render_consensus_tab(auc_all, filtered, vary, key_to_short, disease_names,
                             n_diffs=n_diffs, ref_ck=ref_ck)

    with tab_autoimmune:
        render_autoimmune_tab(auc_all, filtered, key_to_short, disease_names, ref_ck=ref_ck)


if __name__ == "__main__":
    main()
