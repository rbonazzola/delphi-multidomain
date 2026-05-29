"""
Delphi Trajectory Explorer

Streamlit app to explore patient trajectories.
Two modes:
  - Raw dataset (cached tokens, no no-event tokens)
  - DataLoader batch (after collate: with no-event tokens, sorted, global IDs)

Run:
    streamlit run trajectory_explorer.py

Expects to be run from the Delphi project root.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data.dataset import (
    AgeSampler,
    DelphiBatch,
    DelphiCollateFn,
    DelphiDataset,
)
from delphi.model import AttentionMaskBuilder, Delphi
from utils import load_domain_config
from utils.cv_utils import get_data_partitions

DELPHI_DIR = Path(__file__).resolve().parent.parent
DAYS_PER_YEAR = 365.25

# ── Domain colors ─────────────────────────────────────────────────────────

DOMAIN_COLORS = {
    "diseases": "#E63946",
    "death": "#457B9D",
    "cv_drugs": "#F4A261",
    "ns_drugs": "#2A9D8F",
    "lifestyle": "#9B5DE5",
    "hla_alleles": "#FFBE0B",
    "sex": "#00B4D8",
    "rare_variants": "#FB5607",
    "genetic_pcs": "#06D6A0",
    "padding": "#ADB5BD",
    # per-locus HLA
    "hla_a": "#FFD166",
    "hla_b": "#EF8C8C",
    "hla_c": "#56CFE1",
    "hla_dpa": "#118AB2",
    "hla_dpb": "#073B4C",
    "hla_dqa": "#FF9F1C",
    "hla_dqb": "#80B918",
    "hla_drb": "#FF595E",
}

DEFAULT_COLOR = "#CCCCCC"

# Okabe-Ito palette as [0,1] RGB arrays for the blended attention mask view
_DOMAIN_COLORS_RGB: dict[str, np.ndarray] = {
    "diseases": np.array([0.00, 0.45, 0.70]),
    "death": np.array([0.84, 0.37, 0.00]),
    "cv_drugs": np.array([0.80, 0.47, 0.74]),
    "ns_drugs": np.array([0.94, 0.89, 0.26]),
    "lifestyle": np.array([0.00, 0.62, 0.45]),
    "hla_alleles": np.array([0.34, 0.71, 0.91]),
    "sex": np.array([0.94, 0.89, 0.26]),
    "rare_variants": np.array([0.36, 0.15, 0.49]),
    "genetic_pcs": np.array([0.50, 0.80, 0.20]),
    "padding": np.array([0.40, 0.40, 0.40]),
    "hla_a": np.array([0.34, 0.71, 0.91]),
    "hla_b": np.array([0.94, 0.55, 0.55]),
    "hla_c": np.array([0.34, 0.81, 0.88]),
    "hla_dpa": np.array([0.07, 0.54, 0.70]),
    "hla_dpb": np.array([0.03, 0.23, 0.30]),
    "hla_dqa": np.array([1.00, 0.62, 0.11]),
    "hla_dqb": np.array([0.50, 0.72, 0.09]),
    "hla_drb": np.array([1.00, 0.35, 0.37]),
}
_DEFAULT_DOMAIN_COLOR_RGB = np.array([0.60, 0.60, 0.60])


def get_color(domain_name):
    return DOMAIN_COLORS.get(domain_name, DEFAULT_COLOR)


# ── Streamlit config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Delphi Trajectory Explorer",
    page_icon="🔮",
    layout="wide",
)

st.markdown(
    """
<style>
    .block-container { padding-top: 1rem; }
    .stDataFrame { font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════
#  Cached loading
# ══════════════════════════════════════════════════════════════════════════


@st.cache_resource
def load_everything(
    domains_str,
    test_fold,
    block_size,
    no_event_token_rate,
    no_event_insertion_mode,
    seed,
    date_cutoff=None,
    birth_dates_file=None,
):

    root_path = DELPHI_DIR / "data" / "transforms"
    domain_config_yaml = DELPHI_DIR / "config" / "domain_config_default.yaml"

    domains = domains_str.split(",")
    default_cfg = load_domain_config(domain_config_yaml, root_path / "tokens")
    domain_cfg = {k: v for k, v in default_cfg.items() if k in domains}

    domain_to_int = Delphi._build_domain_to_int(list(domain_cfg.keys()))
    int_to_domain = {v: k for k, v in domain_to_int.items()}
    domain_offsets, global_vocab_size = Delphi._build_domain_offsets(domain_to_int, domain_cfg)

    continuous_domains = {
        dname: cfg.n_latent_tokens or 1 for dname, cfg in domain_cfg.items() if cfg.type == "continuous"
    }

    # Subject splits
    train_ids, _val_ids, _test_ids = get_data_partitions("./data/transforms/subject_lists", fold=test_fold)

    train_ids = train_ids[:100]

    # Dataset
    dataset_kwargs: dict[str, Any] = dict(
        root=root_path,
        domains_cfg=domain_cfg,
        domain_to_int=domain_to_int,
        block_size=block_size,
        exclusions=[],
        required_domains=["diseases"],
        no_event_token_rate=no_event_token_rate,
        no_event_insertion_mode=no_event_insertion_mode,
        continuous_domains=continuous_domains,
        age_domains=["diseases", "death"],
    )

    train_dataset = DelphiDataset(
        subjects=train_ids,
        date_cutoff=date_cutoff or None,
        birth_dates_file=birth_dates_file or None,
        **dataset_kwargs,
    )

    # Collate
    age_sampler = AgeSampler(
        insertion_mode=no_event_insertion_mode,
        token_rate=no_event_token_rate,
        seed=seed,
    )

    age_jitter_config = {
        domain_to_int[dname]: (cfg.age_jitter_min, cfg.age_jitter_max)
        for dname, cfg in domain_cfg.items()
        if getattr(cfg, "age_jitter", False) and dname in domain_to_int
    }

    collate = DelphiCollateFn(
        age_sampler=age_sampler,
        block_size=block_size,
        domain_to_int=domain_to_int,
        domain_offsets=domain_offsets,
        padding_domain_id=domain_to_int["padding"],
        no_event_token_id=1,
        continuous_domains=continuous_domains,
        age_jitter=age_jitter_config,
    )

    return SimpleNamespace(
        dataset=train_dataset,
        collate=collate,
        domain_to_int=domain_to_int,
        int_to_domain=int_to_domain,
        domain_offsets=domain_offsets,
        global_vocab_size=global_vocab_size,
        domain_cfg=domain_cfg,
        continuous_domains=continuous_domains,
    )


# ══════════════════════════════════════════════════════════════════════════
#  Helper: raw dataset → DataFrame for one subject
# ══════════════════════════════════════════════════════════════════════════


def raw_subject_to_df(data, subject_idx):
    ds = data.dataset
    item = ds[subject_idx]

    domain_ids = item["domain_ids"]
    token_ids = item["local_token_ids"]
    ages = item["ages"]
    real_count = int(item["real_count"])
    subject_id = int(item["subject_id"])
    eval_mask = item.get("eval_mask")

    # Compute birth_date for absolute date column when cutoff data is available
    birth_date = None
    if ds._cutoff_ages is not None and ds._cutoff_date is not None:
        cutoff_age_days = float(ds._cutoff_ages[subject_idx])
        if not np.isinf(cutoff_age_days):
            birth_date = ds._cutoff_date - pd.Timedelta(days=cutoff_age_days)

    rows = []
    for pos in range(real_count):
        d_id = int(domain_ids[pos])
        dname = data.int_to_domain.get(d_id, f"unknown_{d_id}")
        tok_id = int(token_ids[pos])
        age_days = float(ages[pos])

        tokenizer = ds.tokenizers.get(dname, {})
        row = {
            "subject_id": subject_id,
            "position": pos,
            "age_years": round(age_days / DAYS_PER_YEAR, 2),
            "age_days": age_days,
            "domain": dname,
            "domain_id": d_id,
            "token_id": tok_id,
            "token_name": tokenizer.get(tok_id, f"id_{tok_id}"),
        }
        if birth_date is not None:
            row["date"] = (birth_date + pd.Timedelta(days=age_days)).date()
        if eval_mask is not None:
            row["is_post_cutoff"] = bool(eval_mask[pos])
        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
#  Helper: batch → DataFrame
# ══════════════════════════════════════════════════════════════════════════


def batch_subject_to_df(data, batch, batch_subject_idx):
    subject_id = int(batch.subject_ids[batch_subject_idx])
    T = batch.seq_len

    rows = []
    for pos in range(T):
        g_id = int(batch.global_token_ids[batch_subject_idx, pos])
        d_id = int(batch.domain_ids[batch_subject_idx, pos])
        age_days = float(batch.ages[batch_subject_idx, pos])
        dname = data.int_to_domain.get(d_id, f"unknown_{d_id}")

        offset = data.domain_offsets.get(d_id, 0)
        local_id = g_id - offset

        tokenizer = data.dataset.tokenizers.get(dname, {})

        # Identify token type
        pad_offset = data.domain_offsets.get(data.domain_to_int["padding"], 0)
        if g_id == pad_offset:
            token_type = "padding"
        elif g_id == pad_offset + 1:
            token_type = "no_event"
        else:
            token_type = "real"

        rows.append(
            {
                "subject_id": subject_id,
                "position": pos,
                "age_years": round(age_days / DAYS_PER_YEAR, 2),
                "age_days": age_days,
                "domain": dname,
                "domain_id": d_id,
                "token_id": local_id,
                "global_token_id": g_id,
                "token_name": tokenizer.get(local_id, f"id_{local_id}"),
                "token_type": token_type,
            }
        )

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════


def plot_timeline(df, title="Trajectory Timeline"):
    """
    Horizontal timeline: x = age, y = domain, colored by domain.
    """
    if df.empty:
        return go.Figure()

    # Filter out padding with age < 0
    plot_df = df[df["age_days"] >= 0].copy()
    if plot_df.empty:
        return go.Figure()

    color_map = {d: get_color(d) for d in plot_df["domain"].unique()}

    fig = px.strip(
        plot_df,
        x="age_years",
        y="domain",
        color="domain",
        color_discrete_map=color_map,
        hover_data=["token_name", "token_id", "age_years"],
        stripmode="overlay",
        title=title,
    )

    fig.update_traces(
        marker=dict(size=10, opacity=0.8, line=dict(width=1, color="white")),
        jitter=0.3,
    )

    fig.update_layout(
        height=max(250, len(plot_df["domain"].unique()) * 70 + 100),
        xaxis_title="Age (years)",
        yaxis_title="",
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="#FAFAFA",
        font=dict(size=13),
    )

    fig.update_xaxes(gridcolor="#E0E0E0", zeroline=True, zerolinecolor="#CCCCCC")
    fig.update_yaxes(gridcolor="#E0E0E0")

    return fig


def add_cutoff_line(fig, cutoff_age_years):
    """Add a vertical dashed line at the cutoff age."""
    fig.add_vline(
        x=cutoff_age_years,
        line_dash="dash",
        line_color="#FF6B6B",
        line_width=2,
        annotation_text=f"cutoff ({cutoff_age_years:.1f}y)",
        annotation_position="top right",
        annotation_font_color="#FF6B6B",
    )
    return fig


def style_table(df):
    """Color rows by domain."""

    def row_color(row):
        c = get_color(row.get("domain", ""))
        return [f"background-color: {c}22; color: #333"] * len(row)

    return df.style.apply(row_color, axis=1)


# ══════════════════════════════════════════════════════════════════════════
#  Attention mask plotting
# ══════════════════════════════════════════════════════════════════════════


def build_attention_mask(batch, scheme_str, domain_to_int):

    builder = AttentionMaskBuilder(scheme_str, domain_to_int)

    mask = builder.build(
        batch.domain_ids[:3],
        batch.global_token_ids[:3],
        batch.ages[:3],
    )

    mask = mask[0]  # [T, T]

    # ensure at least self-attention
    row_sum = mask.sum(dim=1)
    empty_rows = row_sum == 0
    mask[empty_rows, empty_rows] = 1

    return mask


def plot_attention_mask(mask, df, title="Attention Mask"):
    """
    Heatmap of the attention mask [T, T] with token labels on axes.
    """
    T = mask.shape[0]
    mask_np = mask.cpu().numpy().astype(float)

    # Build labels: "pos: domain (token_name)" truncated
    labels = []
    for _, row in df.iterrows():
        dname = row.get("domain", "?")
        tname = row.get("token_name", "?")
        age = row.get("age_years", "?")
        label = f"{dname}: {tname} ({age}y)"
        if len(label) > 30:
            label = label[:27] + "..."
        labels.append(label)

    # Pad labels if df is shorter than T (shouldn't happen, but safety)
    while len(labels) < T:
        labels.append("padding")

    labels = labels[:T]

    # Color: 1 = allowed (blue-ish), 0 = blocked (white)
    fig = go.Figure(
        data=go.Heatmap(
            z=mask_np,
            x=labels,
            y=labels,
            colorscale=[[0, "#F5F5F5"], [1, "#2A6F97"]],
            showscale=False,
            hovertemplate="Query: %{y}<br>Key: %{x}<br>Attention: %{z}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        height=max(500, T * 8 + 100),
        width=max(500, T * 8 + 100),
        xaxis=dict(
            title="Key",
            tickangle=45,
            tickfont=dict(size=7),
            side="bottom",
        ),
        yaxis=dict(
            title="Query",
            tickfont=dict(size=7),
            autorange="reversed",
        ),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


def plot_attention_mask_compact(mask, df, domain_colors):
    """
    Compact attention mask: color-coded domain strips on both axes,
    binary heatmap, and a domain legend in the upper-right triangle.
    """
    T = mask.shape[0]
    mask_np = mask.cpu().numpy().astype(float)

    domain_list = df["domain"].tolist()[:T]
    while len(domain_list) < T:
        domain_list.append("padding")

    unique_domains = list(dict.fromkeys(domain_list))
    dom_to_num = {d: i for i, d in enumerate(unique_domains)}
    dom_strip = [dom_to_num[d] for d in domain_list]

    from plotly.subplots import make_subplots

    STRIP = 0.025
    GAP = 0.004

    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[STRIP, 1 - STRIP],
        column_widths=[STRIP, 1 - STRIP],
        horizontal_spacing=GAP,
        vertical_spacing=GAP,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    n_doms = len(unique_domains)
    dom_colorscale = [
        [i / max(n_doms - 1, 1), domain_colors.get(d, DEFAULT_COLOR)] for i, d in enumerate(unique_domains)
    ]

    fig.add_trace(
        go.Heatmap(z=[dom_strip], colorscale=dom_colorscale, showscale=False, hoverinfo="skip"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(z=[[d] for d in dom_strip], colorscale=dom_colorscale, showscale=False, hoverinfo="skip"),
        row=2,
        col=1,
    )

    labels = []
    for _, row in df.iterrows():
        labels.append(f"{row.get('domain', '?')}: {row.get('token_name', '?')} ({row.get('age_years', '?')}y)")
    while len(labels) < T:
        labels.append("padding")
    labels = labels[:T]

    fig.add_trace(
        go.Heatmap(
            z=mask_np,
            colorscale=[[0, "#F5F5F5"], [1, "#1A3A5C"]],
            showscale=False,
            hovertemplate="Q pos %{y} → K pos %{x}: %{z}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    # ── Legend in upper-right triangle (blank in causal attention) ─────
    plot_domains = [d for d in unique_domains if d != "padding"]
    box_h = max(2, T // 18)
    box_w = max(4, T // 10)
    gap = max(1, T // 40)
    x_right = T - 2
    x_left = x_right - box_w

    for i, dname in enumerate(plot_domains):
        y_top = i * (box_h + gap)
        y_bot = y_top + box_h
        # stay inside the upper triangle
        if y_bot >= x_left - gap:
            break
        color = domain_colors.get(dname, DEFAULT_COLOR)
        fig.add_shape(
            type="rect",
            xref="x4",
            yref="y4",
            x0=x_left,
            x1=x_right,
            y0=y_top,
            y1=y_bot,
            fillcolor=color,
            line=dict(width=0.5, color="#ffffff"),
            layer="above",
        )
        fig.add_annotation(
            xref="x4",
            yref="y4",
            x=x_left - 1,
            y=(y_top + y_bot) / 2,
            text=dname,
            showarrow=False,
            font=dict(size=max(8, T // 12), color="#222"),
            xanchor="right",
            yanchor="middle",
            bgcolor="rgba(255,255,255,0.7)",
        )

    size = max(480, T * 7 + 80)
    fig.update_layout(
        height=size,
        width=size,
        margin=dict(l=5, r=5, t=5, b=5),
        xaxis4=dict(showticklabels=False, autorange=True),
        yaxis4=dict(showticklabels=False, autorange="reversed"),
        xaxis=dict(showticklabels=False),
        xaxis2=dict(showticklabels=False),
        xaxis3=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        yaxis2=dict(showticklabels=False),
        yaxis3=dict(showticklabels=False),
    )

    return fig


def plot_attention_mask_blended(mask, domain_ids, ages, int_to_domain, exclude_padding=True):
    """
    Domain-blended attention mask (style from debug_18vs12.py).

    Each cell's color = average of its row-domain color and column-domain color,
    then multiplied by the mask (black = blocked). Tokens are sorted by age on
    both axes; white lines mark domain-group boundaries.
    Returns a matplotlib Figure.
    """
    d_arr = domain_ids.cpu().numpy()
    a_arr = ages.cpu().numpy()

    if exclude_padding:
        keep = a_arr >= 0
        idx = np.where(keep)[0]
        d_arr = d_arr[keep]
        a_arr = a_arr[keep]
        mask_np = mask.cpu().float().numpy()[np.ix_(idx, idx)]
    else:
        mask_np = mask.cpu().float().numpy()

    order = np.argsort(a_arr, kind="stable")
    m = mask_np[np.ix_(order, order)]
    d_ord = d_arr[order]

    row_colors = np.array(
        [_DOMAIN_COLORS_RGB.get(int_to_domain.get(int(d), "padding"), _DEFAULT_DOMAIN_COLOR_RGB) for d in d_ord]
    )
    img = (row_colors[:, None, :] + row_colors[None, :, :]) / 2
    img = img * m[:, :, None]
    boundaries = np.where(np.diff(d_ord))[0] + 1

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(img, aspect="auto", origin="upper", interpolation="nearest")
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", lw=0.5)
        ax.axvline(b - 0.5, color="white", lw=0.5)
    ax.set_xlabel("Key (sorted by age)")
    ax.set_ylabel("Query (sorted by age)")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    present_domains = list(dict.fromkeys(int_to_domain.get(int(d), "padding") for d in d_ord))
    legend_handles = [
        mpatches.Patch(color=_DOMAIN_COLORS_RGB.get(d, _DEFAULT_DOMAIN_COLOR_RGB), label=d)
        for d in present_domains
        if d != "padding"
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles), fontsize=8, framealpha=0.85)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  Main app
# ══════════════════════════════════════════════════════════════════════════


def main():

    st.title("Delphi Trajectory Explorer")

    # ── Sidebar config ────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")

        domains_str = st.text_input(
            "Domains",
            value="hla_alleles,diseases,death,lifestyle,sex,padding",
        )
        test_fold = st.number_input("Test fold", min_value=1, max_value=5, value=1)
        block_size = st.number_input("Block size", min_value=16, max_value=512, value=96)
        no_event_rate = st.slider("No-event token rate", 0.5, 10.0, 2.0, 0.5)
        insertion_mode = st.selectbox("Insertion mode", ["random", "regular", "grid_jitter"])
        seed = st.number_input("Seed", value=42)

        st.markdown("---")
        st.header("Date Cutoff (longitudinal)")
        enable_cutoff = st.checkbox("Enable date cutoff")
        date_cutoff_str = None
        birth_dates_file_str = None
        if enable_cutoff:
            date_cutoff_str = st.text_input(
                "Cutoff date (YYYY-MM-DD)",
                value="2018-01-01",
            )
            default_dob_path = str(DELPHI_DIR / "data" / "datasets" / "year_and_month_of_birth.txt")
            birth_dates_file_str = st.text_input(
                "Birth dates file",
                value=default_dob_path,
            )

        st.markdown("---")
        st.header("Attention Scheme")
        attention_scheme = st.text_area(
            "Scheme DSL",
            value="all:causal(mask_ties=True)",
            height=80,
            help="e.g. [hla_alleles,sex]:bidirectional,[sex,diseases,lifestyle,death,padding]:causal(mask_ties=True)",
        )

        st.markdown("---")
        st.header("Domain Dropout")
        st.caption("DataLoader batch mode only. Each rerun draws a new dropout sample.")

        domain_names_for_dropout = [d.strip() for d in domains_str.split(",") if d.strip() not in ("padding", "")]
        raw_dropout = {}
        for dname in domain_names_for_dropout:
            enabled = st.checkbox(dname, key=f"do_enable_{dname}")
            if enabled:
                mode = st.radio(
                    "Mode",
                    ["token", "block"],
                    key=f"dm_{dname}",
                    horizontal=True,
                    help="token: drop tokens individually · block: drop the entire domain for the subject",
                )
                rate = st.slider(
                    "Rate",
                    0.0,
                    1.0,
                    0.1,
                    0.05,
                    key=f"dr_{dname}",
                )
                raw_dropout[dname] = (mode, rate)

        st.markdown("---")
        if st.button("Load / Reload data", type="primary"):
            st.cache_resource.clear()

    # ── Load data ─────────────────────────────────────────────────────
    data = load_everything(
        domains_str,
        test_fold,
        block_size,
        no_event_rate,
        insertion_mode,
        seed,
        date_cutoff=date_cutoff_str,
        birth_dates_file=birth_dates_file_str,
    )

    st.sidebar.success(f"Loaded {len(data.dataset)} subjects")

    # Convert domain-name dropout settings to domain-int keyed dict
    dropout_config = {
        data.domain_to_int[dname]: (mode, rate)
        for dname, (mode, rate) in raw_dropout.items()
        if dname in data.domain_to_int
    }

    # Age-jitter toggle — only shown when at least one domain has jitter configured
    apply_age_jitter = False
    if data.collate.age_jitter:
        with st.sidebar:
            st.markdown("---")
            apply_age_jitter = st.checkbox(
                "Apply age jitter",
                value=True,
                help=", ".join(
                    f"{data.int_to_domain.get(d, d)}: [{lo:.0f}, {hi:.0f}] days"
                    for d, (lo, hi) in data.collate.age_jitter.items()
                ),
            )

    def make_live_collate(training: bool) -> DelphiCollateFn:
        return DelphiCollateFn(
            age_sampler=data.collate.age_sampler,
            block_size=data.collate.block_size,
            domain_to_int=data.domain_to_int,
            domain_offsets=data.domain_offsets,
            padding_domain_id=data.domain_to_int["padding"],
            no_event_token_id=1,
            continuous_domains=data.continuous_domains,
            domain_dropout=dropout_config,
            age_jitter=data.collate.age_jitter if apply_age_jitter else {},
            training=training,
        )

    # ── Mode selector ─────────────────────────────────────────────────
    mode = st.radio(
        "View mode",
        ["Raw dataset (no no-event tokens)", "DataLoader batch (with no-event tokens)"],
        horizontal=True,
    )

    # ── Subject selector ──────────────────────────────────────────────
    subject_list = data.dataset.subject_list
    n_subjects = len(subject_list)

    col1, col2 = st.columns([1, 3])
    with col1:
        subject_idx = st.number_input(
            "Subject index",
            min_value=0,
            max_value=n_subjects - 1,
            value=0,
        )
    with col2:
        st.markdown(
            f"**Subject ID:** `{subject_list[subject_idx]}`  |  "
            f"**Max age:** `{float(data.dataset._max_ages[subject_idx]):.0f}` days "
            f"(`{float(data.dataset._max_ages[subject_idx]) / DAYS_PER_YEAR:.1f}` years)  |  "
            f"**Real tokens:** `{int(data.dataset._real_counts[subject_idx])}`"
        )

    # ── Domain filter ─────────────────────────────────────────────────
    all_domains = [data.int_to_domain[i] for i in sorted(data.int_to_domain.keys())]
    selected_domains = st.multiselect(
        "Filter domains",
        all_domains,
        default=[d for d in all_domains if d != "padding"],
    )

    st.markdown("---")

    # ── Build the batch (needed for both views) ──────────────────────
    if mode.startswith("Raw"):
        df = raw_subject_to_df(data, subject_idx)
        batch: DelphiBatch | None = None
    else:
        item = data.dataset[subject_idx]
        batch: DelphiBatch = make_live_collate(training=True)([item])
        df = batch_subject_to_df(data, batch, 0)

        if dropout_config:
            parts = []
            for d_int, (m, r) in dropout_config.items():
                dname = data.int_to_domain.get(d_int, str(d_int))
                parts.append(f"**{dname}** ({m}, p={r})")
            st.info("Dropout active: " + " · ".join(parts))

    # Apply domain filter
    df_filtered = df[df["domain"].isin(selected_domains)] if selected_domains else df

    # ── Tabs ──────────────────────────────────────────────────────────
    tab_timeline, tab_table, tab_attn = st.tabs(["📈 Timeline", "📋 Token Table", "🎯 Attention Mask"])

    # ── Tab: Timeline ─────────────────────────────────────────────────
    with tab_timeline:
        fig = plot_timeline(df_filtered, title=f"Subject {subject_list[subject_idx]}")
        if data.dataset._cutoff_ages is not None:
            cutoff_age_days = float(data.dataset._cutoff_ages[subject_idx])
            if not np.isinf(cutoff_age_days):
                fig = add_cutoff_line(fig, cutoff_age_days / DAYS_PER_YEAR)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Age distribution by domain"):
            real_df = df_filtered[df_filtered["age_days"] >= 0]
            if not real_df.empty:
                fig_hist = px.histogram(
                    real_df,
                    x="age_years",
                    color="domain",
                    color_discrete_map={d: get_color(d) for d in real_df["domain"].unique()},
                    barmode="overlay",
                    opacity=0.7,
                    nbins=30,
                    title="Token age distribution",
                )
                fig_hist.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor="#FAFAFA",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    # ── Tab: Token Table ──────────────────────────────────────────────
    with tab_table:
        if not df_filtered.empty:
            cols = st.columns(min(len(selected_domains), 6) if selected_domains else 1)
            for i, dname in enumerate(selected_domains or df_filtered["domain"].unique()):
                count = len(df_filtered[df_filtered["domain"] == dname])
                with cols[i % len(cols)]:
                    st.metric(dname, count)

        display_cols = ["position", "age_years", "domain", "token_id", "token_name"]
        if "date" in df_filtered.columns:
            display_cols.insert(2, "date")
        if "is_post_cutoff" in df_filtered.columns:
            display_cols.append("is_post_cutoff")
        if "token_type" in df_filtered.columns:
            display_cols.append("token_type")
        if "global_token_id" in df_filtered.columns:
            display_cols.append("global_token_id")

        available_cols = [c for c in display_cols if c in df_filtered.columns]
        st.dataframe(
            style_table(df_filtered[available_cols]),
            use_container_width=True,
            height=min(600, len(df_filtered) * 35 + 50),
        )

    # ── Tab: Attention Mask ───────────────────────────────────────────
    with tab_attn:
        if batch is None:
            st.info(
                "Switch to **DataLoader batch** mode to see the attention mask "
                "(the mask depends on no-event tokens and sorting)."
            )
        else:
            st.caption(f"Scheme: `{attention_scheme}`")

            try:
                mask = build_attention_mask(batch, attention_scheme, data.domain_to_int)

                mask_style = st.radio(
                    "Visualization style",
                    ["Domain-blended (age-sorted)", "Compact (domain-colored axes)", "Detailed (labeled axes)"],
                    horizontal=True,
                )

                if mask_style.startswith("Domain-blended"):
                    fig_mask = plot_attention_mask_blended(
                        mask,
                        batch.domain_ids[0],
                        batch.ages[0],
                        data.int_to_domain,
                    )
                    st.pyplot(fig_mask)
                    plt.close(fig_mask)
                elif mask_style.startswith("Compact"):
                    fig_mask = plot_attention_mask_compact(mask, df, DOMAIN_COLORS)
                    st.plotly_chart(fig_mask, use_container_width=True)
                else:
                    fig_mask = plot_attention_mask(
                        mask, df, title=f"Attention Mask — Subject {subject_list[subject_idx]}"
                    )
                    st.plotly_chart(fig_mask, use_container_width=True)

                # Stats
                T = mask.shape[0]
                n_allowed = mask.sum().item()
                n_total = T * T
                st.caption(
                    f"Mask: {T}×{T} = {n_total} pairs | "
                    f"Allowed: {n_allowed} ({100 * n_allowed / n_total:.1f}%) | "
                    f"Blocked: {n_total - n_allowed} ({100 * (n_total - n_allowed) / n_total:.1f}%)"
                )

            except Exception as e:
                st.error(f"Error building attention mask: {e}")
                st.exception(e)


if __name__ == "__main__":
    main()
