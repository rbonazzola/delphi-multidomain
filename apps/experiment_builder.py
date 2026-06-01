"""
Delphi Experiment Grid Builder

Streamlit app to visually configure training experiment grids,
preview the generated parameter table, edit it, and export as TSV
ready for cluster submission via sarray_params.

Run:
    streamlit run experiment_builder.py
"""

import csv
import io
import itertools
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# ═══════════════════════════════════════════════════════════════════════════════
#  Load attention schemes from config/attention_schemes.yaml
# ═══════════════════════════════════════════════════════════════════════════════

_SCHEMES_YAML = Path(__file__).resolve().parent.parent / "config" / "attention_schemes.yaml"


def _load_attention_schemes():
    if not _SCHEMES_YAML.exists():
        return {"Custom (edit below)": {"scheme": "", "domains": ""}}
    with _SCHEMES_YAML.open() as f:
        data = yaml.safe_load(f)
    schemes = {}
    for name, entry in data.items():
        if isinstance(entry, dict) and "scheme" in entry:
            label = f"{name} — {entry['description']}"
            schemes[label] = {
                "scheme": entry["scheme"],
                "domains": entry.get("domains", ""),
            }
    schemes["Custom (edit below)"] = {"scheme": "", "domains": ""}
    return schemes


PREDEFINED_ATTENTION_SCHEMES = _load_attention_schemes()
_FIRST_ATTN_PRESET = next(iter(PREDEFINED_ATTENTION_SCHEMES))

# ═══════════════════════════════════════════════════════════════════════════════
#  at_birth alias
# ═══════════════════════════════════════════════════════════════════════════════

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _get_at_birth_domains(domain_config_path: str) -> list[str]:
    """Return domain names with at_birth: true from a domain config YAML."""
    path = Path(domain_config_path)
    if not path.exists():
        return []
    with path.open() as f:
        data = yaml.safe_load(f)
    return [name for name, cfg in data.items() if isinstance(cfg, dict) and cfg.get("at_birth")]


def resolve_at_birth(s: str, at_birth_domains: list[str]) -> str:
    return s.replace("at_birth", ",".join(at_birth_domains))


# ═══════════════════════════════════════════════════════════════════════════════
#  Predefined domain sets
# ═══════════════════════════════════════════════════════════════════════════════

ALL_DOMAINS = [
    "diseases",
    "death",
    "lifestyle",
    "sex",
    "hla_alleles",
    "hla_a",
    "hla_b",
    "hla_c",
    "hla_dpa",
    "hla_dpb",
    "hla_dqa",
    "hla_dqb",
    "hla_drb",
    "genetic_pcs",
    "cv_drugs",
    "ns_drugs",
    "rare_variants",
]

PREDEFINED_DOMAIN_SETS = {
    "Base (no HLA, no PCs)": "diseases,death,lifestyle,sex",
    "With genetic PCs": "diseases,death,lifestyle,genetic_pcs,sex",
    "With HLA": "diseases,death,lifestyle,hla_alleles,sex",
    "With HLA + PCs": "diseases,death,lifestyle,hla_alleles,genetic_pcs,sex",
    "With drugs (CV)": "diseases,death,cv_drugs,lifestyle,sex",
    "With drugs (NS)": "diseases,death,ns_drugs,lifestyle,sex",
    "With drugs (both)": "diseases,death,cv_drugs,ns_drugs,lifestyle,sex",
    "Full (drugs + HLA + PCs)": "diseases,death,cv_drugs,ns_drugs,lifestyle,hla_alleles,genetic_pcs,sex",
    "Custom (edit below)": "",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Load params from TSV
# ═══════════════════════════════════════════════════════════════════════════════


def _load_params_from_tsv(path: str) -> str | None:
    """Parse a params TSV and populate session state for sidebar widgets and configs."""
    p = Path(path)
    if not p.exists():
        return f"File not found: `{path}`"

    try:
        df = pd.read_csv(p, sep="\t")
    except Exception as e:
        return f"Could not read file: {e}"

    def _unique_ints(col):
        return sorted(df[col].dropna().astype(int).unique().tolist())

    def _unique_floats(col):
        return sorted(df[col].dropna().astype(float).unique().tolist())

    if "experiment_name" in df.columns:
        st.session_state["ti_experiment_name"] = str(df["experiment_name"].iloc[0])

    if "n_layer" in df.columns:
        st.session_state["ms_n_layers"] = _unique_ints("n_layer")
    if "n_embd" in df.columns:
        st.session_state["ms_n_embds"] = _unique_ints("n_embd")
    if "n_head" in df.columns:
        st.session_state["ms_n_heads"] = _unique_ints("n_head")
    if "test_fold" in df.columns:
        st.session_state["ms_test_folds"] = _unique_ints("test_fold")
    if "block_size" in df.columns:
        raw = df["block_size"].dropna().unique().tolist()
        parsed = []
        for v in raw:
            try:
                parsed.append(int(v))
            except (ValueError, TypeError):
                parsed.append(str(v))
        st.session_state["ms_block_sizes"] = parsed
    if "learning_rate" in df.columns:
        st.session_state["ms_learning_rates"] = _unique_floats("learning_rate")
    if "batch_size_schedule" in df.columns:
        schedules = df["batch_size_schedule"].dropna().unique().tolist()
        st.session_state["radio_batch_mode"] = "Schedule"
        st.session_state["ta_schedules"] = "\n".join(str(s) for s in schedules)
    elif "batch_size" in df.columns:
        st.session_state["radio_batch_mode"] = "Fixed"
        st.session_state["ms_batch_sizes"] = _unique_ints("batch_size")
    if "num_workers" in df.columns:
        st.session_state["ni_num_workers"] = int(df["num_workers"].iloc[0])
    if "seed" in df.columns:
        seeds = _unique_ints("seed")
        st.session_state["ti_seeds"] = ",".join(str(s) for s in seeds)
    if "subjects" in df.columns and df["subjects"].notna().any():
        st.session_state["ti_subjects"] = str(df["subjects"].dropna().iloc[0])
    if "use_amp" in df.columns:
        st.session_state["cb_use_amp"] = bool(df["use_amp"].iloc[0])
    if "compute_aucs" in df.columns:
        st.session_state["cb_compute_aucs"] = bool(df["compute_aucs"].iloc[0])

    # Rebuild configs from unique (domains, attention_scheme) pairs
    pair_cols = [c for c in ["domains", "attention_scheme"] if c in df.columns]
    if len(pair_cols) == 2:
        pairs = df[pair_cols].drop_duplicates()
        new_configs = []
        for idx, (_, row) in enumerate(pairs.iterrows()):
            domains_str = str(row["domains"])
            attn_str = str(row["attention_scheme"])

            domains_preset = next(
                (label for label, v in PREDEFINED_DOMAIN_SETS.items() if v == domains_str),
                "Custom (edit below)",
            )
            attn_preset = next(
                (label for label, info in PREDEFINED_ATTENTION_SCHEMES.items() if info["scheme"] == attn_str),
                "Custom (edit below)",
            )
            suffix = f"_config{idx + 1}" if idx > 0 else "_base"
            new_configs.append(
                {
                    "domains_preset": domains_preset,
                    "domains_custom": domains_str if domains_preset == "Custom (edit below)" else "",
                    "attn_preset": attn_preset,
                    "attn_custom": attn_str if attn_preset == "Custom (edit below)" else "",
                    "suffix": suffix,
                    "_domains": domains_str,
                    "_attn": attn_str,
                }
            )
        st.session_state.configs = new_configs

    return None  # success


# ═══════════════════════════════════════════════════════════════════════════════
#  Page config
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Delphi Experiment Builder",
    page_icon="⚗️",
    layout="wide",
)

st.title("⚗️ Delphi Experiment Grid Builder")

# ═══════════════════════════════════════════════════════════════════════════════
#  Sidebar: Experiment configurations
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Experiment configurations")

    # ── Load from previous params ─────────────────────────────────────────────
    with st.expander("📂 Load from params TSV", expanded=False):
        prev_tsv = st.text_input("Path to params TSV", value="", key="ti_prev_tsv")
        if st.button("Load params"):
            if prev_tsv.strip():
                err = _load_params_from_tsv(prev_tsv.strip())
                if err:
                    st.error(err)
                else:
                    st.success("Params loaded — reloading...")
                    st.rerun()
            else:
                st.warning("Enter a file path first.")

    st.subheader("Experiment name")
    experiment_name = st.text_input("MLflow experiment name", value="Delphi-experiment", key="ti_experiment_name")

    st.subheader("Architecture grid")
    n_layers = st.multiselect("n_layer", [1, 2, 4, 6, 8, 12, 16, 24], default=[12], key="ms_n_layers")
    n_embds = st.multiselect("n_embd", [60, 120, 180, 240, 360, 480], default=[240], key="ms_n_embds")
    n_heads = st.multiselect("n_head", [1, 2, 3, 4, 6, 8, 12, 16], default=[12], key="ms_n_heads")

    st.subheader("Training grid")
    batch_size_mode = st.radio("Batch size mode", ["Fixed", "Schedule"], horizontal=True, key="radio_batch_mode")
    if batch_size_mode == "Fixed":
        batch_sizes = st.multiselect("batch_size", [32, 64, 128, 256, 512], default=[128], key="ms_batch_sizes")
        batch_size_schedules = []
    else:
        batch_sizes = []
        _schedules_raw = st.text_area(
            "batch_size_schedule(s) — one per line",
            value="10:32,10:64,10:128,*:256x4",
            key="ta_schedules",
            help=(
                "Format: n_epochs:batch_size or n_epochs:batch_sizexgrad_accum. "
                "Use * for the last (open-ended) stage. "
                "Example: 10:32,10:64,*:256x4"
            ),
        )
        batch_size_schedules = [s.strip() for s in _schedules_raw.splitlines() if s.strip()]

    block_sizes = st.multiselect("block_size", ["auto", 32, 64, 96, 128, 192, 256], default=[128], key="ms_block_sizes")
    learning_rates = st.multiselect(
        "learning_rate",
        [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
        default=[3e-4],
        key="ms_learning_rates",
        format_func=lambda x: f"{x:.0e}",
    )
    test_folds = st.multiselect("test_fold", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5], key="ms_test_folds")

    st.subheader("Domain config")
    _available_configs = sorted(_CONFIG_DIR.glob("domain_config*.yaml"))
    _config_labels = {p.name: str(p) for p in _available_configs}
    _config_choice = st.selectbox("domain_config", list(_config_labels.keys()))
    domain_config_path = _config_labels.get(_config_choice, "")
    at_birth_domains = _get_at_birth_domains(domain_config_path)
    if at_birth_domains:
        st.caption(f"at_birth → {', '.join(at_birth_domains)}")

    st.subheader("Other")
    num_workers = st.number_input("num_workers", min_value=0, max_value=16, value=4, key="ni_num_workers")
    subjects_path = st.text_input("subjects (path, leave empty for all)", value="", key="ti_subjects")
    seeds_input = st.text_input("seed(s) (comma-separated)", value="142", key="ti_seeds")
    seeds = [int(s.strip()) for s in seeds_input.split(",") if s.strip().isdigit()]

    st.subheader("Date cutoff (longitudinal)")
    enable_date_cutoff = st.checkbox("Enable date cutoff", value=False)
    date_cutoff = None
    birth_dates_file = None
    if enable_date_cutoff:
        date_cutoff = st.text_input("date_cutoff (YYYY-MM-DD)", value="2018-01-01")
        birth_dates_file = st.text_input(
            "birth_dates_file",
            value=str(Path(__file__).resolve().parent.parent / "data" / "datasets" / "year_and_month_of_birth.txt"),
        )

    st.subheader("Boolean flags")
    use_amp = st.checkbox("use_amp (mixed precision)", value=True, key="cb_use_amp")
    compute_aucs = st.checkbox("compute_aucs (evaluate after training)", value=True, key="cb_compute_aucs")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main area: Domain + Attention scheme configs
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("Domain & Attention configurations")
st.caption(
    "Define one or more (domains, attention_scheme) pairs. "
    "Each pair will be crossed with the architecture/training grid above."
)

# Session state for configs
if "configs" not in st.session_state:
    st.session_state.configs = [
        {
            "domains_preset": "Base (no HLA, no PCs)",
            "domains_custom": "",
            "attn_preset": _FIRST_ATTN_PRESET,
            "attn_custom": "",
            "suffix": "_base",
        }
    ]


def add_config():
    st.session_state.configs.append(
        {
            "domains_preset": "Base (no HLA, no PCs)",
            "domains_custom": "",
            "attn_preset": _FIRST_ATTN_PRESET,
            "attn_custom": "",
            "suffix": f"_config{len(st.session_state.configs) + 1}",
        }
    )


def remove_config(idx):
    st.session_state.configs.pop(idx)


# Render each config
for i, cfg in enumerate(st.session_state.configs):
    with st.expander(f"Configuration {i + 1}: {cfg['suffix']}", expanded=(i == 0)):
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            domain_preset = st.selectbox(
                "Domain preset",
                list(PREDEFINED_DOMAIN_SETS.keys()),
                key=f"dom_preset_{i}",
                index=list(PREDEFINED_DOMAIN_SETS.keys()).index(cfg["domains_preset"])
                if cfg["domains_preset"] in PREDEFINED_DOMAIN_SETS
                else 0,
            )
            cfg["domains_preset"] = domain_preset

            if domain_preset == "Custom (edit below)":
                cfg["domains_custom"] = st.text_input(
                    "Custom domains (comma-separated)",
                    value=cfg["domains_custom"],
                    key=f"dom_custom_{i}",
                )
                domains_str = cfg["domains_custom"]
            else:
                domains_str = PREDEFINED_DOMAIN_SETS[domain_preset]
                st.code(domains_str, language=None)

        with col2:
            attn_preset = st.selectbox(
                "Attention scheme preset",
                list(PREDEFINED_ATTENTION_SCHEMES.keys()),
                key=f"attn_preset_{i}",
                index=list(PREDEFINED_ATTENTION_SCHEMES.keys()).index(cfg["attn_preset"])
                if cfg["attn_preset"] in PREDEFINED_ATTENTION_SCHEMES
                else 0,
            )
            cfg["attn_preset"] = attn_preset

            scheme_info = PREDEFINED_ATTENTION_SCHEMES[attn_preset]
            if attn_preset == "Custom (edit below)":
                cfg["attn_custom"] = st.text_area(
                    "Custom attention scheme",
                    value=cfg["attn_custom"],
                    key=f"attn_custom_{i}",
                    height=80,
                )
                attn_str = cfg["attn_custom"]
            else:
                attn_str = scheme_info["scheme"]
                st.code(attn_str, language=None)

            # Suggest domains from YAML if available and different from current
            suggested_domains = scheme_info.get("domains", "")
            if suggested_domains and attn_preset != "Custom (edit below)":
                current_domains = PREDEFINED_DOMAIN_SETS.get(domain_preset, cfg.get("domains_custom", ""))
                if suggested_domains != current_domains:
                    st.caption(f"Suggested domains: `{suggested_domains}`")
                    if st.button("↑ Use these domains", key=f"use_scheme_domains_{i}"):
                        preset_match = next(
                            (label for label, v in PREDEFINED_DOMAIN_SETS.items() if v == suggested_domains),
                            "Custom (edit below)",
                        )
                        cfg["domains_preset"] = preset_match
                        cfg["domains_custom"] = suggested_domains if preset_match == "Custom (edit below)" else ""
                        st.rerun()

        with col3:
            cfg["suffix"] = st.text_input("Suffix", value=cfg["suffix"], key=f"suffix_{i}")
            if len(st.session_state.configs) > 1:
                st.button("🗑️ Remove", key=f"remove_{i}", on_click=remove_config, args=(i,))

        # Store resolved values (expand at_birth alias)
        cfg["_domains"] = resolve_at_birth(domains_str, at_birth_domains)
        cfg["_attn"] = resolve_at_birth(attn_str, at_birth_domains)

col_add, _spacer = st.columns([1, 4])
with col_add:
    st.button("➕ Add configuration", on_click=add_config)


# ═══════════════════════════════════════════════════════════════════════════════
#  Generate grid
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("Generated parameter grid")


def generate_grid():
    rows = []
    batch_dim = batch_size_schedules if batch_size_mode == "Schedule" else batch_sizes
    for fold in test_folds:
        for n_layer, n_embd, n_head, batch_val, block_size, lr, seed in itertools.product(
            n_layers, n_embds, n_heads, batch_dim, block_sizes, learning_rates, seeds
        ):
            for cfg in st.session_state.configs:
                domains = cfg["_domains"]
                attn = cfg["_attn"]
                suffix = cfg["suffix"]

                if not domains or not attn:
                    continue

                row = {
                    "run_name": f"fold{fold}_L{n_layer}_E{n_embd}_H{n_head}_bs{block_size}{suffix}",
                    "experiment_name": experiment_name,
                    "n_layer": n_layer,
                    "n_embd": n_embd,
                    "n_head": n_head,
                    "test_fold": fold,
                    "block_size": block_size,
                    "learning_rate": lr,
                    "domains": domains,
                    "attention_scheme": attn,
                    "domain_config": domain_config_path,
                    "num_workers": num_workers,
                    "seed": seed,
                    "use_amp": use_amp,
                    "compute_aucs": compute_aucs,
                }

                if batch_size_mode == "Schedule":
                    row["batch_size_schedule"] = batch_val
                else:
                    row["batch_size"] = batch_val

                if subjects_path:
                    row["subjects"] = subjects_path

                if date_cutoff:
                    row["date_cutoff"] = date_cutoff
                if birth_dates_file:
                    row["birth_dates_file"] = birth_dates_file

                rows.append(row)

    return pd.DataFrame(rows)


df = generate_grid()

if df.empty:
    st.warning("No experiments generated. Check your configuration.")
else:
    _batch_dim_vals = batch_size_schedules if batch_size_mode == "Schedule" else batch_sizes
    _batch_label = "schedules" if batch_size_mode == "Schedule" else "batch_sizes"
    st.info(
        f"**{len(df)} experiments** = "
        f"{len(test_folds)} folds × "
        f"{len(n_layers)}×{len(n_embds)}×{len(n_heads)} arch × "
        f"{len(_batch_dim_vals)} {_batch_label}×{len(block_sizes)}×{len(learning_rates)} training × "
        f"{len(seeds)} seeds × "
        f"{len(st.session_state.configs)} configs"
    )

    # ── Editable preview ──────────────────────────────────────────────
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        height=min(600, len(df) * 35 + 50),
    )

    # ── Summary stats ─────────────────────────────────────────────────
    with st.expander("Grid summary"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Unique values per parameter:**")
            batch_col = "batch_size_schedule" if batch_size_mode == "Schedule" else "batch_size"
            for col in ["n_layer", "n_embd", "n_head", batch_col, "block_size", "learning_rate"]:
                if col in edited_df.columns:
                    vals = sorted(edited_df[col].unique())
                    st.text(f"  {col}: {vals}")
        with col2:
            st.markdown("**Configurations:**")
            for _, cfg in enumerate(st.session_state.configs):
                st.text(f"  {cfg['suffix']}: {cfg['_domains'][:40]}...")
        with col3:
            st.markdown("**Folds:**")
            st.text(f"  {sorted(edited_df['test_fold'].unique().tolist())}")
            st.markdown("**Total jobs:**")
            st.metric("", len(edited_df))

    # ── Dry-run preview ───────────────────────────────────────────────
    with st.expander("Dry-run preview (first 5 commands)"):
        for _, row in edited_df.head(5).iterrows():
            args = []
            for col in edited_df.columns:
                val = row[col]
                if pd.notna(val) and str(val).strip() != "":
                    if str(val) == "True":
                        args.append(f"--{col}")
                    elif str(val) == "False":
                        pass  # skip
                    else:
                        args.append(f"--{col} {val}")
            cmd = "python train_v2.py " + " ".join(args)
            st.code(cmd, language="bash")

    # ── Export ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Export")

    col_fname, col_savedir, col_savebtn = st.columns([2, 2, 1])
    with col_fname:
        filename = st.text_input("Filename", value="params.tsv")

    # TSV content
    tsv_buffer = io.StringIO()
    edited_df.to_csv(tsv_buffer, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
    tsv_content = tsv_buffer.getvalue()

    with col_savedir:
        save_dir = st.text_input("Save directory (server)", value=str(Path(__file__).parent))

    with col_savebtn:
        st.write("")  # vertical alignment
        if st.button("💾 Save on server"):
            save_path = Path(save_dir) / filename
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_text(tsv_content)
                st.success(f"Saved to `{save_path}`")
            except Exception as e:
                st.error(f"Error: {e}")

    # Show the submission command
    st.markdown("**Then submit with:**")
    st.code(
        f"sarray_params train_v2.py {filename} --gpus=1 --gpu-type=a100 --mem=16G --time=12:00:00",
        language="bash",
    )
