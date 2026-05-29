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
        return {"Custom (edit below)": ""}
    with _SCHEMES_YAML.open() as f:
        data = yaml.safe_load(f)
    schemes = {
        f"{name} — {entry['description']}": entry["scheme"]
        for name, entry in data.items()
        if isinstance(entry, dict) and "scheme" in entry
    }
    schemes["Custom (edit below)"] = ""
    return schemes


PREDEFINED_ATTENTION_SCHEMES = _load_attention_schemes()

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

    st.subheader("Experiment name")
    experiment_name = st.text_input("MLflow experiment name", value="Delphi-experiment")

    st.subheader("Architecture grid")
    n_layers = st.multiselect("n_layer", [1, 2, 4, 6, 8, 12, 16, 24], default=[12])
    n_embds = st.multiselect("n_embd", [60, 120, 180, 240, 360, 480], default=[240])
    n_heads = st.multiselect("n_head", [1, 2, 3, 4, 6, 8, 12, 16], default=[12])

    st.subheader("Training grid")
    batch_sizes = st.multiselect("batch_size", [32, 64, 128, 256, 512], default=[128])
    block_sizes = st.multiselect("block_size", ["auto", 32, 64, 96, 128, 192, 256], default=[128])
    learning_rates = st.multiselect(
        "learning_rate",
        [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
        default=[3e-4],
        format_func=lambda x: f"{x:.0e}",
    )
    test_folds = st.multiselect("test_fold", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])

    st.subheader("Domain config")
    _available_configs = sorted(_CONFIG_DIR.glob("domain_config*.yaml"))
    _config_labels = {p.name: str(p) for p in _available_configs}
    _config_choice = st.selectbox("domain_config", list(_config_labels.keys()))
    domain_config_path = _config_labels.get(_config_choice, "")
    at_birth_domains = _get_at_birth_domains(domain_config_path)
    if at_birth_domains:
        st.caption(f"at_birth → {', '.join(at_birth_domains)}")

    st.subheader("Other")
    num_workers = st.number_input("num_workers", min_value=0, max_value=16, value=4)
    subjects_path = st.text_input("subjects (path, leave empty for all)", value="")
    seeds_input = st.text_input("seed(s) (comma-separated)", value="142")
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
    use_amp = st.checkbox("use_amp (mixed precision)", value=True)
    compute_aucs = st.checkbox("compute_aucs (evaluate after training)", value=True)


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
            "attn_preset": "Full causal (mask ties)",
            "attn_custom": "",
            "suffix": "_base",
        }
    ]


def add_config():
    st.session_state.configs.append(
        {
            "domains_preset": "Base (no HLA, no PCs)",
            "domains_custom": "",
            "attn_preset": "Full causal (mask ties)",
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

            if attn_preset == "Custom (edit below)":
                cfg["attn_custom"] = st.text_area(
                    "Custom attention scheme",
                    value=cfg["attn_custom"],
                    key=f"attn_custom_{i}",
                    height=80,
                )
                attn_str = cfg["attn_custom"]
            else:
                attn_str = PREDEFINED_ATTENTION_SCHEMES[attn_preset]
                st.code(attn_str, language=None)

        with col3:
            cfg["suffix"] = st.text_input("Suffix", value=cfg["suffix"], key=f"suffix_{i}")
            if len(st.session_state.configs) > 1:
                st.button("🗑️ Remove", key=f"remove_{i}", on_click=remove_config, args=(i,))

        # Store resolved values (expand at_birth alias)
        cfg["_domains"] = resolve_at_birth(domains_str, at_birth_domains)
        cfg["_attn"] = resolve_at_birth(attn_str, at_birth_domains)

col_add, _ = st.columns([1, 4])
with col_add:
    st.button("➕ Add configuration", on_click=add_config)


# ═══════════════════════════════════════════════════════════════════════════════
#  Generate grid
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("Generated parameter grid")


def generate_grid():
    rows = []
    for fold in test_folds:
        for n_layer, n_embd, n_head, batch_size, block_size, lr, seed in itertools.product(
            n_layers, n_embds, n_heads, batch_sizes, block_sizes, learning_rates, seeds
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
                    "batch_size": batch_size,
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
    st.info(
        f"**{len(df)} experiments** = "
        f"{len(test_folds)} folds × "
        f"{len(n_layers)}×{len(n_embds)}×{len(n_heads)} arch × "
        f"{len(batch_sizes)}×{len(block_sizes)}×{len(learning_rates)} training × "
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
            for col in ["n_layer", "n_embd", "n_head", "batch_size", "block_size", "learning_rate"]:
                if col in edited_df.columns:
                    vals = sorted(edited_df[col].unique())
                    st.text(f"  {col}: {vals}")
        with col2:
            st.markdown("**Configurations:**")
            for i, cfg in enumerate(st.session_state.configs):
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
