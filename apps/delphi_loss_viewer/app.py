import ast
from pathlib import Path

import mlflow
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import streamlit as st

# local modules
from helpers import exponential_moving_average, load_labels
from mlflow_loader import filter_runs_with_loss_files, load_runs, validate_loss_files
from plot_loss import add_plotly_legend, add_run_trace_plotly, load_token_loss_for_run
from styling import _normalize_value, build_map

from utils import setup_mlflow

setup_mlflow()

print(mlflow.get_tracking_uri())
# ---------------------------------------------------------
# STREAMLIT LAYOUT
# ---------------------------------------------------------
st.set_page_config(page_title="Delphi Loss Evolution Viewer", layout="wide")
st.title("Delphi Loss Evolution Viewer")

# ---------------------------------------------------------
# SIDEBAR — EXPERIMENTS
# ---------------------------------------------------------
with st.sidebar:
    st.header("Controls")

    experiment_names = [(exp.experiment_id, exp.name) for exp in mlflow.search_experiments()]
    exp_ids = st.multiselect("Experiment", [x[0] for x in experiment_names])

    # Load runs
    runs_df = load_runs(exp_ids)

    MIN_FILES = 10
    runs_df = filter_runs_with_loss_files(runs_df, min_files=MIN_FILES)
    runs_df = validate_loss_files(runs_df)

    if runs_df.empty:
        st.error(f"No runs with ≥ {MIN_FILES} valid loss files.")
        st.stop()

    # Parameter-based filtering
    st.subheader("Run selection by parameter grid")

    raw_vals = sorted(runs_df["params.attention_scheme"].dropna().unique())
    attn_schemes = set([ast.literal_eval(v)[0] for v in raw_vals])
    selected_attn = st.multiselect(
        "attention_scheme",
        options=[*list(attn_schemes), "(any)"],
    )

    selected_layers = st.multiselect(
        "n_layer", options=["(any)"] + [str(v) for v in sorted(runs_df["params.n_layer"].dropna().unique())]
    )

    filtered = runs_df.copy()
    filtered["params.attention_scheme"] = filtered["params.attention_scheme"].apply(lambda x: ast.literal_eval(x)[0])

    if selected_attn and "(any)" not in selected_attn:
        filtered = filtered[filtered["params.attention_scheme"].isin(selected_attn)]

    if selected_layers and "(any)" not in selected_layers:
        filtered = filtered[filtered["params.n_layer"].astype(str).isin(selected_layers)]

    if filtered.empty:
        st.warning("No runs match selected grid parameters.")
        st.stop()

    selected_runs = st.multiselect(
        "Select runs",
        options=filtered["run_id"],
        default=filtered["run_id"].tolist(),
    )

    st.dataframe(filtered[["run_id", "experiment_id", "status", "start_time", "end_time", "n_loss_files"]])

    if not selected_runs:
        st.warning("Select at least one run.")
        st.stop()


# ---------------------------------------------------------
# TOKEN SELECTOR
# ---------------------------------------------------------
labels_path = "tokenizer.yaml"
labels = load_labels(labels_path)
labels = [*labels, "Death"]

token_display = [f"{i}: {labels[i]}" if i < len(labels) else str(i) for i in range(1257)]

selected_token = st.sidebar.selectbox("Select token:", token_display)
token_id = int(selected_token.split(":")[0])

loss_metric = st.sidebar.radio(
    "Loss metric",
    options=["log_p_total", "log_p_mean"],
    help=(
        "log_p_total: sum of per-batch mean log-p (contribution = frequency × difficulty)\n"
        "log_p_mean:  mean of per-batch mean log-p (difficulty, frequency-independent)"
    ),
)


# ---------------------------------------------------------
# STYLING PARAMETERS
# ---------------------------------------------------------
styleable_params = [c for c in runs_df.columns if c.startswith("params.") and runs_df[c].nunique() > 1]

style_choices = st.sidebar.multiselect("Select up to 3 styling parameters:", options=styleable_params, max_selections=3)

style_choices_padded: list[str | None] = list(style_choices)
while len(style_choices_padded) < 3:
    style_choices_padded.append(None)
attr_color, attr_marker, attr_linestyle = style_choices_padded

color_palette = sns.color_palette("tab10")
marker_palette = ["o", "s", "D", "^", "v", "P", "X", "*", "+", "1"]
line_palette = ["-", "--", "-.", ":"]

color_map = build_map(attr_color, color_palette, runs_df)
marker_map = build_map(attr_marker, marker_palette, runs_df)
linestyle_map = build_map(attr_linestyle, line_palette, runs_df)

# st.write("DEBUG color_map:", color_map)
# st.write("DEBUG marker_map:", marker_map)
# st.write("DEBUG linestyle_map:", linestyle_map)

# ---------------------------------------------------------
# PLOTS (Plotly)
# ---------------------------------------------------------
kk = st.slider("EMA alpha (0–100)", min_value=1, max_value=100, value=5)
ema_alpha = kk / 100.0

fig_token = go.Figure()
fig_total = go.Figure()

fig_token.update_layout(width=900, height=600)

runs_indexed = runs_df.set_index("run_id")

# ---------------------------------------------------------
# ADD RUN TRACES
# ---------------------------------------------------------
for runid in selected_runs:
    runinfo = runs_indexed.loc[runid]

    # TOTAL LOSS
    val_total_file = Path(runinfo["artifact_uri"]) / "metrics" / "val_total"
    try:
        val_total_loss = pd.read_csv(val_total_file, sep=" ", header=None).iloc[:, 1]
        ema_total = exponential_moving_average(val_total_loss.values, ema_alpha)

        fig_total.add_trace(
            go.Scatter(
                x=list(range(len(ema_total))),
                y=ema_total,
                mode="lines",
                name=runid,
                showlegend=False,
                line=dict(color="rgba(100,100,100,0.4)"),
            )
        )
    except Exception:
        pass

    # TOKEN LOSS
    df_sel = load_token_loss_for_run(runinfo, token_id)

    runinfo[attr_color] = _normalize_value(attr_color, runinfo[attr_color])

    if df_sel is not None:
        add_run_trace_plotly(
            fig_token,
            runinfo,
            df_sel,
            ema_alpha,
            attr_color,
            attr_marker,
            attr_linestyle,
            color_map,
            marker_map,
            linestyle_map,
            loss_col=loss_metric,
        )

        # st.write("DEBUG PARAM VALS FOR", runid)
        # st.write("RAW:", runinfo[attr_color])
        # st.write("NORM:", _normalize_value(attr_color, runinfo[attr_color]))


# ---------------------------------------------------------
# LEGENDS (SEPARATED)
# ---------------------------------------------------------
add_plotly_legend(
    fig_token,
    attr_color,
    attr_marker,
    attr_linestyle,
    color_map,
    marker_map,
    linestyle_map,
)


# ---------------------------------------------------------
# UPDATE LAYOUTS
# ---------------------------------------------------------
fig_total.update_layout(
    title="Validation total loss (EMA)",
    xaxis_title="Step",
    yaxis_title="Loss",
    template="plotly_white",
)

label_name = labels[token_id + 1] if token_id < len(labels) else f"Token {token_id}"

fig_token.update_layout(
    title=f"Loss evolution for {label_name}  [{loss_metric}]",
    xaxis_title="Epoch",
    yaxis_title="Loss (log scale)",
    yaxis_type="log",
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.35,
        xanchor="left",
        x=0,
    ),
)

# ---------------------------------------------------------
# DISPLAY PLOTS
# ---------------------------------------------------------
# st.plotly_chart(fig_total, use_container_width=True)
st.plotly_chart(fig_token, use_container_width=True)


# ---------------------------------------------------------
# METADATA
# ---------------------------------------------------------
# st.subheader("Selected runs metadata")
# st.dataframe(
# runs_df[runs_df["run_id"].isin(selected_runs)][
# ["run_id", "experiment_id", "status", "start_time", "end_time", "n_loss_files", "artifact_uri"]
# ]
# )
