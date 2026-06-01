import re
from pathlib import Path

import pandas as pd
import plotly.graph_objs as go
from helpers import exponential_moving_average
from styling import resolve_visuals

pattern = re.compile(r"epoch(\d+)_([\d]+)")


# ==========================================================
# LOAD TOKEN LOSS
# ==========================================================
def load_token_loss_for_run(runinfo, token_id):
    data_dir = Path(runinfo["artifact_uri"]) / "val_loss_per_disease"
    files = sorted(data_dir.glob("losses_epoch*_*.csv"))
    if not files:
        return None

    dfs = []
    for f in files:
        m = pattern.search(f.name)
        if not m:
            continue

        epoch, fraction = map(int, m.groups())
        df = pd.read_csv(f)
        if df.empty:
            continue

        # New format: [token_id, log_p_total, log_p_mean]
        # Old format: [token_id, <unnamed value column>]  — keep backwards compat
        df = df.rename(columns={df.columns[0]: "token_id"})
        if "log_p_total" not in df.columns:
            df = df.rename(columns={df.columns[1]: "log_p_total"})
        if "log_p_mean" not in df.columns:
            df["log_p_mean"] = df["log_p_total"]  # old runs: treat total as mean too

        df["epoch"] = epoch
        df["subepoch"] = fraction
        dfs.append(df)

    if not dfs:
        return None

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values(["epoch", "subepoch"]).reset_index(drop=True)

    df_sel = df_all[(df_all["token_id"] == token_id) & (df_all["subepoch"] == 0)]
    return df_sel if not df_sel.empty else None


# ==========================================================
# ADD A RUN TRACE TO FIG (PLOTLY)
# ==========================================================
def add_run_trace_plotly(
    fig,
    runinfo,
    df_sel,
    ema_alpha,
    attr_color,
    attr_marker,
    attr_linestyle,
    color_map,
    marker_map,
    linestyle_map,
    loss_col="log_p_total",
):

    epochs = df_sel["epoch"].values
    losses = (-df_sel[loss_col]).clip(lower=1e-8).values
    ema = exponential_moving_average(losses, alpha=ema_alpha)

    color, mk_mpl, ls_mpl = resolve_visuals(
        runinfo,
        attr_color,
        attr_marker,
        attr_linestyle,
        color_map,
        marker_map,
        linestyle_map,
    )

    marker_mpl2plotly = {
        "o": "circle",
        "s": "square",
        "D": "diamond",
        "^": "triangle-up",
        "v": "triangle-down",
        "P": "cross",
        "X": "x",
        "*": "star",
        "+": "cross-thin",
        "1": "triangle-down",
    }
    ls_mpl2plotly = {
        "-": "solid",
        "--": "dash",
        "-.": "dashdot",
        ":": "dot",
    }

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=ema,
            mode="lines+markers",
            name=str(runinfo.name),
            showlegend=False,
            line=dict(color=color, dash=ls_mpl2plotly.get(ls_mpl, "solid")),
            marker=dict(symbol=marker_mpl2plotly.get(mk_mpl, "circle"), size=6),
            text=[str(runinfo.name)] * len(epochs),
            hovertemplate=("run: %{text}<br>epoch: %{x}<br>EMA loss: %{y:.4f}<extra></extra>"),
        )
    )


# ==========================================================
# ADD SEPARATED LEGEND BLOCKS (COLOR / MARKER / STYLE)
# ==========================================================
def add_plotly_legend(fig, attr_color, attr_marker, attr_linestyle, color_map, marker_map, linestyle_map):

    marker_mpl2plotly = {
        "o": "circle",
        "s": "square",
        "D": "diamond",
        "^": "triangle-up",
        "v": "triangle-down",
        "P": "cross",
        "X": "x",
        "*": "star",
        "+": "cross-thin",
        "1": "triangle-down",
    }

    ls_mpl2plotly = {
        "-": "solid",
        "--": "dash",
        "-.": "dashdot",
        ":": "dot",
    }

    # -------- COLORS --------
    if attr_color:
        for val, col in color_map.items():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color=col, width=4),
                    showlegend=True,
                    name=f"{attr_color.replace('params.', '')} = {val}",
                )
            )

    # -------- MARKERS --------
    if attr_marker:
        for val, mk in marker_map.items():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        symbol=marker_mpl2plotly.get(mk, "circle"),
                        size=12,
                        color="black",
                    ),
                    showlegend=True,
                    name=f"{attr_marker.replace('params.', '')} = {val}",
                )
            )

    # -------- LINE STYLES --------
    if attr_linestyle:
        for val, ls in linestyle_map.items():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color="black", dash=ls_mpl2plotly.get(ls, "solid"), width=3),
                    showlegend=True,
                    name=f"{attr_linestyle.replace('params.', '')} = {val}",
                )
            )
