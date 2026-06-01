import ast

import matplotlib.pyplot as plt


def _normalize_value(attr, value):
    if attr == "params.attention_scheme":
        try:
            return ast.literal_eval(value)[0] if isinstance(value, str) else value
        except Exception:
            return value
    return value


def _to_plotly_color(c):
    """
    Takes a seaborn/matplotlib RGB tuple in 0–1 range and converts to Plotly rgb string.
    If already a valid color (hex or name), returns unchanged.
    """
    if isinstance(c, tuple) and len(c) == 3:
        r, g, b = [int(float(x) * 255) for x in c]
        return f"rgb({r},{g},{b})"
    return c


def build_map(attr, palette, runs_df):
    if attr is None:
        return {}

    raw_vals = runs_df[attr].dropna().unique()
    vals = [_normalize_value(attr, v) for v in raw_vals]

    # convert palette colors to plotly-friendly format
    converted_palette = [_to_plotly_color(c) for c in palette]

    return {v: converted_palette[i % len(converted_palette)] for i, v in enumerate(vals)}


def resolve_visuals(runinfo, attr_color, attr_marker, attr_linestyle, color_map, marker_map, linestyle_map):
    """
    Given a runinfo row and the selected attributes,
    return (color, marker, linestyle) for plotting.
    """
    # COLOR
    if attr_color:
        val = runinfo[attr_color]
        color = color_map.get(val, "gray")
    else:
        color = "gray"

    # MARKER
    if attr_marker:
        val = runinfo[attr_marker]
        marker = marker_map.get(val, "o")
    else:
        marker = "o"

    # LINESTYLE
    if attr_linestyle:
        val = runinfo[attr_linestyle]
        linestyle = linestyle_map.get(val, "-")
    else:
        linestyle = "-"

    return color, marker, linestyle


def render_legends(ax, attr_color, attr_marker, attr_linestyle, color_map, marker_map, linestyle_map):
    """
    Draw legend blocks for color, marker, and linestyle.
    """
    legend_blocks = []

    # COLOR LEGEND
    if attr_color:
        handles = [plt.Line2D([0], [0], color=col, lw=3, label=str(val)) for val, col in color_map.items()]
        if handles:
            legend_blocks.append(("Color = " + attr_color.replace("params.", ""), handles))

    # MARKER LEGEND
    if attr_marker:
        handles = [
            plt.Line2D([0], [0], color="black", marker=mk, linestyle="", markersize=8, label=str(val))
            for val, mk in marker_map.items()
        ]
        if handles:
            legend_blocks.append(("Marker = " + attr_marker.replace("params.", ""), handles))

    # LINESTYLE LEGEND
    if attr_linestyle:
        handles = [
            plt.Line2D([0], [0], color="black", linestyle=ls, lw=2, label=str(val)) for val, ls in linestyle_map.items()
        ]
        if handles:
            legend_blocks.append(("Line = " + attr_linestyle.replace("params.", ""), handles))

    # RENDER ALL BLOCKS
    for idx, (title, handles) in enumerate(legend_blocks):
        ax.legend(
            handles=handles,
            title=title,
            frameon=False,
            fontsize=9,
            loc="upper right",
            bbox_to_anchor=(1.15, 1 - 0.22 * idx),
        )
