"""
visualization.py
-----------------
Builds the 2D Plotly map that makes the trilateration idea visible:

    Beacon -> Measured Distance -> Circle -> Multiple Circles
           -> Best-Fit Position -> Estimated Position

Markers used:
    Orange square  -> Bluetooth beacon
    Dotted circle  -> distance circle (radius = measured distance)
    Red circle     -> estimated device position
    Green X        -> ground truth (labeled clearly, shown only for
                       educational comparison -- never an input)
    Purple star    -> selected destination (optional)
"""

import numpy as np
import plotly.graph_objects as go


def _circle_points(cx, cy, r, n=120):
    """Trace n points around a circle of radius r centered at (cx, cy)."""
    theta = np.linspace(0, 2 * np.pi, n)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    return x, y


def build_map(beacons_df, measured_distances, x_est, y_est,
              ground_truth=None, destination=None):
    """
    Build the INTRIX Plotly figure.

    Parameters
    ----------
    beacons_df : DataFrame with columns ["beacon", "x", "y"]
    measured_distances : array-like, simulated distances in the same
        order as beacons_df's rows
    x_est, y_est : estimated device position
    ground_truth : optional (x, y) tuple -- shown for educational
        comparison only, clearly labeled, never used as an input
    destination : optional dict {"name": str, "x": float, "y": float}
    """
    fig = go.Figure()

    # --- Distance circles: one per beacon, radius = measured distance ---
    for i, row in beacons_df.reset_index(drop=True).iterrows():
        cx, cy, r = row["x"], row["y"], measured_distances[i]
        cx_pts, cy_pts = _circle_points(cx, cy, r)
        fig.add_trace(go.Scatter(
            x=cx_pts, y=cy_pts,
            mode="lines",
            line=dict(color="rgba(255,140,0,0.45)", width=1.5, dash="dot"),
            name=f"{row['beacon']} distance circle",
            hoverinfo="skip",
            showlegend=False,
        ))

    # --- Beacons ---
    fig.add_trace(go.Scatter(
        x=beacons_df["x"], y=beacons_df["y"],
        mode="markers+text",
        marker=dict(symbol="square", size=16, color="#FF8C00",
                    line=dict(color="#8a4b00", width=1)),
        text=beacons_df["beacon"],
        textposition="top center",
        name="Bluetooth Beacon",
    ))

    # --- Ground truth (educational comparison only) ---
    if ground_truth is not None:
        gx, gy = ground_truth
        fig.add_trace(go.Scatter(
            x=[gx], y=[gy],
            mode="markers+text",
            marker=dict(symbol="x", size=14, color="#2ca02c",
                        line=dict(width=2)),
            text=["Ground Truth"],
            textposition="bottom center",
            name="Ground Truth (evaluation only)",
        ))

    # --- Estimated position ---
    fig.add_trace(go.Scatter(
        x=[x_est], y=[y_est],
        mode="markers+text",
        marker=dict(symbol="circle", size=16, color="#d62728",
                    line=dict(color="#7a0000", width=2)),
        text=["Estimated Position"],
        textposition="bottom center",
        name="Estimated Device Position",
    ))

    # --- Selected destination (optional, secondary feature) ---
    if destination is not None:
        fig.add_trace(go.Scatter(
            x=[destination["x"]], y=[destination["y"]],
            mode="markers+text",
            marker=dict(symbol="star", size=15, color="#9467bd",
                        line=dict(color="#4b2e6b", width=1)),
            text=[destination["name"]],
            textposition="top center",
            name="Selected Destination",
        ))

    fig.update_layout(
        title="INTRIX — Beacons, Distance Circles & Estimated Position",
        xaxis_title="X (meters)",
        yaxis_title="Y (meters)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35),
        height=620,
        margin=dict(t=60, b=20),
        plot_bgcolor="#fafafa",
    )

    return fig
