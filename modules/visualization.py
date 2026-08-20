import numpy as np
import plotly.graph_objects as go

CATEGORY_COLORS = {
    "Public": "#cfe8fc",
    "Emergency": "#ffd3d3",
    "Service": "#dcf3dc",
    "Ward": "#fde7c2",
    "Patient Room": "#e8dcf7",
    "Critical Care": "#ffc2c2",
    "Diagnostic": "#c9e6ff",
    "Surgery": "#ffb3b3",
    "Staff": "#e2e2e2",
    "Consultation": "#d6ecff",
}
DEFAULT_ROOM_COLOR = "#eeeeee"
ROOM_BORDER_COLOR = "#9a9a9a"


def _circle_points(cx, cy, r, n=120):
    theta = np.linspace(0, 2 * np.pi, n)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    return x, y


def _nice_tick_step(span, target_ticks=6):
    """Pick a 'round' tick spacing (1/2/5 x 10^n) for a given axis span
    so the coordinate grid reads cleanly regardless of floor size."""
    if span <= 0:
        return 1
    raw_step = span / target_ticks
    magnitude = 10 ** np.floor(np.log10(raw_step))
    residual = raw_step / magnitude
    if residual <= 1:
        step = 1
    elif residual <= 2:
        step = 2
    elif residual <= 5:
        step = 5
    else:
        step = 10
    return step * magnitude


def build_floor_map(floor_label, locations_df, beacons_df,
                     floor_width, floor_height,
                     measured_distances=None, show_circles=True,
                     x_est=None, y_est=None,
                     destination=None,
                     ground_truth=None, show_ground_truth=False):
    fig = go.Figure()

    fig.add_shape(
        type="rect", x0=0, y0=0, x1=floor_width, y1=floor_height,
        line=dict(color="#555555", width=2),
        fillcolor="#f7f7f7",
        layer="below",
    )

    room_centers_x, room_centers_y, room_hover = [], [], []
    for _, room in locations_df.iterrows():
        x0 = room["x"] - room["width"] / 2
        x1 = room["x"] + room["width"] / 2
        y0 = room["y"] - room["height"] / 2
        y1 = room["y"] + room["height"] / 2
        color = CATEGORY_COLORS.get(room["category"], DEFAULT_ROOM_COLOR)

        fig.add_shape(
            type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color=ROOM_BORDER_COLOR, width=1.5),
            fillcolor=color,
            layer="below",
        )
        fig.add_annotation(
            x=room["x"], y=room["y"],
            text=f"<b>{room['name']}</b><br><span style='font-size:10px'>{room['category']}</span>",
            showarrow=False,
            font=dict(size=12, color="#333333"),
            align="center",
        )
        room_centers_x.append(room["x"])
        room_centers_y.append(room["y"])
        room_hover.append(f"{room['name']} ({room['category']})")

    fig.add_trace(go.Scatter(
        x=room_centers_x, y=room_centers_y,
        mode="markers",
        marker=dict(size=1, color="rgba(0,0,0,0)"),
        hovertext=room_hover,
        hoverinfo="text",
        showlegend=False,
    ))

    if show_circles and measured_distances is not None:
        for i, row in beacons_df.reset_index(drop=True).iterrows():
            cx, cy, r = row["x"], row["y"], measured_distances[i]
            cx_pts, cy_pts = _circle_points(cx, cy, r)
            fig.add_trace(go.Scatter(
                x=cx_pts, y=cy_pts,
                mode="lines",
                line=dict(color="rgba(255,140,0,0.55)", width=1.5, dash="dot"),
                name=f"{row['beacon_id']} distance circle",
                hoverinfo="skip",
                showlegend=False,
            ))

    fig.add_trace(go.Scatter(
        x=beacons_df["x"], y=beacons_df["y"],
        mode="markers+text",
        marker=dict(symbol="square", size=13, color="#FF8C00",
                    line=dict(color="#8a4b00", width=1)),
        text=beacons_df["beacon_id"],
        textposition="top center",
        textfont=dict(size=10),
        name="Bluetooth Beacon",
    ))

    if show_ground_truth and ground_truth is not None:
        gx, gy = ground_truth
        fig.add_trace(go.Scatter(
            x=[gx], y=[gy],
            mode="markers+text",
            marker=dict(symbol="x", size=14, color="#2ca02c", line=dict(width=2)),
            text=["Ground Truth (simulation)"],
            textposition="bottom center",
            name="Ground Truth (for verification only)",
        ))

    if x_est is not None and y_est is not None:
        fig.add_trace(go.Scatter(
            x=[x_est], y=[y_est],
            mode="markers+text",
            marker=dict(symbol="circle", size=20, color="#d62728",
                        line=dict(color="#ffffff", width=2)),
            text=["YOU ARE HERE"],
            textposition="bottom center",
            textfont=dict(size=13, color="#d62728"),
            name="Your Estimated Position",
        ))

    if destination is not None:
        fig.add_trace(go.Scatter(
            x=[destination["x"]], y=[destination["y"]],
            mode="markers+text",
            marker=dict(symbol="star", size=17, color="#9467bd",
                        line=dict(color="#4b2e6b", width=1)),
            text=[destination["name"]],
            textposition="top center",
            textfont=dict(size=12, color="#5e3d8a"),
            name="Destination",
        ))

    x_dtick = _nice_tick_step(floor_width)
    y_dtick = _nice_tick_step(floor_height)

    fig.update_layout(
        title=f"{floor_label} — Hospital Map",
        xaxis=dict(
            visible=True,
            range=[-1, floor_width + 1],
            title=dict(text="X (m)", font=dict(size=12, color="#666666")),
            tick0=0, dtick=x_dtick,
            tickfont=dict(size=11, color="#555555"),
            showgrid=True, gridcolor="rgba(0,0,0,0.08)", gridwidth=1,
            zeroline=True, zerolinecolor="#999999", zerolinewidth=1.5,
            showline=True, linecolor="#444444", linewidth=1.5,
            mirror=True,
        ),
        yaxis=dict(
            visible=True,
            range=[-1, floor_height + 1],
            title=dict(text="Y (m)", font=dict(size=12, color="#666666")),
            tick0=0, dtick=y_dtick,
            tickfont=dict(size=11, color="#555555"),
            showgrid=True, gridcolor="rgba(0,0,0,0.08)", gridwidth=1,
            zeroline=True, zerolinecolor="#999999", zerolinewidth=1.5,
            showline=True, linecolor="#444444", linewidth=1.5,
            mirror=True,
            scaleanchor="x", scaleratio=1,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        height=560,
        margin=dict(t=50, b=50, l=60, r=20),
        plot_bgcolor="white",
    )

    return fig