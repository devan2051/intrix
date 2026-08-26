import subprocess

import numpy as np
import streamlit as st

from modules.trilateration import trilaterate, position_error
from modules.simulation import generate_ground_truth, true_distances, simulate_measurements
from modules.locator import distance_to_destination, nearest_location
from modules.visualization import build_floor_map
from modules.pathfinding import compute_route, simplify_path, generate_directions
from modules.floors import (
    FLOOR_CODES,
    FLOOR_ICON,
    FLOOR_WIDTH,
    FLOOR_HEIGHT,
    GROUND_TRUTH_X_RANGE,
    GROUND_TRUTH_Y_RANGE,
    floor_label,
    normalize_floor_code,
    beacons_for_floor,
    locations_for_floor,
)
from modules.qr import build_floor_url, generate_qr_image

st.set_page_config(page_title="INTRIX — Hospital Locator", page_icon="🏥", layout="wide")

DEFAULT_NOISE_PERCENT = 12

PLOTLY_CONFIG = {"displayModeBar": False, "scrollZoom": False}


@st.cache_data
def _running_build():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip()
    except Exception:
        return "unknown"


def simulation_status(error_m):
    if error_m < 1.0:
        return "🟢 Very close"
    elif error_m < 3.0:
        return "🟡 Moderate difference"
    return "🔴 Large difference"


def run_locate_me(floor):
    if floor not in st.session_state.ground_truth_by_floor:
        st.session_state.ground_truth_by_floor[floor] = generate_ground_truth(
            x_range=GROUND_TRUTH_X_RANGE, y_range=GROUND_TRUTH_Y_RANGE
        )
    x_true, y_true = st.session_state.ground_truth_by_floor[floor]

    bdf = beacons_for_floor(floor)
    true_d = true_distances(x_true, y_true, bdf["x"].values, bdf["y"].values)
    measured_d = simulate_measurements(true_d, DEFAULT_NOISE_PERCENT)

    x_est, y_est = trilaterate(bdf["x"].values, bdf["y"].values, measured_d)
    err = position_error(x_true, y_true, x_est, y_est)

    st.session_state.result_by_floor[floor] = {
        "x_est": x_est,
        "y_est": y_est,
        "error": err,
        "measured_d": measured_d,
        "beacons": bdf,
        "ground_truth": (x_true, y_true),
        "noise_percent": DEFAULT_NOISE_PERCENT,
    }

st.session_state.setdefault("ground_truth_by_floor", {})
st.session_state.setdefault("result_by_floor", {})

if "floor" not in st.session_state:
    qp_floor = normalize_floor_code(st.query_params.get("floor"))
    st.session_state.floor = qp_floor if qp_floor else FLOOR_CODES[0]

st.sidebar.title("📡 INTRIX")
st.sidebar.caption("Hospital Indoor Positioning & Wayfinding · v2.2")
st.sidebar.caption(f"Running build: `{_running_build()}`")
st.sidebar.markdown("#### FLOORS")

came_from_qr = normalize_floor_code(st.query_params.get("floor")) is not None

selected_floor = st.sidebar.radio(
    "Select floor",
    FLOOR_CODES,
    format_func=lambda c: f"{FLOOR_ICON} {floor_label(c)}",
    index=FLOOR_CODES.index(st.session_state.floor),
    label_visibility="collapsed",
)
st.session_state.floor = selected_floor
floor = st.session_state.floor

st.query_params["floor"] = floor

st.sidebar.info(f"📍 Currently viewing:\n**{floor_label(floor)}**")
if came_from_qr:
    st.sidebar.caption("✅ Floor selected automatically from your link/QR code.")
else:
    st.sidebar.caption("Selected manually. Arriving from a floor's QR code selects it automatically.")

st.sidebar.divider()
with st.sidebar.expander("📱 Floor QR Codes"):
    st.caption(
        "INTRIX URL"
    )
    base_url = st.text_input(
        "This app's URL", value="https://intrix.streamlit.app",
    )
    qr_floor_code = st.selectbox(
        "Preview QR for floor", FLOOR_CODES, format_func=floor_label, key="qr_floor_choice"
    )
    floor_url = build_floor_url(base_url, qr_floor_code)
    qr_img = generate_qr_image(floor_url)
    st.image(qr_img, caption=floor_url, width=180)
    
st.title("📡 INTRIX")
st.caption("Hospital Indoor Positioning Using Simulated Bluetooth Trilateration")
st.info(
    "🏥 This is a **fictional, simulated** hospital. "
    "INTRIX does not use real Bluetooth hardware and is not connected to any real hospital."
)

st.subheader(f"{FLOOR_ICON} {floor_label(floor)}")

beacons_df = beacons_for_floor(floor)
locations_df = locations_for_floor(floor)

col_locate, col_moved, col_moved_info = st.columns([20, 8, 2])
with col_locate:
    if st.button("📍 LOCATE ME", type="primary", width='stretch'):
        run_locate_me(floor)
with col_moved:
    if st.button("🔄 Simulation", width='stretch',
                  help="Simulates that you've walked to a different spot on this floor."):
        st.session_state.ground_truth_by_floor.pop(floor, None)
        run_locate_me(floor)
with col_moved_info:
    st.markdown(
        """
        <style>
        .intrix-sim-info-wrap {
            display: flex; align-items: center; justify-content: center; height: 2.5rem;
        }
        .intrix-sim-info {
            position: relative; display: inline-flex; align-items: center; justify-content: center;
            width: 1.3rem; height: 1.3rem; border-radius: 50%;
            background: rgba(128, 128, 128, 0.18); border: 1px solid rgba(128, 128, 128, 0.55);
            color: rgba(128, 128, 128, 0.95); font-size: 0.78rem; font-weight: 700; cursor: help;
            user-select: none;
        }
        .intrix-sim-info:hover {
            background: rgba(128, 128, 128, 0.32); color: inherit; border-color: rgba(128, 128, 128, 0.8);
        }
        .intrix-sim-info .intrix-sim-tooltip {
            visibility: hidden; opacity: 0; position: absolute; bottom: 135%; right: 0; width: 200px;
            background-color: #262730; color: #fafafa; text-align: left; padding: 0.5rem 0.65rem;
            border-radius: 0.4rem; border: 1px solid rgba(250, 250, 250, 0.15);
            font-size: 0.75rem; font-weight: 400; line-height: 1.35;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4); transition: opacity 0.15s ease-in-out;
            z-index: 1000; pointer-events: none;
        }
        .intrix-sim-info:hover .intrix-sim-tooltip { visibility: visible; opacity: 1; }
        </style>
        <div class="intrix-sim-info-wrap">
          <span class="intrix-sim-info">i<span class="intrix-sim-tooltip">Simulates a change in your location to demonstrate how INTRIX recalculates your position and route.</span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

result = st.session_state.result_by_floor.get(floor)
destination_point = None
route_points_for_map = None
directions = None
dest_distance_text = None
route_distance_text = None

if result is not None:
    near_name, _near_dist = nearest_location(result["x_est"], result["y_est"], locations_df)
    location_value = f"Near {near_name}" if near_name else "Unknown"
    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; align-items: flex-start;
                    gap: 0.2rem; margin: 0 0 1rem 0; padding: 0;">
            <span style="font-size: 1.75rem; font-weight: 700; line-height: 1.25;">📍Your Location: </span>
            <span style="font-size: 1.5rem; font-weight: 700; opacity: 0.75; line-height: 1.25;">{location_value}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🧭  Where do you want to go?")
    st.markdown("Destination")
    dest_name = st.selectbox("Destination", locations_df["name"].tolist(), width=420)
    dest_row = locations_df[locations_df["name"] == dest_name].iloc[0]
    dest_distance = distance_to_destination(result["x_est"], result["y_est"], dest_row["x"], dest_row["y"])
    dest_distance_text = f"🏥 **{dest_name}** is approximately **{dest_distance:.1f} m** away in a straight line."
    destination_point = {"name": dest_name, "x": dest_row["x"], "y": dest_row["y"]}
    if str(dest_row.get("floor", floor)) != str(floor):
        st.warning(
            f"🏥 **{dest_name}** is on {floor_label(dest_row['floor'])}. "
            "Please switch to that floor first, then generate the path again."
        )
    else:
        route = compute_route(
            result["x_est"], result["y_est"],
            dest_row["x"], dest_row["y"],
            floor, FLOOR_WIDTH, FLOOR_HEIGHT,
        )
        if "error" in route:
            st.warning(f"⚠️ {route['error']}")
        elif route["arrived"]:
            st.success("📍 You have reached your destination.")
            directions = generate_directions(route["points"], dest_name)
        else:
            route_points_for_map = simplify_path(route["points"])
            directions = generate_directions(route["points"], dest_name)
            route_distance_text = (
                f"🚶 **Route distance:** approximately **{route['route_distance']:.1f} m** "
                "along the walkable path."
            )

    show_circles = st.session_state.get("show_circles_toggle", True)
else:
    st.info("👆 Click **📍 LOCATE ME** to find your position on the map.")
    show_circles = False

st.markdown("#### 🗺️ Floor Map")
fig = build_floor_map(
    floor_label(floor),
    locations_df,
    beacons_df,
    FLOOR_WIDTH, FLOOR_HEIGHT,
    measured_distances=result["measured_d"] if result else None,
    show_circles=show_circles,
    x_est=result["x_est"] if result else None,
    y_est=result["y_est"] if result else None,
    destination=destination_point,
    route_points=route_points_for_map,
)
st.plotly_chart(fig, config=PLOTLY_CONFIG)

if result is not None:
    st.write(dest_distance_text)
    if route_distance_text:
        st.write(route_distance_text)
    show_circles = st.checkbox(
        "Show measurement circles on map", value=True,
        help="Shows the Bluetooth distance circles used to calculate your position.",
        key="show_circles_toggle",
    )

if directions:
    st.markdown("#### 🧭 Directions")
    for i, step in enumerate(directions, start=1):
        st.write(f"{i}. {step['icon']} {step['text']}")

with st.expander("🔧 Technical Details — How was my location calculated?"):
    if result is None:
        st.caption("Click **Locate Me** first to see the underlying Bluetooth measurements and math.")
    else:
        st.markdown("**Simulated Bluetooth distance measurements** (not real hardware readings):")
        meas_table = beacons_df[["beacon_id"]].copy()
        meas_table["Estimated Distance (m)"] = np.round(result["measured_d"], 2)
        st.dataframe(meas_table, hide_index=True, width='stretch')

        st.markdown(
            f"""
Each measured distance forms a **circle** around its beacon. The estimated
position above is the point that best fits **all** of these circles at once,
found using mathematical trilateration (linearization + least squares).

- Simulated measurement noise used: **{result['noise_percent']}%**
            """
        )

        st.markdown("##### Simulation Verification")
        st.caption(
            "The actual simulated position below is the hidden position used to "
            "generate the measurements above. It is used only to check how close "
            "the prediction landed — it is **never** given to the trilateration algorithm."
        )

        gx, gy = result["ground_truth"]
        vc1, vc2 = st.columns(2)
        vc1.metric("Predicted Position", f"X = {result['x_est']:.2f} m, Y = {result['y_est']:.2f} m")
        vc2.metric("Actual Simulated Position", f"X = {gx:.2f} m, Y = {gy:.2f} m")

        vc3, vc4 = st.columns(2)
        vc3.metric("Position Error", f"{result['error']:.2f} m")
        vc4.metric("Result", simulation_status(result["error"]))

st.divider()
st.caption(
    "INTRIX — A mathematical and computational simulation demonstrating how distance "
    "measurements from fixed Bluetooth beacons can be used to estimate an unknown "
    "indoor position through trilateration."
)
