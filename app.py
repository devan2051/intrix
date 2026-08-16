"""
app.py
------
INTRIX — Simulated Bluetooth-Based Indoor Positioning
Using Mathematical Trilateration

This file is the Streamlit interface only. All the actual math and
simulation logic lives in the modules/ package -- see:

    modules/trilateration.py  -> distance formula, trilateration, error
    modules/simulation.py     -> hidden ground truth, simulated Bluetooth
                                  measurements, measurement noise
    modules/locator.py        -> secondary destination-distance feature
    modules/visualization.py  -> the Plotly map

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from modules.trilateration import trilaterate, position_error
from modules.simulation import generate_ground_truth, true_distances, simulate_measurements
from modules.locator import load_locations, distance_to_destination
from modules.visualization import build_map

st.set_page_config(page_title="INTRIX", page_icon="📡", layout="wide")


# ---------------------------------------------------------------------
# Load fixed, known data
# ---------------------------------------------------------------------
@st.cache_data
def load_beacons(path="data/beacons.csv"):
    return pd.read_csv(path)


beacons_df = load_beacons()
locations_df = load_locations("data/locations.csv")

# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
# ground_truth is INTRIX's hidden, internal test position. It exists
# only so the simulation has something to generate measurements from
# and something to grade the estimate against. It is never given to
# the trilateration algorithm.
if "ground_truth" not in st.session_state:
    st.session_state.ground_truth = generate_ground_truth()

if "result" not in st.session_state:
    st.session_state.result = None

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("📡 INTRIX")
st.subheader("Simulated Bluetooth-Based Indoor Positioning Using Mathematical Trilateration")

st.info(
    "INTRIX does not use physical Bluetooth hardware. It **simulates** "
    "Bluetooth-based distance measurements to demonstrate how mathematical "
    "trilateration can estimate an unknown indoor position."
)

# ---------------------------------------------------------------------
# Section: Fixed Bluetooth Beacons
# ---------------------------------------------------------------------
st.header("📡 Fixed Bluetooth Beacons")
st.write("These beacon positions are fixed in the building and known in advance.")
st.dataframe(beacons_df, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------
# Section: Measurement Noise
# ---------------------------------------------------------------------
st.header("⚙️ Measurement Noise")
noise_percent = st.slider(
    "Simulated Bluetooth measurement noise",
    min_value=0, max_value=30, value=10, step=1,
    format="%d%%",
    help="Represents uncertainty in Bluetooth-based distance estimation (e.g. from RSSI).",
)

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🎲 New Random Scenario"):
        st.session_state.ground_truth = generate_ground_truth()
        st.session_state.result = None
with col2:
    st.caption(
        "Generates a new hidden device position for the simulation to test against. "
        "This position is not shown to the trilateration algorithm."
    )

x_true, y_true = st.session_state.ground_truth

# ---------------------------------------------------------------------
# Section: Simulated Bluetooth Measurements
# ---------------------------------------------------------------------
true_d = true_distances(x_true, y_true, beacons_df["x"].values, beacons_df["y"].values)
measured_d = simulate_measurements(true_d, noise_percent)

st.header("📶 Simulated Bluetooth Measurements")
st.caption("**Simulated Bluetooth Distance Measurements** — not real Bluetooth hardware readings.")
measurements_df = beacons_df[["beacon"]].copy()
measurements_df["Estimated Distance (m)"] = np.round(measured_d, 2)
st.dataframe(measurements_df, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------
# Section: Locate Device
# ---------------------------------------------------------------------
st.header("📍 Locate Device")
st.write(
    "Clicking the button below sends **only** the beacon coordinates and the "
    "simulated measurements above into the trilateration algorithm. "
    "The hidden device position is never given to it."
)

if st.button("📍 Locate Device", type="primary"):
    x_est, y_est = trilaterate(beacons_df["x"].values, beacons_df["y"].values, measured_d)
    err = position_error(x_true, y_true, x_est, y_est)
    st.session_state.result = {
        "x_est": x_est,
        "y_est": y_est,
        "error": err,
        "measured_d": measured_d,
        "noise_percent": noise_percent,
    }

result = st.session_state.result

if result is not None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated X", f"{result['x_est']:.2f} m")
    c2.metric("Estimated Y", f"{result['y_est']:.2f} m")
    c3.metric("Position Error", f"{result['error']:.2f} m")

    # -------------------------------------------------------------
    # Section: Indoor Destination Locator (secondary feature)
    # -------------------------------------------------------------
    st.header("📍 Indoor Destination Locator")
    st.caption("Secondary feature — distance from the *estimated* device position to a destination.")
    dest_name = st.selectbox("Select a destination", locations_df["name"].tolist())
    dest_row = locations_df[locations_df["name"] == dest_name].iloc[0]
    dest_distance = distance_to_destination(
        result["x_est"], result["y_est"], dest_row["x"], dest_row["y"]
    )
    st.write(f"Distance from estimated position to **{dest_name}**: **{dest_distance:.2f} m**")
    destination_point = {"name": dest_name, "x": dest_row["x"], "y": dest_row["y"]}

    # -------------------------------------------------------------
    # Section: Position Map
    # -------------------------------------------------------------
    st.header("🗺️ Position Map")
    fig = build_map(
        beacons_df,
        result["measured_d"],
        result["x_est"],
        result["y_est"],
        ground_truth=(x_true, y_true),
        destination=destination_point,
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.caption("👆 Click **Locate Device** above to run the trilateration algorithm and see the map.")

# ---------------------------------------------------------------------
# Section: How It Works / Mathematics
# ---------------------------------------------------------------------
with st.expander("📐 How It Works / Mathematics"):
    st.markdown(
        r"""
**1. Bluetooth beacons**
Fixed Bluetooth beacons are placed at known locations inside a building.
A device (like a phone) picks up signals from each nearby beacon.

**2. Distance estimation**
In a real system, Bluetooth signal measurements such as RSSI (signal
strength) can be used to *estimate* the distance between the device and
a beacon — weaker signal generally means further away. INTRIX does not
use real hardware, so it **simulates** these estimated distances instead.

**3. Distance formula**

$$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

**4. Circle representation**
If a beacon at $(x_i, y_i)$ measures a distance $d_i$ to the device, the
device could be *anywhere* on the circle:

$$(x - x_i)^2 + (y - y_i)^2 = d_i^2$$

One beacon alone only narrows the device down to a circle. A second
beacon narrows it to (at most) two intersection points. A third beacon
resolves the position uniquely — and extra beacons help average out
measurement noise.

**5. Trilateration**
INTRIX subtracts the equation of one reference beacon from the equations
of all the others. This cancels out the squared terms and turns the
problem into a system of *linear* equations, which is then solved with
least squares (`numpy.linalg.lstsq`) to find the $(x, y)$ that best fits
every circle at once. Trilateration uses **distances**, not angles —
that's what separates it from triangulation.

**6. Effect of noise**
At 0% noise, the simulated measurements exactly match the true
distances, so the circles meet (almost) perfectly and the estimated
position lands right on the hidden ground truth. As noise increases,
the circles stop agreeing with each other as precisely, and the
best-fit position drifts further from the truth — so positioning error
generally increases.

**7. Position error**

$$\text{Error} = \sqrt{(X_{actual} - X_{estimated})^2 + (Y_{actual} - Y_{estimated})^2}$$

The ground-truth position is used **only** here, to grade the result —
never to calculate it.
        """
    )

st.divider()
st.caption(
    "INTRIX — A mathematical and computational simulation demonstrating how distance "
    "measurements from fixed Bluetooth beacons can be used to estimate an unknown "
    "indoor position through trilateration."
)
