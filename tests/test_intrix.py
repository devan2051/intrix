
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from modules.trilateration import euclidean_distance, trilaterate, position_error
from modules.simulation import generate_ground_truth, true_distances, simulate_measurements
from modules.locator import distance_to_destination, nearest_location
from modules.floors import (
    FLOOR_CODES, GROUND_TRUTH_X_RANGE, GROUND_TRUTH_Y_RANGE,
    beacons_for_floor, locations_for_floor,
)
from modules.qr import build_floor_url

PASS = 0
FAIL = 0


def check(name, condition):
    global PASS, FAIL
    if condition:
        print(f"  [PASS] {name}")
        PASS += 1
    else:
        print(f"  [FAIL] {name}")
        FAIL += 1


GF_BEACONS = beacons_for_floor("ground")
bx, by = GF_BEACONS["x"].values, GF_BEACONS["y"].values

print("1. Distance formula")
d = euclidean_distance(0, 0, 3, 4)
check("distance((0,0),(3,4)) == 5", abs(d - 5) < 1e-9)


print("\n2. Zero-noise trilateration accuracy")
x_true, y_true = generate_ground_truth(x_range=GROUND_TRUTH_X_RANGE, y_range=GROUND_TRUTH_Y_RANGE, seed=42)
true_d = true_distances(x_true, y_true, bx, by)
measured_zero_noise = simulate_measurements(true_d, noise_percent=0)
x_est, y_est = trilaterate(bx, by, measured_zero_noise)
err_zero = position_error(x_true, y_true, x_est, y_est)
check(f"zero-noise error is ~0 (got {err_zero:.6f} m)", err_zero < 1e-6)


print("\n3. Increasing noise generally increases error")
def average_error(noise_percent, trials=60):
    errors = []
    for i in range(trials):
        xt, yt = generate_ground_truth(x_range=GROUND_TRUTH_X_RANGE, y_range=GROUND_TRUTH_Y_RANGE, seed=1000 + i)
        td = true_distances(xt, yt, bx, by)
        md = simulate_measurements(td, noise_percent, seed=2000 + i)
        xe, ye = trilaterate(bx, by, md)
        errors.append(position_error(xt, yt, xe, ye))
    return float(np.mean(errors))

low_noise_err = average_error(5)
high_noise_err = average_error(25)
print(f"  avg error at 5% noise:  {low_noise_err:.3f} m")
print(f"  avg error at 25% noise: {high_noise_err:.3f} m")
check("average error at 25% noise > average error at 5% noise", high_noise_err > low_noise_err)


print("\n4. Works for different hidden ground-truth positions")
ok = True
for seed in range(10):
    xt, yt = generate_ground_truth(x_range=GROUND_TRUTH_X_RANGE, y_range=GROUND_TRUTH_Y_RANGE, seed=seed)
    td = true_distances(xt, yt, bx, by)
    md = simulate_measurements(td, noise_percent=0)
    xe, ye = trilaterate(bx, by, md)
    if position_error(xt, yt, xe, ye) > 1e-6:
        ok = False
        break
check("zero-noise trilateration is accurate across 10 random positions", ok)


print("\n5. Beacon geometry (not collinear)")
x1, y1 = bx[0], by[0]
x2, y2 = bx[1], by[1]
x3, y3 = bx[2], by[2]
area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2
check("first three ground-floor beacons are not collinear (triangle area > 0)", area > 0)


print("\n6. Destination distance calculation")
dist = distance_to_destination(0, 0, 3, 4)
check("distance_to_destination((0,0) -> (3,4)) == 5", abs(dist - 5) < 1e-9)


print("\n7. Every floor loads with beacons and locations")
ok = True
for floor in FLOOR_CODES:
    b = beacons_for_floor(floor)
    l = locations_for_floor(floor)
    if len(b) < 3 or len(l) == 0:
        ok = False
        print(f"  problem on floor '{floor}': {len(b)} beacons, {len(l)} locations")
check("all floors have >=3 beacons and >=1 location", ok)


print("\n8. Beacons are never mixed across floors")
ok = True
for floor in FLOOR_CODES:
    b = beacons_for_floor(floor)
    if not (b["floor"] == floor).all():
        ok = False
check("beacons_for_floor() only ever returns that floor's own beacons", ok)

print("\n9. Trilateration is accurate on each floor's own beacons (0% noise)")
ok = True
for floor in FLOOR_CODES:
    b = beacons_for_floor(floor)
    xt, yt = generate_ground_truth(x_range=GROUND_TRUTH_X_RANGE, y_range=GROUND_TRUTH_Y_RANGE, seed=99)
    td = true_distances(xt, yt, b["x"].values, b["y"].values)
    md = simulate_measurements(td, noise_percent=0)
    xe, ye = trilaterate(b["x"].values, b["y"].values, md)
    err = position_error(xt, yt, xe, ye)
    if err > 1e-6:
        ok = False
        print(f"  problem on floor '{floor}': error {err}")
check("every floor's beacon set localizes correctly at 0% noise", ok)

print("\n10. nearest_location() picks the closest destination")
locs = locations_for_floor("2")
first_row = locs.iloc[0]
name, dist = nearest_location(first_row["x"], first_row["y"], locs)
check(f"nearest_location() at a room's own center returns that room ('{name}')", name == first_row["name"])

print("\n11. QR floor-URL builder produces the expected URL shape")
url = build_floor_url("https://example.streamlit.app", "2")
check("floor URL contains the base URL and floor=2", url == "https://example.streamlit.app/?floor=2")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
