"""
test_intrix.py
---------------
Simple, dependency-free tests for INTRIX (no pytest required).

Run with:
    python tests/test_intrix.py

Covers the checks described in the project spec:
    1. Distance formula
    2. Zero-noise trilateration accuracy
    3. Effect of increasing noise on error
    4. Different hidden ground-truth positions
    5. Beacon geometry (not collinear)
    6. Destination-distance calculation
"""

import os
import sys

# Allow running this file directly (python tests/test_intrix.py) by
# adding the project root to the import path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from modules.trilateration import euclidean_distance, trilaterate, position_error
from modules.simulation import generate_ground_truth, true_distances, simulate_measurements
from modules.locator import distance_to_destination

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


def load_beacons():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "beacons.csv")
    return pd.read_csv(path)


# ---------------------------------------------------------------------
print("1. Distance formula")
d = euclidean_distance(0, 0, 3, 4)
check("distance((0,0),(3,4)) == 5", abs(d - 5) < 1e-9)

# ---------------------------------------------------------------------
print("\n2. Zero-noise trilateration accuracy")
beacons = load_beacons()
bx, by = beacons["x"].values, beacons["y"].values

x_true, y_true = generate_ground_truth(seed=42)
true_d = true_distances(x_true, y_true, bx, by)
measured_zero_noise = simulate_measurements(true_d, noise_percent=0)
x_est, y_est = trilaterate(bx, by, measured_zero_noise)
err_zero = position_error(x_true, y_true, x_est, y_est)
check(f"zero-noise error is ~0 (got {err_zero:.6f} m)", err_zero < 1e-6)

# ---------------------------------------------------------------------
print("\n3. Increasing noise generally increases error")
# Average error over several trials at each noise level, since any
# single random trial can go either way.
def average_error(noise_percent, trials=60):
    errors = []
    for i in range(trials):
        xt, yt = generate_ground_truth(seed=1000 + i)
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

# ---------------------------------------------------------------------
print("\n4. Works for different hidden ground-truth positions")
ok = True
for seed in range(10):
    xt, yt = generate_ground_truth(seed=seed)
    td = true_distances(xt, yt, bx, by)
    md = simulate_measurements(td, noise_percent=0)
    xe, ye = trilaterate(bx, by, md)
    if position_error(xt, yt, xe, ye) > 1e-6:
        ok = False
        break
check("zero-noise trilateration is accurate across 10 random positions", ok)

# ---------------------------------------------------------------------
print("\n5. Beacon geometry (not collinear)")
# Beacons must not all lie on one straight line, or trilateration
# cannot resolve a 2D position. Check using the area of the triangle
# formed by the first three beacons.
x1, y1 = bx[0], by[0]
x2, y2 = bx[1], by[1]
x3, y3 = bx[2], by[2]
area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2
check("first three beacons are not collinear (triangle area > 0)", area > 0)

# ---------------------------------------------------------------------
print("\n6. Destination distance calculation")
dist = distance_to_destination(0, 0, 3, 4)
check("distance_to_destination((0,0) -> (3,4)) == 5", abs(dist - 5) < 1e-9)

# ---------------------------------------------------------------------
print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
