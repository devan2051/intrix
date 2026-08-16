"""
simulation.py
-------------
Everything related to SIMULATING Bluetooth-based distance
measurements. INTRIX does not use real Bluetooth hardware, so this
module stands in for the real-world measurement process.

What this module does:

1. generate_ground_truth() -- internally picks a hidden device
   position. This exists ONLY so the simulation has something to
   generate realistic measurements from, and something to grade the
   final estimate against.

2. true_distances() -- calculates the exact, noise-free distance
   from the hidden position to every beacon, using the plain
   distance formula.

3. simulate_measurements() -- adds random noise to the true
   distances, representing the fact that real Bluetooth signal based
   distance estimation (e.g. from RSSI) is never perfectly accurate.

THE GOLDEN RULE
-----------------
The hidden ground-truth position produced here is used ONLY to:
    - generate realistic simulated measurements
    - evaluate the final positioning error

It is NEVER passed into modules/trilateration.py's trilaterate().
"""

import numpy as np

from modules.trilateration import euclidean_distance


def generate_ground_truth(x_range=(2, 23), y_range=(2, 16), seed=None):
    """
    Randomly generate a hidden device position within a plausible
    indoor area (loosely inside the rectangle formed by the sample
    beacons). This position represents "where the device actually
    is" -- information the positioning system is not allowed to see.

    seed : optional int, for reproducible demos/tests.
    """
    rng = np.random.default_rng(seed)
    x_true = rng.uniform(*x_range)
    y_true = rng.uniform(*y_range)
    return float(x_true), float(y_true)


def true_distances(x_true, y_true, beacon_x, beacon_y):
    """
    Exact (noise-free) distance from the hidden ground-truth position
    to every beacon. This is an internal simulation step, not
    something a real Bluetooth system could measure directly.
    """
    beacon_x = np.asarray(beacon_x, dtype=float)
    beacon_y = np.asarray(beacon_y, dtype=float)
    return euclidean_distance(x_true, y_true, beacon_x, beacon_y)


def simulate_measurements(true_dists, noise_percent, seed=None):
    """
    Simulate imperfect Bluetooth-based distance measurements by
    applying random noise to the true distances.

    noise_percent : 0-30
        Represents how noisy the simulated Bluetooth signal is.
        At 0%, the simulated measurement equals the true distance.
        At higher values, the measured distance can be further off
        from the true distance, in either direction -- similar to
        how real RSSI-based distance estimates vary around the truth.

    Returns
    -------
    numpy array of simulated measured distances, same order/shape as
    true_dists.
    """
    rng = np.random.default_rng(seed)
    true_dists = np.asarray(true_dists, dtype=float)

    if noise_percent <= 0:
        return true_dists.copy()

    max_error_fraction = noise_percent / 100.0
    noise_fraction = rng.uniform(-max_error_fraction, max_error_fraction, size=true_dists.shape)
    measured = true_dists * (1 + noise_fraction)

    # A measured distance can never be zero or negative.
    measured = np.clip(measured, 0.1, None)
    return measured
