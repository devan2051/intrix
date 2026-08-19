import numpy as np

from modules.trilateration import euclidean_distance


def generate_ground_truth(x_range=(2, 23), y_range=(2, 16), seed=None):
    rng = np.random.default_rng(seed)
    x_true = rng.uniform(*x_range)
    y_true = rng.uniform(*y_range)
    return float(x_true), float(y_true)


def true_distances(x_true, y_true, beacon_x, beacon_y):
    beacon_x = np.asarray(beacon_x, dtype=float)
    beacon_y = np.asarray(beacon_y, dtype=float)
    return euclidean_distance(x_true, y_true, beacon_x, beacon_y)


def simulate_measurements(true_dists, noise_percent, seed=None):
    rng = np.random.default_rng(seed)
    true_dists = np.asarray(true_dists, dtype=float)

    if noise_percent <= 0:
        return true_dists.copy()

    max_error_fraction = noise_percent / 100.0
    noise_fraction = rng.uniform(-max_error_fraction, max_error_fraction, size=true_dists.shape)
    measured = true_dists * (1 + noise_fraction)

    measured = np.clip(measured, 0.1, None)
    return measured
