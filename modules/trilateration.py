import numpy as np


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((np.asarray(x2) - np.asarray(x1)) ** 2 +
                    (np.asarray(y2) - np.asarray(y1)) ** 2)


def trilaterate(beacon_x, beacon_y, measured_distances):
    bx = np.asarray(beacon_x, dtype=float)
    by = np.asarray(beacon_y, dtype=float)
    d = np.asarray(measured_distances, dtype=float)

    if len(bx) < 3:
        raise ValueError("Trilateration needs at least 3 beacons.")
    if not (len(bx) == len(by) == len(d)):
        raise ValueError("beacon_x, beacon_y and measured_distances must be the same length.")

    x1, y1, d1 = bx[0], by[0], d[0]

    rows_A = []
    rows_b = []
    for i in range(1, len(bx)):
        xi, yi, di = bx[i], by[i], d[i]
        rows_A.append([2 * (xi - x1), 2 * (yi - y1)])
        rows_b.append((d1 ** 2 - di ** 2) + (xi ** 2 - x1 ** 2) + (yi ** 2 - y1 ** 2))

    A = np.array(rows_A)
    b = np.array(rows_b)

    solution, _residuals, _rank, _sv = np.linalg.lstsq(A, b, rcond=None)
    x_est, y_est = solution
    return float(x_est), float(y_est)


def position_error(x_actual, y_actual, x_estimated, y_estimated):
    return float(euclidean_distance(x_actual, y_actual, x_estimated, y_estimated))
