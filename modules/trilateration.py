"""
trilateration.py
-----------------
The core mathematics of INTRIX.

This file contains exactly three things:

1. euclidean_distance() -> the plain distance formula
2. trilaterate()        -> estimates an unknown (x, y) position using
                            ONLY known beacon coordinates and measured
                            (simulated Bluetooth) distances
3. position_error()     -> compares the estimate with a hidden
                            ground-truth position (used only to grade
                            accuracy, never used to calculate the estimate)

IMPORTANT CONCEPTUAL RULE
--------------------------
trilaterate() never receives the true/hidden device position. It only
ever receives:
    - beacon coordinates (known, fixed)
    - measured distances (simulated Bluetooth measurements)

This keeps the demonstration honest: the algorithm has to "figure out"
the position the same way a real system would, from distances alone.
"""

import numpy as np


def euclidean_distance(x1, y1, x2, y2):
    """
    The basic Class 12 distance formula between two points:

        d = sqrt((x2 - x1)^2 + (y2 - y1)^2)

    Works with plain numbers or with NumPy arrays.
    """
    return np.sqrt((np.asarray(x2) - np.asarray(x1)) ** 2 +
                    (np.asarray(y2) - np.asarray(y1)) ** 2)


def trilaterate(beacon_x, beacon_y, measured_distances):
    """
    Estimate an unknown 2D position from 3 or more known beacon
    positions and their measured distances to that unknown position.

    METHOD: linearization + least squares.

    Each beacon i at (xi, yi) with measured distance di places the
    device somewhere on the circle:

        (x - xi)^2 + (y - yi)^2 = di^2

    Expanding this equation gives x^2 and y^2 terms, which makes the
    system non-linear. But if we SUBTRACT the equation for beacon 1
    from the equation for every other beacon, the x^2 and y^2 terms
    cancel out, leaving a LINEAR equation in (x, y):

        2(xi - x1)*x + 2(yi - y1)*y
            = (d1^2 - di^2) + (xi^2 - x1^2) + (yi^2 - y1^2)

    With 3+ beacons this becomes several linear equations in two
    unknowns (an over-determined system). numpy.linalg.lstsq finds
    the (x, y) that best satisfies all of them at once -- i.e. the
    point that best fits every circle simultaneously, even if the
    circles don't meet at one exact point due to measurement noise.

    Parameters
    ----------
    beacon_x, beacon_y : array-like
        Known, fixed beacon coordinates.
    measured_distances : array-like
        Simulated Bluetooth distance measurements, in the same
        beacon order as beacon_x / beacon_y.

    Returns
    -------
    (x_est, y_est) : tuple of float
        The estimated device position.
    """
    bx = np.asarray(beacon_x, dtype=float)
    by = np.asarray(beacon_y, dtype=float)
    d = np.asarray(measured_distances, dtype=float)

    if len(bx) < 3:
        raise ValueError("Trilateration needs at least 3 beacons.")
    if not (len(bx) == len(by) == len(d)):
        raise ValueError("beacon_x, beacon_y and measured_distances must be the same length.")

    # Use beacon 1 as the reference beacon that we subtract from the rest.
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
    """
    Straight-line error between the hidden ground-truth position and
    the estimated position.

        Error = sqrt((Xactual - Xestimated)^2 + (Yactual - Yestimated)^2)

    This is used ONLY for evaluation, strictly after trilaterate()
    has already produced its estimate.
    """
    return float(euclidean_distance(x_actual, y_actual, x_estimated, y_estimated))
