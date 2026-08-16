"""
locator.py
----------
A small, secondary feature: given the ESTIMATED device position
(never the hidden ground truth), work out how far away a chosen
indoor destination is.

Destinations are loaded from a generic CSV file so INTRIX is not
tied to any specific building type -- swap the CSV and the same code
works for a school, hospital, office, warehouse, etc.
"""

import pandas as pd

from modules.trilateration import euclidean_distance


def load_locations(csv_path="data/locations.csv"):
    """Load the list of generic indoor destinations from CSV."""
    return pd.read_csv(csv_path)


def distance_to_destination(x_estimated, y_estimated, dest_x, dest_y):
    """
    Distance from the estimated device position to a selected
    destination, using the same plain distance formula used
    everywhere else in INTRIX.
    """
    return float(euclidean_distance(x_estimated, y_estimated, dest_x, dest_y))
