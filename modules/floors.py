
import pandas as pd

FLOOR_CODES = ["ground", "1", "2", "3"]

FLOOR_NAMES = {
    "ground": "Ground Floor",
    "1": "1st Floor",
    "2": "2nd Floor",
    "3": "3rd Floor",
}

FLOOR_ICON = "🏥"

FLOOR_WIDTH = 30
FLOOR_HEIGHT = 20
GROUND_TRUTH_X_RANGE = (2, FLOOR_WIDTH - 2)
GROUND_TRUTH_Y_RANGE = (2, FLOOR_HEIGHT - 2)


def floor_label(floor_code):
    return FLOOR_NAMES.get(floor_code, floor_code)


def normalize_floor_code(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    for code in FLOOR_CODES:
        if value == code.lower():
            return code
    return None


def load_all_beacons(csv_path="data/beacons.csv"):
    return pd.read_csv(csv_path, dtype={"floor": str})


def load_all_locations(csv_path="data/locations.csv"):
    return pd.read_csv(csv_path, dtype={"floor": str})


def beacons_for_floor(floor_code, csv_path="data/beacons.csv"):
    df = load_all_beacons(csv_path)
    return df[df["floor"] == floor_code].reset_index(drop=True)


def locations_for_floor(floor_code, csv_path="data/locations.csv"):
    df = load_all_locations(csv_path)
    return df[df["floor"] == floor_code].reset_index(drop=True)
