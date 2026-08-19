
import pandas as pd

from modules.trilateration import euclidean_distance


def load_locations(csv_path="data/locations.csv"):
    return pd.read_csv(csv_path, dtype={"floor": str})


def distance_to_destination(x_estimated, y_estimated, dest_x, dest_y):
    return float(euclidean_distance(x_estimated, y_estimated, dest_x, dest_y))


def nearest_location(x_estimated, y_estimated, locations_df):
    if locations_df is None or len(locations_df) == 0:
        return None, None

    distances = euclidean_distance(
        x_estimated, y_estimated, locations_df["x"].values, locations_df["y"].values
    )
    idx = int(distances.argmin())
    name = locations_df.iloc[idx]["name"]
    return name, float(distances[idx])
