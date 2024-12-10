from p5_simulation.trees import Network, MeterType
import csv
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from math import sqrt


def network_from_file(file, default_meter_type=MeterType.NONE) -> Network:
    with open(file) as csv_file:
        raw = csv.reader(csv_file)
        next(raw)
        connections = [
            [
                int(row[0]),
                int(row[1]),
                default_meter_type,
                compute_impedence(float(row[4]), row[5]),
            ]
            for row in raw
        ]

        net = Network.from_connections(connections)
    return net


def compute_impedence(length, cable_type) -> complex:
    match cable_type:
        case "4x25CU":
            return length * (0.727 + 0.084j) / 1000
            # return length * (0.927 + 0.081j) / 1000
        case "4x50AL":
            return length * (0.641 + 0.081j) / 1000
            # return length * (0.822 + 0.075j) / 1000
        case "4x95AL":
            return length * (0.321 + 0.078j) / 1000
            # return length * (0.411 + 0.073j) / 1000
        case "4x150AL":
            return length * (0.207 + 0.078j) / 1000
            # return length * (0.265 + 0.072j) / 1000
    raise Exception("Invalid cabletype")


def measurements_from_file(file):
    raw = pd.read_excel(
        file,
        names=[
            "ID",
            "timestamp",
            "Voltage L1 - average",
            "Voltage L2 - average",
            "Voltage L3 - average",
            "Active Power P14 - L1 - mean",
            "Active Power P14 - L2 - mean",
            "Active Power P14 - L3 - mean",
            "Active Power P23 - L1 - mean",
            "Active Power P23 - L2 - mean",
            "Active Power P23 - L3 - mean",
            "Reactive Power Q12 - L1 - mean",
            "Reactive Power Q12 - L2 - mean",
            "Reactive Power Q12 - L3 - mean",
            "Reactive Power Q34 - L1 - mean",
            "Reactive Power Q34 - L2 - mean",
            "Reactive Power Q34 - L3 - mean",
            "Frequency - Mean",
        ],
    )
    raw["Active Power L1"] = (
        raw["Active Power P14 - L1 - mean"] - raw["Active Power P23 - L1 - mean"]
    )
    raw["Active Power L2"] = (
        raw["Active Power P14 - L2 - mean"] - raw["Active Power P23 - L2 - mean"]
    )
    raw["Active Power L3"] = (
        raw["Active Power P14 - L3 - mean"] - raw["Active Power P23 - L3 - mean"]
    )

    raw["Reactive Power L1"] = (
        raw["Reactive Power Q12 - L1 - mean"] - raw["Reactive Power Q34 - L1 - mean"]
    )
    raw["Reactive Power L2"] = (
        raw["Reactive Power Q12 - L2 - mean"] - raw["Reactive Power Q34 - L2 - mean"]
    )
    raw["Reactive Power L3"] = (
        raw["Reactive Power Q12 - L3 - mean"] - raw["Reactive Power Q34 - L3 - mean"]
    )

    raw["Voltage"] = raw[
        ["Voltage L1 - average", "Voltage L2 - average", "Voltage L3 - average"]
    ].mean(axis=1) * sqrt(2)
    raw["Active Power"] = raw[
        ["Active Power L1", "Active Power L2", "Active Power L3"]
    ].mean(axis=1) * sqrt(2)
    raw["Reactive Power"] = raw[
        ["Reactive Power L1", "Reactive Power L2", "Reactive Power L3"]
    ].mean(axis=1) * sqrt(2)

    raw["Current"] = (raw["Active Power"] + raw["Reactive Power"] * 1j) / raw["Voltage"]

    selected_data = raw[["ID", "timestamp", "Voltage", "Current"]]
    for time, data in selected_data.groupby(["timestamp"]):
        yield time, data[["ID", "Voltage", "Current"]].set_index("ID")


if __name__ == "__main__":
    # network_from_file("./data/topology.txt")
    for time, data in measurements_from_file("./data/measurements.xlsx"):
        print(data.loc[[7, 49]].to_numpy())
