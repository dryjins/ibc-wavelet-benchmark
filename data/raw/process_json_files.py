#!/usr/bin/env python

#
# This file extracts the measurements from the .json files,
# filters out the bad measurements where noise is comparable with the signal,
# and saves the results in a .csv file which can be more easily processed.
#
# Author: Atis Elsts
#

import json
import os
import math
import glob
import numpy as np
import pandas as pd

from common import *

# set to 0.12 to match the latest value used by Juris
FILTER_THRESHOLD = 0.12

GENERATOR_V = 2.0

# also save in MATLAB format?
SAVE_MATLAB = False

# if True then ignore experiments where only the 1 MOhm measurements are done
INCLUDE_ONLY_WITH_50_OHM_MEASUREMENTS = False

POINTS = [
    ("LE1",0,"Left Ear 1",0.33),
    ("RE1",1,"Right Ear 1",0.33),
    ("LE2",2,"Left Ear 2",0.33),
    ("RE2",3,"Right Ear 2",0.33),
    ("F",4,"Forehead",0.33),
    ("NOSE",5,"Nose",1.27),
    ("NECK",6,"Neck",1.5),
    ("LSB",7,"Left Shoulder Blade",3.7),
    ("RSB",8,"Right Shoulder Blade",3.7),
    ("LHD",9,"Left Hand Delta",3),
    ("RHD",10,"Right Hand Delta",3),
    ("LHF",11,"Left Hand Forearm",0.33),
    ("LHP",12,"Left Hand Palm",0.33),
    ("RHP",13,"Right Hand Palm",0.33),
    ("LHW1",14,"Left Hand Wrist 1",0.33),
    ("RHW1",15,"Right Hand Wrist 1",0.33),
    ("LHW2",16,"Left Hand Wrist 2",0.33),
    ("RHW2",17,"Right Hand Wrist 2",0.33),
    ("LHE",18,"Left Hand Elbow 2",0.33),
    ("LHT",19,"Left Hand Triceps 2",3),
    ("CHEST",20,"Chest/Stomach",4.69),
    ("LOIN",21,"Loin",3.76),
    ("LLQ",22,"Left Leg Quadricep",5.72),
    ("LLB",23,"Left Leg Bicep",5.72),
    ("LLK",24,"Left Leg Knee",0.33),
    ("LLC",25,"Left Leg Calve",2.8),
    ("LLAT",26,"Left Leg Achilles Tendon",0.39),
    ("RLAT",27,"Right Leg Achilles Tendon",0.39),
    ("LLAJ",28,"Left Leg Achilles Joint",0.39),
    ("LLAJ",28,"Left Leg Achilles Joint",0.39),
    ("RHT",29,"Right Hand Tricep",3),
    ("RHE",30,"Right Hand Elbow",0.33),
]

N = len(POINTS)

REDUCED_POINTS = {
    'F': 0,
    'NECK': 1,
    'LSB': 2,
    'RSB': 3,
    'LHD': 4,
    'RHD': 5,
    'LHW1': 6,
    'RHW1': 7,
    'LHW2': 8,
    'RHW2': 9,
    'LHE': 10,
    'LHT': 11,
    'CHEST': 12,
    'LOIN': 13,
    'LLQ': 14,
    'LLB': 15,
    'LLK': 16,
    'LLC': 17,
    'LLAT': 18,
    'RLAT': 19,
    'LLAJ': 20,
    'RHT': 21,
    'RHE': 22,
}


# XXX: these RHT and RHE are not 100% correct, just filled in the actually used points as a quick workaround
REDUCED_POINT_DISTANCES = [
    [0.00,3.00,5.20,5.20,3.93,3.93,9.78,8.93,9.33,9.33,  6.93,5.63,  5.13,7.35,11.03,12.85,13.73,15.93,18.83,17.93,18.03,  5.63,6.93],
    [3.00,0.00,2.20,2.20,3.00,3.00,8.85,8.00,8.40,8.40,  6.00,4.70,  5.70,4.35,11.60,9.85,12.35,14.55,17.45,16.25,16.65,   4.70,6.00],
    [5.20,2.20,0.00,2.65,1.40,4.05,7.25,9.05,6.80,9.45,  4.40,3.10,  4.10,3.00,9.70,8.50,11.00,13.20,16.10,14.90,15.30,    3.10,4.40],
    [5.20,2.20,2.65,0.00,4.05,1.40,9.90,6.40,9.45,6.80,  7.05,5.75,  4.10,3.00,9.70,8.50,11.00,13.20,16.10,14.90,15.30,    5.75,7.05],
    [3.93,3.00,1.40,4.05,0.00,5.45,5.85,10.45,5.40,10.85,  3.00,1.70,  2.70,4.40,8.60,9.90,11.30,13.50,16.40,15.50,15.60,  1.70,3.00],

    [3.93,3.00,4.05,1.40,5.45,0.00,11.30,5.00,10.85,5.40,   8.45,7.15,  2.70,4.40,8.60,9.90,11.30,13.50,16.40,15.50,15.60,      7.15,8.45],
    [9.78,8.85,7.25,9.90,5.85,11.30,0.00,16.30,0.40,16.70,  2.85,4.15,  8.55,10.25,14.45,15.75,17.15,19.35,22.25,21.35,21.45,   4.15,2.85],
    [8.93,8.00,9.05,6.40,10.45,5.00,16.30,0.00,15.85,0.40,  13.45,12.15,  7.70,9.40,13.60,14.90,16.30,18.50,21.40,20.50,20.60,  12.15,13.45],
    [9.33,8.40,6.80,9.45,5.40,10.85,0.40,15.85,0.00,16.25,  2.85,4.15,  8.10,9.80,14.00,15.30,16.70,18.90,21.80,20.90,21.00,    4.15,2.85],
    [9.33,8.40,9.45,6.80,10.85,5.40,16.70,0.40,16.25,0.00,  13.85,12.55,  8.10,9.80,14.00,15.30,16.70,18.90,21.80,20.90,21.00,  12.55,13.85],
    
    [6.93,6.00,4.40,7.05,3.00,8.45,2.85,13.45,2.85,13.85,  0.00,1.30,  5.70,7.40,11.60,12.90,14.30,16.50,19.40,18.50,18.60, -1,-1], # LHT
    [5.63,4.70,3.10,5.75,1.70,7.15,4.15,12.15,4.15,12.55,  1.30,0.00,  4.40,6.10,10.30,11.60,13.00,15.20,18.10,17.20,17.30, -1,-1], # LHE

    [5.13,5.70,4.10,4.10,2.70,2.70,8.55,7.70,8.10,8.10,   5.70,4.40,  0.00,3.20,5.90,6.60,8.60,10.80,13.70,12.80,12.90,     4.40,5.70],
    [7.35,4.35,3.00,3.00,4.40,4.40,10.25,9.40,9.80,9.80,  7.40,6.10,  3.20,0.00,6.70,5.50,8.00,10.20,13.10,11.90,12.30,     6.10,7.40],
    [11.03,11.60,9.70,9.70,8.60,8.60,14.45,13.60,14.00,14.00,  11.60,10.30,  5.90,6.70,0.00,3.90,2.70,4.90,7.80,15.90,7.00, 10.30,11.60],
    [12.85,9.85,8.50,8.50,9.90,9.90,15.75,14.90,15.30,15.30,  12.90,11.60,  6.60,5.50,3.90,0.00,2.50,4.70,7.60,12.00,6.80,  11.60,12.90],
    [13.73,12.35,11.00,11.00,11.30,11.30,17.15,16.30,16.70,16.70,  14.30,13.00,  8.60,8.00,2.70,2.50,0.00,2.20,5.10,14.50,4.30,        13.00,14.30],
    [15.93,14.55,13.20,13.20,13.50,13.50,19.35,18.50,18.90,18.90,  16.50,15.20,  10.80,10.20,4.90,4.70,2.20,0.00,2.90,16.70,3.00,      15.20,16.50],
    [18.83,17.45,16.10,16.10,16.40,16.40,22.25,21.40,21.80,21.80,  19.40,18.10,  13.70,13.10,7.80,7.60,5.10,2.90,0.00,19.60,1.00,      18.10,19.40],
    [17.93,16.25,14.90,14.90,15.50,15.50,21.35,20.50,20.90,20.90,  18.50,17.20,  12.80,11.90,15.90,12.00,14.50,16.70,19.60,0.00,18.80, 17.20,18.50],
    [18.03,16.65,15.30,15.30,15.60,15.60,21.45,20.60,21.00,21.00,  18.60,17.30,  12.90,12.30,7.00,6.80,4.30,3.00,1.00,18.80,0.00,      17.30,18.60],

    [5.63,4.70,3.10,5.75,1.70,7.15,4.15,12.15,4.15,12.55,1.30,0.00,4.40,6.10,10.30,11.60,13.00,15.20,18.10,17.20,17.30, -1, -1], # RHE (copied from LHE)
    [6.93,6.00,4.40,7.05,3.00,8.45,2.85,13.45,2.85,13.85,0.00,1.30,5.70,7.40,11.60,12.90,14.30,16.50,19.40,18.50,18.60, -1, -1], # RHT (copied from LHT)
]

# construct the full distance matrix; put Infinity between points where the distance is not known
FULL_POINT_DISTANCES = [[float("inf") for _ in range(N+1)] for _ in range(N+1)]
for name1, index1, _ ,_ in POINTS:
    if name1 not in REDUCED_POINTS:
        continue
    reduced_index1 = REDUCED_POINTS[name1]
    for name2, index2, _ ,_ in POINTS:
        if name2 not in REDUCED_POINTS:
            continue
        reduced_index2 = REDUCED_POINTS[name2]
        FULL_POINT_DISTANCES[index1][index2] = REDUCED_POINT_DISTANCES[reduced_index1][reduced_index2]


FEATURES = ["subject_id",
            "experiment_id",
            "height",
            "weight",
            "BMI",
            "body_fat_%",
            "age_group",
            "male",
            "tx_point",
            "rx_point",
            "distance",
            "tx_point_fat_level",
            "rx_point_fat_level",
            "total_fat_level",
            "bias"]

HEADER_ROW = list(FEATURES)

for f in FREQUENCIES:
    HEADER_ROW.append(f"tx_abs_Z_{f}")

for f in FREQUENCIES:
    HEADER_ROW.append(f"rx_gain_50_f_{f}")

for f in FREQUENCIES:
    HEADER_ROW.append(f"rx_gain_1M_f_{f}")

# ===================================================

# generator's resistance, ohm
Rg = 50
# oscilloscope's resistance, ohm (on the Tx side, not on the Rx side, where it's either 50 ohm or 1 Mohm)
Ro = 50

def parallel(Ro, Z):
    return Ro * Z / (Ro + Z)

def calc_v_tx(Z):
    p = parallel(Ro, Z)
    return Vg * abs(p / (Rg + p))

def calc_input_abs_z(V, Vg):
    epsilon = 0.01
    assert (Vg / 4 < V < Vg / (2.0 - epsilon))

    R = V * Ro / (Vg - V)
    Z = R * Ro / (Ro - R)
    return Z

# ===================================================

def load_files():
    files = glob.glob(DATA_DIR + "/*/*1M.json") \
        + glob.glob(DATA_DIR + "/*/*1M_50.json")
    files.sort()
    measurements = []
    subjects = []
    iterations = []
    cumulative = 0
    files_per_subject = {}
    for filename in files:
        print(filename)
        with open(filename) as f:
            obj = json.load(f)

        cumulative += len(obj)
        print("num entries", len(obj), "total", cumulative)
        prev = None
        num_used = 0
        for i in range(len(obj)):

            # filter out the metadata in the last entry
            if "TransmitterPoint" not in obj[i]:
                continue

            if prev == None:
                prev = obj[i]
                continue

            # filter out any points without a Rx/Tx record pair
            current = obj[i]
            if current["TransmitterPoint"] == prev["TransmitterPoint"] \
               and current["ReceiverPoint"] == prev["ReceiverPoint"]:
                measurements.append(prev)
                measurements.append(current)
                num_used += 2
            else:
                print("measurement mismatch:")
                print("prev:   ", prev["TransmitterPoint"], prev["ReceiverPoint"])
                print("current:", current["TransmitterPoint"], current["ReceiverPoint"])
            prev = None

        fields = os.path.basename(filename).split("_")
        #print(fields)
        if fields[0][:2] == "ID":
            fields[0] = fields[0][2:]
        subject = int(fields[0])
        files_per_subject[subject] = files_per_subject.get(subject, 0) + 1
        iteration = files_per_subject[subject]
        subjects += [subject] * num_used
        iterations += [iteration] * num_used
    return measurements, subjects, iterations


# filter out bad measurements; for good ones, average the measurement for each frequency
def average_measurement(data, filter_threshold, is_tx):
    if not hasattr(data[0], '__len__'):
        # only single measurement per frequency, nothing to average!

        if is_tx:
            # assuming Z_tx equal to +inf, the R_g functions as 2x voltage divider on the Tx side
            #  -> the voltage should be below (Vg/2 + epsilon) volts
            # assuming Z_tx equal to 25 ohm (way below the expected average value),
            # the R_g functions as 4x voltage divider
            # -> the voltage should be above Vg/4 volts
            epsilon = 0.05
            minimum = GENERATOR_V / 4
            maximum = GENERATOR_V / (2.0 - epsilon)
            for v in data:
                if not (minimum < v < maximum):
                    print(f"Bad tx side voltage {v:.6f}, ignoring")
                    return False, data

        return True, data

    result = []
    for frequency_data in data:
        mean = np.mean(frequency_data)
        scaled_data = [u / mean for u in frequency_data]
        std = np.std(scaled_data)
        if std > filter_threshold:
            return False, None
        s = sorted(frequency_data)[1:-1]
        value = np.mean(s)
        result.append(value)

    return True, result


def main():
    measurements, subjects, iterations = load_files()
    n = len(measurements)

    # read subjects and put them in a dataframe
    subjects_info = pd.read_csv("summary_of_subjects.csv", delimiter=",")
    print(subjects_info)

    rows = []
    rows_by_freq = []
    subjects_included = set()

    for i in range(0, n, 2):

        # validate that there's even number of measurements for each subject
        assert subjects[i] == subjects[i+1]

        if i + 1 >= len(measurements):
            break

        # validate that input and output are in order
        m1 = measurements[i]
        m2 = measurements[i+1]
        if "TransmitterPoint" not in m1 or "TransmitterPoint" not in m2:
            print("m1=", m1)
            print("m2=", m2)
            continue

        if m1["TransmitterPoint"] != m2["TransmitterPoint"]:
            print("bug in measurement order at", i)
            print(m1)
            print(m2)
            print("subjects:", subjects[i], subjects[i+1])
            continue

        if m1["DeviceType"] == m2["DeviceType"]:
            print("bug: same device type at", i)
            continue

        # reformat the data and remove "broken" experiments
        tx_point = measurements[i]["TransmitterPoint"]
        rx_point = measurements[i]["ReceiverPoint"]

        if tx_point == -1:
            # this is a noise level measurement, ignore
            continue

        if tx_point < 0:
            print("unknown tx point", tx_point)
            continue
        if rx_point < 0:
            print("unknown rx point", rx_point)
            continue

        if "Measurements50" in measurements[i]:
            if measurements[i]["Measurements50"][0] == 0.0:
                tx_msrmt = measurements[i]
                rx_msrmt = measurements[i+1]
            else:
                tx_msrmt = measurements[i+1]
                rx_msrmt = measurements[i]

            ok, v_tx = average_measurement(tx_msrmt["Measurements1M"], FILTER_THRESHOLD, True)
            if not ok:
                continue
            ok, v_rx_50 = average_measurement(rx_msrmt["Measurements50"], float("inf"), False)
            if not ok:
                continue
            ok, v_rx_1M = average_measurement(rx_msrmt["Measurements1M"], FILTER_THRESHOLD, False)
            if not ok:
                continue

            path_loss_50 = [20 * math.log(r / t, 10) for r, t in zip(v_rx_50, v_tx)]
            path_loss_1M = [20 * math.log(r / t, 10) for r, t in zip(v_rx_1M, v_tx)]

        else:
            if INCLUDE_ONLY_WITH_50_OHM_MEASUREMENTS:
                # 1 Mohm only, skip this experiment
                continue

            first = measurements[i]["Measurements"][0]
            if hasattr(first, '__len__'):
                first = first[0]

            if first > 0.2:
                tx_msrmt = measurements[i]
                rx_msrmt = measurements[i+1]
            else:
                tx_msrmt = measurements[i+1]
                rx_msrmt = measurements[i]

            ok, v_tx = average_measurement(tx_msrmt["Measurements"], FILTER_THRESHOLD, True)
            if not ok:
                continue
            ok, v_rx = average_measurement(rx_msrmt["Measurements"], FILTER_THRESHOLD, False)
            if not ok:
                continue

            path_loss_50 = [0.0 for _ in v_tx]
            path_loss_1M = [20 * math.log(r / t, 10) for r, t in zip(v_rx, v_tx)]

        #print(i)
        tx_z = [calc_input_abs_z(t, GENERATOR_V) for t in v_tx]

        if min(path_loss_1M) < -70:
            print("suspiciously BIG min path loss at", i, min(path_loss_1M))
            print("path loss:", path_loss_1M)
        if max(path_loss_1M) > -19:
            print("suspiciously SMALL max path loss at", i, max(path_loss_1M))
            print("path loss:", path_loss_1M)

        tx_point_name = POINTS[tx_point][2]
        rx_point_name = POINTS[rx_point][2]

        print(tx_point_name, tx_point, " -> ", rx_point_name, rx_point)

        distance = FULL_POINT_DISTANCES[tx_point][rx_point]
        assert distance >= 0.0
        assert distance < float("inf")

        rx_point_fat = POINTS[rx_point][3]
        tx_point_fat = POINTS[tx_point][3]
        total_fat = rx_point_fat + tx_point_fat

        subject_info = subjects_info.loc[subjects_info["ID"] == subjects[i]]
        weight = subject_info["Weight"].to_list()[0]
        height = subject_info["Height"].to_list()[0]
        age_group = subject_info["Age group"].to_list()[0]
        BMI = subject_info["BMI"].to_list()[0]
        fat = subject_info["Body fat %"].to_list()[0]
        male = subject_info["Male"].to_list()[0]
        bias = 1.0 # add a constant bias term
        row = [subjects[i], iterations[i], height, weight, BMI, fat, age_group, male,
               tx_point, rx_point, distance, tx_point_fat, rx_point_fat, total_fat, bias] \
               + tx_z \
               + path_loss_50 \
               + path_loss_1M
        rows.append(row)

        for j, f in enumerate(FREQUENCIES):
            f_data_50 = path_loss_50[j]
            f_data_1M = path_loss_1M[j]
            row = [subjects[i], iterations[i], height, weight, BMI, fat, age_group, male,
                   tx_point, rx_point, distance, tx_point_fat, rx_point_fat, total_fat,
                   bias, f, f_data_50, f_data_1M]
            rows_by_freq.append(row)

        subjects_included.add(subjects[i])

    total_subjects = set(subjects)
    print(f"in total measured {len(total_subjects)} subjects, included {len(subjects_included)} subjects")

    outfilename = "all_measurements.csv"
    with open(outfilename, "w") as f:
        f.write(",".join(HEADER_ROW) + "\n")
        for row in rows:
            f.write(",".join([str(u) for u in row]) + "\n")

    outfilename = "all_measurements_by_freq.csv"
    with open(outfilename, "w") as f:
        f.write(",".join(FEATURES + ["frequency", "rx_gain_50", "rx_gain_1M"]) + "\n")
        for row in rows_by_freq:
            f.write(",".join([str(u) for u in row]) + "\n")

    if SAVE_MATLAB:
        from scipy.io import savemat
        # also save in MATLAB format
        columns = {name: [] for name in HEADER_ROW}
        for i in range(len(HEADER_ROW)):
            c = [x[i] for x in rows]
            columns[HEADER_ROW[i]] = c

        savemat("all_measurements.mat", columns)


if __name__ == "__main__":
    main()
    print("all done")
