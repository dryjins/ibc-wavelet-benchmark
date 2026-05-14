#!/usr/bin/env python

#
# This files filters out outlier measurements from the .csv file.
# A measurement is considered an outlier if its Z-score is above 2.0.
#
# Author: Atis Elsts
#

import os
from scipy import stats
import numpy as np
import pandas as pd

from common import *

FEATURES = "subject_id,experiment_id,height,weight,BMI,body_fat_%,age_group,male,tx_point,rx_point,distance,tx_point_fat_level,rx_point_fat_level,total_fat_level,bias".split(",")

THRESHOLD_Z_SCORE = 2

def main():
    filename = os.path.join(SELF_DIR, "all_measurements.csv")
    df = pd.read_csv(filename, delimiter=",")

    for f in FREQUENCIES:
        column = f"rx_gain_1M_f_{f}"
        z_score = (df[column] - df[column].mean()) / df[column].std(ddof=0)
        df[f'{f}_z_score'.format(f)] = abs(z_score)

    df['z_score'] = df[[f'{f}_z_score'.format(f) for f in FREQUENCIES]].max(axis=1)

    print("before filtering:", df.shape)
    filtered_df = df[df["z_score"] < THRESHOLD_Z_SCORE]
    print("after filtering: ", filtered_df.shape)

    # don't need to save all the z scores
    for f in FREQUENCIES:
        df = df.drop(f'{f}_z_score'.format(f), axis=1)
    df = df.drop('z_score', axis=1)
    
    filtered_df.to_csv("all_measurements_filtered.csv")

    rows_by_freq = []
    for index, row in filtered_df.iterrows():
        lst = [row[f] for f in FEATURES]
        for f in FREQUENCIES:
            column = f"rx_gain_1M_f_{f}"
            f_data_50 = float(row[f"rx_gain_50_f_{f}"])
            f_data_1M = float(row[f"rx_gain_1M_f_{f}"])
            rows_by_freq.append(lst + [f, f_data_50, f_data_1M])

    outfilename = "all_measurements_by_freq_filtered.csv"
    with open(outfilename, "w") as f:
        f.write(",".join(FEATURES + ["frequency", "rx_gain_50", "rx_gain_1M"]) + "\n")
        for row in rows_by_freq:
            f.write(",".join([str(u) for u in row]) + "\n")


if __name__ == "__main__":
    main()
    print("all done")

