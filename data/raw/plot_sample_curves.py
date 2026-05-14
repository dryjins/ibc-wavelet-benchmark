#!/usr/bin/env python

#
# This file visualizes randomly picked subset of the dataset.
#
# Author: Atis Elsts
#

import os
from matplotlib import pyplot as pl
import pandas as pd
import numpy as np

from common import *

# if true the correlations are computed on times, not decibels
TRANSFORM_LOG_TO_TIMES = False

def plot(df, resistance):
    all_freq = []
    for f in FREQUENCIES:
        column1 = f"rx_gain_{resistance}_f_{f}"
        if TRANSFORM_LOG_TO_TIMES:
            new_column1 = f"rx_gain_{resistance}_f_{f}_times"
            df[new_column1] = 10 ** (df[column1] / 20)
            column1 = new_column1
        all_freq.append(column1)

    pl.figure(figsize=(8, 5))
    pl.grid(False)

    for index, row in df.iterrows():
        if str(resistance) == "50":
            if TRANSFORM_LOG_TO_TIMES:
                if row["rx_gain_50_f_50000_times"] == 0.0:
                    continue
            else:
                if row["rx_gain_50_f_50000"] == 0.0:
                    continue

        data = row[all_freq]
        pl.plot(FREQUENCIES, data, marker="o", markersize=3)

    
    pl.xscale('log')
    pl.ylabel('Gain, dB')
    pl.xlabel('Frequency, Hz')
    #pl.ylim(-0.7, +0.7)
    
    figname = f"sample_{resistance}.pdf"
    pl.savefig(figname, format="pdf", bbox_inches="tight")
    pl.close()


def plot_both(df):
    all_freq_50 = [f"rx_gain_50_f_{f}" for f in FREQUENCIES]
    all_freq_1M = [f"rx_gain_1M_f_{f}" for f in FREQUENCIES]

    pl.figure(figsize=(8, 5))
    pl.grid(False)

    had_label = False
    for index, row in df.iterrows():
        if row["rx_gain_50_f_50000"] == 0.0:
            # 50 ohm load impedance values were not recorded
            continue

        if had_label:
            label50 = None
            label1M = None
        else:
            had_label = True
            label50 = "50 ohm load"
            label1M = "1M ohm load"

        pl.plot(FREQUENCIES, row[all_freq_50], marker="o", markersize=3, color="green", label=label50)
        pl.plot(FREQUENCIES, row[all_freq_1M], marker="o", markersize=3, color="darkblue", label=label1M)


    pl.legend()
    pl.xscale('log')
    pl.ylabel('Gain, dB')
    pl.xlabel('Frequency, Hz')
    figname = f"sample_both.pdf"
    pl.savefig(figname, format="pdf", bbox_inches="tight")
    pl.close()


def main():
    filename = os.path.join(SELF_DIR, "all_measurements_filtered.csv")
    df = pd.read_csv(filename, delimiter=",")

    sample = df.sample(n=50)
    plot(sample, "50")
    plot(sample, "1M")

    plot_both(sample)


if __name__ == "__main__":
    main()
    print("done")
