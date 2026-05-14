#!/usr/bin/env python

#
# This file visualizes and analyzes some information about the test subjects.
#
# Author: Atis Elsts
#

import os
from matplotlib import pyplot as pl
import pandas as pd
import seaborn as sns

from common import *


def visualize(df):
    try:
        os.mkdir("measurement_group_plots")
    except:
        pass

    for column in df:
        pl.figure(figsize=(5, 3))

        if column == "Male":
            # plot a barblot instead
            df["Number of subjects"] = 15
            sns.barplot(data=df, x="Sex", y="Number of subjects", palette=["#ff88b0", "#8888ff"])
            pl.savefig(os.path.join("measurement_group_plots", f"barplot_sex_distr.pdf"),
                       format="pdf", bbox_inches="tight")
            pl.close()
            del df["Number of subjects"]
            continue

        if column in ["ID", "Sex"]:
            continue

        print(f"{column} {df[column].mean():.2f} ({df[column].std():.2f})")
        sns.kdeplot(data=df, x=column, hue="Sex", fill=True, palette=["#ff0033", "#0000ff"])
        pl.savefig(os.path.join("measurement_group_plots", f"kde_{column}.pdf"),
                    format="pdf", bbox_inches="tight")
        pl.close()


def analyze(df):
    print(df.info())


def main():
    filename = os.path.join(SELF_DIR, "summary_of_subjects.csv")
    df = pd.read_csv(filename, delimiter=",")
    df["Sex"] = df["Male"].map({0: "Female", 1: "Male"})
    
    print(df)

    visualize(df)
    analyze(df)

if __name__ == "__main__":
    main()
    print("done")

