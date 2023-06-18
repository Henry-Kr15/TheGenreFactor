#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import QuantileTransformer


def plot_columns(df):
    numeric_df = df.select_dtypes(include=np.number)  # select only numeric columns
    if "Unnamed: 0" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["Unnamed: 0"])
    for column in numeric_df.columns:
        print(column)
        plt.figure()
        if (
            column == "Views"
            or column == "Likes"
            or column == "Comments"
            or column == "Stream"
            or column == "Duration_ms"
        ):
            bins = np.logspace(0, np.log10(numeric_df[column].max()), 20)
            sns.histplot(numeric_df[column], bins=bins, kde=False)
            plt.xscale("log")
        elif column == "Instrumentalness":
            sns.histplot(numeric_df[column], bins=20, kde=True)
            plt.yscale("log")
        else:
            sns.histplot(numeric_df[column], bins=20, kde=True)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.savefig(f"../figures/Hist-{column}.pdf")
        plt.close()


def visualize_features(df: pd.DataFrame, Folder_name: str):

    plot_columns(df)

    print(df.head())
    print("---")
    print(df.describe())

    # Scatterplot
    print("Creating scatterplot...")
    numeric_df = df.select_dtypes(include=np.number)
    selected_columns = numeric_df.iloc[:, 1:16]
    sampled_data = selected_columns.sample(frac=1.0)
    scatter_matrix = pd.plotting.scatter_matrix(
        sampled_data, alpha=0.8, figsize=(10, 10), s=10, hist_kwds={"bins": 20}
    )
    for ax in scatter_matrix.ravel():
        ax.xaxis.label.set_rotation(90)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha("right")
    plt.savefig(f"../figures/{Folder_name}/scatter.png")

    # Correlation
    print("Creating correlation plot...")
    plt.figure(figsize=(20, 20))
    sns.heatmap(selected_columns.corr(), annot=True, square=True, cmap="coolwarm")
    plt.savefig(f"../figures/{Folder_name}/correlation.pdf")


# Beispielverwendung:
# IMMER IN EINEM ANDEREN SKRIPT IN DIESER DATEI NUR FUNKTIONSDEFINITONEN
# df = pd.read_csv("../data/Spotify_Youtube.csv")
# df = pd.read_csv("../data/data_selected_v1.csv")

# visualize_features(df, "Test1")
