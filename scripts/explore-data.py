#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_columns(df):
    numeric_df = df.select_dtypes(include=np.number)  # select only numeric columns
    if 'Unnamed: 0' in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=['Unnamed: 0'])
    for column in numeric_df.columns:
        plt.figure()
        sns.histplot(numeric_df[column], bins=20, kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.savefig(f"../figures/Hist-{column}.pdf")
        plt.clf()


df = pd.read_csv("../data/Spotify_Youtube.csv")

plot_columns(df)

print(df.head())
print("---")
print(df.describe())
