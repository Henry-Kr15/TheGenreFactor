#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas.plotting import scatter_matrix


def plot_columns(df):
    numeric_df = df.select_dtypes(include=np.number)  # select only numeric columns
    if 'Unnamed: 0' in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=['Unnamed: 0'])
    for column in numeric_df.columns:
        print(column)
        plt.figure()
        if column == 'Views' or column == 'Likes' or column == 'Comments' or column == 'Stream' or column == "Duration_ms":
            bins = np.logspace(0, np.log10(numeric_df[column].max()), 20)
            sns.histplot(numeric_df[column], bins=bins, kde=False)
            plt.xscale('log')
        elif column == 'Instrumentalness':
            sns.histplot(numeric_df[column], bins=20, kde=True)
            plt.yscale('log')
        else:
            sns.histplot(numeric_df[column], bins=20, kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.savefig(f"../figures/Hist-{column}.pdf")
        plt.close()


df = pd.read_csv("../data/Spotify_Youtube.csv")

plot_columns(df)

print(df.head())
print("---")
print(df.describe())

# Scatterplot
numeric_df = df.select_dtypes(include=np.number) 
selected_columns = numeric_df.iloc[:, 1:16]  
scatter_matrix(selected_columns, alpha=0.8, figsize=(40, 40), s=20)
plt.savefig("../figures/scatter.pdf")

# Correlation
plt.figure(figsize=(20,20))
sns.heatmap(selected_columns.corr(), annot=True, square=True, cmap='coolwarm')
plt.savefig("../figures/correlation.pdf")