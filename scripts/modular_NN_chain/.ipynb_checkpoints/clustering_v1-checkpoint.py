#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Erstmal nur Pop betrachten
df = pd.read_csv("../../data/data_selected_v1.csv", index_col=0)
df = df.loc[df['Genre'] == 'pop']

# print(df.describe())

# sns.pairplot(df["Danceability", "Energy", "Key", "Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo"])
sns.pairplot(df)
plt.savefig("../../figures/Pop_clustering_pairplot.pdf")
