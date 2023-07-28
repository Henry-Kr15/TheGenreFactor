#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas.plotting import scatter_matrix


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


df = pd.read_csv("../data/Spotify_Youtube.csv")

plot_columns(df)

print(df.head())
print("---")
print(df.describe())
print(df.iloc[0])

# Scatterplot
print("create scatterplot")
numeric_df = df.select_dtypes(include=np.number)
selected_columns = numeric_df.iloc[:, 1:16]
sampled_data = selected_columns.sample(frac=1.0)
scatter_matrix = scatter_matrix(
    sampled_data, alpha=0.8, figsize=(10, 10), s=10, hist_kwds={"bins": 20}
)
for ax in scatter_matrix.ravel():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha("right")
plt.savefig("../figures/scatter.png")

# Correlation
print("create correlation plot")
plt.figure(figsize=(20, 20))
sns.heatmap(selected_columns.corr(), annot=True, square=True, cmap="coolwarm")
plt.savefig("../figures/correlation.pdf")

# Plots für die Präsentation
print("create performance plot")
plt.clf()
models = ["Random guessing", "Naive Guessing", "Knn", "SVMs", "Neural Network"]
accuracies = [16.67, 36.63, 60.62, 63.88, 65.87]

data = sorted(zip(accuracies, models))  # Daten sortieren
colors = sns.color_palette("Oranges", len(data))  # Farbpalette erstellen

# Diagramm erstellen
for i, (value, category) in enumerate(data):
    plt.bar(category, value, color=colors[i])

# Setze den Stil auf "darkgrid"
sns.set_style("darkgrid")

# Farbe festlegen
orange = sns.color_palette("Oranges", 10)[5]

# plt.bar(models, accuracies, color=orange)
plt.xlabel("Models", fontsize=25)
plt.ylabel("Achieved accuracy [%]", fontsize=25)
plt.title("Performance Comparison", fontsize=25)
# Ticks vergrößern
plt.tick_params(axis="both", which="major", labelsize=25)
# y-Achse Bereich festlegen
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("../figures/performance-comparison.pdf")
plt.clf()

df = pd.read_csv("../data/data_selected_v1.csv")
print(df["Genre"].unique())

# Genres für das Training auswählen
genres_to_keep = ["classic", "metal", "rock", "hip hop", "electronic", "pop"]
df = df[df["Genre"].isin(genres_to_keep)]

# Farben für den nächsten Plot
colors = sns.color_palette("Oranges", len(genres_to_keep))

# Anzahl der Einträge pro Genre
genre_counts = df['Genre'].value_counts()
print(genre_counts)

# Sortiere die Farben nach der Größe der Werte
sorted_colors = [color for _, color in sorted(zip(genre_counts.values, colors), reverse=True)]


# Histogramm erstellen
plt.figure(figsize=(10, 6))

for i, genre in enumerate(genre_counts.index):
    plt.bar(genre, genre_counts[genre], color=sorted_colors[-(i+1)])

plt.xlabel('Genre')
plt.ylabel('Count')
# plt.title('bar chart of class imbalance')
plt.tight_layout()
plt.savefig('../figures/genre_hist_orange.pdf')
