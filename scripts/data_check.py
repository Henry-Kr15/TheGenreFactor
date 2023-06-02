import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_file = "../data/data.csv"
df = pd.read_csv(csv_file)

df_lim = df.head(500)

# Gesamtanzahl der Einträge
total_count = df_lim.shape[0]

# Zähle die Anzahl der Einträge mit Genre "unknown" und "error"
unknown_count = df_lim[df_lim["Genre"] == "unknown"].shape[0]
error_count = df_lim[df_lim["Genre"] == "error"].shape[0]
nan_count = df_lim['Genre'].isna().sum()
other_count = df_lim[~df_lim["Genre"].isin(["unknown", "error"])].shape[0]

# Anzahl der verschiedenen Genres
genre_counts = df_lim['Genre'].value_counts()

accuracy = (total_count-unknown_count-error_count-nan_count)/total_count

# Gib die Anzahl aus
print(f"Gesamtanzahl der Einträge: {total_count}")
print(f"Anzahl an verschiedenen Genres: {genre_counts}")
print(f"Anzahl Einträge mit Genre 'unknown': {unknown_count}")
print(f"Anzahl Einträge mit Genre 'error': {error_count}")
print(f"Anzahl Einträge mit Genre 'NaN': {nan_count}")
print(f"Anzahl Einträge mit anderen Genres: {other_count}")
print(f"Accuracy = {accuracy:.5}")

# Plot
unknown_counts = df["Genre"].eq("unknown").cumsum()
plt.plot(df.index, unknown_counts)
plt.xlim(0,400)
plt.xlabel("Index")
plt.ylabel("Kumulative Anzahl von 'unknown'")
plt.title("Kumulative Anzahl von 'unknown' nach Index")
plt.show()