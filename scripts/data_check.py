import pandas as pd
import matplotlib.pyplot as plt

csv_file = "../data/data.csv"
df = pd.read_csv(csv_file)

df_lim = df

print(df_lim.tail())

# Gesamtanzahl der Einträge
total_count = df_lim.shape[0]

# Zähle die Anzahl der Einträge mit Genre "unknown" und "error"
notfound_count = df_lim[df_lim["Genre"] == "Not Found"].shape[0]
error_count = df_lim[df_lim["Genre"] == "Error"].shape[0]
nan_count = df_lim['Genre'].isna().sum()
other_count = df_lim[~df_lim["Genre"].isin(["Not Found", "Error"])].shape[0]

# Anzahl der verschiedenen Genres
genre_counts = df_lim['Genre'].nunique()
genre_value = df_lim['Genre'].value_counts()
print(genre_value)


accuracy = (total_count-notfound_count-error_count-nan_count)/total_count

# Gib die Anzahl aus
print(f"\nGesamtanzahl der Einträge: {total_count}")
print(f"Anzahl an verschiedenen Genres: {genre_counts}")
print(f"Anzahl Einträge mit Genre 'Not Found': {notfound_count}")
print(f"Anzahl Einträge mit Genre 'error': {error_count}")
print(f"Anzahl Einträge mit Genre 'NaN': {nan_count}")
print(f"Anzahl Einträge mit anderen Genres: {other_count}")
print(f"Accuracy = {accuracy:.5}\n")

# Zähle die NaN-Werte pro Spalte
nan_counts = df.isna().sum()
print(f"Anzahl der NaN Werte pro Spalte: \n{nan_counts}")


# Plot der kumulativen Anzahl von nicht gefundenen Genres
unknown_counts = df_lim["Genre"].eq("Not Found").cumsum()
plt.plot(df_lim.index, unknown_counts)
plt.xlim(0,total_count)
plt.xlabel("Index")
plt.ylabel("Anzahl 'Not Found'")
plt.title("Kumulative Anzahl von 'Not Found' nach Index")
plt.show()

# Plot der kumulativen Anzahl von error Genres
error_counts = df_lim["Genre"].eq("Error").cumsum()
plt.plot(df_lim.index, error_counts)
plt.xlim(0,total_count)
plt.xlabel("Index")
plt.ylabel("Anzahl 'Error'")
plt.title("Kumulative Anzahl von 'Error' nach Index")
plt.show()

# Plot der kumulativen Anzahl von NaN-Werten
views_nan_counts = df["Views"].isna().cumsum()
plt.plot(df.index, views_nan_counts)
plt.xlim(0, len(df))
plt.xlabel("Index")
plt.ylabel("Kumulative Anzahl NaN")
plt.title("Kumulative Anzahl von NaN-Werten im Feature 'Views'")
plt.show()

# Plot der kumulativen Anzahl von NaN-Werten
views_nan_counts = df["Stream"].isna().cumsum()
plt.plot(df.index, views_nan_counts)
plt.xlim(0, len(df))
plt.xlabel("Index")
plt.ylabel("Kumulative Anzahl NaN")
plt.title("Kumulative Anzahl von NaN-Werten im Feature 'Stream'")
plt.show()