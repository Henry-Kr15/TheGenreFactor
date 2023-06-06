import pandas as pd

# Einlesen der CSV-Datei
df = pd.read_csv("../data/data.csv")

# Selektion der Zeilen ohne NaN-Werte
df_selected = df.dropna().loc[~df["Genre"].isin(["Not Found", "Error"])]

# Filtern der Daten und Behalten nur derjenigen mit Häufigkeit > n
n = 5
genre_counts = df_selected['Genre'].value_counts()
df_selected = df_selected[df_selected['Genre'].isin(genre_counts[genre_counts > n].index)]

# Abspeichern des resultierenden DataFrames als CSV
df_selected.to_csv("../data/data_selected.csv", index=False)

genre_value = df_selected['Genre'].value_counts()
print(genre_value)
print(f"Übriges Datenset hat {df_selected.shape[0]} Einträge")
