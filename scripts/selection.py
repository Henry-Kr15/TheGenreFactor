import pandas as pd
import matplotlib.pyplot as plt

def group_genres(genre):
    if 'pop' in genre.lower():
        return 'pop'
    elif 'alternative rock' in genre.lower() or 'grunge' in genre.lower() or 'progressive rock' in genre.lower() or 'neue deutsche härte' in genre.lower():
        return 'alternative rock'
    elif 'rock' in genre.lower():
        return 'rock'
    elif 'punk' in genre.lower():
        return 'punk'
    elif 'hip hop' in genre.lower() or 'rap' in genre.lower() or 'hop' in genre.lower():
        return 'hip hop'
    elif 'reggaeton' in genre.lower():
        return 'reggaeton'
    elif 'R&B' in genre:
        return 'R&B'
    elif 'classic' in genre.lower() or 'symphony' in genre.lower():
        return 'classic'
    elif 'jazz' in genre.lower():
        return 'jazz'
    elif 'blues' in genre.lower():
        return 'blues'
    elif 'soul' in genre.lower():
        return 'soul'
    elif 'metal' in genre.lower() or 'death' in genre.lower() or 'post-hardcore' in genre.lower() or 'grindcore' in genre.lower():
        return 'metal'
    elif 'country' in genre.lower():
        return 'country'
    elif 'folk' in genre.lower():
        return 'folk'
    elif 'electro' in genre.lower() or 'hardstyle' in genre.lower() or 'house' in genre.lower() or 'dubstep' in genre.lower() or 'techno' in genre.lower():
        return 'electronic'
    elif 'funk' in genre.lower():
        return 'funk'
    elif 'swing' in genre.lower():
        return 'swing'
    elif 'schlager' in genre.lower():
        return 'schlager'
    elif 'opera' in genre.lower():
        return 'opera'
    elif 'gospel' in genre.lower():
        return 'gospel'
    elif 'disco' in genre.lower() or 'dance' in genre.lower():
        return 'disco'
    elif 'slasa' in genre.lower():
        return 'slasa'
    elif 'latin' in genre.lower():
        return 'latin'
    elif 'christian' in genre.lower() or 'worship' in genre.lower():
        return 'christian'
    elif 'literature' in genre.lower():
        return 'literature'
    elif 'podcast' in genre.lower():
        return 'podcast'
    elif 'christmas' in genre.lower():
        return 'christmas'
    elif 'children' in genre.lower():
        return 'children'
    elif 'film score' in genre.lower():
        return 'film score'
    elif 'film' in genre.lower() or 'movie' in genre.lower() or 'television' in genre.lower() or 'drama' in genre.lower() or 'sitcom' in genre.lower() or 'comedy' in genre.lower() or 'series' in genre.lower() or 'show' in genre.lower() or 'game' in genre.lower() or 'shooter' in genre.lower():
        return 'bullshit'
    else:
        return genre

# Einlesen der CSV-Datei
df = pd.read_csv("../data/data.csv")

# Selektion der Zeilen ohne NaN-Werte
df_selected = df.dropna().loc[~df["Genre"].isin(["Not Found", "Error"])]

# Filtern der Daten und Behalten nur derjenigen mit Häufigkeit > n
n = 10
genre_counts = df_selected['Genre'].value_counts()
df_selected = df_selected[df_selected['Genre'].isin(genre_counts[genre_counts > n].index)]

# Extrahieren aller Genres aus dem Datensatz
genres = df_selected['Genre'].unique()

# Erstellen der Übergenres und Gruppierung mit group_genres
df_selected['Genre'] = df_selected['Genre'].apply(group_genres)

# Abspeichern des resultierenden DataFrames als CSV
df_selected.to_csv("../data/data_selected.csv", index=False)

genre_value = df_selected['Genre'].value_counts()
print(genre_value)
#genre_value = df_selected['Genre'].value_counts()
#for genre, count in genre_value.items():
#    print(f"Genre: {genre}, Count: {count}")

print(f"Übriges Datenset hat {df_selected.shape[0]} Einträge")

# Plot erstellen
plt.figure(figsize=(12, 6))
genre_counts = df_selected['Genre'].value_counts()
plt.bar(genre_counts.index, genre_counts.values)
plt.xlabel('Genres')
plt.ylabel('Anzahl')
plt.xticks(rotation=90)
plt.title('Verteilung der Genres')
plt.tight_layout()
plt.show()