#!/usr/bin/env python3
from get_genre import get_genre
import pandas as pd

# Datensatz einlesen
file_path = "../data/Spotify_Youtube.csv"
df_spotify_youtube = pd.read_csv(file_path)

# Dataframe reduzieren für einfacheren Funktionsaufruf
df_spotify_youtube_subset = df_spotify_youtube[["Artist", "Track", "Album_type"]]

# Für Testzwecke reduzieren auf die ersten 20 Zeilen reduzieren
df_spotify_youtube_subset = df_spotify_youtube_subset.head(20)

# Genre holen und in ein neues Dataframe schreiben
genres = df_spotify_youtube_subset.apply(lambda row: get_genre(row["Artist"], row["Track"], row["Album_type"]), axis=1)
df_genres = pd.DataFrame(genres, columns=["genres"])

# Testausgabe
print(df_genres.head())

# Manuelle Tests
# print(get_genre("Metallica", "Ride the Lightning", "album"))
# print(get_genre("Bring me the Horizon", "Sempiternal", "album"))
# print(get_genre("Lorna Shore", "Pain Remains", "album"))
# print(get_genre("Gorillaz", "New Gold (feat. Tame Impala and Bootie Brown)", "single"))
# print(get_genre("Ludovico Einaudi", "Una Mattina", "album"))
# print(get_genre("Rolf Zuckowski", "Theo", "single"))
# print(get_genre("Death", "Scream Bloody Gore", "album"))
# print(get_genre("Movie", "Evil Dead", "album")) # nice dass das auch mit Filmen klappt :D
