#!/usr/bin/env python3
#!/usr/bin/env python3
from get_genre import get_genre
import pandas as pd
import multiprocessing as mp
import time

# Datensatz einlesen
file_path = "../data/Spotify_Youtube.csv"
df_spotify_youtube = pd.read_csv(file_path)

# Dataframe reduzieren für einfacheren Funktionsaufruf
df_spotify_youtube_subset = df_spotify_youtube[["Artist", "Track", "Album_type"]]


def apply_get_genre(args):
    """Hilfsfunktion zum Extrahieren von Argumenten und Aufrufen der Funktion get_genre."""
    return get_genre(*args)


# Erstellen Sie eine Liste von Tupeln, die als Argumente für die Funktion get_genre verwendet werden können
args_list = df_spotify_youtube_subset.to_records(index=False).tolist()

start = time.perf_counter()

# Erstellen eines Pools von Prozessen
# Die Wahl der Anzahl der Kerne ist ein bisschen schwierig; mp.cpu_count() wählt alle zur Verfügung stehenden
# Prozesse aus, das ist aber zuviel für den Wikidata Server (Error Code 429).
# Daher manuelle Angabe:
# 6:
with mp.Pool(6) as pool:
    # Verwende map(), um die Funktion get_genre auf die Liste von Argumenten anzuwenden
    genres = pool.map(apply_get_genre, args_list)

end = time.perf_counter()
print(f"Wikidata-Abfragen durchgeführt in {end-start} Sekunden.")

# Genre holen und in ein neues Dataframe schreiben
df_genres = pd.DataFrame(genres, columns=["genres"])

# DataFrame in eine CSV-Datei schreiben
df_genres.to_csv('../data/genres_versuch2.csv', index=False)

print(df_genres.describe())
