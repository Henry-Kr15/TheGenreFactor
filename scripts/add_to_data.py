import pandas as pd
from get_genre import get_genre
import concurrent.futures #für Parallelisierung
import time #für Stoppuhr
from tqdm import tqdm #für Fortschrittbalken
import multiprocessing #für Anzahl der verfügbaren Kerne

# Stoppuhr
start_time = time.time()

# Dateinamen und Pfad zum CSV-Datensatz
csv_file = "../data/Spotify_Youtube.csv"

df = pd.read_csv(csv_file)

# Anzahl der Prozesse für die parallele Ausführung

if(multiprocessing.cpu_count() <= 14):
    num_processes = multiprocessing.cpu_count()
else:
    num_processes = 14

# TODO delete?
# Funktion zum Abfragen des Genres und Aktualisieren des DataFrames
def query_genre_and_update_df(index, artist, album, album_type):
    genre = get_genre(artist, album, album_type)
    return (index, genre)

#Funktion zur parallelen Genresuche in einem batch
def process_data_batch(batch_df, num_processes):
    # Extrahiere die erforderlichen Spalten als NumPy-Arrays
    indexes = batch_df.index.to_numpy()
    artists = batch_df["Artist"].to_numpy()
    albums = batch_df["Album"].to_numpy()
    album_types = batch_df["Album_type"].to_numpy()

    # Parallele Ausführung der Abfragen
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(query_genre_and_update_df, idx, artist, album, album_type) for idx, artist, album, album_type in zip(indexes, artists, albums, album_types)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            index, genre = future.result()
            results.append((index, genre))

    return results

# Anzahl der Einträge pro Paket
batch_size = 200

# Gesamtzahl der Einträge im DataFrame
total_entries = len(df)
# total_entries = 600

# Schleife zur Durchführung der Genreabfrage für jedes Paket
results = []
for i in range(0, total_entries, batch_size):
    batch_df = df[i:i+batch_size]
    batch_results = process_data_batch(batch_df, num_processes)
    results.extend(batch_results)


# Aktualisiere das DataFrame mit den Ergebnissen
for index, genre in results:
    df.loc[index, "Genre"] = genre

# DataFrame als CSV-Datei speichern
df.to_csv("../data/data.csv", index=False)


# Berechne die Dauer der Ausführung in Sekunden
duration = time.time() - start_time

# Gib die Dauer in Sekunden aus
print(f"Ausführungsdauer: {duration/60.:.3} Minuten")
