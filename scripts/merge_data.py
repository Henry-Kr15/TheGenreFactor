import pandas as pd
import glob

# Liste aller CSV-Dateien im Verzeichnis
csv_files = glob.glob("../data/batch_*.csv")

# Leerer DataFrame zum Speichern aller Daten
combined_df = pd.DataFrame()

# Schleife zum Lesen und Zusammenfügen der CSV-Dateien
for file in csv_files:
    df = pd.read_csv(file)
    combined_df = pd.concat([combined_df, df])

# Speichern des kombinierten DataFrames in einer einzigen CSV-Datei
combined_df.to_csv("../data/data.csv", index=False)
