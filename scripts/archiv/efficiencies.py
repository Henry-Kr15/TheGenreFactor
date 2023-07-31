#!/usr/bin/env python3
import pandas as pd
#from selection_functions import grouping, clean_data, show_results

# Rohdaten einlesen
df = pd.read_csv("../data/data.csv")

df_clean = df.loc[~df["Genre"].isin(["Not Found", "Error"])]

eff = len(df_clean)/len(df)

print("vorher: ", len(df))
print("nachher: ", len(df_clean))
print("effizienz: ", eff)

print("Anzahl an unterschiedlichen Genres: ", df.nunique())