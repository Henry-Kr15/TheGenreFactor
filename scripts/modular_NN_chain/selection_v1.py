#!/usr/bin/env python3
import pandas as pd
from selection_functions import grouping, clean_data, show_results

# Rohdaten einlesen
df = pd.read_csv("../../data/data.csv")

df = grouping(df)
df = clean_data(df, 50, 5)

# Abspeichern des resultierenden DataFrames als CSV
df.to_csv("../../data/data_selected_v1.csv", index=False)

show_results(df)
