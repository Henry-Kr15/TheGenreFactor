#!/usr/bin/env python3
import pandas as pd
from selection_functions import grouping, clean_data, show_results

# Rohdaten einlesen
df = pd.read_csv("../../data/data.csv")

df = grouping(df)

df = clean_data(df, 50, 5)

show_results(df)
