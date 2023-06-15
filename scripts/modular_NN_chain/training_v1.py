#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sklearn

df = pd.read_csv("../../data/data_selected_v1.csv", index_col=0)

# Aufteilen
X = df.drop("Genre", axis=1)
y = df["Genre"]

