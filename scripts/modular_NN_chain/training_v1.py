#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from explore_data_functions import visualize_features


df = pd.read_csv("../../data/data_selected_v1.csv", index_col=0)

# Aufteilen
X = df.drop("Genre", axis=1)
y = df["Genre"]

# Genre Kodieren
genre_encoder = OneHotEncoder(sparse_output=False)
one_hot_y = genre_encoder.fit_transform(y.unique().reshape(-1,1)) # muss wieder in Dataframe umgewandelt werden
y = pd.DataFrame(one_hot_y, columns=genre_encoder.get_feature_names_out())

# Datensatz aufteilen, mit shuffle und stratify
# Trainingsatz:     64%
# Testsatz:         20%
# validierungssatz: 16%

# Trainings- und Testsatz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=True, random_state=42)

# Validierungssatz für die Hyperparameteroptimierung
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=True , random_state=42)


# Skalierung auf dem Trainigssatz und Anwendung dieser Skalierung auf den Test- und Validierungssatz
# Was sollte die Skalierung tun?
#
# Neuronale Netze "mögen" Wertebereiche mit mean=0, daher auf [-1,1] skalieren
# Jede Verteilung sollte letztlich Gauß-änhlich sein?
