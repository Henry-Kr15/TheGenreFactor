#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)
from sklearn.neighbors import KNeighborsClassifier
from itertools import cycle

df = pd.read_csv("../../data/data_selected_v1.csv", index_col=0)

genres_to_keep = ["classic", "metal", "rock", "hip hop", "electronic", "pop"]
df = df[df["Genre"].isin(genres_to_keep)]


# Aufteilen
X = df.drop("Genre", axis=1)
y = df["Genre"]


# Genre Kodieren
genre_encoder = LabelEncoder()
# Muss leider erst in Numpy Array umgewandelt werden
# genres = y.to_numpy().reshape(-1, 1)
one_hot_y = genre_encoder.fit_transform(y)
print(one_hot_y)
print("Class labels:", np.unique(one_hot_y))

# Datensatz aufteilen, mit shuffle und stratify
# Trainingsatz:     64%
# Testsatz:         20%
# validierungssatz: 16%

# Trainings- und Testsatz
X_train, X_test, y_train, y_test = train_test_split(
    X, one_hot_y, test_size=0.2, stratify=one_hot_y, random_state=42
)

# Validierungssatz für die Hyperparameteroptimierung
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

features_to_use = [
    "Danceability",
    "Energy",
    "Key",
    "Loudness",
    "Speechiness",
    "Acousticness",
    "Instrumentalness",
    "Liveness",
    "Valence",
    "Tempo",
    "Duration_ms",
    "Views",
    "Likes",
    "Comments",
    "Stream",
    # "Artist_encoded", # Die hier machen Probleme
    # "Album_type_encoded",
    # "Licensed_encoded",
    # "official_video_encoded",
]
print(X_train.columns)
X_train = X_train[features_to_use]
X_test = X_test[features_to_use]
X_val = X_val[features_to_use]

# KNNs klöppeln
kn = KNeighborsClassifier(n_neighbors=6)
kn.fit(X_train, y_train)
print(kn.classes_)
print(X_test)

y_pred = kn.predict(X_test)

# # Genauigkeit berechnen
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy on test set is {accuracy*100:.2f}%")
