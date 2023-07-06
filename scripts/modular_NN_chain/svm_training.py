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
from sklearn import svm
from itertools import cycle

df = pd.read_csv("../../data/data_selected_v1.csv", index_col=0)

# Aufteilen
X = df.drop("Genre", axis=1)
y = df["Genre"]

# Genre Kodieren
genre_encoder = LabelEncoder()
# Muss leider erst in Numpy Array umgewandelt werden
genres = y.to_numpy().reshape(-1, 1)
one_hot_y = genre_encoder.fit_transform(genres)

# Datensatz aufteilen, mit shuffle und stratify
# Trainingsatz:     64%
# Testsatz:         20%
# validierungssatz: 16%

# Trainings- und Testsatz
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Validierungssatz für die Hyperparameteroptimierung
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# Support Vector Machine als Multiclass Classifier mit "One-Versus-One" Ansatz
clf = svm.SVC(decision_function_shape="ovo", probability=True)
clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)

# Genauigkeit berechnen
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy on test set is {accuracy*100:.2f}%")

# Plots der Precision-Recall Kurve für jede Klasse
