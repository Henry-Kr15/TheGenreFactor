#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
)
from sklearn import svm

df = pd.read_csv("../../data/data_selected_v1.csv", index_col=0)

genres_to_keep = ["classic", "metal", "rock", "hip hop", "electronic", "pop"]
df = df[df["Genre"].isin(genres_to_keep)]

# Aufteilen
X = df.drop("Genre", axis=1)
y = df["Genre"]

# Genre Kodieren
genre_encoder = LabelEncoder()
one_hot_y = genre_encoder.fit_transform(y)

# Datensatz aufteilen, mit shuffle und stratify
# Trainingsatz:     64%
# Testsatz:         20%
# validierungssatz: 16%

# Trainings- und Testsatz
X_train, X_test, y_train, y_test = train_test_split(
    X, one_hot_y, test_size=0.2, stratify=one_hot_y, random_state=42
)

# Validierungssatz
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
X_train = X_train[features_to_use]
X_test = X_test[features_to_use]
X_val = X_val[features_to_use]

# Erstmal Transformieren (TODO Ausprobieren, welche Trafo hier die besten Ergebnisse liefert)
# Fit NUR auf train...
transformer = QuantileTransformer(output_distribution="normal")
transformer.fit(X_train)

# ... dann transform auf alle
X_train = transformer.transform(X_train)
X_val = transformer.transform(X_val)
X_test = transformer.transform(X_test)

# Werte der Attribute mit MinMax auf [-1,1] skalieren
scaler = MinMaxScaler(feature_range=(-1, 1))
# Fit wieder auf train...
scaler.fit(X_train)

# ... dann wieder transform
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# SVM
clf = svm.SVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(y_pred)

# Genauigkeit berechnen
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy on test set is {accuracy*100:.2f}%")

# Pikachu ist VERWIRRT
cm = confusion_matrix(y_test, y_pred, normalize="true")
plt.figure(figsize=(20, 20))
sns.heatmap(
    cm,
    annot=True,
    xticklabels=genre_encoder.classes_,
    yticklabels=genre_encoder.classes_,
)
plt.xlabel("Vorhergesagtes Genre")
plt.ylabel("Tatsächliches Genre")
plt.savefig("../../figures/svm/confusion_matrix.png")
plt.clf()

# Berechne Precision, Recall und Schwellenwerte

# Die Funktion braucht leider einen binären Wahrheitswert :/
# Fake it till you make it
# weiß nicht ob man das so machen kann oder ob das übel dumm ist
# Spoiler alert
# das ist übel dumm
y_pred_binary = np.zeros_like(y_pred)
y_test_binary = np.full_like(y_test, 1)

for prediction in y_pred:
    if y_pred[prediction] == y_test[prediction]:
        y_pred_binary[prediction] = 1
    else:
        y_pred_binary[prediction] = 0

precision, recall, thresholds = precision_recall_curve(y_test_binary, y_pred_binary)

print(y_test_binary)
print(y_pred_binary)

# Berechne den AUC-PR
auc_pr = auc(recall, precision)

plt.plot(recall, precision, label=f"AUC-PR = {auc_pr:.3}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision-Recall Curve")
plt.savefig("../../figures/svm/auc.png")
