#!/usr/bin/env python3
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from explore_data_functions import visualize_features
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Conv2D,
    MaxPooling2D,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from training_functions import plot_history
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc


# Manuell auf CPU einschränken
tf.config.set_visible_devices([], "GPU")

df = pd.read_csv("../../data/data_selected_v1.csv", index_col=0)

# Aufteilen
X = df.drop("Genre", axis=1)
y = df["Genre"]

# Genre Kodieren
genre_encoder = OneHotEncoder(sparse_output=False)
# Muss leider erst in Numpy Array umgewandelt werden
genres = y.to_numpy().reshape(-1, 1)
one_hot_y = genre_encoder.fit_transform(genres)
# Die codierten Daten zurück in ein DataFrame umwandeln
y = pd.DataFrame(one_hot_y, columns=genre_encoder.get_feature_names_out(["Genre"]))

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

# TODO Die Ordinalen/Nominalen Attribute die hier kodiert sind sollten eigentlich
# nicht skaliert/transformiert sondern direkt passend gesetzt werden

# Neuronale Netze "mögen" Wertebereiche mit mean=0, daher auf [-1,1] skalieren
# Jede Verteilung sollte letztlich Gauß-änhlich sein?
#
# Erstmal Transformieren (TODO Ausprobieren, welche Trafo hier die besten Ergebnisse liefert)
# Fit NUR auf train...
transformer = QuantileTransformer(output_distribution="normal")
transformer.fit(X_train)
# ... dann transform auf alle
X_train = transformer.transform(X_train)
X_val = transformer.transform(X_val)
X_test = transformer.transform(X_test)

# TODO Ausprobieren, ob der StandardScaler hier nicht besser ist
# Werte der Attribute mit MinMax auf [-1,1] skalieren
scaler = MinMaxScaler(feature_range=(-1, 1))
# Fit wieder auf train...
scaler.fit(X_train)

# ... dann wieder transform
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Dann wieder zurück zum Dataframe weil Sklearn dumm ist
X_train = pd.DataFrame(X_train, columns=transformer.get_feature_names_out())
X_val = pd.DataFrame(X_val, columns=transformer.get_feature_names_out())
X_test = pd.DataFrame(X_test, columns=transformer.get_feature_names_out())

# visualize_features(X_train, "QuantileTransformer")

# Das schöne Modell
num_classes = len(y.value_counts())

model = Sequential()
model.add(Dense(units=64, activation="relu", input_dim=X_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=num_classes, activation="softmax"))

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
    jit_compile=False,
)
model.summary()
batch_size = 32
nb_epoch = 300

# Definiere die Early Stopping-Bedingungen
early_stopping = EarlyStopping(
    monitor="val_loss", patience=15, mode="min", restore_best_weights=True
)

# Definiere die Reduzierung der Lernrate, falls die Verbesserung stagniert
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=5, mode="min", min_lr=1e-7
)

hist = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=nb_epoch,
    batch_size=batch_size,
    callbacks=[early_stopping, reduce_lr],
)
plot_history(hist)

y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(y_pred, axis=1)
Y_test_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(Y_test_classes, Y_pred_classes, normalize="true")
plt.figure(figsize=(20, 20))
sns.heatmap(
    cm,
    annot=True,
    xticklabels=genre_encoder.get_feature_names_out(),
    yticklabels=genre_encoder.get_feature_names_out(),
)
plt.xlabel("Vorhergesagtes Genre")
plt.ylabel("Tatsächliches Genre")
plt.savefig("../../figures/confusion_matrix.png")
plt.clf()

# Precision Recall Curve für jede Klasse einzeln
if isinstance(y_test, np.ndarray):
    y_test = pd.DataFrame(y_test)
if isinstance(y_pred, np.ndarray):
    y_pred = pd.DataFrame(y_pred)

n_classes = y_test.shape[1]

# Für jede Klasse
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test.iloc[:, i], y_pred.iloc[:, i])

    # Berechne den AUC-PR
    auc_pr = auc(recall, precision)

    plt.plot(
        recall,
        precision,
        label=f"{genre_encoder.get_feature_names_out()[i]}, AUC-PR = {auc_pr:.3}",
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision-Recall Curve")
plt.savefig("../../figures/Precision-Recall.png")
