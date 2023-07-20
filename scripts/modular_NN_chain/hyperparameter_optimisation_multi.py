#!/usr/bin/env python3
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import QuantileTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.constraints import MaxNorm
from keras.wrappers.scikit_learn import KerasClassifier
from training_functions import plot_history
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc


# Manuell auf CPU einschränken
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")


df = pd.read_csv("../../data/data_selected_v1.csv", index_col=0)

# Genres für das Training auswählen
genres_to_keep = ["classic", "metal", "rock", "hip hop", "electronic", "pop"]
df = df[df["Genre"].isin(genres_to_keep)]

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

# Neuronale Netze "mögen" Wertebereiche mit mean=0, daher auf [-1,1] skalieren
# Jede Verteilung sollte letztlich Gauß-änhlich sein?
#
# Erstmal Transformieren
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

# Dann wieder zurück zum Dataframe weil Sklearn dumm ist
X_train = pd.DataFrame(X_train, columns=transformer.get_feature_names_out())
X_val = pd.DataFrame(X_val, columns=transformer.get_feature_names_out())
X_test = pd.DataFrame(X_test, columns=transformer.get_feature_names_out())


def create_model(
    optimizer="adam",
    activation="relu",
    dropout_rate=0.3,
    num_hidden_layers=2,
    neurons_1=64,
    neurons_2=128,
    neurons_3=256,
    neurons_4=512,
    weight_constraint=1.0,
):
    neurons_per_layer = [neurons_1, neurons_2, neurons_3, neurons_4]

    model = Sequential()
    model.add(
        Dense(
            units=neurons_per_layer[0],
            activation=activation,
            input_dim=X_train.shape[1],
            kernel_constraint=MaxNorm(weight_constraint),
        )
    )
    model.add(Dropout(dropout_rate))

    for i in range(1, num_hidden_layers + 1):
        if i < len(neurons_per_layer):
            model.add(Dense(units=neurons_per_layer[i], activation=activation))
            model.add(Dropout(dropout_rate))

    model.add(Dense(units=6, activation="softmax"))  # Anzahl der versch. Genres = 6

    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    return model


# Erstelle KerasClassifier-Objekt
model = KerasClassifier(build_fn=create_model, verbose=0)

# Alle möglichen Hyperparameter
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
weight_constraint = [1.0, 2.0, 3.0, 4.0, 5.0]
optimizer = ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"]
activation = ["relu", "elu", "swish"]
batch_size = [128, 256, 512, 1024]
epochs = [10, 50, 100]
num_hidden_layers = [2, 3]
neurons_1 = [64, 128]
neurons_2 = [128, 256]
neurons_3 = [256, 512]
neurons_4 = [512, 1024]

# Laut ChatGPT zählen die folgenden Parametern nicht zur klassischen Hyperparameteroptimierung;
# die sollten nach der eigentlichen HPO optimiert werden.
early_stopping_patience = [15, 20]
reduce_lr_factor = [0.1, 0.5]
reduce_lr_patience = [5, 7]
reduce_lr_min_lr = [1e-7, 1e-5]
param_grid = dict(
    dropout_rate=dropout_rate,
    weight_constraint=weight_constraint,
    optimizer=optimizer,
    activation=activation,
    batch_size=batch_size,
    epochs=epochs,
    neurons_1=neurons_1,
    neurons_2=neurons_2,
    neurons_3=neurons_3,
    neurons_4=neurons_4,
    num_hidden_layers=num_hidden_layers,
    # early_stopping_patience=early_stopping_patience,
    # reduce_lr_factor=reduce_lr_factor,
    # reduce_lr_patience=reduce_lr_patience,
    # reduce_lr_min_lr=reduce_lr_min_lr,
)

# Führe GridSearch durch
start_time = time.time()
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
grid_result = grid.fit(X_train, y_train, return_train_score=True)
end_time = time.time()

# Berechnung der verstrichenen Zeit in Stunden
elapsed_time_hours = (end_time - start_time) / 3600

# Drucken Sie die besten gefundenen Parameter
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Validieren Sie das Modell mit den besten gefundenen Parametern auf dem Validierungsset
best_model = grid_result.best_estimator_
validation_score = best_model.score(X_val, y_val)

print("Validation Score: ", validation_score)

# initialisiere das DataFrame
model_df = pd.DataFrame(
    columns=[
        "params",
        "mean_train_score",
        "std_train_score",
        "mean_test_score",
        "std_test_score",
        "delta_acc",
    ]
)

# Schleife durch die Modelle in GridSearchCV
for i in range(len(grid.cv_results_["params"])):
    # extrahiere die Parameter
    params = grid.cv_results_["params"][i]
    # extrahiere die durchschnittliche Trainings-Genauigkeit (über die Cross-Validation-Folds)
    mean_train_score = grid.cv_results_["mean_train_score"][i]
    # extrahiere die Standardabweichung der Trainings-Genauigkeit (über die Cross-Validation-Folds)
    std_train_score = grid.cv_results_["std_train_score"][i]
    # extrahiere die durchschnittliche Validierungs-Genauigkeit (über die Cross-Validation-Folds)
    mean_test_score = grid.cv_results_["mean_test_score"][i]
    # extrahiere die Standardabweichung der Validierungs-Genauigkeit (über die Cross-Validation-Folds)
    std_test_score = grid.cv_results_["std_test_score"][i]
    # berechne die Differenz zwischen der Trainings- und Validierungs-Genauigkeit
    delta_acc = mean_train_score - mean_test_score
    # füge die Informationen zu diesem Modell in das DataFrame ein
    model_df.loc[i] = [
        params,
        mean_train_score,
        std_train_score,
        mean_test_score,
        std_test_score,
        delta_acc,
    ]

# sortiere das DataFrame nach der Validierungs-Genauigkeit
model_df = model_df.sort_values(by="mean_test_score", ascending=False)

# Graphische Darstellung der Performance-Änderung
# Mapping der activation_function auf numerische Werte
activation_mapping = {"relu": 0, "elu": 1, "swish": 2}
model_df["activation_numerical"] = model_df["activation_function"].map(
    activation_mapping
)

x_vars = [
    "dropout_rate",
    "weight_constraint",
    "optimizer",
    "activation_numerical",
    "batch_size",
    "epochs",
    "neurons_1",
    "neurons_2",
    "neurons_3",
    "neurons_4",
    "num_hidden_layers",
]

y_vars = ["mean_train_score", "mean_test_score", "delta_acc"]

sns.pairplot(model_df, x_vars=x_vars, y_vars=y_vars, kind="reg", height=2)
plt.savefig("../../figures/HPO_parameter_v2_test.pdf", format="pdf")

# speichere das DataFrame
model_df.to_csv("../../optimization_results_v2_test_by_val_acc.csv", index=False)


# Jetzt noch die ein bisschen fishige Christopher-Score Berechnung
def calculate_christopher_score(row):
    # Bestmöglicher Score ist
    best_score = 1.0

    # Berechnung des Scores basierend auf Delta-Accuracy und Best Validation Accuracy
    delta_acc_score = 1.0 - abs(
        row["delta_acc"]
    )  # Je kleiner das Delta, desto besser der Score
    mean_test_acc_score = row[
        "mean_test_accuracy"
    ]  # Je größer die Validation Accuracy, desto besser der Score

    # Gewichte
    w1 = 0.5
    w2 = 1

    # Gesamtscore berechnen
    score = (w1 * delta_acc_score + w2 * mean_test_acc_score) / 2

    # Normalisierung des Scores auf den Bereich [0, 1]
    normalized_christopher_score = score / best_score

    return normalized_christopher_score


# Speichere den Christopher-Score in dem Dataframe
model_df["Christopher-Score"] = model_df.apply(calculate_christopher_score, axis=1)

# Sortiere nach dem neuen Score
model_df = model_df.sort_values("Christopher-Score", ascending=False)

# speichere das DataFrame
model_df.to_csv(
    "../../optimization_results_v2_test_by_christopher_score.csv", index=False
)

# Das beste Modell nach dem Christopher-Score auswählen
best_params = model_df.head(1)

# Scatterplot erstellen
sns.scatterplot(data=model_df, x="mean_test_accuracy", y="delta_acc", alpha=0.5)
sns.scatterplot(data=best_params, x="mean_test_accuracy", y="delta_acc", color="red")

# Achsentitel hinzufügen
plt.xlabel("validation accuracy")
plt.ylabel("delta accuracy")

# Plot anzeigen
plt.savefig("../../figures/HPO_scatter_v2_test.pdf")
