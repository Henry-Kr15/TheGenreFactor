#!/usr/bin/env python3
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import time
import seaborn as sns
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc

# Manuell auf CPU einschränken
tf.config.set_visible_devices([], "GPU")

# Daten laden
csv_file = "../../data/data_selected_v1.csv"
df = pd.read_csv(csv_file)

# Genres für das Training auswählen
genres_to_keep = ["classic", "metal", "rock", "hip hop", "electronic", "pop"]
df = df[df["Genre"].isin(genres_to_keep)]

# Anzahl der Einträge pro Genre
genre_counts = df["Genre"].value_counts()

# Histogramm erstellen
plt.figure(figsize=(10, 6))
genre_counts.plot(kind="bar")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.savefig("../../figures/genre_hist.pdf")
print(f"Anzahl an Songs: {df.shape[0]}")

# Designmatrix und Target erstellen
features_drop = ["Genre", "Unnamed: 0"]

df = df.sample(frac=1, random_state=42)  # daten mischen

X = df.drop(features_drop, axis=1)
Y = df["Genre"]

# One-Hot-Encoding durchführen
Y = pd.get_dummies(Y)
num_classes = len(Y.value_counts())

# label um One-Hot-Encoding hinterher wieder zu übersetzen
genre_mapping = Y.columns
label = []
for index, genre in enumerate(genre_mapping):
    label.append(genre)
    print("Genre", index, ":", genre)

# Splitten
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, stratify=Y, random_state=42
)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.3, stratify=Y_train, random_state=42
)

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

# Das Modell


def build_model_from_params(params):
    num_hidden_layers = params["num_hidden_layers"]
    activation_function = params["activation_function"]
    dropout_rate = params["dropout_rate"]
    units_per_layer = [
        params["units_1"],
        params["units_2"],
        params["units_3"],
        params["units_4"],
    ]

    model = Sequential()
    model.add(
        Dense(
            units=units_per_layer[0],
            activation=activation_function,
            input_dim=X_train.shape[1],
        )
    )
    model.add(Dropout(dropout_rate))

    for i in range(1, num_hidden_layers + 1):
        if i < len(units_per_layer):
            model.add(Dense(units=units_per_layer[i], activation=activation_function))
            model.add(Dropout(dropout_rate))

    model.add(Dense(units=num_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model


# Kombination aller Hyperparameter
param_space = {
    "num_hidden_layers": [2, 3],
    "units_1": [64, 128],
    "units_2": [128, 256],
    "units_3": [256, 512],
    "units_4": [512, 1024],
    "dropout_rate": [0.3, 0.5],
    "activation_function": ["LeakyReLU", "relu"],
    "batch_size": [128, 256, 512, 1024],
    "early_stopping_patience": [15, 20],
    "reduce_lr_factor": [0.1, 0.5],
    "reduce_lr_patience": [5, 7],
    "reduce_lr_min_lr": [1e-7, 1e-5],
}

# param_space = {
#    'num_hidden_layers': [2, 3],
#    'units_1': [64, 128, 256],
#    'units_2': [128, 256, 512],
#    'units_3': [256, 512, 1024],
#    'units_4': [512, 1024],
#    'dropout_rate': [0.3, 0.5, 0.7],
#    'activation_function': ['relu', 'LeakyReLU'],
#    'batch_size' : [128, 256, 512],
#    'early_stopping_patience': [5, 10, 15],
#    'reduce_lr_factor': [0.1, 0.5, 0.9],
#    'reduce_lr_patience': [5, 10, 15],
#    'reduce_lr_min_lr': [1e-7, 1e-5, 1e-3],
# }

# Speichere alle möglichen Kombinationen von Parametern
value_combis = itertools.product(*[v for v in param_space.values()])

# Erstelle Liste von Dictionarys dieser Parameterkombis
param_combis = [
    {key: value for key, value in zip(param_space.keys(), combi)}
    for combi in value_combis
]

print(f"We have a total of {len(param_combis)} combinations")

# Hyperparameteroptimierung

# Start der Zeitmessung
start_time = time.time()

search_results = []

k_folds = 3
skf = StratifiedKFold(n_splits=k_folds)

for idx, params in enumerate(param_combis):
    print(f"Start run {idx+1}/{len(param_combis)}: Parameters: {params}")

    val_accuracies = []
    val_losses = []
    train_accuracies = []
    train_losses = []

    for fold_idx, (train_index, val_index) in enumerate(
        skf.split(X_train, np.argmax(Y_train, axis=1))
    ):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        Y_train_fold, Y_val_fold = (
            Y_train.reset_index(drop=True).iloc[train_index],
            Y_train.reset_index(drop=True).iloc[val_index],
        )

        filepath = f"../model_fold/model_fold_{idx+1}_fold_{fold_idx+1}.h5"
        checkpoint = ModelCheckpoint(
            filepath, monitor="val_accuracy", verbose=0, save_best_only=True, mode="max"
        )

        this_model = build_model_from_params(params)

        batch_size = params["batch_size"]
        nb_epoch = 60

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=params["early_stopping_patience"],
            mode="min",
            restore_best_weights=True,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=params["reduce_lr_factor"],
            patience=params["reduce_lr_patience"],
            mode="min",
            min_lr=params["reduce_lr_min_lr"],
        )
        fit_results = this_model.fit(
            X_train_fold,
            Y_train_fold,
            validation_data=(X_val_fold, Y_val_fold),
            epochs=nb_epoch,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            verbose=0,
        )

        # Extract the best validation scores
        best_val_epoch = np.argmax(fit_results.history["val_accuracy"])
        val_accuracies.append(np.max(fit_results.history["val_accuracy"]))
        val_losses.append(fit_results.history["val_loss"][best_val_epoch])

        # Get training accuracy and loss
        best_model = load_model(filepath)
        train_loss, train_acc = best_model.evaluate(X_train_fold, Y_train_fold)
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)

    # Store results
    search_results.append(
        {
            **params,
            "best_val_accuracy": np.mean(val_accuracies),
            "val_accuracy_std": np.std(val_accuracies),
            "best_val_loss": np.mean(val_losses),
            "val_loss_std": np.std(val_losses),
            "best_train_accuracy": np.mean(train_accuracies),
            "train_accuracy_std": np.std(train_accuracies),
            "best_train_loss": np.mean(train_losses),
            "train_loss_std": np.std(train_losses),
        }
    )

# Ende der Zeitmessung
end_time = time.time()

# Berechnung der verstrichenen Zeit in Stunden
elapsed_time_hours = (end_time - start_time) / 3600

print(f"Die Ausführung dieser Zelle hat {elapsed_time_hours} Stunden gedauert.")

resultsDF = pd.DataFrame(search_results)
resultsDF["delta_acc"] = (
    resultsDF["best_train_accuracy"] - resultsDF["best_val_accuracy"]
) / resultsDF["best_val_accuracy"]
resultsDF.to_csv("../../data/optimization_results_final_größer.csv", index=False)
resultsDF = pd.read_csv("../../data/optimization_results_final_größer.csv")

# Graphische Darstellung der Performance-Änderung
# Mapping der activation_function auf numerische Werte
activation_mapping = {"relu": 0, "LeakyReLU": 1}
resultsDF["activation_numerical"] = resultsDF["activation_function"].map(
    activation_mapping
)

x_vars = [
    "num_hidden_layers",
    "units_1",
    "units_2",
    "units_3",
    "units_4",
    "dropout_rate",
    "activation_numerical",
    "batch_size",
    "early_stopping_patience",
    "reduce_lr_factor",
    "reduce_lr_patience",
    "reduce_lr_min_lr",
]

y_vars = ["best_val_accuracy", "best_train_accuracy", "delta_acc"]

sns.pairplot(resultsDF, x_vars=x_vars, y_vars=y_vars, kind="reg", height=2)
plt.savefig("../../figures/HPO_parameter.pdf", format="pdf")

# Auswählen der besten Hyperparameter-Kombination


def calculate_score(row):
    # Bestmöglicher Score ist
    best_score = 1.0

    # Berechnung des Scores basierend auf Delta-Accuracy und Best Validation Accuracy
    delta_acc_score = 1.0 - abs(
        row["delta_acc"]
    )  # Je kleiner das Delta, desto besser der Score
    best_val_acc_score = row[
        "best_val_accuracy"
    ]  # Je größer die Best Validation Accuracy, desto besser der Score

    # Gewichte
    w1 = 0.5
    w2 = 1

    # Gesamtscore berechnen
    score = (w1 * delta_acc_score + w2 * best_val_acc_score) / 2

    # Normalisierung des Scores auf den Bereich [0, 1]
    normalized_score = score / best_score

    return normalized_score


# Score-Spalte hinzufügen
resultsDF["score"] = resultsDF.apply(calculate_score, axis=1)
resultsDF = resultsDF.sort_values("score", ascending=False)
best_params = resultsDF.head(1)

# Scatterplot erstellen
sns.scatterplot(data=resultsDF, x="best_val_accuracy", y="delta_acc", alpha=0.5)
sns.scatterplot(data=best_params, x="best_val_accuracy", y="delta_acc", color="red")

# Achsentitel hinzufügen
plt.xlabel("best_val_accuracy")
plt.ylabel("delta_acc")

# Plot anzeigen
plt.show()
plt.savefig("../../figures/HPO_scatter.pdf")

best_params = resultsDF.iloc[1]
model = build_model_from_params(best_params)

print(model.summary())

# Das beste Modell
batch_size = best_params["batch_size"]
nb_epoch = 300

# Definiere die Early Stopping-Bedingungen
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=best_params["early_stopping_patience"],
    mode="min",
    restore_best_weights=True,
)

# Definiere die Reduzierung der Lernrate, falls die Verbesserung stagniert
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=best_params["reduce_lr_factor"],
    best_patience=best_params["reduce_lr_patience"],
    mode="min",
    min_lr=best_params["reduce_lr_min_lr"],
)

hist = model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    epochs=nb_epoch,
    batch_size=batch_size,
    callbacks=[early_stopping, reduce_lr],
    verbose=0,
)

Y_pred = model.predict(X_test)


def plot_history(network_history):
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(network_history.history["loss"])
    plt.plot(network_history.history["val_loss"])
    plt.legend(["Training", "Validation"])

    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(network_history.history["accuracy"])
    plt.plot(network_history.history["val_accuracy"])
    plt.legend(["Training", "Validation"], loc="lower right")
    plt.show()


plot_history(hist)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_test_classes = np.argmax(Y_test, axis=1)

cm = confusion_matrix(Y_test_classes, Y_pred_classes, normalize="true")
plt.figure(figsize=(20, 20))
sns.heatmap(cm, annot=True, xticklabels=label, yticklabels=label)
plt.xlabel("Vorhergesagtes Genre")
plt.ylabel("Tatsächliches Genre")
plt.savefig("../../figures/confusion_matrix_NN.png")
plt.show()

if isinstance(Y_test, pd.DataFrame):
    Y_test = Y_test.values
if isinstance(Y_pred, pd.DataFrame):
    Y_pred = Y_pred.values

precision, recall, thresholds = precision_recall_curve(Y_test.ravel(), Y_pred.ravel())


# Berechne Precision, Recall und Schwellenwerte
precision, recall, thresholds = precision_recall_curve(Y_test.ravel(), Y_pred.ravel())

# Berechne den AUC-PR
auc_pr = auc(recall, precision)

plt.plot(recall, precision, label=f"AUC-PR = {auc_pr:.3}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision-Recall Curve")
plt.show()

# Precision Recall Curve für jede Klasse einzeln

n_classes = Y_test.shape[1]

if isinstance(Y_test, np.ndarray):
    Y_test = pd.DataFrame(Y_test)
if isinstance(Y_pred, np.ndarray):
    Y_pred = pd.DataFrame(Y_pred)

auc_pr_values = []

# Für jede Klasse
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(Y_test.iloc[:, i], Y_pred.iloc[:, i])

    # Berechne den AUC-PR
    auc_pr = auc(recall, precision)
    auc_pr_values.append((label[i], auc_pr))

    plt.plot(
        recall,
        precision,
        label=f"{label[i]}, AUC-PR = {auc_pr:.3}",
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision-Recall Curve")

# Genres nach AUC-PR-Wert sortieren
auc_pr_values_sorted = sorted(auc_pr_values, key=lambda x: x[1], reverse=True)

# Ausgabe der Genres und AUC-PR-Werte
for genre, auc_pr in auc_pr_values_sorted:
    print(f"{genre}: AUC-PR = {auc_pr:.3}")

Y_pred = model.predict(X_test)

# Konvertiere die Vorhersagen in diskrete Klassen
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Konvertiere die Ground-Truth-Labels in diskrete Klassen
Y_test_classes = np.argmax(Y_test, axis=1)

# Berechne die Accuracy
accuracy = np.mean(Y_pred_classes == Y_test_classes)
print("Accuracy:", accuracy)
