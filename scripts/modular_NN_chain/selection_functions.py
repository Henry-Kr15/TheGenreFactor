import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer


def group_genres(genre):
    """
    Diese Funktion soll die Spalte Genres etwas homogenisieren und die Aufteilungen gröber machen.
    """
    #    if 'indie pop' in genre.lower() or 'bedroom pop' in genre.lower():
    #        return 'indie pop'
    #    elif 'K-pop' in genre or 'J-pop' in genre:
    #        return 'asian pop'
    #    elif 'synth-pop' in genre:
    #        return 'synth pop'
    if "pop" in genre.lower():
        return "pop"
    elif (
        "alternative rock" in genre.lower()
        or "punk" in genre.lower()
        or "grunge" in genre.lower()
        or "progressive rock" in genre.lower()
        or "neue deutsche härte" in genre.lower()
    ):
        return "alternative rock"
    elif "rock" in genre.lower():
        return "rock"
    elif (
        "hip hop" in genre.lower()
        or "rap" in genre.lower()
        or "hop" in genre.lower()
        or "uk drill" in genre.lower()
    ):
        return "hip hop"
    elif "reggaeton" in genre.lower():
        return "reggaeton"
    elif "reggae" in genre.lower() or "ska" in genre.lower():
        return "reggae"
    elif "R&B" in genre:
        return "R&B"
    elif "classic" in genre.lower() or "symphony" in genre.lower():
        return "classic"
    elif "jazz" in genre.lower():
        return "jazz"
    elif "blues" in genre.lower():
        return "blues"
    elif "soul" in genre.lower():
        return "soul"
    elif (
        "metal" in genre.lower()
        or "death" in genre.lower()
        or "post-hardcore" in genre.lower()
        or "grindcore" in genre.lower()
    ):
        return "metal"
    elif "country" in genre.lower():
        return "country"
    elif "folk" in genre.lower():
        return "folk"
    elif (
        "electro" in genre.lower()
        or "hardstyle" in genre.lower()
        or "house" in genre.lower()
        or "dubstep" in genre.lower()
        or "techno" in genre.lower()
        or "drum and bass" in genre.lower()
    ):
        return "electronic"
    elif "funk" in genre.lower():
        return "funk"
    elif "swing" in genre.lower():
        return "swing"
    elif "schlager" in genre.lower():
        return "schlager"
    elif "opera" in genre.lower():
        return "opera"
    elif "gospel" in genre.lower():
        return "gospel"
    elif (
        "disco" in genre.lower()
        or "dance" in genre.lower()
        or "trance" in genre.lower()
    ):
        return "disco"
    elif (
        "latin" in genre.lower()
        or "salsa" in genre.lower()
        or "bachata" in genre.lower()
        or "samba" in genre.lower()
        or "vallenato" in genre.lower()
        or "mariachi" in genre.lower()
        or "grupera" in genre.lower()
    ):
        return "latin"
    elif (
        "mexican" in genre.lower()
        or "ranchera" in genre.lower()
        or "norteña" in genre.lower()
        or "tejano" in genre.lower()
    ):
        return "mexican"
    elif "christian" in genre.lower() or "worship" in genre.lower():
        return "christian"
    elif "literature" in genre.lower():
        return "literature"
    elif "podcast" in genre.lower():
        return "podcast"
    elif "christmas" in genre.lower():
        return "christmas"
    elif "children" in genre.lower():
        return "children"
    elif "film score" in genre.lower():
        return "film score"
    elif (
        "film" in genre.lower()
        or "movie" in genre.lower()
        or "television" in genre.lower()
        or "drama" in genre.lower()
        or "sitcom" in genre.lower()
        or "comedy" in genre.lower()
        or "series" in genre.lower()
        or "show" in genre.lower()
        or "game" in genre.lower()
        or "shooter" in genre.lower()
    ):
        return "bullshit"
    else:
        return genre

def grouping(df: pd.DataFrame):
    """
    Dummy für etwas netteren Aufruf
    """

    df["Genre"] = df["Genre"].apply(group_genres)

    return df


def clean_data(df: pd.DataFrame, n: int, k: int):
    """
    Diese Funktion soll verschiedene Strategien anwenden, um den Datensatz zu reinigen.

    1. Einige der nominalen/ordinalen Attribute können wir in kardinale umwandeln.
    2. Attribute, die nicht kardinal sind, können (Stand jetzt) nicht verarbeitet werden.
       Daher werden wir diese vorerst entfernen.
    3. Manche Attribute tragen keine Information und werden daher entfernt (z.B. die URLs).
    4. Vorerst werden wir nur die Genres behalten, welche öfter als n-mal auftreten.
    5. Das Genre-Attribut soll besser verarbeitbar werden. Dafür muss die Aufteilung gröber werden.
    6. Spalten mit NaN-Werten sollen nach Möglichkeit mt verschiedenen Strategien aufgeteilt werden.

    Argumente:

    df: Das vollständige Dataframe mit den Rohdaten die gereinigt werden sollen

    n: Mindestanzahl der zu Genres, um das Genre zu behalten

    k: Anzahl der nächsten Nachbarn, anhand denen die NaNs ersetzt werden

    Rückgabe:

    df_clean: das bereinigte Dataframe
    """

    df_clean = df.copy() # das ist scheiße

    # 1.
    # Dieser Ansatz könnte dazu führen, dass das NN annimmt der Künstlername wäre Ordinal; TESTEN
    # Erstelle Objekte des LabelEncoder
    artist_encoder = LabelEncoder()
    album_type_encoder = LabelEncoder()
    licensed_encoder = LabelEncoder()
    official_video_encoder = LabelEncoder()

    # Fit und transform auf die Daten anwenden
    df_clean["Artist_encoded"] = artist_encoder.fit_transform(df_clean["Artist"])
    df_clean["Album_type_encoded"] = album_type_encoder.fit_transform(df_clean["Album_type"])
    df_clean["Licensed_encoded"] = licensed_encoder.fit_transform(df_clean["Licensed"])
    df_clean["official_video_encoded"] = official_video_encoder.fit_transform(df_clean["official_video"])

    # 2. & 3.
    features_to_drop = [
        "Artist", # Wird encoded
        "Url_spotify",
        "Track",
        "Album",
        "Album_type", # Wird encoded
        "Uri",
        "Url_youtube",
        "Title",
        "Channel",
        # 'Views',
        # 'Likes',
        # 'Comments',
        "Description",
        "Licensed", # Wird encoded
        "official_video", # Wird encoded
        # 'Stream'
    ]

    df_clean = df_clean.drop(features_to_drop, axis=1)

    # 4.
    genre_counts = df_clean["Genre"].value_counts()
    df_clean= df_clean[
        df_clean["Genre"].isin(genre_counts[genre_counts > n].index)
    ]

    # 5.
    df_clean["Genre"] = df_clean["Genre"].apply(group_genres)
    df_clean = df_clean.loc[~df_clean["Genre"].isin(["bullshit"])] # manche Genres sinds einfach nicht
    df_clean = df_clean.loc[~df_clean["Genre"].isin(["Not Found", "Error"])]


    # 6.
    # Dieser Ansatz verwendet die Knns, um die NaNs zu ersetzen
    # Dies ist ein experimentelles Feature von sklearn, kann in verschiedenen Fällen nicht konvergieren
    imputer = KNNImputer(n_neighbors=k)

    # Nur numerische Spalten auswählen
    num_cols = df.select_dtypes(include=[np.number]).columns

    # Kopie, um den Imputer zu trainieren
    df_clean_copy = df_clean.copy()

    # fit_transform auf den numerischen Spalten anwenden...
    df_clean_copy[num_cols] = imputer.fit_transform(df_clean_copy[num_cols])

    # ...und das Ergebnis in das ursprüngliche Dataframe den ursprünglichen Spalten zuweisen
    df_clean[num_cols] = df_clean_copy[num_cols]

    df_clean_imputed = df_clean # eigentlich echt unnötig aber ich finde die Benennung hilft beim Verständnis

    return df_clean_imputed


def show_results(df_clean_imputed: pd.DataFrame):
    """
    Diese Funktion soll verschiedene Informationen zu dem gereinigten Datensatz anzeigen.
    """

    genres = df_clean_imputed["Genre"].unique()
    print(f"Es gibt insgesamt {len(genres)} verschiedene Genres")

    genre_value = df_clean_imputed["Genre"].value_counts()
    print(genre_value)

    print(f"Übriges Datenset hat {df_clean_imputed.shape[0]} Einträge")

    # Plot erstellen
    plt.figure(figsize=(12, 6))
    genre_counts = df_clean_imputed["Genre"].value_counts()
    plt.bar(genre_counts.index, genre_counts.values)
    plt.xlabel("Genres")
    plt.ylabel("Anzahl")
    plt.xticks(rotation=90)
    plt.title("Verteilung der Genres")
    plt.tight_layout()
    plt.show()
