import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

csv_file = "../data/data.csv"
df = pd.read_csv(csv_file)
genres = df['Genre'].unique()

features = ['Danceability', 'Energy', 'Key', 'Speechiness', 
            'Instrumentalness', 'Liveness', 'Valence', 
            'Tempo', 'Duration_ms', 'Views', 'Stream']

genre_liste = ['pop', 'rock']

for idx in genre_liste:

    print(idx)

    selected_genres = [genre for genre in genres if idx in genre.lower()]
    genre_counts = Counter(selected_genres)
    most_common_genres = genre_counts.most_common(5) # hier einstellen wie viele Genres in den Plot sollen
    selected_genres = [genre for genre, _ in most_common_genres]

    genre_data = []
    for genre in selected_genres:
        genre_df = df[df['Genre'] == genre]
        genre_features = genre_df[features]
        genre_data.append(genre_features)

    # DataFrame für den Pairplot erstellen
    pairplot_data = pd.concat(genre_data)
    genre_labels = []
    for i, genre_features in enumerate(genre_data):
        genre_labels.extend([selected_genres[i]] * len(genre_features))
    pairplot_data['Genre'] = genre_labels

    # Pairplot erstellen
    pairplot = sns.pairplot(pairplot_data, hue='Genre', plot_kws={'s': 5})
    

    pairplot.savefig(f'../../figures/pairplots/pairplot_{idx}.png')
