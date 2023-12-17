import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def summary_stats(csv_name):
    data = pd.read_csv(csv_name)
    summary_stats_data = [[round(data['duration_ms'].mean()/60000, 2), round(data['duration_ms'].std()/60000, 2), data['artist_name'].mode()[0], 'Major' if (data['mode'].mode()[0] == 1) else 'Minor']]
    df = pd.DataFrame(summary_stats_data, columns=['Average Duration (min)', 'Duration SD (min)', 'Most Common Artist', 'Most Common Mode'])
    return df

# to test with Henry's data
#summary_stats("spotifyhbritton@haverford.edu.csv")

def feature_score_boxplot(csv_name):
    data = pd.read_csv(csv_name)
    new_data = data[['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']]
    new_data = pd.melt(new_data, value_vars=['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence'])
    
    plot = sns.boxplot(x = 'variable', y = 'value', data = new_data)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=45)
    plot.set(xlabel='Music Feature',
           ylabel='Score',
           title='Distributions of Music Feature Scores')
    plt.savefig('summarystats.png', format = 'png')

# to test with Henry's data
#feature_score_boxplot("spotifyhbritton@haverford.edu.csv")