#Py que serve pra testar qual o numero de estimators mais eficiente para o modelo


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


from sklearn.preprocessing import OneHotEncoder


file_path = './data/spotify-2023.csv'

spotify = pd.read_csv(file_path)

features_list = ['artist_count','in_spotify_playlists', 'streams','released_year','in_spotify_charts', 'in_apple_playlists', 'in_apple_charts', 'key']
X = spotify[features_list]
y = spotify.danceability

#função pra definir o score de determinado modelo baseado no numero de N_estimators
def get_score(n_estimators):
     pipes = Pipeline(steps = [
         ('preprocess', OneHotEncoder(handle_unknown='ignore')),
         ('model', RandomForestRegressor(n_estimators = n_estimators, random_state = 0))
     ])
     score = -1 * cross_val_score(pipes,X,y,cv = 3, scoring = 'neg_mean_absolute_error', error_score='raise')
     return score.mean()

#loop pra testar varios estimators diferentes
results = {}
for i in range (1,9):
    results[50*i] = get_score(50*i)



plt.plot(list(results.keys()), list(results.values()))
plt.show()
