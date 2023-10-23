#py que roda o model em si e faz as previsões

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


file_path = './data/spotify-2023.csv'

spotify = pd.read_csv(file_path)

features_list = ['artist_count','in_spotify_playlists', 'streams','released_year','in_spotify_charts','in_apple_playlists', 'in_apple_charts', 'key']
X = spotify[features_list]
y = spotify.danceability


spotify_model = RandomForestRegressor(n_estimators=100)

spot_pipeline = Pipeline(steps=[('preprocessor', OneHotEncoder(handle_unknown='ignore')),
                              ('model', spotify_model)
                             ])

spot_pipeline.fit(X,y)

print(y.head())

print(spot_pipeline.predict(X))