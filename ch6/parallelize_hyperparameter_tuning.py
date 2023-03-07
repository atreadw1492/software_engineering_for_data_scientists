
from ch4.ml_model import MlModel
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import pandas as pd
import time

parameters = {"n_estimators": (50, 100, 150, 200),
              "max_depth": (1, 2),
              "min_samples_split": (100, 250, 500, 750, 1000)
                 }

forest = MlModel(ml_model = RandomForestRegressor(),
                 parameters = parameters,
                 n_jobs = 3,
                 scoring = "neg_mean_squared_error",
                 n_iter = 5,
                 random_state = 0)


artists = pd.read_csv("data/artists.csv")


artists = artists.dropna().reset_index(drop = True)

artists['num_genres'] = artists.genres.map(len)
features = ["followers", "num_genres"]


start = time.time()
forest.tune(artists[features], artists.popularity)
end = time.time()
print("Completed in ", end - start, " seconds")