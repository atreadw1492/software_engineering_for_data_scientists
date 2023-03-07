
from ch4.ml_model import MlModel
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import time

parameters = {"n_estimators": (50, 100, 150, 200),
              "max_depth": (1, 2),
              "learning_rate": (0.01, 0.05, 0.1),
              "subsample": (0.5, 0.6, 0.7, 0.8)}

gbm = MlModel(ml_model = GradientBoostingRegressor(),
                 parameters = parameters,
                 n_jobs = 4,
                 scoring = "roc_auc",
                 n_iter = 30,
                 random_state = 0)


artists = pd.read_csv("data/artists.csv")

artists = artists.dropna().reset_index(drop = True)

artists['num_genres'] = artists.genres.map(len)
features = ["followers", "num_genres"]


start = time.time()
gbm.tune(artists[features], artists.popularity)
end = time.time()
print("Completed in ", end - start, " seconds")