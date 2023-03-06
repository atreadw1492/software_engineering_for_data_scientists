


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import time
from functools import lru_cache

artists = pd.read_csv("data/artists.csv")


artists = artists.dropna().reset_index(drop = True)

artists['num_genres'] = artists.genres.map(len)
features = ["followers", "num_genres"]


start = time.time()
forest_model = RandomForestRegressor(n_estimators = 300,
                                       min_samples_split = 40,
                                       max_depth = 3,
                                       verbose = 1,
                                       n_jobs = 3
                                       )

forest_model.fit(artists[features], artists.popularity)
end = time.time()
print("Model training runtime: ", end - start)


@lru_cache(maxsize=10)
def get_model_predictions(model, inputs):
    
    
    formatted_inputs = pd.DataFrame.from_dict({"followers": inputs[0], 
                            "num_genres": inputs[1]}, orient = "index").transpose()
    #pd.DataFrame(artists[features].iloc[10]).transpose()
    return model.predict(formatted_inputs)


inputs = (2, 4)

start = time.time()
get_model_predictions(forest_model, inputs)
end = time.time()
print("Model predictions runtime (w/o) cache: ", end - start)

start = time.time()
get_model_predictions(forest_model, inputs)
end = time.time()
print("Model predictions runtime (w/) cache: ", end - start)



