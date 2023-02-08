
import pandas as pd
import time


artists = pd.read_csv("data/artists.csv")


start = time.time()
artists["high_followers_has_genre"] = artists.apply(lambda df: 1 if df.followers > 10 and len(df.genres) > 0 else 0 , axis = 1)
end = time.time()
print(end - start)



start = time.time()
artists["high_followers_has_genre"] = [1 if followers > 10 and len(genres) > 0 else 0
                                       for followers, genres in 
                                       zip(artists.followers, artists.genres)]
end = time.time()
print(end - start)



import numpy as np

def create_follower_genre_feature(followers, genres):
    
    return 1 if followers > 10 and len(genres) > 0 else 0
    

vec_create_follower_genre_feature = np.vectorize(create_follower_genre_feature)


start = time.time()
artists["high_followers_has_genre"] = vec_create_follower_genre_feature(artists.followers, artists.genres)
end = time.time()
print(end - start)



start = time.time()
check = ["Ace", "Rico", "Luna"]
artists[artists.name.isin(check)]
end = time.time()
print(end - start)



start = time.time()
check = {"Ace", "Rico", "Luna"}
artists[artists.name.isin(check)]
end = time.time()
print(end - start)




