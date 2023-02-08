

import pandas as pd
import time


artists = pd.read_csv("data/artists.csv")


start = time.time()
high_followers = artists.followers.quantile(.999)

print(artists[artists.followers > high_followers].name.tolist())

end = time.time()
print("\n",end - start, " seconds")
