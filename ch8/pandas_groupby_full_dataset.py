
import pandas as pd
import time


start = time.time()
ad_frame = pd.read_csv("data/ads_train_data.csv", 
                      usecols=("click", "banner_pos", "site_category"))


print(ad_frame[["banner_pos", "site_category"]].groupby("site_category").sum())

end = time.time()

print("Time to execute code: ", end - start)