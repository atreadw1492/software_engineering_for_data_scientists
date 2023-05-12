
import dask.dataframe as dd
import time

start = time.time()
ad_frame = dd.read_csv("data/ads_train_data.csv", 
                      usecols=("click", "banner_pos", "site_category"))


ad_frame = ad_frame[ad_frame.click == 1]

end = time.time()

print(end - start)