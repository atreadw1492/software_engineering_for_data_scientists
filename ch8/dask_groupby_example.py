
import dask.dataframe as dd
import time
from dask.distributed import LocalCluster, Client

cluster = LocalCluster()
client = Client(cluster)

start = time.time()
ad_frame = dd.read_csv("data/ads_train_data.csv", 
                      usecols=("click", "banner_pos", "site_category"))


ad_frame = ad_frame[ad_frame.click == 1]


print(ad_frame[["banner_pos", "site_category"]].groupby("site_category").sum().compute())

end = time.time()

print("Time to execute code: ", end - start)