
import dask.dataframe as dd
import time
from dask.distributed import LocalCluster, Client

cluster = LocalCluster()
client = Client(cluster)

start = time.time()
ad_frame = dd.read_csv("data/ads_train_data.csv", 
                      usecols=("click", "banner_pos", "site_category"))



print(ad_frame[["click", "site_category"]].groupby("site_category").mean().compute())

end = time.time()

print("Time to execute code: ", end - start)