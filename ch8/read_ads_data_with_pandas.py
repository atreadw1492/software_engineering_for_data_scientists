
import pandas as pd
import time
import dask.dataframe as dd
from dask.distributed import LocalCluster, Client

cluster = LocalCluster()
client = Client(cluster)

start = time.time()
ad_frame = pd.read_csv("data/ads_train_data.csv", 
                       usecols=("click", "banner_pos", "site_category"))
end = time.time()
print("Read data in ", end - start, " seconds")




start = time.time()
ad_frame = dd.read_parquet("data/ads_train_data.parquet", 
                       columns=("click", "banner_pos", "site_category"))
ad_frame[["click", "site_category"]].groupby("site_category").mean().compute()
end = time.time()
print("Read data in ", end - start, " seconds")


start = time.time()
ad_frame = pd.read_parquet("data/ads_train_data.parquet", 
                       columns=("click", "banner_pos", "site_category"))
ad_frame[["click", "site_category"]].groupby("site_category").mean()
end = time.time()
print("Read data in ", end - start, " seconds")





start = time.time()
ad_frame = pd.read_parquet("data/ads_train_data.parquet", 
                       columns=("click", "banner_pos", "site_category"))
end = time.time()
print("Read data in ", end - start, " seconds")



