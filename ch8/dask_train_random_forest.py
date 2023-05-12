
from dask.distributed import LocalCluster, Client
import dask.dataframe as dd
#import dask
#import coiled
from sklearn.ensemble import RandomForestClassifier
import joblib
import time

keep = ["dask", "coiled", "xarray", "pandas", "numpy", "sklearn", "joblib"]


cluster = coiled.Cluster(
    name="new",
    n_workers=8,
    worker_memory='8Gib',
    shutdown_on_close=False,
    package_sync = keep
)



cluster = LocalCluster()
client = Client(cluster)



ad_frame = dd.read_csv("data/ad_needed_columns.csv")
ad_frame = ad_frame.head(1000)

ad_frame["site_category_28905ebd"] = ad_frame.site_category.map(lambda val: 1 if val == "28905ebd" else 0)
ad_frame["site_category_50e219e0"] = ad_frame.site_category.map(lambda val: 1 if val == "50e219e0" else 0)

#ad_frame = dd.read_csv("s3://thisismynewbucket109/ad_needed_columns.csv")

inputs = ["banner_pos", "site_category_28905ebd", "site_category_50e219e0"]

start = time.time()
with joblib.parallel_backend("dask"):
    forest = RandomForestClassifier(n_estimators = 100,
                                    max_depth = 2,
                                    min_samples_split = 40,
                                    verbose = 1)
    forest.fit(ad_frame[inputs], ad_frame.click)
    end = time.time()
    
    
start = time.time()
with joblib.parallel_backend("dask"):
    forest2 = RandomForestClassifier(n_estimators = 100,
                                    max_depth = 2,
                                    min_samples_split = 40,
                                    verbose = 1)
    forest2.fit(ad_frame[["banner_pos", "site_category"]], ad_frame.click)
    end = time.time()
    




import dask.dataframe as dd
import time
from dask.distributed import LocalCluster, Client

cluster = LocalCluster()
client = Client(cluster)


ad_frame = dd.read_csv("ad_needed_columns.csv", 
                      usecols=("click", "banner_pos", "site_category"))


ad_frame = ad_frame.head(10000)

start = time.time()
with joblib.parallel_backend("dask"):
    forest = RandomForestClassifier(n_estimators = 100,
                                    max_depth = 2,
                                    min_samples_split = 40,
                                    verbose = 1)
    forest.fit(ad_frame[["banner_pos"]], ad_frame.click)
    end = time.time()





start = time.time()
forest = RandomForestClassifier(n_estimators = 100,
                                max_depth = 2,
                                min_samples_split = 40,
                                verbose = 1)
forest.fit(ad_frame[["banner_pos"]], ad_frame.click)
end = time.time()
print("Time to train model = ", end - start)






