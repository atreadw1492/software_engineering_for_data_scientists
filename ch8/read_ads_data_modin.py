
import modin.pandas as pd
import time

start = time.time()
ad_frame = pd.read_csv("data/ads_train_data.csv")
end = time.time()
print("Read data in ", end - start, " seconds")