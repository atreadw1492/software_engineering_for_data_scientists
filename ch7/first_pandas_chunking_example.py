


import pandas as pd
import time

start = time.time()
ad_data = pd.read_csv("/Users/amily/Downloads/avazu-ctr-prediction/train", chunksize=100)
end = time.time()
print(end - start)