

import pandas as pd
import numpy as np

ad_data = pd.read_csv("data/ads_train_data.csv",
                      chunksize=1000000, 
                      dtype = {"site_category": "category",
                               "C1": np.int32,
                               "C14": np.int32})


ad_frame = ad_data.get_chunk()


print(ad_frame.memory_usage(deep = True))