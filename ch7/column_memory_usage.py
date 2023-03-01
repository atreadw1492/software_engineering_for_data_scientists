
import pandas as pd
import numpy as np



ad_data = pd.read_csv("data/ads_train_data.csv",
                      chunksize=1000000)


ad_frame = ad_data.get_chunk()



ad_frame.banner_pos.memory_usage(deep = True)


ad_frame["banner_pos"] = ad_frame["banner_pos"].astype(np.int8)

ad_frame.C1.memory_usage(deep=True)



ad_frame.site_category.memory_usage(deep = True)

ad_frame["site_category"] = ad_frame.site_category.astype("category")

ad_frame.site_category.memory_usage(deep = True)