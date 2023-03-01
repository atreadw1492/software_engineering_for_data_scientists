


import pandas as pd

ad_data = pd.read_csv("data/ads_train_data.csv", chunksize=1000000)

ad_frame = ad_data.get_chunk()


ad_frame['click_pos_interaction'] = ad_frame.click * ad_frame.banner_pos


print("Bytes used before transformation: ", 
      ad_frame.click_pos_interaction.memory_usage(deep = True))


ad_frame['click_pos_interaction'] = ad_frame.click_pos_interaction.\
                                    astype(pd.SparseDtype("float", 0))


print("Bytes after before transformation: ", 
      ad_frame.click_pos_interaction.memory_usage(deep = True))
                                    