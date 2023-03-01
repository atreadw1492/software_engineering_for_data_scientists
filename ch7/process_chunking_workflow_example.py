
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier


def train_model():
    
    filename = "data/ads_train_data.csv"
    fields = ["click", "site_category", "banner_pos"]
    
    ad_data = pd.read_csv(filename, 
                           chunksize=1000000,
                           usecols = fields)
  
    sgd_model = SGDClassifier(random_state = 0,
                              loss = "log")
    
    
    fields = []
    for ad_frame in tqdm(ad_data):

        features = pd.concat([ad_frame["banner_pos"], 
                              pd.get_dummies(ad_frame.site_category,
                                             prefix = "site_category")],
                             axis = 1)
        
        
        if not fields:
            
            top_site_categories = set(ad_frame.site_category.value_counts().\
                                  head().index.tolist())
            
            
            fields = [field for field in features.columns.tolist() 
                      if "site_category" in field]
            fields = [field for field in fields 
                      if "site_category_" + field in top_site_categories]
            
            fields.append("banner_pos")
        
        
        sgd_model.partial_fit(features[fields], 
                              ad_frame.click, 
                              classes = (0, 1))


    return sgd_model


sgd_model = train_model()


