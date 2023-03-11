

import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier

class Chunk_pipeline:

    def __init__(self, filename: str, label: str, 
                 numeric_signals: list, cat_signals: list,
                 chunksize: int):
        self.filename = filename
        self.label = label
        self.numeric_signals = numeric_signals
        self.cat_signals = cat_signals
        self.keep_cols = [self.label] + self.numeric_signals + self.cat_signals
        self.chunksize = chunksize
    
    def get_chunks(self):
        return pd.read_csv(self.filename, 
                            chunksize=self.chunksize,
                            usecols = self.keep_cols)
    
    @profile
    def train_model(self):
        
        data_chunks = self.get_chunks()
      
        sgd_model = SGDClassifier(random_state = 0,
                                  loss = "log")
        
        
        keep_fields = []
        for frame in tqdm(data_chunks):
    
            features = pd.concat([frame[self.numeric_signals], 
                                  pd.get_dummies(frame[self.cat_signals])],
                                 axis = 1)
            
            
            if not keep_fields:
                
                for signal in self.cat_signals:
                    top_categories = set(frame[signal].value_counts().\
                                      head().index.tolist())
                
                
                    fields = [field for field in features.columns.tolist() 
                              if field in top_categories]
                
                    
                    keep_fields.extend(fields)
                
                keep_fields.extend(self.numeric_signals)
            
            
            sgd_model.partial_fit(features[keep_fields], 
                                  frame[self.label], 
                                  classes = (0, 1))
    
    
        self.sgd_model = sgd_model

        
        
        
pipeline = Chunk_pipeline(filename = "data/ads_train_data.csv",
                          label = "click",
                          numeric_signals=["banner_pos"],
                          cat_signals=["site_category"],
                          chunksize=1000000)

pipeline.train_model()


