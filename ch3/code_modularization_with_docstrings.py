

import pandas as pd

def read_data(file_name: str) -> pd.DataFrame:
    
    """Inputs:
              file_name: str
       
       Takes a file name as input and reads in data from the file.  
       Returns a data frame (pd.DataFrame)
    """

    df = pd.read_csv(file_name)
    return df

def cap_outliers(feature:pd.Series, upper_percentile: float, 
                 lower_percentile: float) -> pd.Series:

    """Inputs:
              feature: pd.Series\n
              upper_percentile: float\n
              lower_percentile: float\n
              
       
       Takes a file name as input and reads in data from the file.  
       Returns a pandas Series (pd.Series)
    """
    
    lower_thresh = feature.quantile(lower_percentile)
    upper_thresh = feature.quantile(upper_percentile)

    feature = feature.map(lambda val: \
                          lower_thresh if val < lower_thresh \
                          else upper_thresh if val > upper_thresh \
                          else val)

    return feature


def clean_data(df: pd.DataFrame, upper_percentile: float, 
               lower_percentile: float) -> pd.DataFrame:

    """Inputs:
              df: pd.DataFrame\n
              upper_percentile: float\n
              lower_percentile: float\n
              
       
       Cleans an input data frame, including replacing missing values and
       capping outliers.
       
       Returns the processed data frame after cleaning.
    """
    
    
    df = df.fillna(df.median(numeric_only=True))

    for col in df.columns:
        if df.dtypes[col] in ("float64", "int64"):
            df[col] = cap_outliers(df[col], upper_percentile, lower_percentile)

    # additional cleaning...
    # [code block]

    return df

if __name__ == "__main__":

    customer_data = read_data("customer_data.csv")

    cleaned_data = clean_data(customer_data)

    # additional code
    # [code block]
