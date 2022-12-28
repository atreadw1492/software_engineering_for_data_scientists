
import pandas as pd

def process_data(conn):

    customer_key_features = pd.read_sql(conn,
                                        """SELECT churn,
                                                  total_day_charge,
                                                  total_intl_minutes,
                                                  number_customer_service_calls
                                           FROM customer_data""")

    assert customer_key_features.shape[0] > 0

    # [...additional code]
    
    return customer_key_features