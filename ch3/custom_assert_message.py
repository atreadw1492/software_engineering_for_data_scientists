
import pandas as pd

def process_data(conn):

    customer_key_features = pd.read_sql(conn,
                                        """SELECT churn,
                                                  total_day_charge,
                                                  total_intl_minutes,
                                                  number_customer_service_calls
                                           FROM customer_data""")

    # if zero rows are returned, raise an AssertionError
    # (this time with a custom error message)
    assert customer_key_features.shape[0] > 0, "Empty data returned"

    # [...additional code]
    return customer_key_features