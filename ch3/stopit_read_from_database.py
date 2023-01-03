
import stopit
import pandas as pd
import pyodbc


conn = pyodbc.connect("DRIVER={SQLite3 ODBC Driver};SERVER=localhost;" \
                      "DATABASE=customers.db;Trusted_connection=yes")

with stopit.ThreadingTimeout(300) as context_manager:

    customer_key_features = pd.read_sql(conn,
                                        """SELECT churn,
                                                  total_day_charge,
                                                  total_intl_minutes,
                                                  number_customer_service_calls
                                           FROM customer_data""")


# Did code finish running in under 300 seconds (5 minutes)?
if context_manager.state == context_manager.EXECUTED:
    print("FINISHED READING DATA...")

# Did code timeout?
elif context_manager.state == context_manager.TIMED_OUT:
    raise Exception("DID NOT FINISH READING DATA WITHIN TIME LIMIT")
