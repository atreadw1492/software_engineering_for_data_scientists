
import pandas as pd

customer_data = pd.read_csv("train.csv")


customer_data["total_day_minutes"] = customer_data["total_day_minutes"].fillna(customer_data.total_day_minutes.median())

customer_data["total_day_calls"] = customer_data["total_day_calls"].\
                                fillna(customer_data.total_day_calls.median())

customer_data["total_day_charge"] = customer_data["total_day_charge"].\
                                  fillna(customer_data.total_day_charge.median())

customer_data["total_eve_minutes"] = customer_data["total_eve_minutes"].\
                                 fillna(customer_data.total_eve_minutes.median())


customer_data = customer_data.fillna(customer_data.median())
