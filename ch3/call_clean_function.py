

from code_modularization_example import clean_data
import pandas as pd

customer_data = pd.read_csv("../data/customer_churn_data.csv")

customer_data = clean_data(customer_data, 0.99, 0.01)