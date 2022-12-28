

from yahoo_fin import stock_info as si

amzn_data = si.get_data("AMZN")

print(amzn_data.head())