
from yahoo_fin import stock_info as si

all_data = {}
failures = {}


tickers = ["amzn", "meta", "goog",
           "not_a_real_ticker", "msft", "aapl", "nflx"]


for ticker in tickers:

    try:
        all_data[ticker] = si.get_data(ticker)
    except Exception as error:
        failures[ticker] = type(error)
    finally:
        print(ticker)