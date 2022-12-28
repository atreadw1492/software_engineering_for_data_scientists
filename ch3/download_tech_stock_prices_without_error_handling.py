
from yahoo_fin import stock_info as si

tickers = ["amzn", "meta", "goog",
           "not_a_real_ticker", "msft", "aapl", "nflx"]

all_data = {}
for ticker in tickers:
    all_data[ticker] = si.get_data(ticker)
