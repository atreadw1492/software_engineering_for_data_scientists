
from tqdm import tqdm
from yahoo_fin import stock_info as si


tickers = ["amzn", "meta", "goog",
                "not_a_real_ticker", "msft", "aapl", "nflx"]

all_data = {}
failures = []


for ticker in tqdm(tickers):
    try:
        all_data[ticker] = si.get_data(ticker)

    except Exception:
        failures.append(ticker)
        print(ticker)