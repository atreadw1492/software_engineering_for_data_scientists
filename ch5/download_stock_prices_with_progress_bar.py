


from yahoo_fin import stock_info as si
from tqdm import tqdm

tickers = ["goog", "aapl", "meta", "nflx", "amzn"]

stock_prices = {}
failures = []
for ticker in tqdm(tickers):
    try:
        stock_prices[ticker] = si.get_data(ticker)
    except Exception:
        failures.append(ticker)