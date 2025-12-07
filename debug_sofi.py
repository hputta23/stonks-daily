import yfinance as yf

def test_ticker(ticker):
    print(f"Testing {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        if hist.empty:
            print(f"FAIL: No data for {ticker}")
        else:
            print(f"SUCCESS: Got {len(hist)} rows for {ticker}")
            print(hist.head())
    except Exception as e:
        print(f"ERROR: {e}")

test_ticker("AAPL")
test_ticker("SOFI")
