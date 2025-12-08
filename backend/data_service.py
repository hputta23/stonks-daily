import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(ticker: str, period: str = "2y"):
    """
    Fetches historical stock data for the given ticker.
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        period: Data period to fetch (default '2y' for sufficient training data)
    Returns:
        DataFrame with Date and Close price.
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Enforce minimum period of 6mo for models (need 60 days look_back)
        fetch_period = period
        if period in ["1mo", "2mo", "3mo"]:
            fetch_period = "6mo"
            
        # Fetch history
        hist = stock.history(period=fetch_period)
        
        if hist.empty:
            raise ValueError(f"No data found for ticker {ticker}")
            
        # Reset index to get Date as a column
        hist = hist.reset_index()
        
        # Keep relevant columns for visualization and modeling
        data = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Ensure Date is timezone naive or consistent
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
        
        # Add Technical Indicators
        data = add_technical_indicators(data)
        
        return data
    except Exception as e:
        raise ValueError(f"Error fetching data for {ticker}: {str(e)}")

def add_technical_indicators(data):
    """
    Adds RSI, SMA, and EMA to the dataframe.
    """
    df = data.copy()
    
    # SMA (Simple Moving Average)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # EMA (Exponential Moving Average)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    # EMA 12 is already calculated
    # EMA 26 is already calculated
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    # SMA 20 is already calculated
    std_dev = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (std_dev * 2)
    df['Lower_Band'] = df['SMA_20'] - (std_dev * 2)
    
    # Fill NaN values (resulting from rolling windows)
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

def get_current_price(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        # fast_info is often faster/more reliable for current price than history
        return stock.fast_info.last_price
    except:
        return None
