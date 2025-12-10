import pandas as pd
import numpy as np
from stockstats import StockDataFrame as Sdf

def main():
    n = 50
    # Use varying prices
    prices1 = 100 + np.cumsum(np.random.randn(n))
    
    data = {
        'open': list(prices1),
        'close': list(prices1),
        'high': list(prices1 + 1),
        'low': list(prices1 - 1),
        'volume': [1000] * n,
        'tic': ['AAPL'] * n
    }
    
    df = pd.DataFrame(data)
    print("DataFrame head:")
    print(df.head())
    
    stock = Sdf.retype(df.copy())
    print("\nStockDataFrame Wrapper created.")
    
    macd = stock['macd']
    print("\nMACD Head:")
    print(macd.head())
    print("\nMACD Tail:")
    print(macd.tail())
    
    print(f"\nAny non-NaN? {not macd.isnull().all()}")

if __name__ == "__main__":
    main()
