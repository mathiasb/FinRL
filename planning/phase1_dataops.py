import pandas as pd
import numpy as np
import os
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor

# Configuration
TRAIN_START_DATE = '2021-01-01'
TRAIN_END_DATE = '2022-12-31'
TIME_INTERVAL = '1D'
# Yahoo Finance Tickers:
# EURUSD=X: Euro / US Dollar
# GBPUSD=X: British Pound / US Dollar
# JPY=X: US Dollar / Japanese Yen (Note: Inverted quote compared to EUR/GBP)
# SEK=X: US Dollar / Swedish Krona (Note: Inverted quote)
TICKER_LIST = ['EURUSD=X', 'GBPUSD=X', 'JPY=X', 'SEK=X']

TECHNICAL_INDICATORS_LIST = [
    "macd",
    "rsi_30",
    "cci_30",
    "dx_30",
    "boll_ub",
    "boll_lb"
]

def main():
    print(f"Starting DataOps Phase 1...")
    print(f"Tickers: {TICKER_LIST}")
    print(f"Range: {TRAIN_START_DATE} to {TRAIN_END_DATE}")

    # 1. Download Data
    dp = YahooFinanceProcessor()
    df = dp.download_data(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=TICKER_LIST,
        time_interval=TIME_INTERVAL
    )
    
    print(f"Data Downloaded. Shape: {df.shape}")
    print(df.head())

    # 2. Clean Data
    df = dp.clean_data(df)
    print(f"Data Cleaned. Shape: {df.shape}")

    # 3. Feature Engineering
    print("Adding Technical Indicators...")
    df = dp.add_technical_indicator(df, TECHNICAL_INDICATORS_LIST)
    print(f"Features Added. Shape: {df.shape}")
    print(df.head())

    # 4. Normalization (Log Returns)
    # We calculate log returns for the 'close' price to make the data stationary
    # However, for the environment, we usually feed the raw features + returns.
    # The PRD requested "Returns-based normalization".
    # We will add a 'log_return' column.
    
    # Sort to ensure shift works correctly
    df = df.sort_values(['tic', 'timestamp'])
    
    # Calculate Log Returns per ticker
    df['log_return'] = df.groupby('tic')['close'].apply(
        lambda x: np.log(x / x.shift(1))
    ).reset_index(level=0, drop=True)

    # Drop the first row of each ticker which will be NaN
    df = df.dropna()
    print(f"NaNs dropped. Final Shape: {df.shape}")

    # 5. Save to Parquet
    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'fx_data_2021_2022.parquet')
    df.to_parquet(output_path)
    print(f"Data saved to {output_path}")

    # Verification
    print("\nVerification:")
    print(df.groupby('tic')['timestamp'].count())

if __name__ == "__main__":
    main()
