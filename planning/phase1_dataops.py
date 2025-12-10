import pandas as pd
import numpy as np
import os
import sys

# Add the root directory to sys.path to ensure we can import finrl
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor

def main():
    # 1. Define Tickers
    # tickers = ['EURUSD=X', 'GBPUSD=X', 'JPY=X', 'SEK=X']
    # NOTE: JPY=X in Yahoo Finance is typically USD/JPY. SEK=X is USD/SEK.
    tickers = ['EURUSD=X', 'GBPUSD=X', 'JPY=X', 'SEK=X']
    
    start_date = '2021-01-01'
    end_date = '2023-01-01' # Include 2022 fully
    time_interval = '1D'
    
    print(f"Fetching data for {tickers} from {start_date} to {end_date}...")

    # 2. Fetch Data
    p = YahooFinanceProcessor()
    
    # download_data returns the dataframe
    df = p.download_data(ticker_list=tickers, 
                         start_date=start_date, 
                         end_date=end_date, 
                         time_interval=time_interval)
    
    # clean_data takes df and returns df
    df = p.clean_data(df)
    
    # 3. Feature Engineering
    print("Adding technical indicators...")
    technical_indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30', 'boll_ub', 'boll_lb']
    
    # add_technical_indicator takes df
    df = p.add_technical_indicator(data=df, tech_indicator_list=technical_indicators)
    
    # p.dataframe is NOT used/stored in the processor class based on inspection, we use the local df variable
    # df = p.dataframe # REMOVED
    
    # 4. Normalization (Log Returns)
    print("Computing log returns and normalization...")
    
    # We need to compute log returns for the 'close' price.
    # However, the dataframe structure from FinRL usually has columns like 'tic', 'date', 'close', etc.
    # We need to process per ticker.
    
    normalized_dfs = []
    
    for ticker in tickers:
        temp_df = df[df['tic'] == ticker].copy()
        temp_df.sort_values('date', inplace=True)
        
        # Log Returns: ln(close_t / close_{t-1})
        # We can implement this as np.log(temp_df['close']) - np.log(temp_df['close'].shift(1))
        # equivalent to np.log(temp_df['close'] / temp_df['close'].shift(1))
        
        temp_df['log_return'] = np.log(temp_df['close'] / temp_df['close'].shift(1))
        
        # Drop NaNs created by shifting
        temp_df.dropna(inplace=True)
        
        normalized_dfs.append(temp_df)
    
    final_df = pd.concat(normalized_dfs)
    
    # 5. Save
    output_path = 'data/fx_data_2021_2022.parquet'
    print(f"Saving data to {output_path}...")
    final_df.to_parquet(output_path)
    
    print("DataOps pipeline completed successfully.")
    print(f"Shape: {final_df.shape}")
    print(final_df.head())

if __name__ == "__main__":
    main()
