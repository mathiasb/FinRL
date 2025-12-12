import sys
import os
import pandas as pd

# Add root directory to path to import dataops
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataops.pipeline import FXDataPipeline

import argparse

def main(tickers=None, start_date='2018-01-01', end_date='2023-01-01'):
    pipeline = FXDataPipeline()
    
    if tickers is None:
        tickers = ['EURUSD=X', 'GBPUSD=X', 'JPY=X', 'SEK=X', 'EURSEK=X']
    
    interval = '1d'
    
    print(f"Fetching data from {start_date} to {end_date} for {tickers}...")
    df = pipeline.fetch_data(tickers, start_date, end_date, interval)
    print(f"Raw shape: {df.shape}")
    
    # ... (Rest of logic remains similar, ensuring 'start_date' and 'end_date' variables are used) ...
    # Wait, I need to preserve the inner logic but using the function arguments.
    # To reduce complexity for replace_file_content, I will replace the top block and ensure variables align.
    
    # yfinance multi-ticker download returns MultiIndex columns.
    if isinstance(df.columns, pd.MultiIndex):
        print("Detected MultiIndex columns, converting to long format...")
        df = df.stack(level=1).reset_index().rename(columns={'level_1': 'tic', 'Ticker': 'tic'})
        df.columns = df.columns.astype(str).str.lower()
        if 'date' not in df.columns and 'Date' in df.columns:
            df.rename(columns={'Date': 'date'}, inplace=True)
            
    print(f"Long format shape: {df.shape}")
    
    print("Cleaning data...")
    df = pipeline.clean_data(df)
    
    print("Adding features...")
    features = ['macd', 'rsi_30', 'cci_30', 'dx_30', 'boll_ub', 'boll_lb', 'atr', 'adx', 'wr']
    df = pipeline.add_features(df, features)
    
    print("Merging Macro Data (^TNX)...")
    df = pipeline.merge_macro_data(df, macro_ticker='^TNX', start_date=start_date, end_date=end_date)
    
    print("Normalizing data...")
    normalized_dfs = []
    for tic in df['tic'].unique():
        temp_df = df[df['tic'] == tic].copy()
        temp_df = pipeline.normalize_data(temp_df)
        normalized_dfs.append(temp_df)
    
    final_df = pd.concat(normalized_dfs)
    
    # Enforce Square Data
    param_tic_count = len(tickers)
    date_counts = final_df.index.value_counts()
    valid_dates = date_counts[date_counts == param_tic_count].index
    
    print(f"Filtering non-square data. Dropping {len(date_counts) - len(valid_dates)} dates.")
    final_df = final_df[final_df.index.isin(valid_dates)]
    
    final_df = final_df.sort_index()
    
    print(f"Final shape: {final_df.shape}")
    print(final_df.head())
    
    output_path = 'data/fx_data_2018_2023.parquet'
    final_df.to_parquet(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run FX Data Pipeline')
    parser.add_argument('--tickers', type=str, help='Comma-separated list of tickers', default='EURUSD=X,GBPUSD=X,JPY=X,SEK=X,EURSEK=X')
    parser.add_argument('--start', type=str, default='2018-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-01-01', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Parse tickers list
    ticker_list = [t.strip() for t in args.tickers.split(',')]
    
    main(tickers=ticker_list, start_date=args.start, end_date=args.end)
