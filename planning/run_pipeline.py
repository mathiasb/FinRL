import sys
import os
import pandas as pd

# Add root directory to path to import dataops
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataops.pipeline import FXDataPipeline

def main():
    pipeline = FXDataPipeline()
    
    tickers = ['EURUSD=X', 'GBPUSD=X', 'JPY=X', 'SEK=X']
    start_date = '2021-01-01'
    end_date = '2023-01-01' # through 2022
    interval = '1d'
    
    print(f"Fetching data for {tickers}...")
    df = pipeline.fetch_data(tickers, start_date, end_date, interval)
    print(f"Raw shape: {df.shape}")
    
    # yfinance multi-ticker download returns MultiIndex columns.
    # We need to handle this.
    # The current pipeline.clean_data assumes a single level index or doesn't explicitly handle MultiIndex reshaping.
    # Let's inspect the DF structure first in a real run, but typically yfinance returns (Price, Ticker) levels.
    # We probably need to stack it to long format for FinRL.
    
    # However, our TDD was simple. Let's see what happens.
    # If the DF is MultiIndex, cleaning might fail or effectively do nothing if not iterating correctly.
    
    # NOTE: FinRL usually expects a long format with 'tic' column.
    # Let's add that logic here or in the pipeline if we discover we need it.
    # For now, let's just run it and see.
    
    # Transforming to long format if necessary
    if isinstance(df.columns, pd.MultiIndex):
        print("Detected MultiIndex columns, converting to long format...")
        df = df.stack(level=1).reset_index().rename(columns={'level_1': 'tic', 'Ticker': 'tic'})
        df.columns = df.columns.astype(str).str.lower()
        # Rename 'date' if needed, usually yfinance puts Date in index, reset_index puts it as 'Date'
        if 'date' not in df.columns and 'Date' in df.columns:
            df.rename(columns={'Date': 'date'}, inplace=True)
            
    print(f"Long format shape: {df.shape}")
    print(df.head())
    
    print("Cleaning data...")
    df = pipeline.clean_data(df)
    
    print("Adding features...")
    features = ['macd', 'rsi_30', 'cci_30', 'dx_30', 'boll_ub', 'boll_lb']
    df = pipeline.add_features(df, features)
    
    print("Normalizing data...")
    # Normalize per ticker
    normalized_dfs = []
    for tic in df['tic'].unique():
        temp_df = df[df['tic'] == tic].copy()
        temp_df = pipeline.normalize_data(temp_df)
        normalized_dfs.append(temp_df)
    
    final_df = pd.concat(normalized_dfs)
    
    print(f"Final shape: {final_df.shape}")
    print(final_df.head())
    
    output_path = 'data/fx_data_2021_2022.parquet'
    final_df.to_parquet(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
