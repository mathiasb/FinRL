import pandas as pd
import yfinance as yf
import numpy as np
from stockstats import StockDataFrame as Sdf

class FXDataPipeline:
    def __init__(self):
        pass

    def fetch_data(self, tickers: list[str], start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
        """
        Fetches data from Yahoo Finance using yfinance.download.
        """
        df = yf.download(tickers, start=start_date, end=end_date, interval=interval)
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the DataFrame by forward filling missing values and dropping remaining NaNs.
        """
        df = df.copy()
        df = df.ffill()
        df = df.dropna()
        return df

    def add_features(self, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        """
        Adds technical indicators to the DataFrame using stockstats.
        Handles both single-ticker and multi-ticker dataframes (requires 'tic' column for multi-ticker).
        """
        df = df.copy()
        # stockstats requires lowercase column names
        df.columns = df.columns.astype(str).str.lower()
        
        # If 'tic' column exists, process per ticker
        if 'tic' in df.columns:
            processed_dfs = []
            for tic in df['tic'].unique():
                tic_df = df[df['tic'] == tic].copy()
                
                # Sort by date if possible to ensure correct technical indicator calculation
                if 'date' in tic_df.columns:
                    tic_df = tic_df.sort_values('date')
                
                # StockDataFrame modifies in place or returns new wrapper
                stock = Sdf.retype(tic_df)
                
                for feature in features:
                    # Accessing the column triggers calculation
                    _ = stock[feature]
                    if feature == 'macd':
                        _ = stock['macds']
                        _ = stock['macdh']
                
                # Convert back to standard DataFrame to avoid Sdf metadata issues later
                processed_dfs.append(pd.DataFrame(stock))
            
            # Recombine
            df = pd.concat(processed_dfs)
        else:
            # Single ticker / no 'tic' column
            if 'date' in df.columns:
                df = df.sort_values('date')
                
            stock = Sdf.retype(df)
            for feature in features:
                _ = stock[feature]
                if feature == 'macd':
                    _ = stock['macds']
                    _ = stock['macdh']
            df = pd.DataFrame(stock)
            
        return df

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes log returns for the 'close' price and drops NaN values created by shifting.
        """
        df = df.copy()
        
        # Ensure we have a 'close' column (case insensitive check done in add_features, but strictly here we might want to check)
        # Assuming clean_data/add_features normalized columns to lowercase
        if 'close' in df.columns:
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        # If 'Close' (Cap) exists (if add_features wasn't called)
        elif 'Close' in df.columns:
            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
            
        df = df.dropna()
        return df

    def merge_macro_data(self, fx_df: pd.DataFrame, macro_ticker: str = '^TNX', start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetches macro data (e.g. Bond Yields), reindexes to match FX trading days, 
        and merges it as a new feature 'us_roi' (or similar).
        """
        fx_df = fx_df.copy()
        
        # Determine dates from fx_df if not provided
        if start_date is None:
            start_date = fx_df['date'].min().strftime('%Y-%m-%d')
        if end_date is None:
            end_date = fx_df['date'].max().strftime('%Y-%m-%d')
            
        # 1. Fetch Macro Data
        macro_df = self.fetch_data([macro_ticker], start_date, end_date)
        
        # 2. Process Macro Data
        # Keep only Close (Yield)
        # Handle MultiIndex if present (yfinance often returns (Price, Ticker))
        if isinstance(macro_df.columns, pd.MultiIndex):
             # Try to get Close level
             try:
                 macro_series = macro_df.xs('Close', level=0, axis=1) # Get Close for all tickers
                 # If we have multiple tickers or just one, it might be DF or Series
                 if isinstance(macro_series, pd.DataFrame):
                     macro_series = macro_series.iloc[:, 0] # Take first column (assuming 1 ticker)
             except KeyError:
                 # Try capitalized or lowercase
                 try:
                     macro_series = macro_df.xs('close', level=0, axis=1).iloc[:, 0]
                 except:   
                     raise ValueError(f"Could not find Close/close in MultiIndex columns: {macro_df.columns}")
        elif 'Close' in macro_df.columns:
            macro_series = macro_df['Close']
        elif 'close' in macro_df.columns:
            macro_series = macro_df['close']
        else:
            raise ValueError(f"Macro data for {macro_ticker} has no Close column. Cols: {macro_df.columns}")
            
        # Ensure it is a Series
        if isinstance(macro_series, pd.DataFrame):
            macro_series = macro_series.iloc[:, 0]

        macro_series.name = 'us_roi'
        
        # 3. Align Dates
        # Get unique FX dates
        if 'date' in fx_df.columns:
            unique_dates = fx_df['date'].unique()
        else:
            unique_dates = fx_df.index.unique()
            
        unique_dates = np.sort(pd.to_datetime(unique_dates))
        
        # Reindex Macro to match FX dates
        macro_aligned = macro_series.reindex(unique_dates)
        
        # 4. Fill Missing (Forward Fill then 0)
        macro_aligned = macro_aligned.ffill().fillna(0)
        
        # 5. Merge
        if isinstance(macro_aligned, pd.Series):
            macro_aligned_df = macro_aligned.to_frame()
        else:
            macro_aligned_df = macro_aligned
            
        macro_aligned_df.columns = ['us_roi'] # Force rename to ensure merge works
        macro_aligned_df.index.name = 'date'
        
        # Reset index to make 'date' a column if needed
        macro_aligned_df = macro_aligned_df.reset_index()
        
        if 'date' in fx_df.columns:
            # Ensure types match
            fx_df['date'] = pd.to_datetime(fx_df['date'])
            merged_df = pd.merge(fx_df, macro_aligned_df, on='date', how='left')
        else:
            # Merge on index
            merged_df = pd.merge(fx_df, macro_aligned_df, left_index=True, right_on='date', how='left').set_index('date')

        return merged_df
