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
