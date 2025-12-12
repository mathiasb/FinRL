import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from dataops.pipeline import FXDataPipeline

class TestFXDataPipeline(unittest.TestCase):
    def setUp(self):
        self.tickers = ['EURUSD=X', 'GBPUSD=X']
        self.start_date = '2021-01-01'
        self.end_date = '2021-01-05'
        self.interval = '1d'
        self.pipeline = FXDataPipeline()

    @patch('yfinance.download')
    def test_fetch_data_calls_yfinance_correctly(self, mock_download):
        # Setup mock return value
        mock_df = pd.DataFrame({
            'Open': [1.1, 1.2],
            'High': [1.3, 1.4],
            'Low': [1.0, 1.1],
            'Close': [1.2, 1.3],
            'Volume': [1000, 2000]
        })
        # yfinance returns a MultiIndex columns if multiple tickers, but we generally want to test standard behavior first.
        # However, for simplicity in MVP, let's assume yfinance returns a DataFrame.
        # We need to verify how FinRL expects it, but for our custom pipeline we can define our own contract.
        
        mock_download.return_value = mock_df

        df = self.pipeline.fetch_data(self.tickers, self.start_date, self.end_date, self.interval)

        # Assert yfinance was called with correct arguments
        mock_download.assert_called_with(
            self.tickers, 
            start=self.start_date, 
            end=self.end_date, 
            interval=self.interval
        )
        
        # Assert return value is what we expect
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertEqual(len(df), 2)

    def test_clean_data_handles_nans(self):
        # Create a DataFrame with NaNs
        df_with_nans = pd.DataFrame({
            'Open': [1.0, float('nan'), 1.2],
            'Close': [1.1, 1.1, 1.3],
            'Volume': [100, 100, 100]
        })
        
        cleaned_df = self.pipeline.clean_data(df_with_nans)
        
        # Expectation: NaNs should be filled (ffill) or dropped depending on strategy.
        # For FX, ffill is often safer than dropping if gaps are small.
        # Let's assume we want to ffill first, then drop any remaining (leading) NaNs.
        
        self.assertFalse(cleaned_df.isnull().values.any())
        self.assertEqual(len(cleaned_df), 3) # Should keep all rows if ffill works
        self.assertEqual(cleaned_df.iloc[1]['Open'], 1.0) # Forward filled from index 0

    def test_add_features_adds_columns(self):
        # Create a simple DataFrame locally
        df = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104] * 10,
            'Close': [100, 101, 102, 103, 104] * 10,
            'High': [100, 101, 102, 103, 104] * 10,
            'Low': [100, 101, 102, 103, 104] * 10,
            'Volume': [1000] * 50
        })
        
        # Test adding MACD
        df_with_features = self.pipeline.add_features(df, ['macd'])
        
        self.assertIn('macd', df_with_features.columns)
        self.assertIn('macds', df_with_features.columns) # Signal line usually added by stockstats
        self.assertIn('macdh', df_with_features.columns) # Histogram usually added by stockstats

    def test_normalize_computes_log_returns(self):
        df = pd.DataFrame({
            'close': [100.0, 110.0, 121.0]
        })
        
        # ln(110/100) = ln(1.1) approx 0.0953
        # ln(121/110) = ln(1.1) approx 0.0953
        
        normalized_df = self.pipeline.normalize_data(df)
        
        self.assertIn('log_return', normalized_df.columns)
        self.assertAlmostEqual(normalized_df.iloc[1]['log_return'], 0.09531, places=5)
        # First row usually NaN after diff/shift, ensure it is handled (dropped or kept as NaN)
        self.assertEqual(len(normalized_df), 2)

    def test_add_features_handles_multiple_tickers(self):
        # Create a DataFrame with two tickers
        # Need enough points for MACD (usually > 26)
        n = 50
        dates = pd.date_range(start='2021-01-01', periods=n)
        # Use varying prices
        import numpy as np
        prices1 = 100 + np.cumsum(np.random.randn(n))
        prices2 = 200 + np.cumsum(np.random.randn(n))
        
        data = {
            'open': list(prices1) + list(prices2),
            'close': list(prices1) + list(prices2),
            'high': list(prices1 + 1) + list(prices2 + 1),
            'low': list(prices1 - 1) + list(prices2 - 1),
            'volume': [1000] * (2*n),
            'tic': ['AAPL'] * n + ['GOOG'] * n
        }
        # Repeat dates for each ticker
        all_dates = list(dates) + list(dates)
        
        df = pd.DataFrame(data)
        df['date'] = all_dates
        
        # This implementation should fail if add_features doesn't handle grouping or duplicate dates
        df_with_features = self.pipeline.add_features(df, ['macd'])
        
        print(df_with_features[['tic', 'macd']].head())
        print(df_with_features[['tic', 'macd']].tail())
        
        self.assertIn('macd', df_with_features.columns)
        # Verify MACD was calculated (not all NaNs)
        self.assertFalse(df_with_features['macd'].isnull().all())

    def test_add_features_phase7_indicators(self):
        # Create Dummy Data sufficient for ATR/ADX/WR
        n = 50
        dates = pd.date_range(start='2021-01-01', periods=n)
        import numpy as np
        prices = 100 + np.cumsum(np.random.randn(n))
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices + 1,
            'low': prices - 1,
            'close': prices,
            'volume': 1000,
            'tic': 'EURUSD',
            'date': dates
        })
        
        # Test adding Phase 7 features
        features = ['atr', 'adx', 'wr']
        df_out = self.pipeline.add_features(df, features)
        
        self.assertIn('atr', df_out.columns)
        self.assertIn('adx', df_out.columns)
        self.assertIn('wr', df_out.columns) # Williams %R
        
        # Verify values are populated (not all NaN)
        # Note: First few rows might be NaN due to windowing, but last rows should be valid
        self.assertFalse(df_out['atr'].iloc[-1] is None)
        self.assertFalse(np.isnan(df_out['adx'].iloc[-1]))

    @patch('yfinance.download')
    def test_merge_macro_data_merges_and_ffills(self, mock_download):
        # Setup FX DataFrame (24/5 days)
        # Dates: Mon, Tue (Bond Hol), Wed
        dates = pd.to_datetime(['2021-10-11', '2021-10-12', '2021-10-13']) 
        fx_df = pd.DataFrame({
            'close': [1.1, 1.1, 1.1],
            'tic': 'EURUSD'
        }, index=dates)
        fx_df['date'] = dates 
        
        # Setup Macro Data (Bond Holiday on 12th)
        # Dates: Mon, Wed
        macro_dates = pd.to_datetime(['2021-10-11', '2021-10-13'])
        macro_df = pd.DataFrame({
            'Close': [1.5, 1.6]
        }, index=macro_dates)
        
        mock_download.return_value = macro_df
        
        # Call merge_macro_data 
        if hasattr(self.pipeline, 'merge_macro_data'):
            merged_df = self.pipeline.merge_macro_data(fx_df, macro_ticker='^TNX', start_date='2021-01-01', end_date='2021-01-01')
            
            self.assertIn('us_roi', merged_df.columns) 
            
            # Check 12th (Bond Holiday) is filled
            # 11th = 1.5 -> 12th should be 1.5 (ffill) -> 13th = 1.6
            val_12th = merged_df.loc[merged_df['date'] == pd.Timestamp('2021-10-12')]['us_roi'].values[0]
            self.assertEqual(val_12th, 1.5)
        else:
            self.fail("merge_macro_data not implemented yet")


