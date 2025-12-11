import pandas as pd
import numpy as np
import os

def main():
    # Create a deterministic dataset for testing
    # 2 Tickers, 10 Days
    dates = pd.date_range(start='2021-01-01', periods=10, freq='D')
    
    # Ticker A: Linear growth
    prices_a = np.linspace(100, 110, 10)
    
    # Ticker B: Linear decline
    prices_b = np.linspace(100, 90, 10)
    
    data = {
        'date': list(dates) * 2,
        'tic': ['TICKER_A'] * 10 + ['TICKER_B'] * 10,
        'close': np.concatenate([prices_a, prices_b]),
        'open': np.concatenate([prices_a, prices_b]), # Simplify: open=close
        'high': np.concatenate([prices_a + 1, prices_b + 1]),
        'low': np.concatenate([prices_a - 1, prices_b - 1]),
        'volume': [1000] * 20,
        # Log returns approx
        'log_return': [0.0] * 20 # Fill with dummy, tests should calculate or we pre-calc
    }
    
    # Pre-calc log returns properly for testing "state"
    # log(p_t / p_{t-1})
    log_ret_a = np.diff(np.log(prices_a), prepend=np.log(prices_a[0]))
    log_ret_b = np.diff(np.log(prices_b), prepend=np.log(prices_b[0]))
    
    data['log_return'] = np.concatenate([log_ret_a, log_ret_b])
    
    df = pd.DataFrame(data)
    
    # Ensure sorted by date then ticker as is standard in FinRL
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    output_path = 'tests/fixtures/fx_test_data.parquet'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving fixture to {output_path}")
    print(df.head())
    
    df.to_parquet(output_path)

if __name__ == "__main__":
    main()
