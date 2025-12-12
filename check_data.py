import pandas as pd

def check_data():
    df = pd.read_parquet('data/fx_data_2018_2023.parquet')
    
    # Check tickers per date
    counts = df.groupby('date')['tic'].count()
    
    print(f"Total dates: {len(counts)}")
    print(f"Target tickers: 5")
    
    mismatch = counts[counts != 5]
    
    if len(mismatch) > 0:
        print(f"Found {len(mismatch)} dates with missing tickers!")
        print(mismatch.head())
        print("Example missing date data:")
        bad_date = mismatch.index[0]
        bad_date = mismatch.index[0]
        # Handle if date is index
        try:
             subset = df[df['date'] == bad_date]
        except KeyError:
             subset = df.loc[bad_date]
             
        print(subset)
    else:
        print("Data is square (all dates have 5 tickers).")

if __name__ == "__main__":
    check_data()
