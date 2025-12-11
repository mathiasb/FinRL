import pandas as pd
import numpy as np
from finrl.meta.env_fx_trading.env_fx_portfolio import FXPortfolioEnv
from finrl.agents.fx_agent import FXAgent
import matplotlib.pyplot as plt

def backtest():
    # Load Data
    df = pd.read_parquet('data/fx_data_2021_2022.parquet')
    df = df.reset_index()
    
    # Filter for Backtest Period (e.g., 2022)
    # Assuming 'date' column is datetime
    df['date'] = pd.to_datetime(df['date'])
    test_df = df[df['date'] >= '2022-01-01'].reset_index(drop=True)
    
    # Configuration (Must match training env)
    stock_dim = len(test_df['tic'].unique())
    lookback = 10
    tech_indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30']
    state_space = stock_dim * lookback * len(tech_indicators)
    
    env_kwargs = {
        "stock_dim": stock_dim,
        "hmax": 100,
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.001,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": stock_dim,
        "tech_indicator_list": tech_indicators,
        "lookback": lookback,
        "print_verbosity": 0
    }
    
    # Create Environment
    env = FXPortfolioEnv(df=test_df, **env_kwargs)
    
    # Load Agent
    agent = FXAgent(env=env)
    agent.load("models/fx_agent_mvp")
    
    # Run Backtest
    print("Starting backtest on 2022 data...")
    obs, info = env.reset()
    done = False
    
    portfolio_values = []
    dates = []
    
    while not done:
        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        portfolio_values.append(info['portfolio_value'])
        dates.append(info['date'])
        
    print("Backtest finished.")
    
    # Analysis
    portfolio_values = np.array(portfolio_values)
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    # Sharpe Ratio (Assuming daily data, risk-free=0)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    
    print(f"Final Portfolio Value: {portfolio_values[-1]:,.2f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")

if __name__ == "__main__":
    backtest()
