import pandas as pd
import numpy as np
import gymnasium as gym
from finrl.meta.env_fx_trading.env_fx_portfolio import FXPortfolioEnv
from finrl.agents.fx_agent import FXAgent
import os

def train():
    # Load Data
    df = pd.read_parquet('data/fx_data_2021_2022.parquet')
    # Reset index to ensure 'date' is a column
    df = df.reset_index()
    
    # Configuration
    # Ensure columns match what env expects
    # Env expects 'date', 'tic' and technicals
    
    # Configuration
    stock_dim = len(df['tic'].unique())
    lookback = 10
    tech_indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30']
    
    # State space = stock_dim * lookback * n_features
    state_space = stock_dim * lookback * len(tech_indicators)
    print(f"Stock Dim: {stock_dim}, State Space: {state_space}")
    
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
        "print_verbosity": 1
    }
    
    # Create Environment
    env = FXPortfolioEnv(df=df, **env_kwargs)
    
    # Create Agent
    agent = FXAgent(env=env)
    
    # Train
    print("Starting training...")
    agent.train(total_timesteps=5000)
    print("Training finished.")
    
    # Save
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    agent.save(f"{models_dir}/fx_agent_mvp")
    print(f"Model saved to {models_dir}/fx_agent_mvp")

if __name__ == "__main__":
    train()
