import pandas as pd
import numpy as np
import gymnasium as gym
from finrl.meta.env_fx_trading.env_fx_portfolio import FXPortfolioEnv
from finrl.agents.fx_agent import FXAgent
from stable_baselines3 import PPO
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
    
    # 4. Initialize Agent (Tuned)
    # Best Params from Optuna (Study: no-name-e205799b...)
    # Value: -2.32e-06
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=1.0189e-05,
        n_steps=512,
        batch_size=64,
        gamma=0.9259,
        ent_coef=0.0095
    )
    
    # 5. Train
    print("Training Agent...")
    model.learn(total_timesteps=5000) # Keep short for MVP, normally increase to 100k+
    
    # 6. Save
    models_dir = "models" # Keep models_dir for consistency, though hardcoded path is used
    os.makedirs(models_dir, exist_ok=True)
    model.save("models/fx_agent_tuned")
    print("Training finished.")
    print("Model saved to models/fx_agent_tuned")

if __name__ == "__main__":
    train()
