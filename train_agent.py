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
    
    # Define environment parameters
    stock_dim = len(df['tic'].unique())
    state_space = stock_dim # Simplified state space
    
    # We need to refine state space calculation. 
    # In Phase 2 we defaulted to returning zeros based on 'state_space' int.
    # To make this useful, we should ideally have a real state.
    # But for MVP integration test, as long as shapes match, it runs.
    
    # For MVP, let's fix state_space to match what PPO expects (MlpPolicy).
    # If using MlpPolicy, observation can be Flat Box.
    # Let's say we pass state_space = stock_dim * 10 (just a guess for now)
    # The Env returns np.zeros(state_space).
    
    env_kwargs = {
        "stock_dim": stock_dim,
        "hmax": 100,
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.001,
        "reward_scaling": 1e-4,
        "state_space": stock_dim * 10, # Dummy size
        "action_space": stock_dim,
        "tech_indicator_list": ['macd', 'rsi_30'], # Dummy list
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
