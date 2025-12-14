import pandas as pd
import numpy as np
import gymnasium as gym
from finrl.meta.env_fx_trading.env_fx_portfolio import FXPortfolioEnv
from finrl.agents.fx_agent import FXAgent
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os
import argparse

def train(total_timesteps=30000, learning_rate=2.27e-5, batch_size=256, n_steps=2048, lookback=10, ent_coef=0.00015, device='auto'):
    # Load Data
    df = pd.read_parquet('data/fx_data_2018_2023.parquet')
    # Reset index to ensure 'date' is a column
    df = df.reset_index()
    
    # Configuration
    # Ensure columns match what env expects
    # Env expects 'date', 'tic' and technicals
    
    # Configuration
    stock_dim = len(df['tic'].unique())
    # lookback = 10 # NOW USING ARGUMENT
    tech_indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30', 'boll_ub', 'boll_lb', 'atr', 'adx', 'wr', 'us_roi']
    
    # State space = stock_dim * lookback * n_features
    state_space = stock_dim * lookback * len(tech_indicators)
    print(f"Stock Dim: {stock_dim}, State Space: {state_space}, Lookback: {lookback}")
    
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
    
    # Wrap in Monitor for logging
    log_dir = "results"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    
    # 4. Initialize Agent (Tuned Phase 7)
    # Best Params from Optuna (Phase 7 w/ Macro Features)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=0.914,
        ent_coef=ent_coef,
        device=device
    )
    
    # 5. Train
    print(f"Training Agent for {total_timesteps} steps | LR={learning_rate} | Ent Coef={ent_coef} | Lookback={lookback} | Device={device}...")
    model.learn(total_timesteps=total_timesteps) # Increased for larger n_steps
    
    # 6. Save
    models_dir = "models" # Keep models_dir for consistency, though hardcoded path is used
    os.makedirs(models_dir, exist_ok=True)
    model.save("models/fx_agent_tuned")
    print("Training finished.")
    print("Model saved to models/fx_agent_tuned")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train FX Agent')
    parser.add_argument('--timesteps', type=int, default=30000, help='Total training timesteps')
    parser.add_argument('--lr', type=float, default=2.27e-5, help='Learning Rate')
    parser.add_argument('--lookback', type=int, default=10, help='Lookback window size')
    parser.add_argument('--ent-coef', type=float, default=0.00015, help='Entropy coefficient (Exploration)')
    parser.add_argument('--device', type=str, default='auto', help='Device to run on (auto, cuda, cpu, mps)')
    
    args = parser.parse_args()
    
    train(total_timesteps=args.timesteps, learning_rate=args.lr, lookback=args.lookback, ent_coef=args.ent_coef, device=args.device)
