import optuna
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import os

from finrl.meta.env_fx_trading.env_fx_portfolio import FXPortfolioEnv

def optimize_agent(trial):
    # 1. Hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.01)
    
    # 2. Data Setup
    # Load Data
    df = pd.read_parquet('data/fx_data_2018_2023.parquet')
    df = df.reset_index()
    df['date'] = pd.to_datetime(df['date'])
    
    # Split Train/Val
    train_df = df[df['date'] < '2022-01-01']
    val_df = df[df['date'] >= '2022-01-01']
    
    # Env Config
    stock_dim = len(train_df['tic'].unique())
    lookback = 10
    tech_indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30', 'boll_ub', 'boll_lb', 'atr', 'adx', 'wr', 'us_roi']
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
    
    # 3. Create Environment
    # Use valid split? For MVP tuning, we might just use the whole training set 
    # and evaluate on a hold-out or cross-val. 
    # Let's keep it simple: Train on 2021, Eval on 2022-Q1 (subset).
    
    # Simplified: Train on whole dataset for short period and check episodic reward stability
    env = FXPortfolioEnv(df=df, **env_kwargs)
    
    # 4. Define Agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        ent_coef=ent_coef,
        verbose=0
    )
    
    # 5. Train
    # Short training for speed in this demo
    try:
        model.learn(total_timesteps=5000)
    except Exception as e:
        # Prune invalid parameter combos that crash
        print(f"Trial failed: {e}")
        return -float('inf')
    
    # 6. Evaluate
    # Evaluate on the same env (or a validation env)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    
    return mean_reward

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    
    print("Starting Optimization...")
    # n_trials=10 is small for demo, normally 50-100
    study.optimize(optimize_agent, n_trials=10) 
    
    print("Best trials:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Optional: Save best params
    with open("best_params.txt", "w") as f:
        f.write(str(trial.params))
