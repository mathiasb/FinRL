import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from finrl.meta.env_fx_trading.env_fx_portfolio import FXPortfolioEnv
import sys
import os

def run_benchmark():
    # 1. Load Data
    # 1. Load Data
    data_path = 'data/fx_data_2018_2023.parquet'
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        return
        
    df = pd.read_parquet(data_path)
    # Ensure date is a column
    if 'date' not in df.columns:
        df = df.reset_index()
        
    df['date'] = pd.to_datetime(df['date'])
    
    # Validation Data (2022)
    validation_df = df[df['date'] >= '2022-01-01'].reset_index(drop=True)
    
    # 2. Setup Environment
    stock_dim = len(validation_df['tic'].unique())
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
    
    # 3. Initialize Env
    env = FXPortfolioEnv(df=validation_df, **env_kwargs)
    
    # 4. Load Agent
    # Prefer tuned model if exists, else MVP
    model_path = "models/fx_agent_tuned.zip"
    if not os.path.exists(model_path):
        model_path = "models/fx_agent_mvp.zip"
        
    try:
        model = PPO.load(model_path)
        print(f"Loaded model from {model_path}")
    except:
        print("Model not found. Please train agent first.")
        sys.exit(1)
        
    # 5. Run Evaluations
    print("Running Agent...")
    agent_returns = []
    obs, info = env.reset()
    done = False
    
    # Track portfolio values
    agent_portfolio_values = [env.initial_amount]
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        agent_portfolio_values.append(info['portfolio_value'])
        
    # 6. Run Baseline (Buy & Hold)
    print("Running Buy & Hold Baseline...")
    # B&H: Equal weight at start, held forever.
    # Logic: Reset env, set action to equal weights ONCE, then hold? 
    # Or just rebalance every step? "Buy and Hold" implies holding.
    # But simplified B&H in portfolio env: Rebalancing to equal weight daily = "Equal Weight Strategy".
    # True B&H: Buy at step 0, never trade again. 
    # Let's do Equal Weight Rebalancing for simplicity as a robust baseline.
    
    env.reset()
    done = False
    baseline_values = [env.initial_amount]
    
    equal_weights = np.ones(stock_dim) / stock_dim
    
    while not done:
        # Action: equal weights
        obs, reward, terminated, truncated, info = env.step(equal_weights)
        done = terminated or truncated
        baseline_values.append(info['portfolio_value'])

    # 7. Metrics Calculation
    def calculate_metrics(values, name):
        values = np.array(values)
        returns = (values[1:] - values[:-1]) / values[:-1]
        
        cum_return = (values[-1] - values[0]) / values[0]
        
        # Sharpe (Annualized, assuming daily data)
        # If intraday, scale factor changes. Assuming daily for standardized reporting.
        # Data is 1D? (From Phase 1 check: "Resolution: Daily").
        if np.std(returns) == 0:
            sharpe = 0
        else:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            
        # Max Drawdown
        cum_returns_series = (1 + returns).cumprod()
        peaks = np.maximum.accumulate(cum_returns_series)
        drawdowns = (cum_returns_series - peaks) / peaks
        max_dd = np.min(drawdowns)
        
        return {
            "Name": name,
            "Cum Return": f"{cum_return*100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd*100:.2f}%",
            "Final Value": f"{values[-1]:.2f}"
        }

    agent_metrics = calculate_metrics(agent_portfolio_values, "PPO Agent")
    baseline_metrics = calculate_metrics(baseline_values, "Equal Weight")
    
    # 8. Report
    results_df = pd.DataFrame([agent_metrics, baseline_metrics])
    print("\nBenchmark Results (2022 Validation):")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    run_benchmark()
