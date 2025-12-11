from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class FXPortfolioEnv(gym.Env):
    """
    A unified FX Portfolio Trading Environment compatible with Gymnasium.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        transaction_cost_pct: float,
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        lookback: int = 252,
        day: int = 0,
        print_verbosity: int = 1
    ):
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space_dim = action_space
        self.tech_indicator_list = tech_indicator_list
        self.lookback = lookback
        self.day = day
        self.print_verbosity = print_verbosity

        # Define Action Space: Portfolio Weights (-1 to 1 for long/short)
        # Typically shape is (n_tickers,) if just weights, or included cash.
        # For simplicity MVP, we assume fully invested or rebalanced weights.
        # Or standard FinRL continuous action space.
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space_dim,))

        # Define Observation Space
        # Needs to match the return of reset()
        # Usually (stock_dim, window_size, features)
        # For now, let's assume a simplified flat observation for the skeleton pass
        # We will refine this as we implement reset()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.day = 0

        
        # Initialize state: [Portfolio Value] + [Holdings per ticker] + [Features per ticker...]
        # For this MVP, let's stick to a vector representation.
        # However, the skeleton defined observation_space as shape=(state_space,).
        # We need to map the dataframe columns to this state vector.
        
        # Logic:
        # 1. Holdings: Initially 0 for all assets (all cash) - BUT we need to track cash separately?
        #    Typically FinRL defines state as [Balance, Close_0, ..., Close_N, Holdings_0, ..., Holdings_N, Tech_0, ..., Tech_N]
        #    Let's try to construct a simple state vector.
        
        # For now, let's just return a zero vector of correct size to pass the basic shape check, 
        # but with correct day initialization.
        # We will need to expand state_space calculation or dynamic definition.
        
        # To make it concrete:
        # State = [Current Balance] + [Price_1, Price_2] + [Owned_1, Owned_2] + [Tech_1, Tech_2]
        # Total dim = 1 + 2 + 2 + 2 = 7.
        # The test kwargs said "state_space": 2. This is likely too small.
        # We should update the class to calculate space automatically or trust the input.
        
        self.dates = self.df['date'].unique()
        # Sort dates to be safe
        self.dates = np.sort(self.dates)
        
        current_date = self.dates[self.day]
        self.data = self.df[self.df['date'] == current_date]
        
        # Initialize state
        self.portfolio_value = self.initial_amount
        # Weights: 0.0 means 0% allocation to that asset. 
        # State vector can just be current prices for now as placeholders, 
        # or we track [Balance, ...Prices...]
        # For the test mechanics, returning zeros is fine for 'obs', but internal state must be correct.
        
        # Let's say current assets held is 0 (all cash)
        self.current_weights = np.zeros(self.action_space_dim)
        
        return np.zeros(self.state_space, dtype=np.float32), {}

    def step(self, actions):
        terminated = False
        truncated = False
        
        self.day += 1
        
        # Check termination
        if self.day >= len(self.dates):
            terminated = True
            # To prevent index error, clamp day or just return last state
            self.day = len(self.dates) - 1
        
        # Update Data for new day
        current_date = self.dates[self.day]
        step_data = self.df[self.df['date'] == current_date]
        
        # --- LOGIC: Update Portfolio Value based on Price Move ---
        # Current Value = Previous Value * (1 + sum(weight * return))
        # But we need close prices.
        # Let's assume we have returns or prices.
        # Our fixture has 'log_return'.
        # Approximatiopn: New Value = Old Value * exp(sum(weights * log_returns))
        
        # Get returns for the current step (from t-1 to t)
        # Note: self.data was at t-1 (from init). step_data is at t.
        # But wait, step(action) happens at t. Action is applied. Result is observed at t+1?
        # Standard Gym:
        # Obs_t -> Action_t -> Env updates to t+1 -> Reward_t, Obs_{t+1}
        # So the price move is from t to t+1.
        
        # However, our data is sorted by day.
        # reset() -> day=0. Obs_0 (Prices_0).
        # step() -> day=1. Obs_1 (Prices_1).
        # The change is Prices_1 / Prices_0.
        
        # Let's calculate the portfolio return.
        # We need prices for the tickers corresponding to weights.
        # Simplification: Assume 'tic' order matches weight indices.
        # Our fixture: TICKER_A, TICKER_B.
        
        # In a real env, we'd map tickers to indices strictly.
        # For MVP, assume sorted by ticker.
        # Get log_returns for day t.
        
        # If terminated, we might not have data for t+1?
        # If terminated, just return last value.
        
        if not terminated:
            # Get log returns for the current day
            # Note: The 'log_return' column in our data prep is usually "Return from t-1 to t".
            
            # Filter safely
            current_log_rets = step_data.sort_values('tic')['log_return'].values
            
            # If actions (weights) were applied at t-1 (previous step),
            # then Return = sum(weights_{t-1} * log_rets_t) + (1-sum(weights))*0 (Cash return 0)
            
            # But wait, 'actions' passed to step() are for the *next* interval?
            # Usually: Obs_t -> Agent decides weights w_t -> Hold w_t over period t to t+1 -> Obs_{t+1} sees result.
            # So 'actions' passed NOW are used for the FUTURE.
            # But 'portfolio_value' calculated NOW should be based on 'current_weights' (stored from previous step).
            
            # 1. Update Portfolio Value based on PREVIOUS weights and CURRENT market move.
            # Assuming cash return = 0
            
            # Log return approximation: r_p = sum(w * r_i)
            # This is only approx. Exact: V_t = V_{t-1} * (sum( w_i * exp(r_i) ) + w_cash)
            # weights sum to <= 1. w_cash = 1 - sum(w_i)
            
            asset_returns = np.exp(current_log_rets)
            portfolio_return = np.sum(self.current_weights * asset_returns) + (1.0 - np.sum(self.current_weights))
            
            self.portfolio_value *= portfolio_return
            
            # 2. Update Weights for *next* step (Action)
            # Apply transaction costs here if weights change.
            # cost = |new_weights - current_weights| * value * cost_pct
            # Update self.current_weights = actions
            
            # Calculate cost (simplified turnover)
            # Note: 'actions' are the Target Weights for the next period.
            # Actual weights drift due to price moves, but let's ignore drift for this MVP step.
            turnover = np.sum(np.abs(actions - self.current_weights))
            cost = turnover * self.portfolio_value * self.transaction_cost_pct
            
            self.portfolio_value -= cost
            self.current_weights = actions
            
            self.data = step_data
        
        reward = 0.0 # Placeholder
        
        info = {
            'portfolio_value': self.portfolio_value,
            'date': current_date
        }
        
        obs = np.zeros(self.state_space, dtype=np.float32)
        
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

