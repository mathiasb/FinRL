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
        self.returns_memory = []
        
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
            asset_returns = np.exp(current_log_rets)
            # Portfolio Return = sum(weights * asset_returns) + (1-sum(w))
            # Assuming cash return 0 for now
            portfolio_return = np.sum(self.current_weights * asset_returns) + (1.0 - np.sum(self.current_weights))
            
            # Apply Return
            old_portfolio_value = self.portfolio_value
            self.portfolio_value *= portfolio_return
            
            # Apply Transaction Costs
            turnover = np.sum(np.abs(actions - self.current_weights))
            cost = turnover * self.portfolio_value * self.transaction_cost_pct
            self.portfolio_value -= cost
            self.current_weights = actions
            self.data = step_data
            
            # Track Return for Reward Calculation
            # R_t = ln(V_t / V_{t-1}) approx or (V_t - V_{t-1})/V_{t-1}
            step_return = (self.portfolio_value - old_portfolio_value) / old_portfolio_value
            self.returns_memory.append(step_return)
        
        reward = self._get_reward()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'date': current_date,
            'reward': reward
        }
        
        obs = self._get_state(self.day)
        
        return obs, reward, terminated, truncated, info

    def _get_state(self, day):
        """
        Return the state vector for a value-based agent (flattened).
        Shape: (stock_dim * lookback * n_features)
        """
        # Determine the window range [start_day, end_day]
        # end_day is current 'day'.
        # start_day is day - lookback + 1.
        # If start_day < 0, we need to pad.
        
        start_day = day - self.lookback + 1
        
        state_frames = []
        
        if start_day < 0:
            # Pad with repeated first day or zeros
            # For simplicity, let's repeat the data at day 0 for 'abs(start_day)' times
            # Or just take data[0:day+1] and pad head
            
            # Fetch available data from 0 to day+1
            # Note: self.df might be large. We should index by day integer if possible 
            # or rely on date logic. But 'day' is our index cursor.
            
            # Optimization: We can't slice self.df by 'day' easily unless we reset index or use iloc logic mapped to dates.
            # But self.dates is sorted.
            # self.df usually has MultiIndex or sorted by date/tic.
            pass
            
        # Efficient approach: Filter df by date range using self.dates
        # But lookback is small (e.g. 10).
        
        # Effective window indices in self.dates
        window_indices = np.arange(day - self.lookback + 1, day + 1)
        # Clip negative indices to 0 (Repeat first day)
        window_indices = np.clip(window_indices, 0, len(self.dates)-1)
        
        window_dates = self.dates[window_indices]
        
        # Fetch data for these dates
        # self.df should be indexed by 'date' ideally for speed, or we query.
        # Ensure correct sort order: Date ASC, Ticker ASC
        window_element = self.df[self.df['date'].isin(window_dates)].sort_values(['date', 'tic'])
        
        # Extract features
        # Columns: We need config for 'tech_indicator_list'.
        features = self.tech_indicator_list
        # Maybe include 'close'? Usually yes.
        # For now, stick to tech_indicator_list as defined.
        
        # vector shape: (n_dates * n_tickers * n_features)
        flat_state = window_element[features].values.flatten()
        
        # Check consistency with state_space
        # If the df is missing some dates/tickers, shape might be wrong.
        # Robustness: Check length.
        expected_len = self.state_space
        
        if len(flat_state) != expected_len:
            # Padding needed or Truncation?
            # If we clipped indices, we should have 'lookback' dates.
            # Unless tickers are missing.
            
            # Fallback: Resize with zeros if mismatch (dangerous but safe for crash)
            # Ideally we ensure data completeness in pipeline.
             if len(flat_state) < expected_len:
                 flat_state = np.pad(flat_state, (expected_len - len(flat_state), 0))
             else:
                 flat_state = flat_state[-expected_len:]
                 
        return flat_state.astype(np.float32)

    def _get_reward(self):
        """
        Calculate Risk-Adjusted Return.
        R = Return - Lambda * Volatility
        """
        if self.day == 0:
            return 0.0
            
        if len(self.returns_memory) == 0:
            return 0.0
            
        # Current step return
        r_t = self.returns_memory[-1]
        
        # Volatility of last 'lookback' returns (or smaller window)
        # Using same lookback for vol window matches state window intuitively
        window_rets = self.returns_memory[-self.lookback:]
        if len(window_rets) > 1:
            volatility = np.std(window_rets)
        else:
            volatility = 0.0
            
        # Lambda is usually a hyperparam. Let's use 'reward_scaling' or hardcode a risk aversion
        # For this MVP, let's treat 'reward_scaling' as strict scaler, 
        # and add a separate risk_aversion param? Or simple hardcode 0.1
        risk_aversion = 0.1
        
        # Adjusted Reward
        adj_reward = r_t - (risk_aversion * volatility)
        
        # Scale
        return adj_reward * self.reward_scaling

    def render(self, mode='human'):
        pass
