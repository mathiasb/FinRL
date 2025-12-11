import unittest
import pandas as pd
import numpy as np
import gymnasium as gym
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from finrl.meta.env_fx_trading.env_fx_portfolio import FXPortfolioEnv

class TestFXPortfolioEnv(unittest.TestCase):
    def setUp(self):
        # Load fixture
        self.fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures/fx_test_data.parquet')
        if not os.path.exists(self.fixture_path):
            # Fallback for running from root
            self.fixture_path = 'tests/fixtures/fx_test_data.parquet'
            
        self.df = pd.read_parquet(self.fixture_path)
        
        self.env_kwargs = {
            # "env_id": "fx_portfolio_v1", # Not needed for direct class instantiation
            "df": self.df,
            "stock_dim": 2,
            "hmax": 100, 
            "initial_amount": 100000,
            "transaction_cost_pct": 0.001,
            "reward_scaling": 1e-4,
            "state_space": 2, # Number of tickers
            "action_space": 2,
            "tech_indicator_list": ["log_return"], # simplified features
            "print_verbosity": 0
        }

    def test_env_initialization(self):
        """Test that the environment initializes correctly and inherits from gymnasium."""
        env = FXPortfolioEnv(**self.env_kwargs)
        self.assertIsInstance(env, gym.Env)
        self.assertIsNotNone(env.action_space)
        self.assertIsNotNone(env.observation_space)
        
    def test_reset(self):
        """Test reset returns correct observation shape."""
        env = FXPortfolioEnv(**self.env_kwargs)
        obs, info = env.reset(seed=42)
        
        # Check observation shape
        # For a portfolio env, obs is usually (n_tickers, window_size, n_features) or similar
        # Let's assume a simplified flat observation for MVP1 first or check what we define.
        # Implementation Plan says: Observation Space: Box(shape=(3, window_size, features))
        # Here we only have 2 tickers.
        # But we need to see what the code actually implements.
        
        self.assertTrue(isinstance(obs, np.ndarray))
        # self.assertEqual(obs.shape, ...) # We will refine this after defining the contract
        self.assertTrue(isinstance(info, dict))

    def test_step(self):
        """Test step returns correct tuple and increments day."""
        env = FXPortfolioEnv(**self.env_kwargs)
        env.reset(seed=42)
        
        # Test basic step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Basic assertions
        self.assertEqual(len(obs), env.state_space)
        self.assertTrue(isinstance(reward, (float, int, np.number)))
        self.assertIn(terminated, [True, False])
        self.assertIn(truncated, [True, False])
        self.assertTrue(isinstance(info, dict))
        
        # Check day increment? 
        # The env skeleton sets day=0 on reset.
        # step logic likely increments index.
        # Check internal state if possible or rely on return values.
        # For now, just ensuring it runs without error is the first step.
    
    def test_run_full_episode(self):
        """Test running until termination."""
        env = FXPortfolioEnv(**self.env_kwargs)
        env.reset()
        
        done = False
        steps = 0
        limit = 100 # Safety break
        while not done and steps < limit:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
        # Our fixture has 10 days of data.
        # Hmax is 100, but data length limits it.
        # Logic should terminate when it runs out of data.
        self.assertTrue(done, "Episode should terminate")
        self.assertLessEqual(steps, 11, "Should not run longer than data size")

    def test_invariant_no_actions(self):
        """Test invariant: Zero actions -> No change in holdings/balance."""
        env = FXPortfolioEnv(**self.env_kwargs)
        env.reset(seed=42)
        
        # Initial state should show full cash (if we implement that)
        # For now, just step with zeros
        actions = np.zeros(env.action_space_dim)
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # If we have logic for 'portfolio_value' in info, check it.
        # Currently placeholder.
        # We expect to implement 'portfolio_value' tracking.
        if 'portfolio_value' in info:
            self.assertAlmostEqual(info['portfolio_value'], env.initial_amount)
        
        # If we implemented state tracking, we would check it here.
        # But this test serves as a driver to implement that tracking.
        # So I will assert that 'portfolio_value' IS in info
        self.assertIn('portfolio_value', info)

    def test_invariant_cost_deduction(self):
        """Test invariant: Trading costs are deducted correctly."""
        env = FXPortfolioEnv(**self.env_kwargs)
        env.reset(seed=42)
        
        # Step 1: Stay in Cash (Action [0, 0])
        # Weights before: [0, 0]. Action: [0, 0]. Turnover: 0. Cost: 0.
        # Market Return: 0 (since weights 0).
        # Result: Value = Initial.
        obs, _, _, _, info = env.step(np.zeros(2))
        self.assertAlmostEqual(info['portfolio_value'], env.initial_amount)
        
        # Step 2: Move to [0.5, 0.5]
        # Weights before: [0, 0]. Action: [0.5, 0.5].
        # Market Return applied to OLD weights ([0, 0]): 0.
        # Turnover: |0.5-0| + |0.5-0| = 1.0.
        # Cost: 1.0 * CurrentValue * 0.001.
        # Expected Value = Initial - (Initial * 0.001).
        
        target_value = env.initial_amount * (1 - 0.001)
        
        obs, _, _, _, info = env.step(np.array([0.5, 0.5]))
        
        self.assertAlmostEqual(info['portfolio_value'], target_value, places=2)
        self.assertAlmostEqual(info['portfolio_value'], target_value, places=2)

    def test_state_shape(self):
        """Test that observation matches (stock_dim, lookback, features) flattened or shaped."""
        # For this test, we expect the Env to handle state_space calc or we must provide it correctly.
        # Let's say we rely on the env to give us a vector of size:
        # stock_dim * lookback * (len(tech_list) + 1 for close price? OR just tech list)
        # Fixture only has log_return. let's assume features = tech_list.
        
        # We need to update env_kwargs to have a lookback > 1 to test windowing
        kwargs = self.env_kwargs.copy()
        lookback = 3
        kwargs['lookback'] = lookback
        # Features: log_return (1). Stock_dim: 2.
        # Expected size: 2 * 3 * 1 = 6.
        kwargs['state_space'] = 6 
        
        env = FXPortfolioEnv(**kwargs)
        obs, _ = env.reset(seed=42)
        
        self.assertEqual(obs.shape, (6,))
        # Also check observation space match
        self.assertEqual(env.observation_space.shape, (6,))

    def test_window_logic(self):
        """Test that the state contains the correct looking-back data."""
        kwargs = self.env_kwargs.copy()
        lookback = 2
        kwargs['lookback'] = lookback
        kwargs['state_space'] = 4 # 2 stocks * 2 days * 1 feature
        
        env = FXPortfolioEnv(**kwargs)
        
        # Reset (day 0)
        # We need data from t-lookback to t.
        # If today is day 0, do we have history? 
        # Usually padding or we start at day=lookback.
        # Let's assume we start at day 0 and pad with 0 or first day?
        # Or better, environment starts at day = lookback.
        
        obs, _ = env.reset()
        
        # If env starts at day=0, and we look back 2 days. 
        # Implementation choice: zero pad? duplicate?
        # Let's see what we implement.
        # A common robust way: `reset` starts at `day = 0` but data access handles `max(0, day-lookback)`.
        
        # Let's just assert that it is NOT all zeros if we are at day > 0
        # Step once
        obs, _, _, _, _ = env.step(env.action_space.sample())
        # Now at day 1. 
        
        # In our fixture, log_returns are non-zero?. 
        # Check fixture content: 'log_return' = 0.001 * t. 
        # So it should be non-zero.
        self.assertTrue(np.any(obs != 0), "Observation should contain data, not just zeros")

    def test_reward_mechanics(self):
        """Test that reward is calculated (non-zero for a profitable step)."""
        env = FXPortfolioEnv(**self.env_kwargs)
        env.reset(seed=42)
        
        # We need to force a situation where we make money.
        # We know fixture 'log_return' is positive (0.001*t).
        # Actions: Buy and hold.
        actions = np.ones(env.action_space_dim) * 0.5 # 50% each
        
        # Step
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # Since returns are positive and we held assets, reward should be positive.
        # (Assuming risk penalty doesn't outweigh it).
        self.assertNotEqual(reward, 0.0, "Reward should be calculated, not placeholder 0.0")
    def test_sb3_check_env(self):
        """Test strict API compliance with SB3 check_env."""
        from stable_baselines3.common.env_checker import check_env
        env = FXPortfolioEnv(**self.env_kwargs)
        # check_env runs reset, step, checking types and shapes
        check_env(env, warn=False) # Warnings treated as errors? No, just print.
        
        # Explicit pass if check_env doesn't raise
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
