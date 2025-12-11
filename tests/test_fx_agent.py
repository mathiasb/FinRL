import unittest
from unittest.mock import MagicMock, patch
import gymnasium as gym
import numpy as np

# Adjust path if necessary or rely on installed package
from finrl.agents.fx_agent import FXAgent
from stable_baselines3 import PPO

class TestFXAgent(unittest.TestCase):
    def setUp(self):
        # Mock environment
        self.mock_env = MagicMock()
        self.mock_env.observation_space.shape = (10,)
        self.mock_env.action_space.shape = (2,)
        
    @patch('finrl.agents.fx_agent.PPO')
    def test_agent_initialization(self, mock_ppo):
        """Test that Agent initializes PPO with correct environment."""
        agent = FXAgent(env=self.mock_env)
        
        # Verify PPO was initialized
        mock_ppo.assert_called_once()
        args, kwargs = mock_ppo.call_args
        
        # Check that 'MlpPolicy' and env were passed
        # Env might be positional arg at index 1
        self.assertEqual(args[0], 'MlpPolicy')
        self.assertEqual(args[1], self.mock_env)
        
        self.assertEqual(args[1], self.mock_env)
        
        self.assertIsNotNone(agent.model)

    @patch('finrl.agents.fx_agent.PPO')
    def test_train(self, mock_ppo):
        """Test that train calls model.learn."""
        # Setup mock instance
        mock_model_instance = MagicMock()
        mock_ppo.return_value = mock_model_instance
        
        agent = FXAgent(env=self.mock_env)
        agent.train(total_timesteps=5000)
        
        mock_model_instance.learn.assert_called_once_with(total_timesteps=5000)

    @patch('finrl.agents.fx_agent.PPO')
    def test_predict(self, mock_ppo):
        """Test that predict calls model.predict."""
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = (np.array([0.5, 0.5]), None)
        mock_ppo.return_value = mock_model_instance
        
        agent = FXAgent(env=self.mock_env)
        obs = np.random.rand(10)
        action, state = agent.predict(obs)
        
        mock_model_instance.predict.assert_called_once_with(obs, deterministic=True)
        np.testing.assert_array_equal(action, np.array([0.5, 0.5]))

    @patch('finrl.agents.fx_agent.PPO')
    def test_save_load(self, mock_ppo):
        """Test save and load delegation."""
        mock_model_instance = MagicMock()
        mock_ppo.return_value = mock_model_instance
        
        agent = FXAgent(env=self.mock_env)
        agent.save("test_model")
        mock_model_instance.save.assert_called_once_with("test_model")
        
        # Test load
        agent.load("test_model")
        mock_ppo.load.assert_called_once_with("test_model", env=self.mock_env)

if __name__ == '__main__':
    unittest.main()
