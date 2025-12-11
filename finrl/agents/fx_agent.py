from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

class FXAgent:
    """
    FXAgent wraps Stable-Baselines3 algorithms for the FXPortfolioEnv.
    """
    def __init__(self, env: gym.Env):
        self.env = env
        self.model = None
        
        # Initialize PPO model
        self.model = PPO("MlpPolicy", self.env, verbose=1)

    def train(self, total_timesteps: int = 10000):
        if self.model:
            self.model.learn(total_timesteps=total_timesteps)
            
    def predict(self, observation, deterministic: bool = True):
        if self.model:
            return self.model.predict(observation, deterministic=deterministic)
        return None, None
        
    def save(self, path: str):
        if self.model:
            self.model.save(path)
            
    def load(self, path: str):
        self.model = PPO.load(path, env=self.env)
