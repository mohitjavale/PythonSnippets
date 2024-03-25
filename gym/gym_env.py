import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.env_checker import check_env

class EnvName(gym.Env):

    def __init__(self):
        self.observation_space = spaces.Discrete(4)
        self.action_space = spaces.Discrete(4)

    def step(self, action):
        observation = 0
        reward = 0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=0):
        observation = 0
        info = {}
        return observation, info

    
    def render(self):
        pass

    def close(self):
        pass

if __name__=='__main__':
    env = EnvName()
    check_env(env, warn=True)
    