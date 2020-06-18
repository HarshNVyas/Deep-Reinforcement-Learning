import gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC
env = gym.make('HalfCheetah-v3')
model = SAC.load("sac_cheetah1")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
