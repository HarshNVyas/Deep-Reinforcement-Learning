import gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC

env = gym.make('HalfCheetah-v3')

model = SAC(MlpPolicy, env,learning_starts=10000, buffer_size=1000000, learning_rate=0.001, train_freq=1000, gradient_steps=1000, verbose=1)
model.learn(total_timesteps=1000000, log_interval=10)
model.save("sac_cheetah1")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_cheetah1")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

