import gym
from stable_baselines import PPO2
env=gym.make('MountainCarContinuous_5_6_20')
model=PPO2.load('MountainCarContinuous_5_6_20')
obs = env.reset()
for i in range(5000):
      # _states are only useful when using LSTM policies
    action, _ = model.predict(obs)

    obs, reward, done, info = env.step(action)
    env.render()  
 	  
    if done:
        env.close()
        break

