import gym
from stable_baselines.td3 import MlpPolicy
from stable_baselines.td3 import TD3
env = gym.make('HalfCheetah-v3')
model = TD3(MlpPolicy, env,learning_starts=10000,buffer_size=1000000 ,learning_rate=0.001, train_freq=1000,
			 gradient_steps=1000, policy_kwargs= dict(layers=[400, 300]) ,target_policy_noise=0.1,verbose=1)
model.learn(total_timesteps=1000000)
model.save('Cheetah_model')
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()