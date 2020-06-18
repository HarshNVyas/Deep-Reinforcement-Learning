import gym
from stable_baselines.ddpg import LnMlpPolicy
from stable_baselines.ddpg import DDPG
env = gym.make('HalfCheetah-v3')
model = DDPG(LnMlpPolicy, env,gamma=.95,buffer_size=1000000,param_noise_adaption_interval=0.22,batch_size=256,
			 normalize_observations=True,normalize_returns=False, policy_kwargs= dict(layers=[400, 300]) ,verbose=1)
model.learn(total_timesteps=1000000)
model.save('Cheetah_model_DDPG')
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()