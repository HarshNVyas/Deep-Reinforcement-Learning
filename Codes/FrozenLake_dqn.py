#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import gym
import random
import matplotlib.pyplot as plt
np.random.seed = 29


# In[2]:


env = gym.make("FrozenLake-v0")


# In[3]:


direction = {0:'LEFT ', 1:'DOWN ', 2:'RIGHT', 3:'UP   '}


# In[4]:


observation = env.reset()
env.render()
for i in range(100):
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()
    if done: 
        print('{} steps to complete'.format(i+1))
        break
env.close()


# In[5]:


n, max_steps = 10000, 100
total_reward = 0
num_steps = []
for episode in range(n):
    observation = env.reset()
    for i in range(max_steps):
        observation, reward, done, info = env.step(env.action_space.sample())
        if done: 
            total_reward += reward
            num_steps.append(i+1)
            break
env.close()
print('Success Percentage = {0:.2f} %'.format(100*total_reward/n))
print('Average number of steps taken to reach the goal = {0:.2f}'.format(np.mean(num_steps)))


# In[6]:


# Initialize the q-table to all zeros
num_actions = env.action_space.n
num_states = env.observation_space.n
q_table = np.zeros((num_states, num_actions))


# In[7]:


# Q-table update parameters
alpha = 0.8
gamma = 0.95

# Exploration paramters
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001


# In[8]:


rewards = []
epsilon = max_epsilon
n, max_steps = 25000, 100
for episode in range(n):
    s = env.reset()
    total_reward = 0
    for i in range(max_steps):
        if random.uniform(0, 1) < epsilon:
            a = env.action_space.sample()
        else:
            a = np.argmax(q_table[s, :])
        
        s_new, r, done, info = env.step(a)
        
        q_table[s, a] = q_table[s, a] + alpha*(r + gamma*np.max(q_table[s_new, :]) - q_table[s, a])
        s, total_reward = s_new, total_reward+r
        if done: 
            rewards.append(total_reward)
            epsilon = min_epsilon + (max_epsilon-min_epsilon)*np.exp(-decay_rate*episode)
            break
env.close()


# In[9]:


with np.printoptions(precision=5, suppress=True):
    print(q_table)


# In[10]:


# 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP
env.reset()
env.render()


# In[11]:


moving_avg_reward = []
window = 1000
for i in range(window, n):
    moving_avg_reward.append(100*sum(rewards[i-window:i])/window)

fig, axes = plt.subplots(figsize=(8, 8))
plt.plot(range(window, n), moving_avg_reward)
axes.set(xlabel='Episode Idx', ylabel='Success Rate', title='FrozenLake-V0')
plt.show()


# In[12]:


# Print the action the agent takes at each state
env.reset()
env.render()
print(np.array([direction[x] for x in np.argmax(q_table, axis=1)]).reshape(4, 4))


# In[13]:


n, max_steps = 10000, 100
count, num_prints = 0, 1
rewards = []
num_steps = []
for episode in range(n):
    s = env.reset()
    total_reward = 0
    if count<num_prints:
        print('---------EPISODE {}---------'.format(episode))
        env.render()
    for i in range(max_steps):
        a = np.argmax(q_table[s, :])
        s, r, done, info = env.step(a)
        total_reward+=r
        if count<num_prints:
            env.render()
        if done: 
            rewards.append(total_reward)
            num_steps.append(i+1)
            if count<num_prints: 
                if r==1:
                    print('SUCCESS!!!')
                else:
                    print('Failed :( )')
            count+=1
            break
env.close()
print('Success Percentage = {0:.2f} %'.format(100*np.sum(rewards)/len(rewards)))
print('Average number of steps taken to reach the goal = {0:.2f}'.format(np.mean(num_steps)))


# In[14]:


q_table_no_slip = np.array([[0, 1, 0, 0], 
                           [0, 0, 1, 0],
                           [0, 1, 0, 0],
                           [1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 1, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 0],])
# Print the action the agent takes at each state
env.reset()
env.render()
print(np.array([direction[x] for x in np.argmax(q_table_no_slip, axis=1)]).reshape(4, 4))


# In[15]:


n, max_steps = 10000, 100
rewards = []
num_steps = []
for episode in range(n):
    s = env.reset()
    total_reward = 0
    for i in range(max_steps):
        a = np.argmax(q_table_no_slip[s, :])
        s, r, done, info = env.step(a)
        total_reward+=r
        if done: 
            rewards.append(total_reward)
            num_steps.append(i+1)
            break
env.close()
print('Success Percentage = {0:.2f} %'.format(100*np.sum(rewards)/len(rewards)))
print('Average number of steps taken to reach the goal = {0:.2f}'.format(np.mean(num_steps)))


# In[ ]:


#rest of the code...

