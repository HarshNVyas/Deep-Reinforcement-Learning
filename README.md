# Abstract
Deep Reinforcement Learning is at the cutting edge and has finally reached a point where we can apply it in real-world applications. In this project, we use policy gradient methods, which are present in reinforcement learning, to get state-of-the-art performance in continuous control tasks. We aimed to use deep reinforcement learning to allow robots to learn locomotion gaits. We emulated these continuous control tasks in Box2D and MuJoCo (Multi-Joint dynamics with Contact) environment, allowing us to see the results of the algorithms we used on the agents for the task. The policy gradient algorithms that we used are DDPG, SAC and TD-3. The algorithms used on the agents in the environment are obtained from the gym library.  The two agents that we worked with are half cheetah and bipedal walker. After training the models using the algorithms, we tuned the hyperparameters for optimal performance by the agent. This also involves generating the graph based on the agent's performance, which allows us to evaluate the reward to the time-steps trained. 

The results of our project are:- 
1) We successfully taught the agent to learn locomotion gaits
2) We optimized its performance by hyperparameter tuning and performance analysis
3) Graphs representing the performance of the agent.

# Deep-Reinforcement-Learning
Deep Reinforcement Learning is the combination of Reinforcement Learning and Deep Learning. This field has recently solved a wide range of previously impossible decision-making tasks for a machine. Deep RL opens up many new robotics, healthcare, smart grids, and finance applications. Implementing deep learning architectures (deep neural networks) with reinforcement learning algorithms (Q-learning, actor-critic, TD3, etc.) can scale to previously unsolvable problems. Deep Reinforcement Learning is been applied to robotics to overcome the complexity of rule-based programming. The robot learns to navigate the environment after exploring it thousands of times, eliminating the need to program its actions explicitly.

# Open AIGym
Gym is a toolkit developed by OpenAI to develop and evaluate Reinforcement Learning algorithms. It is an open-source library in Python with many environments prebuilt under several categories, such as Box2D ToyText. The agents and environments resembling a robot locomotion problem are Half Cheetah and Bipedal Walker. MuJoCo (multi-joint dynamics in contact) is a proprietary physics engine for robotics, biomechanics, graphics, and animation development. Many environments are built in the gym on MuJoCo, such as Humanoid, Ant-bot, etc. Solving these continuous control problems using state-of-the-art DRL algorithms is the primary task of this work.

 ![image](https://github.com/HarshNVyas/Deep-Reinforcement-Learning-to-train-locomotation-activity-in-Robots/assets/41836190/5560f872-23ca-41c1-914f-8101ef29598f)

We have visualized and analyzed the previous works on these environments. We could see flaws in the movement of the agent, and the actions could have been more effective. So we went on to solve these issues by working on several cutting-edge algorithms such as SAC, DDPG and TD3. We resolved the issues in the previous works through some tweaks.
