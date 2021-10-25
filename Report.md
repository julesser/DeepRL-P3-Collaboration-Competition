# Udacity Deep Reinforcement Learning Nanodegree - <br /> Project 3: Collaboration and Competition
![trained_agent](https://github.com/julesser/DeepRL-P3-Collaboration-Competition/blob/main/fig/trained_agent.gif)

## Solution Method
The following sections describe the main components of the implemented solution along with design choices. 
### Learning Algorithm & Network Architecture
Multi-Agent Deep Deterministic Policy Gradient (MADDPG) is used to solve this environment. In this method, two neural networks are used, one as actor and one as critic. The actor network has the state vector as input and action vector as output. The critic network has both state vector and action vector as inputs and estimates the reward. In particular, both the actor and critic neural networks consist of two hidden layers with 256 and 128 nodes each, respectively. 
### Hyperparameters
The following hyperparameters have been used for training the agent (see `maddpg_agent.py`):

    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 256        # minibatch size
    LEARN_NUM = 2           # number of learning passes
    GAMMA = 0.99            # discount factor
    TAU = 1e-2              # for soft update of target parameters
    LR_ACTOR = 1e-3         # learning rate of the actor 
    LR_CRITIC = 1e-3        # learning rate of the critic
    EPS_START = 5.0         # initial value for epsilon in noise decay process in Agent.act()
    EPS_EP_END = 300        # episode to end the noise decay process
    EPS_FINAL = 0           # final value for epsilon after decay

## Results
The implemented RL algorithm is able to solve the environment in about 1800 episodes:
![training_results](https://github.com/julesser/DeepRL-P3-Collaboration-Competition/blob/main/fig/results.png) 

## Ideas for Future Improvements
- Perform a systematic hyperparameter optimization study to improve convergence characteristics.  
- Benchmark against other algorithms, such as TRPO (Trust Region Policy Optimization), PPO (Proximal Policy Optimization) or D4PG (Distributed Distributional Deterministic Policy Gradients), which may obtain more robust results.
