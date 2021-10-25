# Udacity Deep Reinforcement Learning Nanodegree - <br /> Project 3: Collaboration and Competition
![trained_agent](https://github.com/julesser/DeepRL-P3-Collaboration-Competition/blob/main/fig/trained_agent.gif)

## Introduction
In this project, I use reinforcement learning (RL) to train two agents playing tennis in an environment similar to [Unity's Tennis environment](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#tennis).

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. 
- This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.Consequently, the environment can be categoriezed as an episodic, multi-agent, continous control problem.

## Getting Started
### Install Dependencies
    cd /python
    pip install .
### Instructions
- Run `python3 watch_trained_agent.py` to see the trained agent in action.
- Run `python3 watch_random_agent.py` to see an untrained agent performing random actions.
- Run `python3 train_control_agent.py` to re-train the agent.
- `model.py` defines the actor and critic network architectures.
- `agent.py` defines the MADDPG agent class.
- `*.pth` files each contain the saved network weights after training.

## Solution Method
Details on the training algorithm, network architecture and hyperparameters are discussed in the ![project report](https://github.com/julesser/DeepRL-P3-Collaboration-Competition/blob/main/Report.md) 

## Results
The implemented RL algorithm is able to solve the environment in about 1800 episodes:
![training_results](https://github.com/julesser/DeepRL-P3-Collaboration-Competition/blob/main/fig/results.png) 

## Ideas for Future Improvements
- Perform a systematic hyperparameter optimization study to improve convergence characteristics.  
- Benchmark against other algorithms, such as TRPO (Trust Region Policy Optimization), PPO (Proximal Policy Optimization) or D4PG (Distributed Distributional Deterministic Policy Gradients), which may obtain more robust results.
