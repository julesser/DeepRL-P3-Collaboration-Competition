from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch

from maddpg_agent import Agent

# 1. Create environment
env = UnityEnvironment(file_name="simulator/Tennis.x86")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=False)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]
print('Number of agents:', num_agents)

# 2. Create agents
agent_0 = Agent(state_size, action_size, num_agents=1, random_seed=0)
agent_1 = Agent(state_size, action_size, num_agents=1, random_seed=0)

# 3. Load the previously learned weights from file
agent_0.actor_local.load_state_dict(torch.load('checkpoint_actor_0.pth'))
agent_0.critic_local.load_state_dict(torch.load('checkpoint_critic_0.pth'))
agent_1.actor_local.load_state_dict(torch.load('checkpoint_actor_1.pth'))
agent_1.critic_local.load_state_dict(torch.load('checkpoint_critic_1.pth'))

# 4. Apply trained agent to solve a full episode
scores_deque = deque(maxlen=100)
scores_all = []
avgs = []
for i in range(1,6):
        env_info = env.reset(train_mode=False)[brain_name]         # reset the environment
        states = np.reshape(env_info.vector_observations, (1,48)) # get states and combine them
        agent_0.reset()
        agent_1.reset()
        scores = np.zeros(num_agents)
        while True: 
            action_0 = agent_0.act(states, add_noise=False)    # agent 0 chooses an action
            action_1 = agent_1.act(states, add_noise=False)    # agent 1 chooses an action
            actions = np.concatenate((action_0, action_1), axis=0).flatten()
            env_info = env.step(actions)[brain_name]           # send both agents' actions together to the environment
            next_states = np.reshape(env_info.vector_observations, (1, 48)) # combine the agent next states
            rewards = env_info.rewards                         # get reward
            done = env_info.local_done                         # see if episode finished
            agent_0.step(states, actions, rewards[0], next_states, done, 0) # agent 1 learns
            agent_1.step(states, actions, rewards[1], next_states, done, 1) # agent 2 learns
            scores += np.max(rewards)                          # update the score for each agent
            states = next_states                               # roll over states to next time step
            if np.any(done):                                   # exit loop if episode finished
                break

        ep_best_score = np.max(scores)
        scores_deque.append(ep_best_score)
        scores_all.append(ep_best_score)
        avgs.append(np.mean(scores_deque))

        # print results
        print('Episode {:0>4d}\tMax Reward: {:.3f}\t Average: {:.3f}'.format(i, np.max(scores_all[-10:]), avgs[-1]))

        print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores_all)))

env.close()


