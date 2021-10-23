from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch

from maddpg_agent import Agent

def get_actions(states, add_noise):
    '''gets actions for each agent and then combines them into one array'''
    action_0 = agent_0.act(states, add_noise)    # agent 0 chooses an action
    action_1 = agent_1.act(states, add_noise)    # agent 1 chooses an action
    return np.concatenate((action_0, action_1), axis=0).flatten()

def maddpg(n_episodes=4000, max_t=1000):
    """Deep Deterministic Policy Gradient (maddpg).

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of time steps per episode
    """

    scores_deque = deque(maxlen=100)
    scores_all = []
    avgs = []
    
    for i_episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]         # reset the environment
        states = np.reshape(env_info.vector_observations, (1,48)) # get states and combine them
        agent_0.reset()
        agent_1.reset()
        scores = np.zeros(num_agents)
        while True: 
            actions = get_actions(states, True)           # choose agent actions and combine them
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
        if i_episode % 10 == 0:
            print('Episode {:0>4d}\tMax Reward: {:.3f}\t Average: {:.3f}'.format(
                i_episode, np.max(scores_all[-10:]), avgs[-1]))

        # determine if environment is solved and keep best performing models
        if avgs[-1] >= 0.5:
            print('<-- Environment solved in {:d} episodes! \
            \n<-- Average: {:.3f} over past {:d} episodes'.format(i_episode-100, avgs[-1], 100))
            # save weights
            torch.save(agent_0.actor_local.state_dict(),
                       'checkpoint_actor_0.pth')
            torch.save(agent_0.critic_local.state_dict(),
                       'checkpoint_critic_0.pth')
            torch.save(agent_1.actor_local.state_dict(),
                       'checkpoint_actor_1.pth')
            torch.save(agent_1.critic_local.state_dict(),
                       'checkpoint_critic_1.pth')
            break
    return scores_all, avgs



# 1. Create environment
env = UnityEnvironment(file_name="simulator/Tennis.x86", no_graphics=True)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]
print('Number of agents:', num_agents)

# 2. Create agents
agent_0 = Agent(state_size, action_size, num_agents=1, random_seed=0)
agent_1 = Agent(state_size, action_size, num_agents=1, random_seed=0)

# 3. Roll out MADDPG algorithm
scores, avgs = maddpg()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores, label='DQN')
plt.plot(np.arange(len(scores)), avgs, c='r', label='Average')
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.legend()
plt.show()
