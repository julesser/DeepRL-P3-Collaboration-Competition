from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch

from ddpg_agent import Agent


def ddpg(n_episodes=2000, max_t=1000):
    """Deep Deterministic Policy Gradient (DDPG).

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of time steps per episode
    """

    scores_deque = deque(maxlen=100)
    scores = []
    avgs = []
    for i_episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
        state = env_info.vector_observations                  # get the current state (for each agent)
        score = np.zeros(num_agents)
        for t in range(max_t):
            action = agent.act(state, add_noise=True)
            env_info = env.step(action)[brain_name]           # send all actions to the environment
            next_state = env_info.vector_observations         # get next state (for each agent)
            reward = env_info.rewards                         # get reward (for each agent)
            done = env_info.local_done                        # see if episode finished

            agent.step(state, action, reward, next_state, done)

            state = next_state                               # roll over state to next time step
            score += reward

            if np.any(done):
                break 
        scores_deque.append(score)
        scores.append(np.mean(score))
        avg = np.mean(scores_deque)
        avgs.append(avg)
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % 50 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if np.mean(scores_deque) >= 0.5:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
                break
        
    # torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
    # torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

    return scores, avgs



# 1. Create environment
env = UnityEnvironment(file_name="simulator/Tennis.x86")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]
print('Number of agents:', num_agents)

# 2. Create agent
agent = Agent(state_size, action_size, random_seed=0)

# 3. Roll out DQN algorithm
scores, avgs = ddpg()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores, label='DQN')
plt.plot(np.arange(len(scores)), avgs, c='r', label='Average')
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.legend()
plt.show()
