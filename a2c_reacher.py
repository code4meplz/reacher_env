from unityagents import UnityEnvironment
import numpy as np
from agent import Agent
import torch as T
import matplotlib.pyplot as plt

env = UnityEnvironment(
    file_name='/home/deepmind/PycharmProjects/pytorch_examples/A2C_Reacher/Reacher_Linux/Reacher.x86_64',no_graphics=True)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

agent = Agent(state_size, [4, 1], [0.0001, 0.0003], 0.98)

print(T.cuda.is_available())
episode_scores = []


for episode in range(300):
    NaN = False
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state (for each agent)
    scores = np.zeros(num_agents)  # initialize the score (for each agent)
    while True:
        actions = agent.choose_action(states)  # select an action (for each agent)
        env_info = env.step(actions)[brain_name]  # send all actions to tne environment
        next_states = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards[0]  # get reward (for each agent)
        dones = env_info.local_done[0]  # see if episode finished
        agent.memory.append((states, rewards, next_states, dones, agent.log_prob, agent.entropy))
        if (len(agent.memory) == agent.memory.maxlen) or (np.any(dones)):
            agent.learn_from_exp()

        scores += env_info.rewards  # update the score (for each agent)
        states = next_states  # roll over states to next time step
        if np.isnan(states).any():
            NaN = True
        if np.any(dones):  # exit loop if episode finished
            episode_scores.append(np.mean(scores))
            agent.memory.clear()
            break
    print(f'Total score in episode {episode}: {np.mean(scores)}')
    if NaN:
        print('NaNs in episode')


env.close()
plt.plot(episode_scores)
plt.savefig('fig.png')
plt.show()