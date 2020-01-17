import numpy as np
from _collections import defaultdict
import gym

# environment_name = 'Blackjack-v0'
# env = gym.make(environment_name)
#
# nS = env.observation_space.n
# nA = env.action_space.n

def run_episode(env):
    done = False
    Q_sa = []
    state = env.reset()
    while not done:
        action = np.random.randint(0,env.action_space.n)
        next_state, reward, done, info = env.step(action)
        # Expand the tuple for the state
        Q_sa.append([*state,action, reward])
        state = next_state

    return Q_sa

def get_first_visit(state_action):
    # Create a default dictionary to collect the state action tuples
    episode_sa = defaultdict(list)
    # Loop over each element in the state, action pair
    for ii, item in enumerate(state_action):
        episode_sa[item[0]].append(ii)

    return episode_sa










