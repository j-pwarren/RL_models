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

def get_first_visit(Q_sa, end_pos):
    # Extract the state and action pairs to search for duplicates, these are the 0 - 4th element in the list
    # Convert the list to a tuple so it can be used as a dictionary key
    state_action = [tuple(ele[:end_pos]) for ele in Q_sa]

    # Create a default dictionary to collect the state action tuples
    episode_sa = defaultdict(list)
    # Loop over each element in the state, action pair
    for ii, item in enumerate(state_action):
        episode_sa[item].append(ii)

    first_index = [item[0] for key, item in episode_sa.items()]

    return first_index

def calculate_returns(Q_sa, gamma = 1):
    # Initialise G at 0
    G = 0
    for ii, element in enumerate(Q_sa):
        # The final element in Q_sa is the reward for each state action
        G += gamma**ii*element[-1]

    return G

def a_func():
    return 








