import gym
import numpy as np
import monte_carlo as mc
from collections import defaultdict

environment_name = 'Blackjack-v0'
env = gym.make(environment_name)
gamma = 1

# Create a set of dictionaries to collect the results
total_returns = defaultdict(float)
action_count = defaultdict(float)
averageQ = defaultdict(lambda: np.zeros(env.action_space.n))




for ii in range(100):
    # Steps, generate episode
    # Returns a embedded list of [[state, state, state, action, reward]......[S,S,S,A,R]]
    episode = mc.run_episode(env)

    # re-arrange the state, action reward tuple?
    state_action = [[tuple(sa[:4]), [sa[-1]]] for sa in episode]
    sa_log = []
    # Loop over each element of the state action tuple?!
    for ii, item in enumerate(state_action):
        # log the state action pair
        if item[0] in sa_log:
            print(True)
        else:
            print(False)
            state = item[0][:3]
            action = item[0][-1]
            sa_log.append(item[0])
            total_returns[item[0]] += gamma**ii + item[1][0]
            action_count[item[0]] += 1
            averageQ[state][action] = total_returns[item[0]]/action_count[item[0]]



