import numpy as np
import gym

def value_iteration(env, num_iterations = 10000, epsilon = 1e-20, gamma = 1):

    # Initialise Value function
    V = np.zeros(env.nS)
    num_iterations = num_iterations

    # discount factor
    gamma = gamma

    # Create a convergence criterion
    epsilon = epsilon
    ii = 0
    while (ii < num_iterations):
        ii +=1
        # Log the previous value function
        prev_v = np.copy(V)

        # Loop over all states
        for s in range(env.nS):
            # Loop over each avction
            # Loop over each possible outcome of each action
            Q_s_a = [sum([prob*(reward+ gamma*prev_v[next_s]) for prob, next_s, reward, info in env.P[s][a]]) for a in
                     range(env.nA)]
            V[s] = max(Q_s_a)
        converge = sum(np.fabs(V - prev_v))

        if(converge <= epsilon):
            print('Value iteration has converged after ', ii, ' iterations')
            break

    return V

def get_policy(env, v, gamma = 1):
    "Given the valuation function create the policy"

    # Create a matrix to populate the best actions in for each state
    policy = np.zeros(env.nS)
    action = env.action_space.n
    # For each state
    for s in range(env.nS):
        # Create a temporary matrix for each of the actions within the state
        q_sa = np.zeros(action)
        for a in range(action):
            for info in env.P[s][a]:
                # calculate the expected value for each of the state action pair
                prob, next_s, reward, other = info
                q_sa[a] +=  prob*(reward+ gamma*v[int(next_s)])
        # The state action pair with the highest expected value is the policy
        policy[s] = np.argmax(q_sa)
    return policy

def run_episode(env, policy, gamma, render = False):
    # To ensure we start from a fresh game reset the episode
    state = env.reset()
    total_reward = 0
    step_index = 0

    done = False

    while not done:
        if render == True:
            env.render()

        state, reward, done, info = env.step(int(policy[state]))
        total_reward += (gamma ** step_index * reward)
        step_index += 1

    return total_reward

def evaluate_policy(env, policy, gamma = 1, niter = 1000):
    total_reward = [run_episode(env, policy, gamma = gamma) for ii in range(niter)]

    return np.mean(total_reward)




