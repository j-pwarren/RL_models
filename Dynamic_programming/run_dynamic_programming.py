import gym
import iteration_solutions as its

environment_name = 'FrozenLake8x8-v0'
env = gym.make(environment_name)

# Run the value function iteration example
# Compute the value function to convergence
value = its.value_iteration(env)
VI_policy = its.get_policy(env, value)
VI_evaluate = its.evaluate_policy(env, VI_policy)
print('the average reward for the game using value iteration is ', VI_evaluate)

# Run the policy iteration example
PI_policy = its.policy_iteration(env)
PI_evaluate = its.evaluate_policy(env, PI_policy)
print('the average reward for the game using policy iteration is ', PI_evaluate)

