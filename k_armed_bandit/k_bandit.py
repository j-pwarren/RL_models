import numpy as np


class worker_bandit():
    def __init__(self, nbandits, mean, variance, epsilon, decay, min_epsilon):
        self.nbandits = nbandits
        # Initialise the expected value with very small random numbers
        self.Qn = np.random.randn(self.nbandits)*0.001
        # The underlying distribution of rewards is drawn randomly
        self.variance = variance
        self.mean = mean
        self.reward_dist = mean + np.random.randn(self.nbandits)*variance
        # Rewards for each are are drawn randomly
        self.Rn = np.random.normal(self.reward_dist, self.variance)
        # Initialise the first action as a random draw
        self.action = np.random.randint(0, self.nbandits)

        # Exploitation and exploration parameters
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

        # Logging component
        self.optimal = 0
        self.optimal_occured = 0

    def draw_reward(self):
        self.Rn = np.random.normal(self.reward_dist, self.variance)

    def choose_action(self):
        # Draw for the probability of exploration, keep variable in local scope
        p_explore = np.random.rand()

        # If the random draw is greater than epsilon choose best otherwise chose randomly
        if p_explore > self.epsilon:
            maximum_ev = np.where(np.max(self.Qn) == self.Qn)[0]
            # In case maximum value is shared choose between them randomly
            if len(maximum_ev) > 1:
                self.action = maximum_ev[np.random.randint(0, len(maximum_ev))]
            else:
                # Unpack the tuple from where completely
                self.action = maximum_ev[0]
        else:
            self.action = np.random.randint(0, self.nbandits)

    def decay_epsilon(self):
        # If epsilon is greater than its minimum value multiply it by its decay value
        if self.epsilon <= self.min_epsilon:
            self.epsilon == self.min_epsilon
        else:
            self.epsilon *=self.decay

    def get_optimal_choice(self):
        # Unpack the tuple from where completely
        self.optimal = np.where(np.max(self.reward_dist) == self.reward_dist)[0][0]


class sample_average(worker_bandit):
    def __init__(self, nbandits, niter, mean = 0, variance = 1, epsilon = 0, decay = 1,
                 min_epsilon = 0):
        # Mean and variance parameters are defaulted to the standard normal distribution
        # Epsilon, the exploration parameter is set to 0 by default, i.e. we always chose best unless
        # specified
        # Decay is by default set to 1, i.e. no decay
        # Minimim value of elpsilon is 0
        super().__init__(nbandits, mean, variance, epsilon, decay, min_epsilon)
        self.niter = niter
        self.actions_picked = np.zeros(self.nbandits)
        # Store the rewards and whether the learner chose optimally
        self.reward_by_step = np.zeros(self.niter)
        self.optimal_by_step = np.zeros(self.niter)

    def update_Qn(self):
        # Can be done in a single line, we will split up and keep local to the function for readability
        numerator = 1/self.actions_picked[self.action]
        error_reward = self.Rn[self.action] - self.Qn[self.action]
        self.Qn[self.action] = self.Qn[self.action] + numerator*error_reward

    def run_stationary_bandit(self):

        # prior to running the bandit, get the optimal choice
        self.get_optimal_choice()

        # Log which action was picked, the reward at each step
        for ii in range(self.niter):
            self.actions_picked[self.action] += 1
            self.reward_by_step[ii] = self.Rn[self.action]
            if self.optimal == self.action:
                self.optimal_by_step[ii] = 1

            self.update_Qn()
            # Draw a new reward
            self.draw_reward()
            # Chose an action, either random or best
            self.choose_action()
            # Decay the epsilon value
            self.decay_epsilon()



