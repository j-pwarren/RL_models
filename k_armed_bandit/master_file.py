import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import k_bandit as kb # This is a user defined function
import run_learners as rl

if __name__ == '__main__':
    # Everything that we want to get executed only once we encapsulate within the if statement, as python spawns
    # a new process on importing only the first run will return true, otherwise for process we will re-run anycode
    # outside the if statement

    nsteps = 1000 # Each bandit will run 1000 times
    nbandits = 10 # Each learner will have 10 bandits to chose from


    # Show an illustrative plot of the distribution of the bandits
    bandit = kb.sample_average(nbandits, nsteps)
    distribution = np.zeros((nsteps, nbandits))

    for ii in range(nsteps):
        bandit.draw_reward()
        distribution[ii,:] = bandit.Rn

    distribution = pd.DataFrame(data=distribution, columns=['bandit #' + str(ii) for ii in range(nbandits)],
                                index = [str(ii) for ii in range(nsteps)])

    # Change the shape of the dataframe to long form
    distribution = distribution.unstack().to_frame()
    # Unwind the multi-index into columns
    distribution.reset_index(inplace = True)
    # Add column names for convenience
    column_names = ['bandit', 'step', 'reward']
    distribution.columns = column_names


    mean_value = pd.Series(np.round(bandit.reward_dist,2), ['bandit #' + str(ii) for ii in range(nbandits)])
    ax = sns.violinplot(x="bandit", y="reward" ,  data=distribution, palette="muted")

    for xtick in ax.get_xticks():
        vertical_offset = mean_value[xtick]*0.1
        ax.text(xtick, mean_value[xtick] + vertical_offset, mean_value[xtick],
                horizontalalignment='center',size='x-small',color='w',weight='semibold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    plt.show()


    # create a dictionary of settings
    extractor = ('nbandits', 'niter', 'mean', 'variance', 'epsilon', 'decay', 'min_epsilon')
    configuration_1 = dict(nbandits=10, niter=1000, mean=0, variance=1, epsilon=0, decay=1, min_epsilon=0)
    configuration_2 = dict(nbandits=10, niter=1000, mean=0, variance=1, epsilon=0.1, decay=1, min_epsilon=0)
    configuration_3 = dict(nbandits=10, niter=1000, mean=0, variance=1, epsilon=0.2, decay=1, min_epsilon=0)
    configuration_4 = dict(nbandits=10, niter=1000, mean=0, variance=1, epsilon=0.3, decay=1, min_epsilon=0)

    # Run the different configurations using multiprocessing
    config1_reward, config1_optimal = rl.run_sample_average(configuration_1, extractor, 'stationary', 2000)
    config1_reward = rl.consolidate_results(config1_reward, 'e - 0')
    config1_optimal = rl.consolidate_results(config1_optimal, 'e - 0')

    config2_reward, config2_optimal = rl.run_sample_average(configuration_2, extractor, 'stationary', 2000)
    config2_reward = rl.consolidate_results(config2_reward, 'e - 0.1')
    config2_optimal = rl.consolidate_results(config2_optimal, 'e - 0.1')

    config3_reward, config3_optimal = rl.run_sample_average(configuration_3, extractor, 'stationary', 2000)
    config3_reward = rl.consolidate_results(config3_reward, 'e - 0.2')
    config3_optimal = rl.consolidate_results(config3_optimal, 'e - 0.2')

    config4_reward, config4_optimal = rl.run_sample_average(configuration_4, extractor, 'stationary', 2000)
    config4_reward = rl.consolidate_results(config4_reward, 'e - 0.3')
    config4_optimal = rl.consolidate_results(config4_optimal, 'e - 0.3')

    # collect results in a single dataframe
    reward_frame = pd.concat([config1_reward, config2_reward, config3_reward, config4_reward], axis = 1)
    optimal_frame = pd.concat([config1_optimal, config2_optimal, config3_optimal, config4_optimal], axis = 1)

    # Turn the dataframes into long form and decouple the index
    reward_frame = reward_frame.unstack().to_frame()
    reward_frame.reset_index(inplace=True)
    reward_frame.columns = ['epsilon', 'step', 'value']

    # Plot the rewards of each of the learners
    ax = sns.lineplot(x="step", y="value", hue = "epsilon", data = reward_frame)
    plt.show()

    optimal_frame = optimal_frame.unstack().to_frame()
    optimal_frame.reset_index(inplace=True)
    optimal_frame.columns = ['epsilon', 'step', 'optimal']

    # Plot the rewards of each of the learners
    ax = sns.lineplot(x="step", y="optimal", hue="epsilon", data=optimal_frame)
    plt.show()

    # Works up to this point - next step non-stationary dgp




