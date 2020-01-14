from operator import itemgetter
from multiprocessing import Pool
import pandas as pd
import k_bandit as kb

def run_sample_average(config, extractor, dgp, nlearners):

    # unpack the configuration settings
    nbandits, niter, mean, variance, epsilon, decay, min_epsilon = unpack_parameters(config, extractor)
    # Create a generator expression of iterables, we will initialise instances of the class in the iterable.
    # Initialising the allocating would lead to a class with a single memory referece rather than multiple classes
    num_array = [ii for ii in range(nlearners)]
    iterable = (dict(config=config, extractor=extractor, dgp=dgp,
                     learner=kb.sample_average(nbandits, niter, mean, variance, epsilon, decay, min_epsilon))
                for ii in range(nlearners))

    # run the paralell loop
    # with Pool() as p:
    #     results = p.imap_unordered(paralell_wrapper, iterable)
    #     print(results)
    reward = []
    optimal = []
    p = Pool()
    for r in p.imap_unordered(paralell_wrapper, iterable):
        temp1, temp2 = r
        reward.append(temp1), optimal.append(temp2)
    p.close()
    return reward, optimal

def unpack_parameters(config, extractor):
    # Created as a separate function so it can be reused
    return itemgetter(*extractor)(config)

def paralell_wrapper(iterable):
    # The first argument of the iterable is the looping parameter, this is needed for multiprocessing
    # but unneeded for the actual learning element

    dgp = iterable['dgp']
    config = iterable['config']
    extractor = iterable['extractor']
    learner = iterable['learner']
    if dgp == 'stationary':
        learner.run_stationary_bandit()
    else:
        pass
    return learner.reward_by_step, learner.optimal_by_step

def consolidate_results(results, key):
    # Take the list produced from multiprocessing, turn into a dataframe and take the mean
    results = pd.DataFrame(results).T
    # The mean returns a series, as we wish to concatenate we will turn it back to a frame
    results = results.mean(axis = 1).to_frame()
    results.columns = [key]

    return results
