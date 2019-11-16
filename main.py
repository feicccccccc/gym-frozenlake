"""

testing genetic algorithm on frozen lake problem:

https://gym.openai.com/envs/FrozenLake-v0/

Gym env source code:
https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py

credit:
https://becominghuman.ai/genetic-algorithm-for-reinforcement-learning-a38a5612c4dc

"""

import numpy as np
import gym
from func import *

"""
Hyperparameter:

"""

num_polciy = 100 # numnber of polciy to search
render = True # Render or not
iter_step = 20

"""

16 state (in which place)

0  1  2  3
4  5  6  7
8  9  10 11
11 13 14 15

4 Action:
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
"""

env = gym.make('FrozenLake-v0') # create the gym env
env.seed(0) # https://github.com/openai/gym/blob/master/gym/core.py
np.random.seed(721) # random seed for numpy

# population of different policy
policy_pop = [gen_random_policy() for _ in range(num_polciy)]

for idx in range(iter_step):
    policy_scores = [evaluate_policy(env, p) for p in policy_pop]
    print('Generation {} : max score = {}'.format(idx + 1, max(policy_scores)))

    # EVALUATION
    # Get the idx for the rank base on the reward
    policy_ranks = list(reversed(np.argsort(policy_scores)))
    # The highest 5 rank of the policy
    elite_set = [policy_pop[x] for x in policy_ranks[:5]]

    # SELECTION
    # probability to select that policy base on the normalised scores of the entire population
    select_probs = np.array(policy_scores) / np.sum(policy_scores)


    # CROSSOVER
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.choice.html

    # randomly select two policy base on the probability(normalised scores in the population)
    child_set = [crossover(
        policy_pop[np.random.choice(range(num_polciy), p=select_probs)],
        policy_pop[np.random.choice(range(num_polciy), p=select_probs)])
        for _ in range(num_polciy - 5)]

    # MUTATION
    mutated_list = [mutation(p) for p in child_set]

    # Survived population + mutated population
    policy_pop = elite_set + mutated_list

    policy_score = [evaluate_policy(env, p) for p in policy_pop]
    best_policy = policy_pop[np.argmax(policy_score)]

    print('Best policy score = {}, index = {}'.format(np.max(policy_score),np.argmax(policy_score)))

    # Evaluation
    env = gym.wrappers.Monitor(env, 'frozenlake1', force=True)
    for _ in range(200):
        run_episode(env, best_policy,render = True)
    env.close()