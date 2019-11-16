import numpy as np
import gym

def gen_random_policy():
    # what to do in each 16 state
    return np.random.choice(4, size=((16)))

def run_episode(env, policy, episode_len=100, render = False):

    # episode: maximum length of the step
    total_reward = 0
    obs = env.reset() # inital observation
    for t in range(episode_len):
        if render:
            env.render()
        action = policy[obs] # what is the observation
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            # print('Epside finished after {} timesteps.'.format(t+1))
            break
    return total_reward

def evaluate_policy(env, policy, n_episodes=100):
    # stochastic MDP
    # return sampled expected reward
    total_rewards = 0.0
    for _ in range(n_episodes):
        total_rewards += run_episode(env, policy)
    return total_rewards / n_episodes

def crossover(policy1, policy2):
    new_policy = policy1.copy()
    for i in range(16):
        rand = np.random.uniform(0,1)
        if rand > 0.5:
            new_policy[i] = policy2[i]
    # max the two policy
    return new_policy

def mutation(policy, p=0.05):
    new_policy = policy.copy()
    for i in range(16):
        rand = np.random.uniform()
        if rand < p:
            new_policy[i] = np.random.choice(4)
    return new_policy
