#!/usr/bin/env python3
"""Q-learning task"""


import numpy as np
import gym


def epsilon_greedy(Q, state, epsilon):
    """Use epsilon greedy to determine next action"""
    p = np.random.uniform(0, 1)

    if p < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state, :])

    return action


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1,
          epsilon_decay=0.05):
    """Train an agent"""
    total_rewards = []
    total_eps = epsilon

    for ep in range(episodes):
        state = env.reset()
        done = False
        count = 0

        for step in range(max_steps):
            perform = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done, misc = env.step(perform)

            if done and reward == 0:
                reward = -1
            Q[state, perform] = Q[state, perform] * (1 - alpha) + alpha * \
                (reward + gamma * np.max(Q[next_state, :]))
            count += reward

            if done:
                break
            state = next_state
        epsilon = min_epsilon + (total_eps - min_epsilon) * \
            np.exp(-epsilon_decay * ep)
        total_rewards.append(count)

    return Q, total_rewards
