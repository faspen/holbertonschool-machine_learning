#!/usr/bin/env python3
"""Sarsa lambtha"""


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


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Perform sarsa lambtha algorithm"""
    initial = epsilon
    vals = np.zeros(Q.shape)

    for i in range(episodes):
        e = env.reset()
        action = epsilon_greedy(env, Q, e, epsilon)

        for j in range(max_steps):
            obs, reward, done, info = env.step(action)
            new = epsilon_greedy(env, Q, e, epsilon)
            vals *= gamma * epsilon
            vals[e, action] += 1.0

            delta = reward + gamma * Q[obs, new] - Q[e, action]
            Q += alpha * delta * vals

            if done:
                break
            e = obs
            action = new
        if epsilon < min_epsilon:
            epsilon = min_epsilon
        else:
            epsilon *= initial * np.exp(-epsilon_decay * i)

    return Q
