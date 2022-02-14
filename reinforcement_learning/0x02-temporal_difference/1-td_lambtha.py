#!/usr/bin/env python3
"""TD lambtha algorithm"""


import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Performs the TD lambtha algorithm"""
    states = V.shape[0]
    vals = np.zeros(states)

    for i in range(episodes):
        e = env.reset()

        for j in range(max_steps):
            action = policy(e)
            obs, reward, done, info = env.step(action)

            vals[e] += 1.0
            d = reward + gamma * V[obs] - V[e]
            V += alpha * d * vals
            vals *= lambtha * gamma

            if done:
                break
            e = obs

    return V
