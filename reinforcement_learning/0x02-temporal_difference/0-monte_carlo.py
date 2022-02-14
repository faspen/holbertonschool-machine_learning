#!/usr/bin/env python3
"""Monte carlo task"""


import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    vals = V.shape[0]

    for i in range(episodes):
        e = env.reset()
        episode = []

        for j in range(max_steps):
            action = policy(e)
            obs, reward, done, info = env.step(action)
            episode.append([e, action, reward, obs])

            if done:
                break
            e = obs

        episode = np.array(episode, dtype=int)
        mc = 0
        for x, y in enumerate(episode[::-1]):
            e, action, reward, e_next = y
            mc = gamma * mc + reward
            if e not in episode[:i, 0]:
                V[e] = V[e] + alpha * (mc - V[e])

    return V
