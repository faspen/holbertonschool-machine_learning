#!/usr/bin/env python3
"""Play module"""


import gym
import numpy as np


def play(env, Q, max_steps=100):
    """Play an episode"""
    state = env.reset()
    total_rewards = 0
    env.render()

    for step in range(max_steps):
        perform = np.argmax(Q[state, :])
        next_state, reward, done, misc = env.step(perform)

        state = next_state
        total_rewards += reward
        env.render()
        if done:
            break

    return total_rewards
