#!/usr/bin/env python3


import numpy as np
import gym
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Train using policy gradients"""
    W = np.random.rand(4, 2)
    score = []

    for episode in range(nb_episodes):
        state = env.reset()[None, :]
        grads = []
        rewards = []
        total = 0

        while True:
            if show_result and (episode % 1000 == 0):
                env.render()
            action, gradient = policy_gradient(state, W)
            next, reward, done, info = env.step(action)
            grads.append(gradient)
            rewards.append(reward)
            total += reward

            if done:
                break
            state = next[None, :]

        for i in range(len(grads)):
            W += (alpha * grads[i] *
                  sum([y * (gamma ** y) for x, y in enumerate(
                      rewards[i:]
                  )]))

        score.append(total)
        print("[{}]: [{}]".format(episode, total), end="\r", flush=False)

    return score
