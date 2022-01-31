#!/usr/bin/env python3
"""Load the Environment"""


import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load pre-trained agent thingy"""
    env = gym.make("FrozenLake-v0", desc=desc,
                   map_name=map_name, is_slippery=is_slippery)
    return env
