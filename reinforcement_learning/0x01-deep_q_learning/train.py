#!/usr/bin/env python3


import gym
import numpy as np
import tensorflow.keras as k
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.processors import Processor
from PIL import Image


class AtariProcessor(Processor):
    """Class that defines atari env"""

    def process_observation(self, observation):
        """Method for processing image"""
        assert observation.ndim == 3

        img = Image.fromarray(observation)
        img = img.resize((84, 84), Image.ANTIALIAS).convert('L')
        processed = np.array(img)
        assert processed.shape == (84, 84)

        return processed.astype('uint8')
    
    def process_state_batch(self, batch):
        """Convert imgs to float"""
        processed = batch.astype('float32') / 255.0
        return processed
    
    def process_reward(self, reward):
        """Make reward between -1 and 1"""
        return np.clip(reward, -1., 1.)