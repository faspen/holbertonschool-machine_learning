#!/usr/bin/env python3


import urllib.request
import rl
import tensorflow.keras as K
import gym
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
urllib.request.urlretrieve(
    'http://www.atarimania.com/roms/Roms.rar',
    'Roms.rar')


def create_CNN_q_model():
    """Make rough CNN q model from rubiks code"""
    inputs = K.layers.Input((actions,) + shp)

    # Conv layers
    layer1 = K.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = K.layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = K.layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    # Flatten
    layer4 = K.layers.Flatten()(layer3)
    # Finish off with dense
    layer5 = K.layers.Dense(512, activation="relu")(layer4)
    action = K.layers.Dense(actions, activation="linear")(layer5)

    return K.Model(inputs=inputs, outputs=action)


def create_agent(model, actions):
    """Create agent that plays breakout"""
    memory = SequentialMemory(limit=20000, window_length=actions)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1., value_min=.1,
                                  value_test=.05, nb_steps=20000)
    agent = DQNAgent(model, policy=policy, enable_double_dqn=True,
                     enable_dueling_network=False, dueling_type='avg',
                     nb_actions=actions, memory=memory, nb_steps_warmup=20000,
                     train_interval=4, delta_clip=1.)

    return agent


if __name__ == "__main__":
    # Load in the environment
    env = gym.make("Breakout-v0")
    env.reset()

    # Check observations and actions
    shp = env.observation_space.shape
    print(shp)
    actions = env.action_space.n
    print(actions)

    # Create the model and agent
    model = create_CNN_q_model()
    dqn = create_agent(model, actions)
    dqn.compile(K.optimizers.Adam(lr=0.00025), metrics=['mae'])
    dqn.fit(env, nb_steps=20000, visualize=False, verbose=2)

    # Save weights
    dqn.save_weights('policy.h5', overwrite=True)
