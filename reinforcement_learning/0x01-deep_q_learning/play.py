#!/usr/bin/env python3
"""Play atari's breakout"""


import gym
import tensorflow.keras as K
create_agent = __import__('train.py').create_agent
create_CNN_q_model = __import__('train.py').create_CNN_q_model


if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    env.reset()

    actions = env.action_space.n
    model = create_CNN_q_model()
    dqn = create_agent(model, actions)

    dqn.compile(K.optimizers.Adam(lr=0.00025), metrics=['mae'])
    dqn.load_weights('policy.h5')
    dqn.test(env, nb_episodes=5, visualize=True)
