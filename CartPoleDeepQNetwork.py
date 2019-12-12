import argparse
import sys

import time
import numpy as np
import gym
from gym import wrappers, logger

import matplotlib.pyplot as plt


class Buffer():
    @staticmethod
    def tuple_are_equals(tuple_a, tuple_b):
        return tuple_a[1] == tuple_b[1]                 \
            and tuple_a[3] == tuple_b[3]                \
            and tuple_a[4] == tuple_b[4]                \
            and np.array_equal(tuple_a[0], tuple_b[0])  \
            and np.array_equal(tuple_a[2], tuple_b[2])

    def __init__(self, buffer_size):
        self.list = []
        self.buffer_size = buffer_size

    def register_experience(self, state, action, next_state, reward, end_episode):
        tuple = (state, action, next_state, reward, end_episode)

        for i in range(len(self.list)):
            if Buffer.tuple_are_equals(self.list[i], tuple):
                self.list.pop(i)
                break
        if len(tuple) == self.buffer_size:
            self.list.pop(0)
        
        self.list.append(tuple)


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space
        self.buffer = Buffer(100000)

    def act(self, observation, reward, done):
        return self.action_space.sample()

    def store_experience(self, state, action, next_state, reward, end_episode):
        self.buffer.register_experience(state, action, next_state, reward, end_episode)
        
      

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()

    show_first = True

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    # env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    list_of_rewards = []

    for i in range(episode_count):
        ob = env.reset()

        reward_sum = 0
        
        while True:
            if i == 0 and show_first:
              env.render()
              time.sleep(0.1)
            state = ob
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            reward_sum = reward_sum + reward
            agent.store_experience(state, action, ob, reward, done)
            if done:
                break

        list_of_rewards.append(reward_sum)

    env.close()

    def show_evolution_of_rewards(list_of_rewards):
        x = [i + 1 for i in range(len(list_of_rewards))]
        plt.plot(x, list_of_rewards)
        plt.title("Evolution de la somme des récompenses")
        plt.xlabel("Numéro de l'épisode")
        plt.ylabel("Somme des récompenses")
        plt.show()

    show_evolution_of_rewards(list_of_rewards)
