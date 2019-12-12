import argparse
import sys

import time

import gym
from gym import wrappers, logger

import matplotlib.pyplot as plt

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

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
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            reward_sum = reward_sum + reward
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

        list_of_rewards.append(reward_sum)

    env.close()

    def show_evolution_of_rewards(list_of_rewards):
        x = [i + 1 for i in range(len(list_of_rewards))]
        plt.scatter(x, list_of_rewards)
        plt.title("Evolution de la somme des récompenses")
        plt.xlabel("Numéro de l'épisode")
        plt.ylabel("Somme des récompenses")
        plt.show()

    show_evolution_of_rewards(list_of_rewards)
