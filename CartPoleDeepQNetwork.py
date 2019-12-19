import argparse
import sys

import time
import numpy as np
import gym
from gym import wrappers, logger

import matplotlib.pyplot as plt


import random
import os


class Buffer():
    def __init__(self, buffer_size):
        self.dict = {}
        self.buffer_size = buffer_size
        self.insert_number = 0

    def register_experience(self, state, action, next_state, reward, end_episode):
        tuple = (state.tobytes(), action, next_state.tobytes(), reward, end_episode)

        self.dict[tuple] = self.insert_number

        self.insert_number = self.insert_number + 1

        if len(self.dict) == self.buffer_size:
            to_remove = None
            to_remove_time = 0

            for t in self.dict:
                if to_remove is None or self.dict[t] < to_remove_time:
                    to_remove = t
                    to_remove_time = self.dict[t]

            self.dict.pop(t)

    def get_mini_batch(self, size_of_sample):
         return random.sample(self.dict.keys(), min([len(self.dict), size_of_sample]))


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space
        self.buffer = Buffer(100000)

    def act(self, observation, reward, done):
        return self.action_space.sample()

    def store_experience(self, state, action, next_state, reward, end_episode):
        self.buffer.register_experience(state, action, next_state, reward, end_episode)
    
    def get_mini_batch(self, size_of_sample):
        return self.buffer.get_mini_batch(size_of_sample)
      

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
            zz = agent.get_mini_batch(10)
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
        if(list_of_rewards != None):
            if(type(list_of_rewards[0]) == type([]) and len(list_of_rewards[0]) > 1):
                #Calcul min
                minY = []
                maxY = []
                meanY = []
                for i in range(len(list_of_rewards)):
                    localMin = list_of_rewards[i][0]
                    localMax = list_of_rewards[i][0]
                    localMean = 0
                    
                    for j in range(len(list_of_rewards[i])):
                        localMean += list_of_rewards[i][j]
                        localMin = list_of_rewards[i][j] if list_of_rewards[i][j] < localMin else localMin
                        localMax = list_of_rewards[i][j] if list_of_rewards[i][j] > localMax else localMax

                    localMean = float(localMean / len(list_of_rewards[i]))                   
                    meanY.append(localMean)
                    minY.append(localMin)
                    maxY.append(localMax)
                            
                x = [i + 1 for i in range(len(list_of_rewards))]
                fig, ax = plt.subplots()
                ax.plot(x, minY, 'g') 
                ax.plot(x, maxY, 'b') 
                ax.plot(x, meanY, 'r') 
                ax.fill_between(x, minY, maxY, where=maxY >= minY, facecolor='yellow', interpolate=True)
                plt.title("Evolution de la somme des récompenses")
                plt.xlabel("Numéro de l'épisode")
                plt.ylabel("Somme des récompenses")
                plt.show()
                		
            else:
                x = [i + 1 for i in range(len(list_of_rewards))]
                plt.plot(x, list_of_rewards)
                plt.title("Evolution de la somme des récompenses")
                plt.xlabel("Numéro de l'épisode")
                plt.ylabel("Somme des récompenses")
                plt.show()

    show_evolution_of_rewards(list_of_rewards)

