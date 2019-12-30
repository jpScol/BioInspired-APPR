import argparse
import sys

import time
import numpy as np
import gym
from gym import wrappers, logger

import matplotlib.pyplot as plt

import random
import os

import torch.nn as nn

from collections import OrderedDict

import torch

from enum import Enum

# == Auteurs 
# Julian Bruyat 11706770
# Jean-Philippe Tisserand 11926733

# DONE : 2.3 Deep Q Learning - Question 6

# TODO : 
# .backward => donne l'erreur
# utiliser un optimiseur


class Network(nn.Module):
    def __init__(self, sizes):
        super().__init__()

        d = OrderedDict()

        d['input'] = nn.Linear(sizes[0], sizes[1])

        for i in range(1, len(sizes) - 1):
            d['relu' + str(i)] = nn.ReLU()
            d['hidden' + str(i + 1)] = nn.Linear(sizes[i], sizes[i + 1])

        d['softmax'] = nn.Softmax(dim=0)

        self.model = nn.Sequential(d)
        print(self.model)


    def forward(self, x):
        return self.model(x)


class Buffer():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.queue = []
        self.index = 0

    def register_experience(self, state, action, next_state, reward, end_episode):
        tuple = (state.tobytes(), action, next_state.tobytes(), reward, end_episode)

        if len(self.queue) < self.buffer_size:
            self.queue.append(tuple)
            self.index += 1
        else:
            self.queue[self.index] = tuple
            self.index += 1

        self.index = self.index % self.buffer_size
    
    def get_mini_batch(self, size_of_sample):
         return random.sample(self.queue, min([len(self.queue), size_of_sample]))


class Strategy(Enum):
    EXPLORE = 0
    EXPLOIT = 1

EPSILON = 0.4

class NeuralNetworkAgent(object):
    """The world's simplest agent!"""
    def __init__(self, env, extra_layers_size):
        self.action_space = env.action_space
        self.buffer = Buffer(100000)
        neural_net_structure = [env.observation_space.shape[0]] + extra_layers_size + [env.action_space.n]
        self.neural_network = Network(neural_net_structure)

    def act(self, observation, reward, done):
        strategy = self.get_strategy()

        if strategy == Strategy.EXPLORE:
            act = self.action_space.sample()
            return act
        else:
            x = torch.Tensor(observation)
            qValues = self.neural_network.forward(x)

            chosen_action, value = None, None

            for i, action in enumerate(qValues):
                current_value = action.item()

                if chosen_action is None or value < current_value:
                    chosen_action = i
                    value = current_value

            return chosen_action


    def get_strategy(self):
        rand = random.random()
        return Strategy.EXPLOIT if rand > EPSILON else Strategy.EXPLORE

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
    agent = NeuralNetworkAgent(env, [50])

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

