# AI
import torch
import torch.nn as nn
import gym
from gym import wrappers, logger
import copy

# Display
import matplotlib.pyplot as plt
import time

# Code Structure
from collections import OrderedDict
from enum import Enum
import random


#######################################################################################################################
#### Constants

# Display when the trigger is triggered, which means the network is copied
TRIGGER_VERBOSE = False

GAMMA = 0.1
NUMBER_OF_EPISODES = 200
show_every = 50
learning_rate = 1e-2
buffer_size=100000
batch_size=20
extra_layers_size=[15, 5]       # For each hidden layer, number of neurons
trigger_every = 400


#######################################################################################################################
#### Network implementation


# Raise an error if false when constructing a Network without using a static method
# We use this because python does not support natively multiple constructors
NetworkAllowConstruction = False

# Linear Neural Network
class Network(nn.Module):
    @staticmethod
    def build_initial_network(sizes):
        global NetworkAllowConstruction
        NetworkAllowConstruction = True
        network = Network()

        d = OrderedDict()
        d['input'] = nn.Linear(sizes[0], sizes[1])

        for i in range(1, len(sizes) - 1):
            d['relu' + str(i)] = nn.LeakyReLU()
            d['layer' + str(i + 1)] = nn.Linear(sizes[i], sizes[i + 1])

        network.model = nn.Sequential(d)
        print(network.model)
        network.loss_fn = torch.nn.MSELoss(reduction='sum')
        network.optimizer = torch.optim.Adam(network.model.parameters(), lr=learning_rate)

        return network

    @staticmethod
    def clone_network(original_network):
        global NetworkAllowConstruction
        NetworkAllowConstruction = True
        network = Network()

        network.model = copy.deepcopy(original_network.model)
        network.loss_fn = torch.nn.MSELoss(reduction='sum')
        network.optimizer = torch.optim.Adam(network.model.parameters(), lr=learning_rate)
        
        return network

    def __init__(self):
        global NetworkAllowConstruction
        if not NetworkAllowConstruction:
            raise Exception("Network construction with init is not allowed")
        NetworkAllowConstruction = False

        super().__init__()

    def forward(self, x):
        return self.model(x)

    def learn(self, qValue, expectedqValues):
        loss = self.loss_fn(qValue, expectedqValues)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# A list of the last buffer_size experiences
class Buffer():
    def __init__(self, buffer_size=buffer_size):
        self.buffer_size = buffer_size
        self.queue = []
        self.index = 0

    def register_experience(self, state, action, next_state, reward, end_episode):
        t = (torch.Tensor(state), action, torch.Tensor(next_state), reward, end_episode)

        if len(self.queue) < self.buffer_size:
            self.queue.append(t)
        else:
            self.queue[self.index] = t
            self.index = (self.index + 1) % self.buffer_size
    
    def get_mini_batch(self, size_of_sample):
        return random.sample(self.queue, min([len(self.queue), size_of_sample]))


#######################################################################################################################
#### EXPLORATIONS CLASS

# Epsilon Exploration (random action with a probability of epsilon, else best)
class EpsilonExploration():
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose_action(self, action_space, observation, neural_network):
        if random.random() < self.epsilon:
            return action_space.sample()
        else:
            x = torch.Tensor(observation)
            qValues = neural_network.forward(x)

            chosen_action, value = None, None

            for i, action in enumerate(qValues):
                current_value = action.item()

                if chosen_action is None or value < current_value:
                    chosen_action = i
                    value = current_value

            return chosen_action


# BoltzmannExploration
class BoltzmannExploration():
    def __init__(self, tau):
        self.tau = tau

    def choose_action(self, action_space, observation, neural_network):
        x = torch.Tensor(observation)
        qValues = torch.exp(neural_network.forward(x) / self.tau)
        qValuesSum = torch.sum(qValues) 
        probs = qValues / qValuesSum

        rng = random.random()

        for i, prob in enumerate(probs):
            if rng < prob:
                return i
            rng -= prob
        
        return action_space.sample()


#######################################################################################################################
#### DQNAgent

class PeriodicTrigger():
    def __init__(self, trigger_every=10000):
        self.trigger_every = trigger_every
        self.current = 0
    
    def IsTriggered(self):
        self.current += 1
        if self.current == self.trigger_every:
            if TRIGGER_VERBOSE:
                print("Trigger")
            self.current = 0
        
        return self.current == 0


class DQNAgent(object):
    def __init__(self, env, exploration=EpsilonExploration(epsilon=0), target_update=None):
        self.action_space = env.action_space
        self.buffer = Buffer()
        neural_net_structure = [env.observation_space.shape[0]] + extra_layers_size + [env.action_space.n]
        self.neural_network = Network.build_initial_network(neural_net_structure)
        self.exploration = exploration
        self.batch_size = batch_size
        self.target = self.neural_network if target_update is not None else Network.clone_network(self.neural_network)
        self.target_update = target_update

    def act(self, observation, reward, done):
        return self.exploration.choose_action(self.action_space, observation, self.neural_network)

    def store_experience(self, state, action, next_state, reward, end_episode):
        self.buffer.register_experience(state, action, next_state, reward, end_episode)
        self.learn()

    def learn(self):
        sampled_experiences = self.get_mini_batch(self.batch_size)

        for input, action_id, next_state, reward, end_of_episode in sampled_experiences:
            qValues_pred = self.neural_network.forward(input)
            qValues_real = torch.Tensor(qValues_pred) # Copy all the qValues

            expected_qValue = reward
            if not end_of_episode:
                expected_qValue += GAMMA * torch.max(self.target.forward(next_state))

            qValues_real[action_id] = expected_qValue

            self.neural_network.learn(qValues_pred, qValues_real)

            if self.target_update is not None and self.target_update.IsTriggered():
                self.target = Network.clone_network(self.neural_network)

    def get_mini_batch(self, size_of_sample):
        return self.buffer.get_mini_batch(size_of_sample)


#######################################################################################################################
#### MAIN

def show_evolution_of_rewards(list_of_rewards):
    if not list_of_rewards:
        return
    
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
        _, ax = plt.subplots()
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


if __name__ == '__main__':

    logger.set_level(logger.INFO)

    env = gym.make('CartPole-v1')
    # env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    #exploration = BoltzmannExploration(tau=0.5)
    exploration = EpsilonExploration(epsilon=0.1)
    agent = DQNAgent(env=env, exploration=exploration, target_update=PeriodicTrigger(trigger_every=trigger_every))

    list_of_rewards = []

    for i in range(NUMBER_OF_EPISODES):
        ob = env.reset()

        reward_sum = 0
        reward = 0
        done = False

        if show_every != 0 and i % show_every == 0:
            print("Display experience " + str(i) + " / " + str(NUMBER_OF_EPISODES))

        while True:
            if show_every != 0 and i % show_every == 0:
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

    show_evolution_of_rewards(list_of_rewards)
