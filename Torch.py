"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
import sys
sys.path.append(".")
import numpy as np
import random
from collections import deque
from functions import cache_read, cache_write
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

class DQNNetwork:
    def update_Phi(self, source, tau=0.01):
        """ 'source' is another DQNNetwork. We will update this networks weights towards
        the weights in source, i.e.
        > self.phi <- self.phi + tau * (source.phi - self.phi)
        Note we can overwrite the weights by setting tau = 1. """
        raise NotImplementedError

    def __call__(self, s):
        """
        Assuming s is of dimension batch_size x env.observations_space.n
        this will return an Array of Q-values of size:
        > batch_size x Actions
        Can be invoked as Q(s)
        """
        raise NotImplementedError

    def fit(self, s, target): 
        """ Fit network weights by minimizing
        > |q_\phi(s,:) - target|^2
        where target is a [batch_size x actions] matrix of target values """ 
        raise NotImplementedError

class Buffer:
    def __init__(self, max_size):
        self.max_size = max_size

    def push(self, s, a, r, sp, done):
        raise NotImplementedError()

    def sample(self, batch_size):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

class BasicBuffer(Buffer):
    def __init__(self, max_size):
        super().__init__(max_size)
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return map(lambda x: np.asarray(x), (state_batch, action_batch, reward_batch, next_state_batch, done_batch))

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        cache_write(self.buffer, path)

    def load(self, path):
        self.buffer = cache_read(path)

# Use GPU; If the drivers give you grief you can turn GPU off without a too big hit on performance in the cartpole task
USE_CUDA = torch.cuda.is_available()

Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class TorchNetwork(nn.Module,DQNNetwork):
    def __init__(self, env, trainable=True, learning_rate=0.001, hidden=30):
        nn.Module.__init__(self)
        DQNNetwork.__init__(self)
        self.env = env
        self.hidden = hidden
        self.actions = env.action_space.n
        self.build_model_()
        if trainable:
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        if USE_CUDA:
            self.cuda()

    def build_feature_network(self):
        num_observations = np.prod(self.env.observation_space.shape)
        return (nn.Linear(num_observations, self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.hidden),
                nn.ReLU())

    def build_model_(self):
        num_actions = self.env.action_space.n
        self.model = nn.Sequential(*self.build_feature_network(), nn.Linear(self.hidden,num_actions))

    def forward(self, s):
        s = Variable(torch.FloatTensor(s))
        s = self.model(s)
        return s

    def __call__(self, s):
        return self.forward(s).detach().numpy()

    def fit(self, s, target):
        q_value = self.forward(s)
        loss = (q_value - torch.FloatTensor(target).detach()).pow(2).sum(axis=1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_Phi(self, source, tau=1):
        """
        Polyak adapt weights of this class given source:
        I.e. tau=1 means adopt weights in one step,
        tau = 0.001 means adopt very slowly, tau=1 means instant overwriting
        """
        state = self.state_dict()
        for k, wa in state.items():
            wb = source.state_dict()[k]
            state[k] = wa*(1 - tau) + wb * tau
        self.load_state_dict(state)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        torch.save(self.state_dict(), path+".torchsave")

    def load(self, path):
        self.load_state_dict(torch.load(path+".torchsave"))
        self.eval() # set batch norm layers, dropout, other stuff we don't use

