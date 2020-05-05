"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
from functions import defaultdict2
from agents import Agent

import torch
import torch.nn as nn
import pyro
from pyro.nn import PyroModule, PyroParam
from pyro.contrib.bnn import HiddenLayer
from pyro.distributions import Normal


#l1 = HiddenLayer(Normal)

class BayesianNeuralNetwork(PyroModule):
  def __init__(self, input_size, hidden_size, output_size):
    super(BayesianNeuralNetwork, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

  def model(self, X, y):
    w1_mean = pyro.param("w1.mean", torch.zeros((self.input_size, self.hidden_size)))
    w1_std = pyro.param("w1.std", torch.ones((self.input_size, self.hidden_size)))
    a1 = pyro.sample("a1", HiddenLayer(X, w1_mean, w1_std))
    w2_mean = pyro.param("w2.mean", torch.zeros((self.hidden_size, self.output_size)))
    w2_std = pyro.param("w2.std", torch.zeros((self.hidden_size, self.output_size)))

    batches = X.size()

    return pyro.sample("y", HiddenLayer(a1, w2_mean, w2_std), obs=y)

  def guide(self, X, y):
    pass

  def forward(self, X):
    y = HiddenLayer(X, self.w1_mean, self.w1_std)
    return y


network = BayesianNeuralNetwork(1, 7)
y = network(torch.ones((10, 1)))
print(f"{y}")

class BayesianSarsaLambdaAgent(Agent):
    def __init__(self, env, policy, gamma=0.99, alpha=0.5, lamb=0.9):
        super().__init__(env, policy, gamma=gamma)
        self.lamb = lamb
        self.gamma = gamma
        self.e = defaultdict2(self.Q.default_factory)

    def pi(self, s):
      image = s["image"]
      X = torch.tensor(image.flatten()).view((1, -1))
      #print(X)

      return 0

    def train(self, s, a, r, sp, done=False):
      # Test
      #print("train", s, a, r, sp)
      pass
      
    def __str__(self):
        return f"BayesianSarsaLambda_{self.policy}_{self.gamma}_{self.alpha}_{self.lamb}"