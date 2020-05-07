"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
from functions import defaultdict2
from agents import Agent

class BayesianSarsaLambdaAgent(Agent):
    def __init__(self, env, policy, gamma=0.99, alpha=0.5, lamb=0.9):
        super().__init__(env, policy, gamma=gamma)
        self.lamb = lamb
        self.gamma = gamma
        self.e = defaultdict2(self.Q.default_factory)

    def pi(self, s):
      image = s["image"]
      X = torch.tensor(image.flatten()).view((1, -1))

      return 0

    def train(self, s, a, r, sp, done=False):
      pass
      
    def __str__(self):
        return f"BayesianSarsaLambda_{self.policy}_{self.gamma}_{self.alpha}_{self.lamb}"