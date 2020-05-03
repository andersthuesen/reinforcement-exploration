import numpy as np
from agents import QAgent
from policies import Policy

class SarsaAgent(QAgent):
    def __init__(self, env, policy: Policy, gamma=0.99, alpha=0.5):
        self.t = 0 # indicate we are at the beginning of the episode
        super().__init__(env, policy, gamma=gamma, alpha=alpha)

    def train(self, s, a, r, sp,done=False):
        # Generate A' by being epsilon-greedy
        self.a = self.pi(sp) if not done else -1

        # Perform the update to self.Q[s][a] 
        delta = r + (self.gamma * self.Q[sp][self.a] if not done else 0) - self.Q[s][a]
        self.Q[s][a] += self.alpha * delta
        self.t = 0 if done else self.t + 1 # update current iteration number

    def __str__(self):
        return f"Sarsa_{self.policy}_{self.gamma}_{self.alpha}"