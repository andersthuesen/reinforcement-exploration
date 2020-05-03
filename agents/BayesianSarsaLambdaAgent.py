"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
from functions import defaultdict2
from agents import SarsaAgent

class BayesianSarsaLambdaAgent(SarsaAgent):
    def __init__(self, env, policy, gamma=0.99, alpha=0.5, lamb=0.9):
        super().__init__(env, policy, gamma=gamma, alpha=alpha)
        self.lamb = lamb
        self.e = defaultdict2(self.Q.default_factory)

    def train(self, s, a, r, sp, done=False):
        ap = self.pi(sp)

        # The ordinary Sarsa learning signal
        delta = r + self.gamma * self.Q[sp][ap] - self.Q[s][a]

        # Update the eligibility trace e(s,a)
        self.e[s][a] += 1
        for s, es in self.e.items():
            for a, e_sa in enumerate(es):
                # Update Q values and eligibility trace
                self.Q[s][a] += self.alpha * delta * e_sa
                self.e[s][a] = self.gamma * self.lamb * e_sa
        if done: # Clear eligibility trace after each episode (missing in pseudo code) and update variables for Sarsa
            self.e.clear()
        else:
            self.a = ap
            self.t += 1

    def __str__(self):
        return f"BayesianSarsaLambda_{self.policy}_{self.gamma}_{self.alpha}_{self.lamb}"