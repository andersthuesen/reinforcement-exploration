"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
from functions import defaultdict2
from agents import SarsaAgent

class SarsaLambdaAgent(SarsaAgent):
    def __init__(self, env, gamma=0.99, epsilon=0.1, alpha=0.5, lamb=0.9):
        """
        Implementation of Sarsa(Lambda) in the tabular version, see
        http://incompleteideas.net/book/first/ebook/node77.html
        for details (and note that as mentioned in the exercise description/lecture Sutton forgets to reset the
        eligibility trace after each episode).
        Note 'lamb' is an abbreveation of lambda, because lambda is a reserved keyword in python.

        The constructor initializes e, the eligibility trace, as a datastructure similar to self.Q. I.e.
        self.e[s][a] is the eligibility trace e(s,a).
        """
        super().__init__(env, gamma=gamma, alpha=alpha, epsilon=epsilon)
        self.lamb = lamb
        self.e = defaultdict2(self.Q.default_factory)

    def train(self, s, a, r, sp, done=False):
        ap = self.pi_eps(s)

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
        return f"SarsaLambda_{self.gamma}_{self.epsilon}_{self.alpha}_{self.lamb}"