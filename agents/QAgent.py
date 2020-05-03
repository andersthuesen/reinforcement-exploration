import numpy as np
from agents import Agent
from policies import Policy

class QAgent(Agent):
    """
    Q-learning agent
    """
    def __init__(self, env, policy: Policy, gamma=1.0, alpha=0.5):
        self.alpha = alpha
        self.policy = policy
        super().__init__(env, policy, gamma)
 
    def pi(self, s):
        return self.policy.pi(lambda s: { a: (self.Q[s][a], 0) for a in np.arange(len(self.Q[s])) }, s)

    def train(self, s, a, r, sp, done=False): 
        """
        Q-learning update rule
        """
        self.Q[s][a] += self.alpha * (r + self.gamma * max(self.Q[sp]) - self.Q[s][a])

    def __str__(self):
        return f"QLearner_{self.gamma}_{self.epsilon}_{self.alpha}"