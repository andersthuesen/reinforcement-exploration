import numpy as np
from agents import Agent

class QAgent(Agent):
    """
    Q-learning agent
    """
    def __init__(self, env, gamma=1.0, alpha=0.5, epsilon=0.1):
        self.alpha = alpha
        super().__init__(env, gamma, epsilon)
 
    def pi(self, s): 
        """
        Return current action using epsilon-greedy exploration. 
        """
        return self.random_pi_eps(s) if np.random.rand() < self.epsilon else np.argmax(self.Q[s])

    def train(self, s, a, r, sp, done=False): 
        """
        Q-learning update rule
        """
        self.Q[s][a] += self.alpha * (r + self.gamma * max(self.Q[sp]) - self.Q[s][a])
         


    def __str__(self):
        return f"QLearner_{self.gamma}_{self.epsilon}_{self.alpha}"