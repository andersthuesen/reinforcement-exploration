import numpy as np
from policies import Policy

class EpsilonUCB(Policy):
  def __init__(self, epsilon, c):
    super().__init__()
    self.epsilon = epsilon
    self.c = c

  def pi(self, Q, s):
    actions = Q(s)
    action_keys = list(actions.keys())
    if np.random.rand() < self.epsilon:
      return action_keys[np.argmax([mean + self.c * std for mean, std in actions.values()])]
    else:
      # Return greedy action
      return action_keys[np.argmax([mean for mean, std in actions.values()])]

  def __str__(self):
    return f"epsilon-expected-improvement-{self.epsilon}"
