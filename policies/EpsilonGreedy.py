import numpy as np
from policies import Policy

class EpsilonGreedy(Policy):
  def __init__(self, epsilon):
    super().__init__()
    self.epsilon = epsilon

  def pi(self, Q, s):
    actions = Q(s)
    action_keys = list(actions.keys())
    if np.random.rand() < self.epsilon:
      return np.random.choice(action_keys)
    else:
      # Return greedy action
      return action_keys[np.argmax([mean for mean, std in actions.values()])]

  def __str__(self):
    return f"epsilon-greedy-{self.epsilon}"
