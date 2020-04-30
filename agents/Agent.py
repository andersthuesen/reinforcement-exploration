import Agent

class Agent:
  def __init__(self, env, gamma):
    self.env = env
    self.gamma = gamma

  def pi(self, s):
    pass

  def train(self, s, a, r, sp, done=False):
    pass