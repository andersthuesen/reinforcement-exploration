import Agent

class SarsaLambdaAgent(Agent):
  def __init__(self, env, gamma, alpha, lambda):
    super().__init__(self, env, gamma)
    self.lambda = lambda
    self.alpha = alpha
    self.Q = None # Make Q function
    self.z = None # Elegibility trace
    self.w = ... # Test

  def Q(self, s, a):
    # Implement Q function here
    pass

  def pi(self, s):
    pass

  def train(self, s, a, r, sp, done=False):
    #Q = self.w @ s
    Q = self.w @ s
    Q_prime = self.w @ sp
    Q = Q + self.alpha * ((r + self.gamma * Q_prime) - Q)


    pass