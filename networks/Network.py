class Network:
  def update_phi(self, source, tau=0.01):
    """ 'source' is another DQNNetwork. We will update this networks weights towards
    the weights in source, i.e.
    > self.phi <- self.phi + tau * (source.phi - self.phi)
    Note we can overwrite the weights by setting tau = 1. """
    raise NotImplementedError

  def __call__(self, s):
    """
    Assuming s is of dimension batch_size x env.observations_space.n
    this will return an Array of Q-values of size:
    > batch_size x Actions
    Can be invoked as Q(s)
    """
    raise NotImplementedError

  def fit(self, s, target): 
    """ Fit network weights by minimizing
    > |q_\phi(s,:) - target|^2
    where target is a [batch_size x actions] matrix of target values """ 
    raise NotImplementedError