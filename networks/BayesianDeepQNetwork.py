import torch 
import torch.nn as nn

from networks import Network

class BayesianDeepQNetwork(nn.Module, Network):
  def __init__(self, source, tau=0.01):
    super(TwoLayerNet, self).__init__()
    pass

  def model(self, X, y):
    pass

  def guide(self, X, y):
    pass

  def fit(self, X, y):
    pass

  def forward(self, X):
    pass