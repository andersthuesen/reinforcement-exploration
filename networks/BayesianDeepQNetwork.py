import numpy as np

import torch 
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroModule, DenseNN

from networks import Network

class BayesianDeepQNetwork(PyroModule, Network):
  def __init__(self, env, trainable=True, learning_rate=0.001):
    super(BayesianDeepQNetwork, self).__init__()

    input_size = np.prod(env.observation_space.shape)
    output_size = env.action_space.n

    self.to_latent = PyroModule[nn.Sequential](
      PyroModule[nn.Linear](input_size, 512),
      PyroModule[nn.ReLU](),
      PyroModule[nn.Linear](512, 1024),
      PyroModule[nn.ReLU](),
      PyroModule[nn.Linear](1024, 512),
      PyroModule[nn.ReLU]()
    )

    self.to_loc = PyroModule[nn.Sequential](
      PyroModule[nn.Linear](512, 256),
      PyroModule[nn.ReLU](),
      PyroModule[nn.Linear](256, output_size),
    )

    self.to_scale = PyroModule[nn.Sequential](
      PyroModule[nn.Linear](512, 256),
      PyroModule[nn.ReLU](),
      PyroModule[nn.Linear](256, output_size),
    )

    optimizer = Adam({"lr": 0.01})
    self.svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

  def forward(self, X):
    latent = self.to_latent(X)
    loc = self.to_loc(latent)
    scale = self.to_scale(latent)

    return loc, scale

  # p(x|z)p(z)
  def model(self, X, y):
    with pyro.plate("data", X.shape[0]):
      loc, scale = self.forward(X)
      pyro.sample("y", dist.Normal(loc, scale).to_event(1), obs=y)

  # q(y|x)
  def guide(self, X, y):
    with pyro.plate("data", X.shape[0]):
      loc, scale = self.forward(X)
      pyro.sample("y", dist.Normal(loc, scale).to_event(1))

  def fit(self, X, y):
    X = torch.tensor(X)
    y = torch.tensor(y)
    loss = self.svi.step(X, y)
    print(loss)

  def __call__(self, X):
    X = torch.tensor(X)
    loc, scale = self.forward(X)
    return loc.detach().numpy(), scale.detach().numpy()