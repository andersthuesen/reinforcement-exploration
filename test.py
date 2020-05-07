import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.contrib.bnn import HiddenLayer
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal

from tqdm import tqdm
import matplotlib.pyplot as plt

class SimpleDataset:
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, i):
    return (self.X[i], self.y[i])

class BayesianNN(PyroModule):
  def __init__(self, input_size, hidden_size, output_size):
    super(BayesianNN, self).__init__()
    self.fc1 = PyroModule[nn.Linear](input_size, hidden_size)
    self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_size, input_size]).to_event(2))
    self.fc1.bias = PyroSample(dist.Normal(0., 10.).expand([hidden_size]).to_event(1))

    self.out = PyroModule[nn.Linear](hidden_size, output_size)
    self.out.weight = PyroSample(dist.Normal(0., 2.).expand([output_size, hidden_size]).to_event(2))
    self.out.bias = PyroSample(dist.Normal(0., 10.).expand([output_size]).to_event(1))
      
  def forward(self, X, y=None):
    sigma = pyro.sample("sigma", dist.Uniform(0., 1.))

    mean = self.fc1(X)
    # mean = F.relu(mean)
    # mean = self.out(mean)

    with pyro.plate("data", X.size(0)):
      obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)

    return mean


def model(X, y=None):
  hidden_size = 10
  input_size = 1
  output_size = 1

  w1_mean = torch.zeros((hidden_size, input_size))
  w1_std = torch.ones((hidden_size))

  w2_mean = torch.zeros((output_size, hidden_size))
  w2_std = torch.ones((output_size))

  a1 = pyro.sample("a1", F.relu(X @ dist.Normal(w1_mean, w1_std)))
  y = pyro.sample("y", a1 @ dist.Normal(w2_mean, w2_std), dist=y)

  
def guide(X, y):
  hidden_size = 10
  input_size = 1
  output_size = 1

  w1_mean = pyro.param("w1.mean", dist.Normal(0., 1.).expand([input_size, hidden_size]))
  w1_std = pyro.param("w1.std", dist.Normal(0., 5.).expand([1, hidden_size]))

  w2_mean = pyro.param("w2.mean", dist.Normal(0., 1.).expand([output_size, hidden_size]))
  w2_std = pyro.param("w2.std", dist.Normal(0., 5.).expand([output_size]))

  w1 = pyro.sample("w1", dist.Normal(w1_mean, w1_std))
  w2 = pyro.sample("w2", dist.Normal(w2_mean, w2_std))

  with pyro.iarange("data", X.size(0)) as ids:
    a1 = pyro.sample("a1", F.relu(X[ids] @ w1))
    y = pyro.sample("y", w2 @ a1)


def train(model, guide, dataloader, num_epochs=50):
  optim = Adam({ "lr": 0.01 })
  svi = SVI(model, guide, optim, loss=Trace_ELBO())
  for epoch in range(num_epochs):
    print(f"Training epoch: {epoch + 1}")
    tqdm_dataloader = tqdm(dataloader)
    for i, (X, y) in enumerate(tqdm_dataloader):
      X = X.view((X.shape[0], -1))
      y = y.view((y.shape[0], -1))
      loss = svi.step(X, y)
      if i % 10 == 0:
        tqdm_dataloader.set_description(f"Loss: {loss}")

if __name__ == "__main__":

  X = torch.randn(100) * 5
  y = X * 1.5 + 10 * torch.randn(100)

  dataset = SimpleDataset(X, y)
  dataloader = DataLoader(dataset, batch_size=4)

  #bnn = BayesianNN(1, 1, 1)
  #guide = AutoDiagonalNormal(bnn)

  train(model, guide, dataloader);

  plt.scatter(X, y)
  plt.scatter(X, bnn(X.view((X.shape[0], -1))))
  plt.show()


# class BayesianNeuralNetwork(nn.Module):
#   def __init__(self, input_size, hidden_size, output_size):
#     super(BayesianNeuralNetwork, self).__init__()
#     self.input_size = input_size
#     self.hidden_size = hidden_size
#     self.output_size = output_size
    
#     self.w1_mean = torch.zeros((self.input_size, self.hidden_size))
#     self.w1_std = torch.ones((self.input_size, self.hidden_size))
#     self.w2_mean = torch.zeros((self.hidden_size, self.output_size))
#     self.w2_std = torch.zeros((self.hidden_size, self.output_size))

#   def forward(self, X, y):


#   def model(self, X, y):
#     w1_mean = pyro.param("w1.mean", self.w1_mean)
#     w1_std = pyro.param("w1.std", self.w1_std)
#     w2_mean = pyro.param("w2.mean", self.w2_mean)
#     w2_std = pyro.param("w2.std", self.w2_std)

#     with pyro.iarange("data", X.size(0)) as ids:
#       a1 = pyro.sample("a1", HiddenLayer(X[ids], w1_mean, w1_std, include_hidden_bias=False))
#       return pyro.sample("y", HiddenLayer(a1, w2_mean, w2_std), obs=y[ids])

#   def guide(self, X, y):
#     w1_mean = pyro.param("w1.mean", torch.randn_like(self.w1_mean))
#     w1_std = pyro.param("w1.std", torch.randn_like(self.w1_std))
#     w2_mean = pyro.param("w2.mean", torch.randn_like(self.w2_mean))
#     w2_std = pyro.param("w2.std", torch.randn_like(self.w2_std))

#     with pyro.iarange("data", X.size(0)) as ids:
#       a1 = pyro.sample("a1", HiddenLayer(X[ids], w1_mean, w1_std, include_hidden_bias=False))
#       return pyro.sample("y", HiddenLayer(a1, w2_mean, w2_std))

#   def infer_parameters(self, dataloader, num_epochs):
#     optim = Adam({ "lr": 0.01 })
#     svi = SVI(self.model, self.guide, optim, loss=Trace_ELBO())
#     for i in range(num_epochs):
#       print(f"Training epoch: {i + 1}")
#       for X, y in tqdm(dataloader):
#         X = X.view((X.shape[0], -1))
#         y = y.view((y.shape[0], -1))
#         loss = svi.step(X, y)
#         print(loss)


