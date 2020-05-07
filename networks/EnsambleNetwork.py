import numpy as np
from networks import Network

class EnsambleNetwork(Network):
    def __init__(self, networks, *args, **kwargs):
        Network.__init__(self)
        print(args, kwargs)
        self.networks = [
          Network(*args, **kwargs)
          for Network in networks
        ]

    def __call__(self, s):
        means, stds = zip(*[network(s) for network in self.networks])
        means = np.array(means)
        if len(means.shape) > 2:
            means = means.swapaxes(0, 1)
        
        return np.mean(means, axis=-2), np.std(means, axis=-2)

    def fit(self, s, target):
        for network in self.networks:
            network.fit(s, target)