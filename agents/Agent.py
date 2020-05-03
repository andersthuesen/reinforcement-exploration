
"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv
from functions import defaultdict2
import warnings

class Agent: 
    def __init__(self, env, policy, gamma=0.99):
        self.env, self.policy, self.gamma = env, policy, gamma
        self.Q = defaultdict2(lambda s: np.zeros(len(env.P[s]) if hasattr(env, 'P') and s in env.P else env.action_space.n))
        
    def pi(self, s): 
        """ Should return the Agent's action in s (i.e. an element contained in env.action_space)"""
        raise NotImplementedError("return action") 

    def train(self, s, a, r, sp, done=False): 
        """ Called at each step of the simulation.
        The agent was in s, took a, ended up in sp (with reward r) and done indicates if the environment terminated """
        raise NotImplementedError() 

    def __str__(self):
        warnings.warn("Please implement string method for caching; include ALL parameters")
        return super().__str__()

    def random_pi(self, s):
        """ Generates a random action given s.

        It might seem strange why this is useful, however many policies requires us to to random exploration.
        We will implement the method depending on whether self.env defines an MDP or just contains an action space.
        """
        if isinstance(self.env, DiscreteEnv):
            return np.random.choice(list(self.env.P[s].keys()))
        else:
            return self.env.action_space.sample()

    def value(self, s):
        return np.max(self.Q[s])
