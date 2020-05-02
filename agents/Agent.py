
"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
import Agent
import sys
sys.path.append(".")
import matplotlib.pyplot as plt
from Functions import main_plot, savepdf, train, log_time_series, defaultdict2, existing_runs
import numpy as np
from collections import defaultdict
import gym
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from gym.envs.toy_text.discrete import DiscreteEnv
import warnings
from collections import OrderedDict
import os
import glob
import csv

class Agent(): 
    def __init__(self, env, gamma=0.99, epsilon=0):
        self.env, self.gamma, self.epsilon = env, gamma, epsilon 
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

    def pi_eps(self, s): 
        """ Implement epsilon-greedy exploration. Return random action with probability self.epsilon,
        else be greedy wrt. the Q-values. """
        return self.random_pi(s) if np.random.rand() < self.epsilon else np.argmax(self.Q[s]+np.random.rand(len(self.Q[s]))*1e-8)

    def value(self, s):
        return np.max(self.Q[s])

class QAgent(Agent):
    """
    Q-learning agent
    """
    def __init__(self, env, gamma=1.0, alpha=0.5, epsilon=0.1):
        self.alpha = alpha
        super().__init__(env, gamma, epsilon)
 
    def pi(self, s): 
        """
        Return current action using epsilon-greedy exploration. 
        """
        return self.random_pi_eps(s) if np.random.rand() < self.epsilon else np.argmax(self.Q[s])

    def train(self, s, a, r, sp, done=False): 
        """
        Q-learning update rule
        """
        self.Q[s][a] += self.alpha * (r + self.gamma * max(self.Q[sp]) - self.Q[s][a])
         


    def __str__(self):
        return f"QLearner_{self.gamma}_{self.epsilon}_{self.alpha}"

q_exp = f"experiments/cliffwalk_Q"
def cliffwalk():
    env = gym.make('CliffWalking-v0')
    agent = QAgent(env, epsilon=0.1, alpha=0.5)
    train(env, agent, q_exp, num_episodes=200, max_runs=10)
    return env, q_exp

if __name__ == "__main__":
    env, exp_name = cliffwalk()
    main_plot(exp_name, smoothing_window=10)
    plt.ylim([-100, 0])
    plt.title("Q-learning on " + env.spec._env_name)
    savepdf("Q_learning_cliff")
    plt.show()


class SarsaAgent(QAgent):
    def __init__(self, env, gamma=0.99, alpha=0.5, epsilon=0.1):
        self.t = 0 # indicate we are at the beginning of the episode
        super().__init__(env, gamma=gamma, alpha=alpha, epsilon=epsilon)

    def pi(self, s):
        if self.t == 0: 
            """ we are at the beginning of the episode. Generate a by being epsilon-greedy"""
            return self.pi_eps(s)
        else: 
            return self.a

    def train(self, s, a, r, sp,done=False):
        # Generate A' by being epsilon-greedy
        self.a = self.pi_eps(sp) if not done else -1

        # Perform the update to self.Q[s][a] 
        delta = r + (self.gamma * self.Q[sp][self.a] if not done else 0) - self.Q[s][a]
        self.Q[s][a] += self.alpha * delta
        self.t = 0 if done else self.t + 1 # update current iteration number
        #self.Q[s][a] += self.alpha * (r + self.gamma * max(self.Q[sp]) - self.Q[s][a])

    def __str__(sefl):
        return f"Sarsa{self.gamma}_{self.epsilon}_{self.alpha}"

sarsa_exp = f"experiments/cliffwalk_Sarsa"
if __name__ == "__main__":
    env, q_experiment = cliffwalk()  # get results from Q-learning
    agent = SarsaAgent(env, epsilon=0.1, alpha=0.5)
    train(env, agent, sarsa_exp, num_episodes=200, max_runs=10)
    main_plot([q_experiment, sarsa_exp], smoothing_window=10)
    plt.ylim([-100, 0])
    plt.title("Q and Sarsa learning on " + env.spec._env_name)
    savepdf("QSarsa_learning_cliff")
    plt.show()

"""
class Agent:
  def __init__(self, env, gamma):
    self.env = env
    self.gamma = gamma

  def pi(self, s):
    pass

  def train(self, s, a, r, sp, done=False):
    pass
"""