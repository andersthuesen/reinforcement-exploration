import gym
import matplotlib.pyplot as plt
from gym_minigrid.wrappers import *
from functions import main_plot, savepdf, train, defaultdict2

from policies import EpsilonGreedy
from networks import EnsambleNetwork, DeepQNetwork, BayesianDeepQNetwork
from agents import SarsaLambdaAgent, SarsaAgent, BayesianSarsaLambdaAgent, DeepQAgent
from wrappers import OneHotWrapper

if __name__ == "__main__":
    envn = "MiniGrid-Empty-8x8-v0"

    env = gym.make(envn)
    env = OneHotWrapper(env)

    methods = []

    deepQAgentArgs = {
        "gamma": 0.95,
        "replay_buffer_minreplay": 300,
        "replay_buffer_size": 500000
    }

    policy = EpsilonGreedy(epsilon=0.1)
    network = lambda *args, **kwargs: EnsambleNetwork([DeepQNetwork] * 2, *args, **kwargs)
    EnsambleDeepQAgent = DeepQAgent(env, policy, network=network, **deepQAgentArgs)
    methods.append(("EnsambleDQN", EnsambleDeepQAgent))

    policy = EpsilonGreedy(epsilon=0.1)
    network = DeepQNetwork
    deepQAgent = DeepQAgent(env, policy, network=network, **deepQAgentArgs)
    methods.append(("DQN", EnsambleDeepQAgent))

    experiments = []
    for k, (name, agent) in enumerate(methods):
        expn = f"experiments/{envn}_{name}"
        train(env, agent, expn, num_episodes=200)
        experiments.append(expn)
    main_plot(experiments, units="Unit", estimator=None, smoothing_window=None)
    savepdf("./cliff_sarsa_lambda")
    plt.show()