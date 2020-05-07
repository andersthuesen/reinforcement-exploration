import gym
import matplotlib.pyplot as plt
from gym_minigrid.wrappers import *
from functions import main_plot, savepdf, train, defaultdict2

from policies import EpsilonGreedy, EpsilonUCB, EpsilonMaxVariance
from networks import EnsambleNetwork, DeepQNetwork
from agents import SarsaLambdaAgent, SarsaAgent, DeepQAgent
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
    network = DeepQNetwork
    DQNAgent = DeepQAgent(env, policy, network=network, **deepQAgentArgs)
    methods.append(("DQN", DQNAgent))

    policy = EpsilonUCB(epsilon=0.1, c=1)
    network = lambda *args, **kwargs: EnsambleNetwork([DeepQNetwork] * 10, *args, **kwargs)
    DQNAgentUCB = DeepQAgent(env, policy, network=network, **deepQAgentArgs)
    methods.append(("DQN-UCB", DQNAgentUCB))

    policy = EpsilonMaxVariance(epsilon=0.1)
    network = lambda *args, **kwargs: EnsambleNetwork([DeepQNetwork] * 10, *args, **kwargs)
    DQNAgentMV = DeepQAgent(env, policy, network=network, **deepQAgentArgs)
    methods.append(("DQN-MV", DQNAgentMV))

    experiments = []
    for k, (name, agent) in enumerate(methods):
        expn = f"experiments/{envn}_{name}"
        for i in range(10):
            train(env, agent, expn, num_episodes=200)
        experiments.append(expn)

    main_plot(experiments,  smoothing_window=10)
    savepdf("./results")
    plt.show()