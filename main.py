import gym
import matplotlib.pyplot as plt
from gym_minigrid.wrappers import *
from functions import main_plot, savepdf, train, defaultdict2
from agents import SarsaLambdaAgent, SarsaAgent, BayesianSarsaLambdaAgent
from policies import EpsilonGreedy

if __name__ == "__main__":
    envn = "MiniGrid-Empty-5x5-v0" # "CliffWalking-v0"
    env = gym.make(envn)
    #env = OneHotPartialObsWrapper(env)
    alpha = 0.05
    policy = EpsilonGreedy(epsilon=0.1)
    bayesianSarsaLambda = BayesianSarsaLambdaAgent(env, policy)
    methods = [("BayesianSarsaL", bayesianSarsaLambda)]
    #sarsaLagent = SarsaLambdaAgent(env, policy, gamma=0.99, alpha=alpha, lamb=0.9)
    #sarsa = SarsaAgent(env, policy, gamma=0.99, alpha=alpha)
    #methods = [("SarsaL", sarsaLagent), ("Sarsa", sarsa)]

    experiments = []
    for k, (name, agent) in enumerate(methods):
        expn = f"experiments/{envn}_{name}"
        train(env, agent, expn, num_episodes=500, max_runs=10)
        experiments.append(expn)
    main_plot(experiments, smoothing_window=10, resample_ticks=200)
    plt.ylim([-100, 0])
    savepdf("./cliff_sarsa_lambda")
    plt.show()