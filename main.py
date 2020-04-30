import gym
from gym_minigrid.wrappers import *
from agents import SarsaLambdaAgent


env = gym.make("MiniGrid-Empty-5x5-v0")
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action) 

  if done:
    observation = env.reset()
env.close()


# Code from Tue down here 

import sys
import itertools
import numpy as np
from irlc import log_time_series
from tqdm import tqdm
from irlc.utils.common import defaultdict2
from gym.envs.toy_text.discrete import DiscreteEnv
from irlc.utils.irlc_plot import existing_runs
import warnings
from collections import OrderedDict
import os
import glob
import csv

def train(env, agent, experiment_name=None, num_episodes=None, verbose=True, reset=True, max_steps=1e10,
          max_runs=None, saveload_model=False):

    if max_runs is not None and existing_runs(experiment_name) >= max_runs:
            return experiment_name, None, True
    stats = []
    steps = 0
    ep_start = 0
    if saveload_model:  # Code for loading/saving models
        did_load = agent.load(os.path.join(experiment_name))
        if did_load:
            stats, recent = load_time_series(experiment_name=experiment_name)
            ep_start, steps = stats[-1]['Episode']+1, stats[-1]['Steps']

    done = False
    with tqdm(total=num_episodes, disable=not verbose) as tq:
        for i_episode in range(num_episodes): 
            s = env.reset() if reset else (env.s if hasattr(env, "s") else env.env.s) 
            reward = []
            for _ in itertools.count():
                a = agent.pi(s)
                sp, r, done, _ = env.step(a)
                agent.train(s, a, r, sp, done)
                reward.append(r)
                steps += 1
                if done or steps > max_steps:
                    break
                s = sp 

            stats.append({"Episode": i_episode + ep_start,
                          "Accumulated Reward": sum(reward),
                          "Average Reward": np.mean(reward),
                          "Length": len(reward),
                          "Steps": steps})
            tq.set_postfix(ordered_dict=OrderedDict(stats[-1]))
            tq.update()
    sys.stderr.flush()
    if saveload_model:
        agent.save(experiment_name)
        if did_load:
            os.rename(recent+"/log.txt", recent+"/log2.txt")  # Shuffle old logs

    if experiment_name is not None:
        log_time_series(experiment=experiment_name, list_obs=stats)
        print(f"Training completed. Logging: '{', '.join( stats[0].keys()) }' to {experiment_name}")
    return experiment_name, stats, done

