import gym
from gym_minigrid.wrappers import *
env = gym.make("MiniGrid-Empty-5x5-v0")
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()