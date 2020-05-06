"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
import sys
sys.path.append(".")
import gym
import numpy as np
import os
from matplotlib import pyplot as plt
from agents import Agent
from functions import train, savepdf, cache_write, cache_read, cache_exists
from Torch import BasicBuffer
from Torch import TorchNetwork as QNetwork  # Torch network architechture

class DeepQAgent(Agent):
    def __init__(self, env, network=None, buffer=None, gamma=0.99, epsilon=None, alpha=0.001, batch_size=32,
                    replay_buffer_size=2000, replay_buffer_minreplay=500):
        # Ensure 'epsilon' is a function to allow gradually decreasing exploration rate
        epsilon = epsilon if callable(epsilon) else lambda steps, episodes: epsilon
        super().__init__(env, gamma=gamma, epsilon=epsilon)
        self.memory = BasicBuffer(replay_buffer_size) if buffer is None else buffer 
        """ 
        All the 'deep' stuff is handled by a seperate class. For instance
        self.Q(s) 
        will return a [batch_size x actions] matrix of Q-values
        """ 
        self.Q = network(env, trainable=True) if network else QNetwork(env, trainable=True, learning_rate=alpha)
        self.batch_size = batch_size
        self.replay_buffer_minreplay = replay_buffer_minreplay
        self.steps, self.episodes = 0, 0

    def pi(self, s):
        eps_ = self.epsilon(self.steps, self.episodes) # get the learning rate
        # return action by regular epsilon-greedy exploration
        return self.env.action_space.sample() if np.random.rand() < eps_ else np.argmax(self.Q(s[np.newaxis,...]))

    def train(self, s, a, r, sp, done=False):
        self.memory.push(s, a, r, sp, done) # save current observation 
        if len(self.memory) > self.replay_buffer_minreplay:
            self.experience_replay() # do the actual training step
        self.steps, self.episodes = self.steps + 1, self.episodes + done

    def experience_replay(self):
        """
        Perform the actual deep-Q learning step.

        The actual learning is handled by calling self.Q.fit(s,target)
        where s is defined as below (i.e. all states from the replay buffer)
        and target is the desired value of self.Q(s).

        Note that target must therefore be of size Batch x Actions. In other words fit minimize

        |Q(s) - target|^2

        which must implement the proper cost. This can be done by setting most entries of target equal to self.Q(s)
        and the other equal to y, which is Q-learning target for Q(s,a). """
        """ First we sample from replay buffer. Returns numpy Arrays of dimension 
        > [self.batch_size] x [...]]
        for instance 'a' will be of dimension [self.batch_size x 1]. 
        """
        s,a,r,sp,done = self.memory.sample(self.batch_size) 
        target = self.Q(s)
        target[np.arange(self.batch_size), a] = r.squeeze() + self.gamma * np.max(self.Q(sp), axis=1) * (1-done)
        self.Q.fit(s, target)

# Save/load module
    def save(self, path): 
        if not os.path.isdir(path):
            os.makedirs(path)
        self.Q.save(os.path.join(path, "Q"))
        cache_write(dict(steps=self.steps, episodes=self.episodes), os.path.join(path, "agent.pkl"))
        mpath = os.path.join(path, "memory.pkl")
        import shutil
        if os.path.isfile(mpath):
            shutil.move(mpath, mpath +".backup") # shuffle file
        self.memory.save(mpath)

    def load(self, path): # allows us to save/load model
        if not cache_exists(os.path.join(path, "agent.pkl")):
            return False
        for k, v in cache_read(os.path.join(path, "agent.pkl")).items():
            self.__dict__[k] = v
        self.Q.load(os.path.join(path, "Q"))
        self.memory.load(os.path.join(path, "memory.pkl"))
        return True

    def __str__(self):
        return f"basic_DQN{self.gamma}"

def linear_interp(maxval, minval, delay, miniter):
    """
    Will return a function f(i) with the following signature:

    f(i) = maxval for i < delay
    f(i) = linear interpolate between max/minval until delay+miniter
    f(i) = miniter for i > delay+miniter
    """
    return lambda steps, episodes: min(max([maxval- ((steps-delay)/miniter)*(maxval-minval), minval]), maxval)

cartpole_dqn_options = dict(gamma=0.95, epsilon=linear_interp(maxval=1,minval=0.01,delay=300,miniter=5000),
                            replay_buffer_minreplay=300, replay_buffer_size=500000)

def mk_cartpole():
    env = gym.make("CartPole-v0")
    agent = DeepQAgent(env, **cartpole_dqn_options)
    return env, agent

if __name__ == "__main__":
    env_id = "CartPole-v0"
    ex = f"experiments/torch_cartpole_dqn"
    num_episodes = 20 # we train 20 just episodes at a time
    for j in range(10): # train for a total of 200 episodes
        env, agent = mk_cartpole()
        """
        saveload_model=True means it will store and load intermediate results
        i.e. we can resume training later. It will not be very useful for cartpole, but necesary for e.g. 
        the atari environment which can run for days
        """
        train(env, agent, experiment_name=ex, num_episodes=num_episodes, saveload_model=True)
        from irlc import main_plot
    main_plot([ex], units="Unit", estimator=None, smoothing_window=None)
    savepdf("cartpole_dqn")
    plt.show()

