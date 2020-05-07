import numpy as np
from agents import Agent
from buffers import BasicBuffer
from networks import DeepQNetwork

class DeepQAgent(Agent):
    def __init__(self, env, policy, network=None, buffer=None, gamma=0.99, alpha=0.001, batch_size=32, replay_buffer_size=2000, replay_buffer_minreplay=500):
        super().__init__(env, policy, gamma=gamma)
        self.batch_size = batch_size
        self.memory = BasicBuffer(replay_buffer_size) if buffer is None else buffer 
        self.Q = network(env, trainable=True) if network else QNetwork(env, trainable=True, learning_rate=alpha)
        self.replay_buffer_minreplay = replay_buffer_minreplay
        self.steps, self.episodes = 0, 0

    def pi(self, s):
        return self.policy.pi(lambda s: { a: q for a, q in enumerate(zip(*self.Q(s))) }, s)

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
        s, a, r, sp, done = self.memory.sample(self.batch_size) 
        target, _ = self.Q(s)
        Q_sp, _ = self.Q(sp)
        target[np.arange(self.batch_size), a] = r.squeeze() + self.gamma * np.max(Q_sp, axis=1) * (1 - done)
        self.Q.fit(s, target)

    def __str__(self):
        return f"DQN_{self.gamma}"