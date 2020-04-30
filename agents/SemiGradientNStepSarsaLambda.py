import Agent

class SemiGradientNStepSarsaLambda(Agent):
	def __init__(self, env, n, lambda, epsilon):
		super().__init__(env)
		self.n = n
		self.epsilon = ellipsis
		self.lambda = lambda

	def pi(self, s):
		pass

	def train(self, s, a, r, sp, done=False):
		pass


"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.

References:
	[SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (See sutton2018.pdf). 
"""
import sys
sys.path.append(".")
import gym
import numpy as np
from irlc.ex08.agent import train
from irlc import main_plot, savepdf
import matplotlib.pyplot as plt
from irlc.ex10.semi_grad_sarsa import LinearSemiGradSarsa

class LinearSemiGradSarsaLambda(LinearSemiGradSarsa):
		def __init__(self, env, gamma=0.99, epsilon=0.1, alpha=0.5, lamb=0.9, q_encoder=None):
				"""
				Sarsa(Lambda) with linear feature approximators (see (SB18, Section 12.7)).
				"""
				super().__init__(env, gamma, alpha=alpha, epsilon=epsilon, q_encoder=q_encoder)
				self.Q, self.e = None, None # we will not need these since we are using linear function approximators
				self.z = np.zeros(self.q.d) # Vector to store eligibility trace (same as in
				self.lamb = lamb # lambda in Sarsa(lambda)

		def pi(self, s):
				if self.t == 0:
						self.a = self.pi_eps(s)
						self.x = self.q.x(s,self.a)
						self.Q_old = 0
				return self.a

		def train(self, s, a, r, sp, done=False):
				a_prime = self.pi_eps(s) if not done else -1
				x_prime = self.q.x(sp, a_prime)
				"""
				Update the eligibility trace self.z and the weights self.w here. 
				Note Q-values are approximated as Q = w @ x.
				We use Q_prime = w * x(s', a') to denote the new q-values for (stored for next iteration as in the pseudo code)
				"""
				a_prime = LinearSemiGradSarsa.pi_eps(self, s) if not done else -1 #!b
				x_prime = self.q.x(sp, a_prime)
				Q = self.w @ self.x
				Q_prime = self.w @ x_prime
				delta = r + (self.gamma * Q_prime if not done else 0) - Q
				self.z = self.gamma * self.lamb * self.z + (1-self.alpha * self.gamma * self.lamb *self.z @ self.x) * self.x
				self.w += self.alpha * (delta + Q - self.Q_old) * self.z - self.alpha * (Q-self.Q_old) * self.x

				# TODO: 5 lines missing.
				#raise NotImplementedError("Update z, w")
				if done: # Reset eligibility trace and time step t as in Sarsa.
						self.t = 0
						self.z = self.z * 0
				else:
						self.Q_old, self.x, self.a = Q_prime, x_prime, a_prime
						self.t += 1

		def __str__(self):
				return f"LinearSarsaLambda_{self.gamma}_{self.epsilon}_{self.alpha}_{self.lamb}"
