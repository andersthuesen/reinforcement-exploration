from agents import QAgent

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