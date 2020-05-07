class Buffer:
    def __init__(self, max_size):
        self.max_size = max_size

    def push(self, s, a, r, sp, done):
        raise NotImplementedError()

    def sample(self, batch_size):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()