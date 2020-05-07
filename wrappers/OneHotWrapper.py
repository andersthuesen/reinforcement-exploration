import gym
from gym import spaces
import numpy as np
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX

class OneHotWrapper(gym.core.ObservationWrapper):
  def __init__(self, env):
    super().__init__(env)

    self.image_shape = env.observation_space["image"].shape
    self.num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)

    height, width, depth = self.image_shape
    self.observation_space = spaces.Box(
      low=0,
      high=1,
      shape=(height * width * self.num_bits + 4,),
      dtype="uint8"
    )

  def observation(self, obs):
    image = obs["image"]
    direction = obs["direction"]
    
    out = np.zeros(self.observation_space.shape, dtype='uint8')

    height, width, depth = self.image_shape
    for i in range(height):
      for j in range(width):
        type = image[i, j, 0]
        color = image[i, j, 1]
        state = image[i, j, 2]

        out[i * width + j * self.num_bits + type] = 1
        out[i * width + j * self.num_bits + len(OBJECT_TO_IDX) + color] = 1
        out[i * width + j * self.num_bits + len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1

    out[height * width * self.num_bits + direction] = 1

    return out