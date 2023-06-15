"""
Environments and wrappers for MarioKart training.
"""

import gym
import numpy as np
import torch
import retro
import cv2

from gym.wrappers import FrameStack
from create_game import register_json_folder, register_json_str

import matplotlib.pyplot as plt
plt.rcParams["animation.html"] = "html5"
import matplotlib.animation

def make_env(state, stacks, size, game="SuperMarioKart-Snes", record=False):
    env = gym.make(state)
    env = KartObservation(env, size=size)
    env = FrameStack(env, num_stack=stacks)
    return env


class KartObservation(gym.ObservationWrapper):
    """
    Prior operations done on the input images for the neural network
    """

    def __init__(self, env, size=(128, 56)):

        super(KartObservation, self).__init__(env)

        self.size = size

    def observation(self, obs):
        # To gray scale
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        # Adding a channel to the image
        obs = np.expand_dims(obs, axis=-1)
        # We remove the minimap
        obs = obs[3:106, 0:256, :]
        # Resizing the image
        obs = cv2.resize(obs, self.size)
        # Converting the image to an array
        obs = obs / 255.0
        return obs

