import os
import argparse
import torch
import numpy as np

from src.GymEnv import make_env
from src.utils import load_inference

parser = argparse.ArgumentParser()

parser.add_argument("-c", "--checkpoint", type=str, default="checkpoint.pt")

from create_game import register_json_folder, register_json_str
import gym

import matplotlib.pyplot as plt
plt.rcParams["animation.html"] = "html5"
import matplotlib.animation

if __name__ == "__main__":
    args = parser.parse_args()
    # Check if the files exists
    if (not os.path.isfile(args.state)) or (not os.path.isfile(args.checkpoint)):
        raise ValueError("Arguments are not valid")

    # Load model
    model = load_inference(args.checkpoint).float().to("cuda")
    env = make_env('CreateLevelCustomPush-v0', stacks=1, size=(84,84))
    obs = env.reset()
    done = False

    lstm_hxs = ([torch.zeros((1, 1, 256)).to("cuda")] * 2)
    frames = []
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float).to("cuda")
        obs_tensor = obs_tensor.unsqueeze(0)
        action, lstm_hxs = model.act_greedy(obs_tensor, lstm_hxs)

        action = torch.tensor(action.cpu().tolist()[0])
        obs, reward, done, _ = env.step(action.tolist()[0])
        frames.append(env.render(mode="rgb_array_high_mega_changed_colors"))


    def update(i):
        ax.imshow(frames[i])

    fig, ax = plt.subplots(1,1)
    ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(frames))
    ani.save("result.gif", fps = 20)
        
env.close()
