from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import os.path as path
from datetime import datetime
from pathlib import Path

import numpy as np
from mdp.cartpole_mdp import CartPoleMDP
from mdp.pendulum_mdp import PendulumMDP
from mdp.three_pole_mdp import ThreePoleMDP
from PIL import Image
from tqdm import trange
import dmc2gym
import yaml
import torch
from torchvision.transforms.functional import rgb_to_grayscale



root_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)

widths = {"pendulum": 48, "cartpole": 80, "threepole": 80, "ccartpole": 80}
heights = {"pendulum": 48, "cartpole": 80, "threepole": 80, "ccartpole": 80}
state_dims = {"pendulum": 2, "cartpole": 4, "threepole": 6, "ccartpole": 3}
u_dims = {"ccartpole": 1}
frequencies = {"pendulum": 50, "cartpole": 50, "threepole": 50, "ccartpole": 50}
mdps = {"pendulum": PendulumMDP, "cartpole": CartPoleMDP, "threepole": ThreePoleMDP}


def sample(env_name, sample_size, noise):
    """
    env_name = ccartpole
    return [(x, u, x_next)]
    """
    width, height, frequency = widths[env_name], heights[env_name], frequencies[env_name]
    s_dim = state_dims[env_name]
    u_dim = u_dims[env_name]
    #load yaml configuration
    with open(os.path.join("data/", env_name+'.yaml')) as file:
        config = yaml.safe_load(file)
    env = dmc2gym.make(
        domain_name=config['domain_name'],
        task_name=config['task_name'],
        seed=config['seed'],
        visualize_reward=False,
        from_pixels=(config['env']['encoder_type'] == 'pixel'),
        height=config['env']['pre_transform_image_size'],
        width=config['env']['pre_transform_image_size'],
        frame_skip=config['env']['action_repeat'])
    
    env.seed(config['seed'])

    # Data buffers to fill.
    x_data = np.zeros((sample_size, 2, width, height), dtype="float32")  # todo: change the dimensions
    u_data = np.zeros((sample_size, u_dim), dtype="float32")
    x_next_data = np.zeros((sample_size, 2, width, height), dtype="float32")

    # Generate interaction tuples (random states and actions).
    for sample in trange(sample_size, desc="Sampling " + env_name + " data"):
        x0 = env.reset()  # x0.shape = [3, 80, 80], todo: all the same data?
        u0 = env.action_space.sample()
        x1, _, _, _ = env.step(u0)
        u1 = env.action_space.sample()  # save this
        x2, _, _, _ = env.step(u1)
        x0, x1, x2 = rgb_to_gray(x0, x1, x2)  # shape: [1, 80, 80]
        # Current state
        x_data[sample, 0, :, :] = x0[0, :, :]
        x_data[sample, 1, :, :] = x1[0, :, :]
        # Action
        u_data[sample] = u1
        # Next state
        x_next_data[sample, 0, :, :] = x1[0, :, :]
        x_next_data[sample, 1, :, :] = x2[0, :, :]

    return x_data, u_data, x_next_data


def write_to_file(env_name, sample_size, noise):
    """
    write [(x, u, x_next)] to output dir
    """
    output_dir = root_path + "/data/" + env_name + "/raw_{:d}_{:.0f}".format(sample_size, noise)
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    samples = []
    data = sample(env_name=env_name, sample_size=sample_size, noise=noise)
    x_data, u_data, x_next_data, state_data, state_next_data = data

    for i in range(x_data.shape[0]):
        x_1 = x_data[i, :, :, 0]
        x_2 = x_data[i, :, :, 1]
        before = np.hstack((x_1, x_2))
        before_file = "before-{:05d}.png".format(i)
        Image.fromarray(before * 255.0).convert("L").save(path.join(output_dir, before_file))

        after_file = "after-{:05d}.png".format(i)
        x_next_1 = x_next_data[i, :, :, 0]
        x_next_2 = x_next_data[i, :, :, 1]
        after = np.hstack((x_next_1, x_next_2))
        Image.fromarray(after * 255.0).convert("L").save(path.join(output_dir, after_file))

        initial_state = state_data[i]
        after_state = state_next_data[i]

        samples.append(
            {
                "before_state": initial_state.tolist(),
                "after_state": after_state.tolist(),
                "before": before_file,
                "after": after_file,
                "control": u_data[i].tolist(),
            }
        )

    with open(path.join(output_dir, "data.json"), "wt") as outfile:
        json.dump(
            {
                "metadata": {"num_samples": x_data.shape[0], "time_created": str(datetime.now()), "version": 1},
                "samples": samples,
            },
            outfile,
            indent=2,
        )


def rgb_to_gray(x0, x1, x2):
    """
    Convert the inputs to tensor first, then use rgb_to_grayscale, finally return ndarray
    """
    x0 = torch.from_numpy(x0)
    x1 = torch.from_numpy(x1)
    x2 = torch.from_numpy(x2)
    x0 = rgb_to_grayscale(x0)
    x1 = rgb_to_grayscale(x1)
    x2 = rgb_to_grayscale(x2)
    x0 = x0.numpy() / 255.0
    x1 = x1.numpy() / 255.0
    x2 = x2.numpy() / 255.0
    return x0, x1, x2


def main(args):
    sample_size = args.sample_size
    noise = args.noise
    env_name = args.env
    assert env_name in ["pendulum", "cartpole", "threepole", "ccartpole"]
    write_to_file(env_name, sample_size, noise)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sample pendulum data")

    parser.add_argument("--sample_size", required=True, type=int, help="the number of samples")
    parser.add_argument("--noise", default=0, type=int, help="level of noise")
    parser.add_argument("--env", required=True, type=str, help="pendulum or cartpole or threepole")

    args = parser.parse_args()

    main(args)
