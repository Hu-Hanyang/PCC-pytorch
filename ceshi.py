import dmc2gym
import yaml
import argparse
import json
import os
import random
import time
from os import path
import gym
import cv2


from dm_control import suite
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from datasets import CartPoleDataset, PendulumDataset, PlanarDataset, ThreePoleDataset, CCartpoleDataset
from latent_map_planar import draw_latent_map
from losses import KL, ae_loss, bernoulli, curvature, entropy, gaussian, vae_bound
from mdp.plane_obstacles_mdp import PlanarObstaclesMDP
from pcc_model import PCC
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from xubo_utils import FrameStack, VideoRecorder, make_dir, Logger, evaluate, make_agent
from torchvision.transforms.functional import rgb_to_grayscale
import argparse
import json
import os
import random
from PIL import Image

import numpy as np
import dmc2gym
import yaml
import torch
from ilqr_utils import (
    backward,
    compute_latent_traj,
    forward,
    get_x_data,
    get_x_data_comparison,
    get_x_goal,
    latent_cost,
    random_actions_trajs,
    random_actions_trajs_comparison,
    refresh_actions_trajs,
    save_traj,
    seq_jacobian,
    update_horizon_start,
)
from mdp.cartpole_mdp import CartPoleMDP
from mdp.pendulum_mdp import PendulumMDP
from mdp.plane_obstacles_mdp import PlanarObstaclesMDP
from mdp.three_pole_mdp import ThreePoleMDP
from pcc_model import PCC


torch.set_default_dtype(torch.float64)

device = torch.device("cuda")
datasets = {
    "planar": PlanarDataset,
    "pendulum": PendulumDataset,
    "cartpole": CartPoleDataset,
    "threepole": ThreePoleDataset,
    "ccartpole": CCartpoleDataset,
}
dims = {
    "planar": (1600, 2, 2),
    "pendulum": (4608, 3, 1),
    "cartpole": ((2, 80, 80), 8, 1),
    "threepole": ((2, 80, 80), 8, 3),
    "ccartpole": ((2, 84, 84), 8, 1), # todo: need to change the networks
}



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
    x0 = x0.numpy()
    x1 = x1.numpy()
    x2 = x2.numpy()
    return x0, x1, x2


# test baseline
env_name = "ccartpole"

with open(os.path.join("data/", env_name+'.yaml')) as file:
        config = yaml.safe_load(file)
print("Load the configurations successfully!")


env = dmc2gym.make(
        domain_name=config['domain_name'],
        task_name=config['task_name'],
        seed=config['seed'],
        visualize_reward=False,
        from_pixels=(config['env']['encoder_type'] == 'pixel'),  # (config['env']['encoder_type'] == 'pixel')
        height=config['env']['pre_transform_image_size'],
        width=config['env']['pre_transform_image_size'],
        frame_skip=config['env']['action_repeat'])

# print(env.observation_space)  # Box(0, 255, (3, 80, 80), uint8)
    
x0 = env.reset()  # x0.shape = (3, 80, 80)
# action_spec = env.action_spec()
# print(action_spec)
# print(x0.shape)
x0 = x0.transpose(1, 2, 0)
print(x0.shape)
image0 = Image.fromarray(x0, 'RGB')
# image0.show()
i = 1
name = "test_image.png"
path = f"test_images/task{i}"
if not os.path.exists(path):
      os.makedirs(path)

image0.save(f"{path}/{name}")

    
# x0 = torch.from_numpy(x0)
# print(x0.shape)

# x0 = rgb_to_grayscale(x0).numpy().squeeze()
# print(x0.shape)

# x_data = torch.zeros(size=(2, 80, 80))
# x_data[0, :, :] = torch.from_numpy(x0)
# x_data[1, :, :] = torch.from_numpy(x0)
# x_data = x_data.unsqueeze(0)  # x_data.shape = tensor (1, 2, 80, 80)
# print(x_data.shape)

# u0 = env.action_space.sample()
# print(u0.shape)
# x1, reward, done, info = env.step(u0)
# print(f"The reward is {reward}.")
# print(f"The result of this step is {done}.")
# u1 = env.action_space.sample()  # save this
# x2, _, _, _ = env.step(u1)
# print(x2.shape)
# x0, x1, x2 = rgb_to_gray(x0, x1, x2)  # shape: (1, 80, 80)
# print(x0.shape)
# print(f"The minimum value in x0 is {np.min(x0)}. \n")
# print(x0.shape)
# print(x1)

# test original cartpole
# env_name = "cartpole"
# dataset = datasets[env_name]
# data = dataset(sample_size=10, noise=0.1)

# env = gym.make("CartPole-v1")

# env = dmc2gym.make(
#         domain_name=config['domain_name'],
#         task_name=config['task_name'],
#         seed=config['seed'],
#         visualize_reward=False,
#         from_pixels=(config['env']['encoder_type'] == 'pixel'),  # (config['env']['encoder_type'] == 'pixel')
#         height=config['env']['pre_transform_image_size'],
#         width=config['env']['pre_transform_image_size'],
#         frame_skip=config['env']['action_repeat'])
# x0 = env.reset()  # x0.shape = (3, 80, 80)
# print(x0.shape)
# # images = np.zeros((50, 80, 80, 3), dtype=np.uint8)
# for _ in range(100):
#     action = env.action_space.sample()
#     obs, r, done, info = env.step(action)
#     # image = env.render(mode="rgb_array")
#     # img = plt.imshow(image)
#     # plt.pause(0.01)  # Need min display time > 0.0.
#     # plt.draw()
#     # print(image.shape)
#     if done:
#         env.reset()
# env.close()   



# dm_control
# max_frame = 5

# width = 80
# height = 80
# video = np.zeros((max_frame, height, 2 * width, 3), dtype=np.uint8)

# # Load one task:
# env = suite.load(domain_name="cartpole", task_name="balance")

# # Step through an episode and print out reward, discount and observation.
# action_spec = env.action_spec()
# time_step = env.reset()
# while not time_step.last():
#   for i in range(max_frame):
#     action = np.random.uniform(action_spec.minimum,
#                              action_spec.maximum,
#                              size=action_spec.shape)
#     time_step = env.step(action)
#     # print(env.physics.render(height, width, camera_id=0).shape)  # (80, 80, 3)
#     # video[i] = np.hstack([env.physics.render(height, width, camera_id=0), env.physics.render(height, width, camera_id=1)]) # 
#     video[i] = env.physics.render(height, width, camera_id=0)
#     print(video[-1].shape)
#     # [env.physics.render(height, width, camera_id=0), env.physics.render(height, width, camera_id=1)]
#     #print(time_step.reward, time_step.discount, time_step.observation)
#   for i in range(max_frame):
#     img = plt.imshow(video[i])
#     plt.pause(0.01)  # Need min display time > 0.0.
#     plt.draw()


# env2 = dmc2gym.make(
#         domain_name=config['domain_name'],
#         task_name=config['task_name'],
#         seed=3,
#         visualize_reward=False,
#         from_pixels=(config['env']['encoder_type'] == 'pixel'),  # (config['env']['encoder_type'] == 'pixel')
#         height=2,
#         width=2,
#         frame_skip=config['env']['action_repeat'])

# x0 = env2.reset()
# print(x0.shape)


# with open("ilqr_config/cartpole.json") as f:
#     config = json.load(f) # 这里是加载参数，比如planar的参数

# s_start_min, s_start_max = config["start_min"], config["start_max"]
# config["s_start"] = np.random.uniform(low=s_start_min, high=s_start_max)
# s_goal = config["goal"][np.random.choice(len(config["goal"]))]
# config["s_goal"] = np.array(s_goal)

# s_start = config["s_start"]
# mdp = CartPoleMDP(frequency=config["frequency"], noise=0.0)
# image_data = mdp.render(s_start).squeeze()
# print(image_data.shape)
# x_start = get_x_data(mdp, s_start, config)
# print(x_start.shape)
# x_goal = get_x_data(mdp, s_goal, config)
# print(x_goal.shape)

# all_actions_trajs = random_actions_trajs(mdp, 3, 3, 10) # 6 * 10
# print(len(all_actions_trajs))


# s_start_horizon = np.copy(s_start)  # s_start and z_start change at each horizon
# print(s_start_horizon.shape)

# obs_traj = [mdp.render(s_start).squeeze()] # [x_0]
# print(obs_traj[0].shape)


# x_data = torch.zeros(size=(2, 80, 80))
# ROOT_PATH = "/localhome/hha160/Downloads/goal_images/cartpole"
# tmp = cv2.imread(os.path.join(ROOT_PATH, "ezgif-frame-00{}.png".format(6)))
# tmp_rgb = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
# cropped_tmp_rgb = tmp_rgb[6:6+80, 6:6+80]
# goal_obs = cropped_tmp_rgb
# # from (h, w, c) to (c, h, w) for pytorch
# goal_obs = np.transpose(goal_obs, (2,0,1)) / 255.0
# goal_obs = torch.from_numpy(goal_obs)
# goal_obs = rgb_to_grayscale(goal_obs)
# x_data[0, :, :] = goal_obs
# x_data[1, :, :] = goal_obs
# x_data = x_data.unsqueeze(0) 
# print(x_data.shape)



# actions_trajs = []
# for _ in range(6):  # 6 = num_uniform + num_extreme
#     actions = []
#     for i in range(10):
#         action = env.action_space.sample()
#         actions.append(action)
#     actions_trajs.append(np.array(actions))

# print(len(actions_trajs))
# print(actions_trajs[0].shape)

# a = [None] *6
# print(a)