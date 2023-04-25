import cv2
import pickle
import numpy as np
import os
from torchvision.transforms.functional import rgb_to_grayscale
import torch

ROOT_PATH = "/localhome/hha160/Downloads/goal_images/cartpole"
# num_frame_stack = 2

# imgs_l = []
# for i in range(num_frame_stack):
#     tmp = cv2.imread(os.path.join(ROOT_PATH, "ezgif-frame-00{}.png".format(i+1)))
#     tmp_rgb = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

#     # Crop the image from 96*96 to size 80*80
#     cropped_tmp_rgb = tmp_rgb[6:6+80, 6:6+80]
#     imgs_l.append(cropped_tmp_rgb)

# goal_obs = np.concatenate(imgs_l, axis=2)



tmp = cv2.imread(os.path.join(ROOT_PATH, "ezgif-frame-00{}.png".format(6)))
tmp_rgb = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
cropped_tmp_rgb = tmp_rgb[6:6+80, 6:6+80]
goal_obs = cropped_tmp_rgb
# from (h, w, c) to (c, h, w) for pytorch
goal_obs = np.transpose(goal_obs, (2,0,1)) / 255.0
goal_obs = torch.from_numpy(goal_obs)
goal_obs = rgb_to_grayscale(goal_obs)


# with open(os.path.join(ROOT_PATH, "goal_obs.pkl"), "wb") as f:
#     pickle.dump(goal_obs,f)

print(goal_obs.shape)