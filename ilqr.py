import argparse
import json
import os
import random

import numpy as np
import torch
from ilqr_utils import (
    backward,
    compute_latent_traj,
    forward,
    get_x_data,
    latent_cost,
    random_actions_trajs,
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


seed = 2020
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_default_dtype(torch.float64)

config_path = {
    "planar": "ilqr_config/planar.json",
    "swing": "ilqr_config/swing.json",
    "balance": "ilqr_config/balance.json",
    "cartpole": "ilqr_config/cartpole.json",
    "threepole": "ilqr_config/threepole.json",
    "ccartpole": "ilqr_config/ccartpole.json",
}
env_data_dim = {
    "planar": (1600, 2, 2),
    "pendulum": ((2, 48, 48), 3, 1),
    "cartpole": ((2, 80, 80), 8, 1),
    "threepole": ((2, 80, 80), 8, 3),
    "ccartpole": ((2, 80, 80), 8, 1),
}


def main(args):
    task_name = args.task
    assert task_name in ["planar", "balance", "swing", "cartpole", "threepole", "pendulum_gym", "mountain_car", "ccartpole"]
    env_name = "pendulum" if task_name in ["balance", "swing"] else task_name

    setting_path = args.setting_path  # setting_path = "result/planar"
    setting = os.path.basename(os.path.normpath(setting_path))  # setting = planar
    noise = args.noise  # 0.0
    epoch = args.epoch  # 2000
    x_dim, z_dim, u_dim = env_data_dim[env_name]  # (1600, 2, 2)
    if env_name in ["planar", "pendulum"]:
        x_dim = np.prod(x_dim)

    ilqr_result_path = "iLQR_result/" + "_".join([task_name, str(setting), str(noise), str(epoch)])
    if not os.path.exists(ilqr_result_path):
        os.makedirs(ilqr_result_path)
    with open(ilqr_result_path + "/settings", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # each trained model will perform 10 random tasks
    all_task_configs = []  # 10个tasks的所有配置
    for task_counter in range(10):  # 生成10个不同起点和终点的tasks
        # config for this task
        with open(config_path[task_name]) as f:
            config = json.load(f) # 这里是加载参数，比如planar的参数

        # sample random start and goal state, s is the original nonlinear state
        s_start_min, s_start_max = config["start_min"], config["start_max"]
        config["s_start"] = np.random.uniform(low=s_start_min, high=s_start_max)
        s_goal = config["goal"][np.random.choice(len(config["goal"]))]
        config["s_goal"] = np.array(s_goal)

        all_task_configs.append(config)

    # the folder where all trained models are saved
    log_folders = [
        os.path.join(setting_path, dI)
        for dI in os.listdir(setting_path)
        if os.path.isdir(os.path.join(setting_path, dI))
    ]
    log_folders.sort()  # log_folders = ['result/planar/planar_1']

    # statistics on all trained models
    avg_model_percent = 0.0
    best_model_percent = 0.0
    for log in log_folders:
        with open(log + "/settings", "r") as f:
            settings = json.load(f)
            armotized = settings["armotized"] 

        log_base = os.path.basename(os.path.normpath(log))
        model_path = ilqr_result_path + "/" + log_base  # model_path = "iLQR_result/planar_planar_0.0_2000/planar_1"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        print("iLQR for " + log_base) 

        # load the trained model
        model = PCC(armotized, x_dim, z_dim, u_dim, env_name)
        model.load_state_dict(torch.load(log + "/model_" + str(epoch), map_location="cpu"))
        model.eval()
        dynamics = model.dynamics # z_t+1 = F(z_t, u_t)
        encoder = model.encoder # zt = encoder(xt) where xt is the image

        # run the task with 10 different start and goal states for a particular model
        avg_percent = 0.0
        for task_counter, config in enumerate(all_task_configs):  # 对于每个task的goal state 和start state不同，记录在config中

            print("Performing task %d: " % (task_counter) + str(config["task"]))

            # environment specification
            horizon = config["horizon_prob"] # T = 50
            plan_len = config["plan_len"]  # control_t = 5

            # ilqr specification
            R_z = config["q_weight"] * np.eye(z_dim) # c(zt, ut) = (zt - zgoal)^T*Q*(zt - zgoal) + ut^T*R*ut
            R_u = config["r_weight"] * np.eye(u_dim)
            num_uniform = config["uniform_trajs"] # 3, ?
            num_extreme = config["extreme_trajs"] # 3, ?
            ilqr_iters = config["ilqr_iters"] # 4, ?
            inv_regulator_init = config["pinv_init"] # 1e-5
            inv_regulator_multi = config["pinv_mult"] # 2.0
            inv_regulator_max = config["pinv_max"] # 10
            alpha_init = config["alpha_init"] # 1.0
            alpha_mult = config["alpha_mult"] # 0.5 
            alpha_min = config["alpha_min"] # 1e-4

            s_start = config["s_start"]
            s_goal = config["s_goal"]

            # mdp
            if env_name == "planar":
                mdp = PlanarObstaclesMDP(goal=s_goal, goal_thres=config["distance_thresh"], noise=noise)
            elif env_name == "pendulum":
                mdp = PendulumMDP(frequency=config["frequency"], noise=noise, torque=config["torque"])
            elif env_name == "cartpole":
                mdp = CartPoleMDP(frequency=config["frequency"], noise=noise)
            elif env_name == "threepole":
                mdp = ThreePoleMDP(frequency=config["frequency"], noise=noise, torque=config["torque"])
            # get z_start and z_goal
            x_start = get_x_data(mdp, s_start, config)  # x_data.shape = tensor (1, 2, 80, 80)
            x_goal = get_x_data(mdp, s_goal, config)  # x_goal.shape = tensor (1, 2, 80, 80)
            with torch.no_grad():
                z_start = encoder(x_start).mean
                z_goal = encoder(x_goal).mean
            z_start = z_start.squeeze().numpy()  # 
            z_goal = z_goal.squeeze().numpy()

            # initialize actions trajectories
            all_actions_trajs = random_actions_trajs(mdp, num_uniform, num_extreme, plan_len) # 6 * (10, 1)

            # perform receding horizon iLQR
            s_start_horizon = np.copy(s_start)  # s_start_horizon.shape = (4,)
            z_start_horizon = np.copy(z_start)
            obs_traj = [mdp.render(s_start).squeeze()] # [shape is (80, 80)]
            goal_counter = 0.0
            for plan_iter in range(1, horizon + 1):  # plan_iter = 1, 2,..., 50
                latent_cost_list = [None] * len(all_actions_trajs)  # 6
                # iterate over all trajectories
                for traj_id in range(len(all_actions_trajs)):  # traj_id = 0, 1,..., 5
                    # initialize the inverse regulator
                    inv_regulator = inv_regulator_init # 1e-5
                    for iter in range(1, ilqr_iters + 1): # 1, 2, 3, 4
                        u_seq = all_actions_trajs[traj_id]  # 1 x 10
                        z_seq = compute_latent_traj(z_start_horizon, u_seq, dynamics)  # len(z_seq) = 11?, use PCC.dynamics 
                        # compute the linearization matrices
                        A_seq, B_seq = seq_jacobian(dynamics, z_seq, u_seq)
                        # run backward
                        k_small, K_big = backward(R_z, R_u, z_seq, u_seq, z_goal, A_seq, B_seq, inv_regulator)  # equation (29)
                        current_cost = latent_cost(R_z, R_u, z_seq, z_goal, u_seq)
                        # forward using line search
                        alpha = alpha_init  # alpha_init = 1.0
                        accept = False  # if any alpha is accepted
                        while alpha > alpha_min:  # alpha > 1e-4
                            z_seq_cand, u_seq_cand = forward(
                                z_seq, all_actions_trajs[traj_id], k_small, K_big, dynamics, alpha
                            )
                            cost_cand = latent_cost(R_z, R_u, z_seq_cand, z_goal, u_seq_cand)
                            if cost_cand < current_cost:  # accept the trajectory candidate
                                accept = True
                                all_actions_trajs[traj_id] = u_seq_cand
                                latent_cost_list[traj_id] = cost_cand
                                break
                            else:
                                alpha *= alpha_mult  # alpha * 0.5
                        if accept:
                            inv_regulator = inv_regulator_init  # 1e-5
                        else:
                            inv_regulator *= inv_regulator_multi  # inv_regulator * 2.0
                        if inv_regulator > inv_regulator_max:  # > 10
                            break

                for i in range(len(latent_cost_list)):
                    if latent_cost_list[i] is None:
                        latent_cost_list[i] = np.inf
                traj_opt_id = np.argmin(latent_cost_list)
                action_chosen = all_actions_trajs[traj_opt_id][0]  # the first best u^* in all 6 random actions trajectories
                s_start_horizon, z_start_horizon = update_horizon_start(
                    mdp, s_start_horizon, action_chosen, encoder, config
                )

                obs_traj.append(mdp.render(s_start_horizon).squeeze())  # [shape is (80, 80)]
                goal_counter += mdp.reward_function(s_start_horizon)

                all_actions_trajs = refresh_actions_trajs(
                    all_actions_trajs,
                    traj_opt_id,
                    mdp,
                    np.min([plan_len, horizon - plan_iter]),
                    num_uniform,
                    num_extreme,
                )

            # compute the percentage close to goal
            success_rate = goal_counter / horizon
            print("Success rate: %.2f" % (success_rate))
            percent = success_rate
            avg_percent += success_rate
            with open(model_path + "/result.txt", "a+") as f:
                f.write(config["task"] + ": " + str(percent) + "\n")

            # save trajectory as gif file
            gif_path = model_path + "/task_{:01d}.gif".format(task_counter + 1)
            save_traj(obs_traj, mdp.render(s_goal).squeeze(), gif_path, config["task"])

        avg_percent = avg_percent / 10
        print("Average success rate: " + str(avg_percent))
        print("====================================")
        avg_model_percent += avg_percent
        if avg_percent > best_model_percent:
            best_model = log_base
            best_model_percent = avg_percent
        with open(model_path + "/result.txt", "a+") as f:
            f.write("Average percentage: " + str(avg_percent))

    avg_model_percent = avg_model_percent / len(log_folders)
    with open(ilqr_result_path + "/result.txt", "w") as f:
        f.write("Average percentage of all models: " + str(avg_model_percent) + "\n")
        f.write("Best model: " + best_model + ", best percentage: " + str(best_model_percent))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run iLQR")
    parser.add_argument("--task", required=True, type=str, help="task to perform")
    parser.add_argument("--setting_path", required=True, type=str, help="path to load trained models")
    parser.add_argument("--noise", type=float, default=0.0, help="noise level for mdp")
    parser.add_argument("--epoch", type=int, default=2000, help="number of epochs to load model")
    args = parser.parse_args()

    main(args)
