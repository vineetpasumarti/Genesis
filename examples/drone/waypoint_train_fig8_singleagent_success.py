import argparse
import os
import pickle
import shutil
import wandb

from waypoint_env_fig_8_singleagent_success import SingleAgentEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.004,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "tanh",
            "actor_hidden_dims": [128, 128],
            "critic_hidden_dims": [128, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 100,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": True,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 4,
        # termination
        "termination_if_roll_greater_than": 180,  # degree
        "termination_if_pitch_greater_than": 180,
        "termination_if_close_to_ground": 0.1,
        "termination_if_x_greater_than": 25.0,
        "termination_if_y_greater_than": 25.0,
        "termination_if_z_greater_than": 25.0,
        # base pose
        "base_init_pos": [-1.0, -1.0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 15.0,
        "at_target_threshold": 0.8,
        "resampling_time_s": 3.0,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        # visualization
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
    }
    obs_cfg = {
        "num_obs": 17,
        "obs_scales": {
            "rel_pos": 1 / 12,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
        },
    }
    # reward_cfg = {
    #     "yaw_lambda": -10.0,
    #     "reward_scales": {
    #         "target": 10.0,
    #         "smooth": -1e-4,
    #         "yaw": 0.01,
    #         "angular": -2e-4,
    #         "crash": -10.0,
    #     },
    # }

    reward_cfg = {
        "yaw_lambda": -10.0,
        "reward_scales": {
            "progress": 0.5,
            "commands_lrg": -0.0005,
            "commands_diff": -0.0002,
            "pass": 1.0,
            "crash": 1.0,
            # "perception": 0.025,
        },
    }

    # traj 4 (figure 8)
    command_cfg = {
        "num_commands": 3,
        "target_locations": [
            [0.0, 0.0, 1.0],
            [10.0, 5.0, 1.0],
            [10.0, -5.0, 1.0],
            [0.0, 0.0, 1.0],
            [-10.0, 5.0, 1.0],
            [-10.0, -5.0, 1.0],
            [0.0, 0.0, 1.0]
        ]
    }


    # # traj 1
    # command_cfg = {
    #     "num_commands": 3,
    #     "target_locations": [
    #         [-1.1, -1.6, 3.6],
    #         [9.2, 6.6, 1.0],
    #         [9.2, -4, 1.2],
    #         [-4.5, -6, 3.5],
    #         [-4.5, -6, 0.8],
    #         [4.75, -0.9, 1.2],
    #         [-2.8, 6.8, 1.2],
    #         [1.1, -1.6, 3.6]
    #     ]
    # }

    # command_cfg = {
    #     "num_commands": 3,
    #     "target_locations": [
    #         [1.0, 0.0, 1.0],
    #         [0.0, 1.0, 1.5],
    #         [-1.0, -1.0, 2.0]
    #     ]
    # }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-hovering")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=8192)
    parser.add_argument("--max_iterations", type=int, default=300)
    args = parser.parse_args()

    gs.init(logging_level="error")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.vis:
        env_cfg["visualize_target"] = True

    wandb.init(project=f"{args.exp_name}",
               sync_tensorboard=True,)

    env = SingleAgentEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    # Log model checkpoints
    wandb.save(os.path.join(log_dir, "*.pt"))
    wandb.save(os.path.join(log_dir, "cfgs.pkl"))


if __name__ == "__main__":
    main()

"""
# training
python examples/drone/hover_train.py
"""
