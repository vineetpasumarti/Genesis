import argparse
import os
import pickle
import shutil
import wandb
import torch

from waypoint_env_fig_8_singleagent_success import SingleAgentEnv
from waypoint_env_fig_8_doubleagent import HoverEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic
from rsl_rl.modules import ActorCriticRecurrent


import genesis as gs


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.008,           # previously 0.004
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.00001,        # previously 0.0003, then 0.0001, then 0.00001
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
            "actor_hidden_dims": [256, 128, 128],      # previously [128, 128], then [256, 256, 128, 128, 64], want to try [512, 512, 385, 384, 256, 256, 128, 128, 64, 64]
            "critic_hidden_dims": [256, 128, 128],     # previously [128, 128], then [256, 256, 128, 128, 64], want to try [512, 512, 385, 384, 256, 256, 128, 128, 64, 64]
            "init_noise_std": 1.0,
            # recurrent parameters (comment out if using feed-forward)
            "rnn_type": "lstm",  # either lstm or gru
            "rnn_hidden_size": 128,
            "rnn_num_layers": 2,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 100,
            "policy_class_name": "ActorCriticRecurrent",    # make sure to change this depending on feed-forward or recurrent
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
        "termination_if_x_greater_than": 15.0,
        "termination_if_y_greater_than": 15.0,
        "termination_if_z_greater_than": 15.0,
        # base pose
        "base_init_pos": [-2.0, -2.0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        # adversary pose
        "adv_init_pos": [-1.0, -1.0, 1.0],
        "adv_init_quat": [1.0, 0.0, 0.0, 0.0],
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
        "num_obs": 30,
        "obs_scales": {
            "rel_pos": 1 / 12,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
            "adv_to_ego_pos": 1 / 12,
            "adv_to_ego_lin_vel": 1 / 3.0,
            "adv_to_ego_ang_vel": 1 / 3.14159
        },
    }

    reward_cfg = {
        "yaw_lambda": -10.0,
        "reward_scales": {
            "progress": 0.5,
            "commands_lrg": -0.0005,
            "commands_diff": -0.0002,
            "pass": 1.0,
            "crash": 1.0,
            "competition": 0.01,
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

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-hovering")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=8192)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--adversary_exp_name", type=str, default=None,
                        help="Experiment name for adversary policy (defaults to exp_name)")
    parser.add_argument("--adversary_ckpt", type=int, default=300,
                        help="Checkpoint for adversary drone policy")
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

    # # load adversary policy
    # adversary_policy = None
    # if args.adversary_ckpt > 0:
    #     adversary_exp_name = args.adversary_exp_name if args.adversary_exp_name else args.exp_name
    #     adversary_log_dir = f"logs/{adversary_exp_name}"
    #
    #     # load configs for adversary
    #     adv_env_cfg, adv_obs_cfg, adv_reward_cfg, adv_command_cfg, adv_train_cfg = pickle.load(
    #         open(f"{adversary_log_dir}/cfgs.pkl", "rb"))
    #
    #     # create runner for adversary policy
    #     adversary_runner = OnPolicyRunner(HoverEnv(
    #         num_envs=1,
    #         env_cfg=adv_env_cfg,
    #         obs_cfg=adv_obs_cfg,
    #         reward_cfg=adv_reward_cfg,
    #         command_cfg=adv_command_cfg,
    #         show_viewer=False,
    #     ), adv_train_cfg, adversary_log_dir, device="cuda:0")
    #
    #     # load adversary model
    #     adversary_resume_path = os.path.join(adversary_log_dir, f"model_{args.adversary_ckpt}.pt")
    #     adversary_runner.load(adversary_resume_path)
    #
    #     # get inference policy
    #     adversary_policy = adversary_runner.get_inference_policy(device="cuda:0")

    # load adversary policy
    adversary_policy = None
    if args.adversary_ckpt > 0:
        adversary_exp_name = args.adversary_exp_name if args.adversary_exp_name else args.exp_name
        adversary_log_dir = f"logs/{adversary_exp_name}"

        # load configs for adversary
        adv_env_cfg, adv_obs_cfg, adv_reward_cfg, adv_command_cfg, adv_train_cfg = pickle.load(
            open(f"{adversary_log_dir}/cfgs.pkl", "rb"))

        # Load checkpoint directly
        adversary_resume_path = os.path.join(adversary_log_dir, f"model_{args.adversary_ckpt}.pt")
        checkpoint = torch.load(adversary_resume_path)

        # Extract policy configuration
        policy_cfg = adv_train_cfg["policy"]
        actor_hidden_dims = policy_cfg["actor_hidden_dims"]
        critic_hidden_dims = policy_cfg["critic_hidden_dims"]
        activation = policy_cfg.get("activation", "tanh")
        init_noise_std = policy_cfg.get("init_noise_std", 1.0)

        # Create model
        adversary_model = ActorCritic(
            num_actor_obs=adv_obs_cfg["num_obs"],
            num_critic_obs=adv_obs_cfg["num_obs"],
            num_actions=adv_env_cfg["num_actions"],
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std
        ).to("cuda:0")

        # Load state dict
        adversary_model.load_state_dict(checkpoint['model_state_dict'])
        adversary_model.eval()

        # Create a policy wrapper to handle the inference call correctly
        class PolicyWrapper(torch.nn.Module):
            def __init__(self, actor_critic):
                super().__init__()
                self.actor_critic = actor_critic

            def forward(self, obs):
                with torch.no_grad():
                    actions = self.actor_critic.act(obs, deterministic=True)
                return actions

        adversary_policy = PolicyWrapper(adversary_model)

    # create environment with adversary policy
    env = HoverEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        adversary_policy=adversary_policy,
        show_viewer=args.vis,
    )
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    # log model checkpoints
    wandb.save(os.path.join(log_dir, "*.pt"))
    wandb.save(os.path.join(log_dir, "cfgs.pkl"))


if __name__ == "__main__":
    main()

"""
# training
python examples/drone/hover_train.py
"""
