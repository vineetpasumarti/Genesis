import argparse
import os
import pickle
import wandb
import torch
from waypoint_env_fig_8_doubleagent import HoverEnv
from rsl_rl.runners import OnPolicyRunner
import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-hovering")
    parser.add_argument("--ckpt", type=int, default=300)
    parser.add_argument("--record", action="store_true", default=False)
    parser.add_argument("--adversary_exp_name", type=str, default=None,
                        help="Experiment name for adversary policy (defaults to exp_name)")
    parser.add_argument("--adversary_ckpt", type=int, default=300,
                        help="Checkpoint for adversary drone policy")
    args = parser.parse_args()

    gs.init()
    wandb.init(project=f"{args.exp_name}")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    # visualize the target
    env_cfg["visualize_target"] = True
    # for video recording
    env_cfg["visualize_camera"] = args.record
    # set the max FPS for visualization
    env_cfg["max_visualize_FPS"] = 60
    # increase the episode length to allow time for reaching all targets
    env_cfg["episode_length_s"] = 45.0  # defaults to 15.0s even without this line

    # load adversary policy
    adversary_policy = None
    if args.adversary_ckpt > 0:
        adversary_exp_name = args.adversary_exp_name if args.adversary_exp_name else args.exp_name
        adversary_log_dir = f"logs/{adversary_exp_name}"

        # load configs for adversary
        adv_env_cfg, adv_obs_cfg, adv_reward_cfg, adv_command_cfg, adv_train_cfg = pickle.load(
            open(f"{adversary_log_dir}/cfgs.pkl", "rb"))

        # create runner for adversary policy
        adversary_runner = OnPolicyRunner(HoverEnv(
            num_envs=1,
            env_cfg=adv_env_cfg,
            obs_cfg=adv_obs_cfg,
            reward_cfg=adv_reward_cfg,
            command_cfg=adv_command_cfg,
            show_viewer=False,
        ), adv_train_cfg, adversary_log_dir, device="cuda:0")

        # load adversary model exactly like in waypoint_eval from single agent
        adversary_resume_path = os.path.join(adversary_log_dir, f"model_{args.adversary_ckpt}.pt")
        adversary_runner.load(adversary_resume_path)

        # get inference policy
        adversary_policy = adversary_runner.get_inference_policy(device="cuda:0")

    # create environment with adversary policy
    env = HoverEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        adversary_policy=adversary_policy,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()

    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])
    with torch.no_grad():
        if args.record:
            env.cam.start_recording()
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, _, rews, dones, infos = env.step(actions)
                env.cam.render()
            env.cam.stop_recording(save_to_filename="video.mp4", fps=env_cfg["max_visualize_FPS"])
        else:
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, _, rews, dones, infos = env.step(actions)

    if args.record:
        env.cam.start_recording()

        # capture frames
        frames = []
        for _ in range(max_sim_step):
            # Existing code
            ...

            # Capture RGB array
            frame = env.cam.get_rgb()
            frames.append(frame)

        # log as video
        wandb.log({
            "eval_video": wandb.Video(
                np.array(frames),
                fps=env_cfg["max_visualize_FPS"],
                format="mp4"
            )
        })

if __name__ == "__main__":
    main()

"""
# evaluation
python examples/drone/hover_eval.py

# Note
If you experience slow performance or encounter other issues 
during evaluation, try removing the --record option.
"""
