import argparse
import os
import pickle
import wandb
import torch
from waypoint_env_fig_8_doubleagent import HoverEnv
from rsl_rl.runners import OnPolicyRunner
import numpy as np
import glfw
import genesis as gs
from rsl_rl.modules import ActorCritic



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
    #     # load adversary model exactly like in waypoint_eval from single agent
    #     adversary_resume_path = os.path.join(adversary_log_dir, f"model_{args.adversary_ckpt}.pt")
    #     adversary_runner.load(adversary_resume_path)
    #
    #     # get inference policy
    #     adversary_policy = adversary_runner.get_inference_policy(device="cuda:0")
    #
    # os.environ["SDL_VIDEO_X11_FORCE_EGL"] = "1"
    #
    # # create environment with adversary policy
    # env = HoverEnv(
    #     num_envs=1,
    #     env_cfg=env_cfg,
    #     obs_cfg=obs_cfg,
    #     reward_cfg=reward_cfg,
    #     command_cfg=command_cfg,
    #     adversary_policy=adversary_policy,
    #     show_viewer=True,
    # )

    # Load adversary policy without creating an environment
    adversary_policy = None
    if args.adversary_ckpt > 0:
        adversary_exp_name = args.adversary_exp_name if args.adversary_exp_name else args.exp_name
        adversary_log_dir = f"logs/{adversary_exp_name}"

        # Load configs for adversary
        adv_env_cfg, adv_obs_cfg, adv_reward_cfg, adv_command_cfg, adv_train_cfg = pickle.load(
            open(f"{adversary_log_dir}/cfgs.pkl", "rb"))

        # Explicitly extract network architecture parameters
        policy_cfg = adv_train_cfg["policy"]
        actor_hidden_dims = policy_cfg["actor_hidden_dims"]
        critic_hidden_dims = policy_cfg["critic_hidden_dims"]
        activation = policy_cfg.get("activation", "tanh")  # Safe get with default
        init_noise_std = policy_cfg.get("init_noise_std", 1.0)

        print("Policy config structure:", adv_train_cfg["policy"].keys())
        print("Actor hidden dims type:", type(actor_hidden_dims))
        print("Actor hidden dims value:", actor_hidden_dims)

        # Load policy directly without creating environment
        adversary_resume_path = os.path.join(adversary_log_dir, f"model_{args.adversary_ckpt}.pt")

        # Create policy with explicit architecture parameters
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
        checkpoint = torch.load(adversary_resume_path)
        adversary_model.load_state_dict(checkpoint['model_state_dict'])
        adversary_model.eval()  # Set to evaluation mode

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

    # Now create only ONE environment for evaluation
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
