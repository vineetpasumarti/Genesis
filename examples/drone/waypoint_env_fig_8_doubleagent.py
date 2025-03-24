import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import torch.nn.functional as F
from TrackModel import SplineCenterline
from interp import spline_interpolant
import numpy as np
from pyquaternion import Quaternion
import wandb



def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class HoverEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, adversary_policy=None, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.02  # run in 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        self.track = SplineCenterline(Traj=4)
        self.gate_positions = self.track.x_gates
        self.current_gate_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.trajectory = self.track.f_xc

        # store the adversary policy
        self.adversary_policy = adversary_policy

        # add adversary drone initialization parameters
        self.adversary_init_pos = torch.tensor(self.env_cfg["adv_init_pos"], device=self.device).repeat(self.num_envs, 1)
        self.adversary_init_quat = torch.tensor(self.env_cfg["adv_init_quat"], device=self.device).repeat(self.num_envs, 1)

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(24, -2, 18),
                camera_lookat=(3, 1, 1),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=10),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add gates
        for i in range(self.track.NumOfGates):
            gate_orientation = self.track.get_gate_orientation(i)
            gate_T = np.array(gate_orientation[0]).flatten()  # Convert DM to numpy array
            gate_quat = Quaternion(matrix=np.column_stack([gate_T, *gate_orientation[1:]])).elements

            # for center gate normals to be aligned with world axes instead of track spline
            if i in {0, 6}:
                rot_matrix = np.array([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]])
                gate_quat = Quaternion(matrix=rot_matrix).elements
            if i in {3}:
                rot_matrix = np.array([[-1, 0, 0],
                                       [0, -1, 0],
                                       [0, 0, 1]])
                gate_quat = Quaternion(matrix=rot_matrix).elements

            self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/gate.obj",
                    scale=0.4,
                    pos=self.track.x_gates[i],
                    quat=tuple(gate_quat.tolist()),
                    fixed=True,
                    collision=False,
                ),
            ),

            # Add tangent vector visualization sphere
            self.tangent = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,  # Smaller scale for visibility
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.0, 0.0)  # Red color for visibility
                    ),
                ),
            )

        # add target
        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.5, 0.5),
                    ),
                ),
            )
        else:
            self.target = None

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(1920, 1080),
                pos=(24, -2, 18),
                lookat=(3, 1, 1),
                fov=40,
                GUI=False,
            )

        # add ego drone
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

        # add adversarial drone
        self.adversary_drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

        # build scene
        self.scene.build(n_envs=num_envs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)

        self.extras = dict()  # extra information for logging

        # initialize adversary buffers if policy is provided
        if self.adversary_policy is not None:
            self.adversary_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
            self.adversary_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
            self.adversary_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
            self.adversary_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
            self.adversary_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device,
                                                 dtype=gs.tc_float)
            self.adversary_last_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device,
                                                      dtype=gs.tc_float)
            self.adversary_obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)

            # initialize adversary drone position
            self.adversary_drone.set_pos(self.adversary_init_pos, zero_velocity=True)
            self.adversary_drone.set_quat(self.adversary_init_quat, zero_velocity=True)

            # add competition reward
            self.reward_scales["competition"] = reward_cfg["reward_scales"].get("competition", 0.5) * self.dt

        # define target locations
        self.target_locations = [
            torch.tensor([0.0, 0.0, 1.0], device=self.device),
            torch.tensor([10.0, 5.0, 1.0], device=self.device),
            torch.tensor([10.0, -5.0, 1.0], device=self.device),
            torch.tensor([0.0, 0.0, 1.0], device=self.device),
            torch.tensor([-10.0, 5.0, 1.0], device=self.device),
            torch.tensor([-10.0, -5.0, 1.0], device=self.device),
            torch.tensor([0.0, 0.0, 1.0], device=self.device),
        ]
        self.current_target_index = -1 * torch.ones(self.num_envs, dtype=torch.long, device=self.device)

        # Precompute all gate tangents once during init
        self.gate_tangents = torch.stack([
            torch.tensor(self.track.get_gate_orientation(i)[0].full().flatten(),
                         device=self.device, dtype=torch.float32)
            for i in range(self.track.NumOfGates)
        ])
        self.gate_tangents = F.normalize(self.gate_tangents, dim=1)  # [num_gates, 3]
        # calculate all gate_quaternions
        self.gate_quaternions = torch.zeros((self.track.NumOfGates, 4), device=self.device)
        for i in range(self.track.NumOfGates):
            gate_orientation = self.track.get_gate_orientation(i)
            gate_T = np.array(gate_orientation[0]).flatten()
            gate_quat = Quaternion(matrix=np.column_stack([gate_T, *gate_orientation[1:]])).elements
            # Handle special cases as before
            if i in {0, 6}:
                rot_matrix = np.eye(3)
                gate_quat = Quaternion(matrix=rot_matrix).elements
            if i in {3}:
                rot_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                gate_quat = Quaternion(matrix=rot_matrix).elements
            self.gate_quaternions[i] = torch.tensor(gate_quat, device=self.device)

    def _resample_commands(self, envs_idx):
        # 1. Convert target_locations to pre-stacked tensor
        target_tensor = torch.stack(self.target_locations)  # Add this line during initialization

        # 2. Use PRE-increment logic
        new_indices = (self.current_target_index[envs_idx] + 1) % len(self.target_locations)
        self.commands[envs_idx] = target_tensor[new_indices]  # Direct tensor indexing
        self.current_target_index[envs_idx] = new_indices  # Update AFTER assignment

        if self.target is not None:
            self.target.set_pos(self.commands[envs_idx], zero_velocity=True, envs_idx=envs_idx)

    def _at_target(self):
        # Position threshold check
        position_mask = (
            (self.rel_pos_gate_frame[:, 0] < 0.0)
            & (torch.abs(self.rel_pos_gate_frame[:, 1]) < self.env_cfg["at_target_threshold"])
            & (torch.abs(self.rel_pos_gate_frame[:, 2]) < self.env_cfg["at_target_threshold"])
        )

        valid_envs = position_mask.nonzero(as_tuple=False).flatten()

        return valid_envs

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.actions

        # apply actions to ego drone
        self.drone.set_propellels_rpm((1 + exec_actions * 0.8) * 14468.429183500699)

        # handle adversary drone if policy is provided
        if self.adversary_policy is not None:
            # get adversary observations
            self.adversary_obs_buf = self._get_adversary_observations()

            # get actions from adversary policy
            with torch.no_grad():
                self.adversary_actions = self.adversary_policy(self.adversary_obs_buf)

            # apply actions to adversary drone
            self.adversary_drone.set_propellels_rpm((1 + self.adversary_actions * 0.8) * 14468.429183500699)

        # step the simulation
        self.scene.step()

        # update ego drone buffers
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        # get current gate orientations for all environments
        gate_quats = self.gate_quaternions[self.current_target_index]  # (num_envs, 4)
        # transform rel_pos to gate frame
        inv_gate_quats = inv_quat(gate_quats)
        self.rel_pos_gate_frame = transform_by_quat(self.rel_pos, inv_gate_quats)

        # update adversary drone buffers if policy is provided
        if self.adversary_policy is not None:
            self.adversary_pos[:] = self.adversary_drone.get_pos()
            self.adversary_quat[:] = self.adversary_drone.get_quat()
            inv_adversary_quat = inv_quat(self.adversary_quat)
            self.adversary_lin_vel[:] = transform_by_quat(self.adversary_drone.get_vel(), inv_adversary_quat)
            self.adversary_ang_vel[:] = transform_by_quat(self.adversary_drone.get_ang(), inv_adversary_quat)

            # # Check for collision with adversary
            # distance_to_adversary = torch.norm(self.base_pos - self.adversary_pos, dim=1)
            # collision_threshold = self.env_cfg.get("collision_threshold", 0.3)
            # collision_with_adversary = distance_to_adversary < collision_threshold

        # resample commands
        envs_idx = self._at_target()
        self._resample_commands(envs_idx)

        # check if all targets have been reached
        all_targets_reached = (self.current_target_index == 0).all()

        # check termination and reset
        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
            | (torch.abs(self.rel_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
            | (
                ((self.rel_pos_gate_frame[:, 0]) < 0.0)
                & ((torch.abs(self.rel_pos_gate_frame[:, 1]) >= self.env_cfg["at_target_threshold"])
                | (torch.abs(self.rel_pos_gate_frame[:, 2]) >= self.env_cfg["at_target_threshold"]))
            )
        )
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos_gate_frame * self.obs_scales["rel_pos"], -1, 1),
                self.base_quat,
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
                self.last_actions,
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.adversary_last_actions[:] = self.adversary_actions[:]

        # print(f"rel_pos: {self.rel_pos}")
        # print(f"rel_pos_gate_frame: {self.rel_pos_gate_frame}")
        # print(self._reward_pass())

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def _get_adversary_observations(self):
        # Similar to ego observations but from adversary's perspective
        rel_pos = self.commands - self.adversary_pos

        # Get current gate orientations for all environments
        gate_quats = self.gate_quaternions[self.current_target_index]

        # Transform rel_pos to gate frame
        inv_gate_quats = inv_quat(gate_quats)
        rel_pos_gate_frame = transform_by_quat(rel_pos, inv_gate_quats)

        # Concatenate observations (same structure as ego observations)
        return torch.cat([
            torch.clip(rel_pos_gate_frame * self.obs_scales["rel_pos"], -1, 1),
            self.adversary_quat,
            torch.clip(self.adversary_lin_vel * self.obs_scales["lin_vel"], -1, 1),
            torch.clip(self.adversary_ang_vel * self.obs_scales["ang_vel"], -1, 1),
            self.adversary_last_actions,
        ], axis=-1)

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # reset adversary drone if policy is provided
        if self.adversary_policy is not None:
            self.adversary_pos[envs_idx] = self.adversary_init_pos[envs_idx]
            self.adversary_quat[envs_idx] = self.adversary_init_quat[envs_idx]
            self.adversary_drone.set_pos(self.adversary_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
            self.adversary_drone.set_quat(self.adversary_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
            self.adversary_lin_vel[envs_idx] = 0
            self.adversary_ang_vel[envs_idx] = 0
            self.adversary_drone.zero_all_dofs_velocity(envs_idx)
            self.adversary_actions[envs_idx] = 0.0
            self.adversary_last_actions[envs_idx] = 0.0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self.current_target_index[envs_idx] = -1
        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_progress(self):
        progress_rew = self.last_rel_pos.norm(dim=1) - self.rel_pos.norm(dim=1)
        return progress_rew

    def _reward_commands_lrg(self):
        cmd_lrg_rew = torch.norm(self.actions, dim=1)
        return cmd_lrg_rew

    def _reward_commands_diff(self):
        cmd_diff_rew = torch.square(torch.norm(self.actions - self.last_actions, dim=1))
        return cmd_diff_rew

    def _reward_pass(self):
        passed = torch.zeros(self.num_envs, device=self.device)
        passed[self._at_target()] = 1.0 - torch.norm(self.rel_pos_gate_frame[self._at_target()], dim=1)
        return passed

    def _reward_crash(self):
        crash = torch.zeros(self.num_envs, device=self.device)
        crash[self.crash_condition] = -5.0
        return crash

    # def _reward_perception(self):
    #     # Transform gate direction to body frame
    #     quat = self.base_quat  # (num_envs, 4)
    #     gate_dir_world = F.normalize(self.commands - self.base_pos, dim=1)  # Requires F import
    #     gate_dir_body = transform_by_quat(gate_dir_world, inv_quat(quat))
    #
    #     # Camera optical axis in body frame (x-forward)
    #     camera_forward_body = torch.tensor([1, 0, 1], device=self.device).repeat(self.num_envs, 1)
    #
    #     # Angle calculation
    #     cos_theta = (camera_forward_body * gate_dir_body).sum(dim=1)
    #     angle = torch.acos(torch.clamp(cos_theta, min=-1.0+1e-6, max=1.0-1e-6))
    #     return torch.exp(-(angle ** 4))

    # def _reward_competition(self):
    #     if self.adversary_policy is None:
    #         return torch.zeros(self.num_envs, device=self.device)
    #
    #     # Calculate how far each drone has progressed through the track
    #     ego_gate_dists = torch.norm(self.base_pos.unsqueeze(1) -
    #                                 torch.tensor(self.gate_positions, device=self.device), dim=2)
    #     adv_gate_dists = torch.norm(self.adversary_pos.unsqueeze(1) -
    #                                 torch.tensor(self.gate_positions, device=self.device), dim=2)
    #
    #     # Find closest gate for each drone
    #     ego_closest_gate = torch.argmin(ego_gate_dists, dim=1)
    #     adv_closest_gate = torch.argmin(adv_gate_dists, dim=1)
    #
    #     # Reward for being at a later gate than adversary
    #     gate_advantage = (ego_closest_gate - adv_closest_gate).float()
    #
    #     # For same gate, reward for being closer to next gate
    #     same_gate_mask = (ego_closest_gate == adv_closest_gate)
    #     next_gate = (ego_closest_gate + 1) % self.track.NumOfGates
    #
    #     next_gate_pos = torch.tensor(np.array([self.gate_positions[i.item()]
    #                                            for i in next_gate]), device=self.device)
    #
    #     ego_next_dist = torch.norm(self.base_pos - next_gate_pos, dim=1)
    #     adv_next_dist = torch.norm(self.adversary_pos - next_gate_pos, dim=1)
    #
    #     # Lower distance is better
    #     dist_advantage = (adv_next_dist - ego_next_dist) / 10.0  # Normalize
    #
    #     # Combine both advantages
    #     competition_reward = gate_advantage.clone()
    #     competition_reward[same_gate_mask] = dist_advantage[same_gate_mask]
    #
    #     return torch.tanh(competition_reward)  # Scale to [-1, 1]
