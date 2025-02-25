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
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
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

            # # Add tangent vector visualization sphere
            # self.tangent = self.scene.add_entity(
            #     morph=gs.morphs.Mesh(
            #         file="meshes/sphere.obj",
            #         scale=0.05,  # Smaller scale for visibility
            #         fixed=True,
            #         collision=False,
            #     ),
            #     surface=gs.surfaces.Rough(
            #         diffuse_texture=gs.textures.ColorTexture(
            #             color=(1.0, 0.0, 0.0)  # Red color for visibility
            #         ),
            #     ),
            # )

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

        # add drone
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

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

        # define target locations

        # self.target_locations = [
        #     torch.tensor([1.0, 0.0, 1.0], device=self.device),
        #     torch.tensor([0.0, 1.0, 1.5], device=self.device),
        #     torch.tensor([-1.0, -1.0, 2.0], device=self.device)
        # ]

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
            (torch.abs(self.rel_pos_gate_frame[:, 0]) < 0.2)
            & (torch.abs(self.rel_pos_gate_frame[:, 1]) < self.env_cfg["at_target_threshold"])
            & (torch.abs(self.rel_pos_gate_frame[:, 2]) < self.env_cfg["at_target_threshold"])
        )
        # Get gate tangent vectors for all environments
        gate_tangents = self.gate_tangents[self.current_target_index]  # [num_envs, 3]
        # Velocity direction check
        vel_norm = torch.norm(self.base_lin_vel, dim=1, keepdim=True) + 1e-6
        vel_dir = self.base_lin_vel / vel_norm
        direction_mask = torch.sum(vel_dir * gate_tangents, dim=1) > 0.0  # [num_envs]

        # Combined validation
        valid_envs = (position_mask & direction_mask).nonzero(as_tuple=False).flatten()

        return valid_envs

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.actions

        # 14468 is hover rpm
        self.drone.set_propellels_rpm((1 + exec_actions * 0.8) * 14468.429183500699)
        self.scene.step()

        # update buffers
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

        # Get current gate orientations for all environments
        gate_quats = self.gate_quaternions[self.current_target_index]  # (num_envs, 4)
        # Transform rel_pos to gate frame
        inv_gate_quats = inv_quat(gate_quats)
        self.rel_pos_gate_frame = transform_by_quat(self.rel_pos, inv_gate_quats)

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
                (torch.abs(self.rel_pos_gate_frame[:, 0]) < 0.2)
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
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
                self.base_quat,
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
                self.last_actions,
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

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
        passed[self._at_target()] = 1.0 - torch.norm(self.rel_pos[self._at_target()], dim=1)
        return passed

    def _reward_crash(self):
        crash = torch.zeros(self.num_envs, device=self.device)
        crash[self.crash_condition] = -4.0
        return crash

    def _reward_perception(self):
        # Transform gate direction to body frame
        quat = self.base_quat  # (num_envs, 4)
        gate_dir_world = F.normalize(self.commands - self.base_pos, dim=1)  # Requires F import
        gate_dir_body = transform_by_quat(gate_dir_world, inv_quat(quat))

        # Camera optical axis in body frame (x-forward)
        camera_forward_body = torch.tensor([1, 0, 1], device=self.device).repeat(self.num_envs, 1)

        # Angle calculation
        cos_theta = (camera_forward_body * gate_dir_body).sum(dim=1)
        angle = torch.acos(torch.clamp(cos_theta, min=-1.0+1e-6, max=1.0-1e-6))
        return torch.exp(-(angle ** 4))

    # def _reward_target(self):
    #     target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)
    #     return target_rew
    #
    # def _reward_smooth(self):
    #     smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
    #     return smooth_rew
    #
    # def _reward_yaw(self):
    #     yaw = self.base_euler[:, 2]
    #     yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159  # use rad for yaw_reward
    #     yaw_rew = torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))
    #     return yaw_rew
    #
    # def _reward_angular(self):
    #     angular_rew = torch.norm(self.base_ang_vel / 3.14159, dim=1)
    #     return angular_rew
    #
    # def _reward_crash(self):
    #     crash_rew = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
    #     crash_rew[self.crash_condition] = 1
    #     return crash_rew

