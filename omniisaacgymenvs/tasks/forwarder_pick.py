from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.forwarder import Forwarder
from omniisaacgymenvs.robots.articulations.wood import Wood
from omniisaacgymenvs.robots.articulations.views.forwarder_view import ForwarderView

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.objects import DynamicCapsule, DynamicSphere, DynamicCuboid
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import RigidPrimView

from omni.isaac.core.utils.rotations import *
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *

import numpy as np
import torch
import math

# set printing options to print only 3 decimal places
np.set_printoptions(precision=3, suppress=True)

from omni.isaac.core.utils.prims import get_all_matching_child_prims

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 0] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        torch.cos(angle[idx, :] / 2.0),
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :]
        
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class ForwarderPickTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.train_step = self._task_cfg["env"]["train_step"]
        self.reward_scale = self._task_cfg["env"]["reward_scale"]
        
        self._forwarder_positions = torch.tensor([0.0, 0.0, 0.1])
        self._wood_positions = torch.tensor([-4,  0,   .5])        

        self._num_observations = 60 
        self._num_actions = 6
        self.num_dof_fwd = 9 
        self.dt = self._task_cfg["sim"]["dt"] #1/60 #
       
        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:
        self.get_fwd()
        self.get_wood()

        super().set_up_scene(scene)

        self._fwds = ForwarderView(prim_paths_expr="/World/envs/.*/forwarder", name="forwarder_view")
        scene.add(self._fwds)
        scene.add(self._fwds._grapple_body)
        scene.add(self._fwds._grapple_l)
        scene.add(self._fwds._grapple_r)
        scene.add(self._fwds._base_link)

        self._woods = RigidPrimView(prim_paths_expr="/World/envs/.*/wood/base_link", name="wood_view", reset_xform_properties=False)        
        scene.add(self._woods)

        self.fwd_default_dof_pos = torch.tensor(
                        [0, -0.349066, -0.436332, 0, .2, 0.035, 0.035], device=self._device
        )
        self.active_joints = torch.tensor([0,1,2,3,6,7,8], device=self._device)
        self.action_scale = torch.tensor(self.action_scale, device=self._device)
        self.num_active_dof = len(self.active_joints)
        self.fwd_dof_targets = self.fwd_default_dof_pos.clone().repeat((self.num_envs,1))
        self.grapple_forward_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.wood_forward_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.unloading_forward_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.unloading_up_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.grapple_up_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.wood_sides = torch.zeros((self._num_envs,), device=self._device, dtype=torch.bool)
        
        self.actions= torch.zeros(
            (self._num_envs, self.num_active_dof), dtype=torch.float, device=self._device
        )

        self.old_to_wood_dist = torch.full((self._num_envs,),15,device=self._device)
        self.train_step = torch.tensor((self.train_step,),device=self._device)

        return

    def get_fwd(self):
        fwd = Forwarder(prim_path=self.default_zero_env_path + "/forwarder", name="forwarder", translation=self._forwarder_positions)

        fwd_config = self._sim_config.parse_actor_config("forwarder")
        fwd_config['enable_self_collisions'] = True
        self._sim_config.apply_articulation_settings("forwarder", get_prim_at_path(fwd.prim_path), fwd_config)

    def get_wood(self):
        wood = Wood(prim_path=self.default_zero_env_path + "/wood", name="wood", translation=self._wood_positions)
        self._sim_config.apply_articulation_settings("wood", get_prim_at_path(wood.prim_path), self._sim_config.parse_actor_config("wood"))
    
    def get_observations(self) -> dict:

        self.grapple_body_pos, self.grapple_body_ori = self._fwds._grapple_body.get_world_poses(clone=False)
        self.grapple_l_pos, self.grapple_l_ori = self._fwds._grapple_l.get_world_poses(clone=False)
        self.grapple_r_pos, self.grapple_r_ori = self._fwds._grapple_r.get_world_poses(clone=False)
        self.grapple_body_vel = self._fwds._grapple_body.get_velocities(clone=False)

        # use this instead get_current_dynamic_state()
        self.wood_pos, self.wood_ori = self._woods.get_world_poses(clone=False)
        state = self._woods.get_current_dynamic_state()
        self.wood_pos, self.wood_ori, self.wood_linear_vel = state.positions, state.orientations, state.linear_velocities

        joint_states = self._fwds.get_joints_state()
        self.joints_positions, joints_velocities = joint_states.positions, joint_states.velocities

        to_wood = self.grapple_body_pos - self.wood_pos
        to_unload = self.wood_pos - self.unloading_pos
        to_target = self.wood_pos - self.target_pos

        # track how much distance was made before dropping the wood
        wood_dropped = self.drop_distance_multiplier.reshape(-1,1)

        self.obs_buf = torch.cat(
            (
                self.grapple_body_pos,
                to_wood,            
                self.wood_ori,
                self.wood_pos,
                to_unload,
                to_target,
                self.unloading_pos,
                self.target_pos,
                self.grapple_l_pos,
                self.grapple_r_pos,
                self.grapple_body_vel,
                wood_dropped,
                self.grapple_body_ori,
                self.joints_positions,
                joints_velocities
            ),
            dim=-1,
        )

        observations = {
            self._fwds.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)

        self.actions[:,0:4] = actions[:,0:4]
        self.actions[:,4] = actions[:,4]
        self.actions[:,5] = actions[:,5]
        self.actions[:, 6] = -actions[:,5]

        targets = self.fwd_dof_targets + self.actions * self.action_scale * self.dt #* self.actions * self.action_scale
        self.fwd_dof_targets = tensor_clamp(targets, self.active_dof_lower_limits, self.active_dof_upper_limits)     

        indices = torch.arange(self._fwds.count, dtype=torch.int32, device=self._device)
        self._fwds.set_joint_position_targets(self.fwd_dof_targets, indices=indices, joint_indices=self.active_joints)

        
    def reset_idx(self, env_ids):

        indices = env_ids.to(dtype=torch.int32)
        indices_64 = env_ids.to(dtype=torch.int64)
        num_indices = len(indices)
        num_resets = len(env_ids)

        dof_lower_limits = self.dof_limits[0, :, 0]
        dof_upper_limits = self.dof_limits[0, :, 1]

        pos=torch.zeros(
            (self._num_envs, self.num_active_dof), dtype=torch.float, device=self._device
        )

        dof_vel = torch.zeros((num_indices, self.num_dof_fwd), device=self._device)
        min_d = -0.5
        max_d = 2

        dists = torch_rand_float(min_d, max_d, (num_resets, 1), self._device)
        wood_dirs = torch_random_dir_2((num_resets, 1), self._device)
        hpos = dists * wood_dirs


        wood_pos = self.initial_wood_pos[indices_64, :].clone().to(self._device)
        # Randomizing wood position
        wood_rot = self.initial_wood_rots.clone()

        # Update
        wood_pos[:,0] +=  torch.randint(0, 2, (num_resets, 1), device=self._device).flatten() * 8
        wood_pos[:,1] += torch_rand_float(-3.5, 1, (num_resets, 1), self._device).flatten()        

        min_angle = 0.1
        max_angle = 3.14
        rand_angle = torch_rand_float(min_angle, max_angle, (num_resets, 1), self._device)

        wood_rot_euler = torch.zeros((num_resets, 3), device=self._device)
        wood_rot_euler[:, 2] = rand_angle.squeeze()

        wood_rot = axisangle2quat(wood_rot_euler)
        wood_velocities = torch.zeros((num_resets, 6), device=self._device)

        self.fwd_dof_targets[indices_64] = self.fwd_default_dof_pos.clone().repeat((num_resets,1))

        # Randomize pos of all active joints
        self.fwd_dof_targets[indices_64] = (self.active_dof_lower_limits - self.active_dof_upper_limits) *torch.rand_like(self.fwd_dof_targets[indices_64]) + self.active_dof_upper_limits

        self._fwds.set_joint_position_targets(self.fwd_dof_targets[indices_64], indices=indices, joint_indices=self.active_joints)
        self._fwds.set_joint_positions(self.fwd_dof_targets[indices_64], indices=indices, joint_indices=self.active_joints)
        self._fwds.set_joint_velocities(dof_vel, indices=indices)

        self._woods.set_velocities(wood_velocities, indices=indices)
        self._woods.set_world_poses(wood_pos, wood_rot, indices=indices)

        self.unloading_pos, self.unloading_ori = self._fwds.unloading_point.get_world_poses(clone=False)
        self.target_pos, _ = self._fwds.targets.get_world_poses(clone=False)

        self.target_pos[:, 2] = 1.25 # this is the real height of the target where the log should be dropped
        self.target_pos[:, 1] += 1.30688 # this is the real y position of the target

        #Instead of delivery tracker, let's add a multiplier for how close the wood was dropped
        # The closer to the target, the better
        self.drop_distance_multiplier = torch.zeros((self._num_envs), device=self._device)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):

        self.active_dof_upper_limits = torch.zeros((self.num_active_dof), device=self._device)
        self.active_dof_lower_limits = torch.zeros((self.num_active_dof), device=self._device)
        self.dof_limits = self._fwds.get_dof_limits()

        self.active_dof_lower_limits[:5] = self.dof_limits[0, :5, 0]
        self.active_dof_lower_limits[4:] = self.dof_limits[0, 6:, 0]

        self.active_dof_upper_limits[:4] = self.dof_limits[0, :4, 1]
        self.active_dof_upper_limits[4:] = self.dof_limits[0, 6:, 1]       
        self.fwd_dof_lower_limits = self. dof_limits[0, :, 0].to(device=self._device)
        self.fwd_dof_upper_limits = self.dof_limits[0, :, 1].to(device=self._device)

        self.initial_wood_pos, self.initial_wood_rots = self._woods.get_world_poses(clone=False) 
        self.wood_lifted = torch.ones((self._num_envs))
        self.wood_lift_count = torch.zeros_like(self.wood_lifted, device=self._device)

        # randomize all envs
        indices = torch.arange(self._fwds.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_distance_reward(self, obj1_pos, obj2_pos):        
        dist = torch.norm(obj1_pos-obj2_pos, p=2, dim=-1).flatten()
        return 1 / (1 + dist**2)

    def calculate_metrics(self) -> None:

        target_position = self.target_pos
        # change the z position of the target to be 1.25

        # Get reward for getting grapple close to wood
        wood_grappler_reward = self.calculate_distance_reward(self.grapple_body_pos, self.wood_pos)

        # Add reward for getting close to the target by z distance
        self.wood_target_reward = self.calculate_distance_reward(self.wood_pos, target_position)


        # Add reward for getting grapple close ot the target(important for leaving the wood)
        grapple_target_reward = self.calculate_distance_reward(self.grapple_body_pos, target_position)

        # Dropped at multiplier
        dropped_at =  self.wood_target_reward * wood_grappler_reward * grapple_target_reward

        # Update drop distance multipliers
        self.drop_distance_multiplier = torch.where(dropped_at > self.drop_distance_multiplier, dropped_at, self.drop_distance_multiplier)
        wood_unloading_reward = self.calculate_distance_reward(self.unloading_pos,self.wood_pos)

        r1 = wood_grappler_reward 
        r2 = self.wood_pos[:, 2] + wood_unloading_reward*self.reward_scale
        r3 = (self.wood_target_reward**2 / (torch.abs(self.wood_linear_vel[:,2])+1)) * self.reward_scale**3

        reward = torch.where(self.train_step==0, r1+r2, r1+r2+r3)

        #print(f'Amount of the delivered logs: {torch.where(self.wood_target_reward > 0.6, 1,0).sum()}')

        reward -= 0.1

        # Update reward buffer
        self.rew_buf[:] = reward
        
    def is_done(self) -> None:

        ones = torch.ones_like(self.reset_buf, device=self._device)
        resets = torch.zeros_like(self.reset_buf, device=self._device)
        resets = torch.where(self.progress_buf >= self._max_episode_length, ones, resets)
        # Update reset buffer
        self.reset_buf[:] = resets

