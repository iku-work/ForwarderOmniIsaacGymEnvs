from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.forwarder import Forwarder
from omniisaacgymenvs.robots.articulations.wood import Wood
from omniisaacgymenvs.robots.articulations.views.forwarder_view import ForwarderView
from omniisaacgymenvs.robots.articulations.views.wood_view import WoodView

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.objects import DynamicCapsule, DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.rotations import *

from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *

import numpy as np
import torch
import math


from omni.isaac.core.utils.prims import get_all_matching_child_prims


# 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

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
        self.action_scale = self._task_cfg["env"]["actionScale"]
        
        self.grapple_2_wood_dist_scale = self._task_cfg["env"]["grappleToWoodDistScale"]
        self.wood_2_unloading_dist_scale = self._task_cfg["env"]["woodToUnloadingPointDistScale"]
        self.woodToTargetDistScale = self._task_cfg["env"]["woodToTargetDistScale"]
        self.woodLiftScale = self._task_cfg["env"]["woodLiftScale"]

        self.stf = self._task_cfg["env"]["a"]
        self.dam = self._task_cfg["env"]["b"]
        
        self._forwarder_positions = torch.tensor([0.0, 0.0, 0.1])
        self._unloading_point_position = torch.tensor([0, -1, 4.5])
        self._ball_position = torch.tensor([0, -1, 1.0])
        #self._wood_initial_positions = torch.tensor([-5.0,  .0,   1.0], device=self._device)
        self._wood_positions = torch.tensor([5.0,  -2.0,   0.3])
        '''[-5.0,   .0,   1.0],
                                                     [5.0,   2.0,  1.0],
                                                     [5.0,  -2.0,  1.0]
                                                     ], device=self._device)'''

        #self._reset_dist = self._task_cfg["env"]["resetDist"]
        #self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]


        self._num_observations = 44 #41#35
        self._num_actions = 7
        
        self.num_dof_fwd = 9
        
        self.dt = 1/60
        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:
        self.get_fwd()
        self.get_wood()
        self.get_unloading_point()
        self.get_target()
        super().set_up_scene(scene)

        self._fwds = ForwarderView(prim_paths_expr="/World/envs/.*/forwarder", name="forwarder_view")
        scene.add(self._fwds)
        scene.add(self._fwds._grapple_body)
        #scene.add(self._fwds._grapple_l)
        #scene.add(self._fwds._grapple_r)
        scene.add(self._fwds._base_link)



        #self._woods = RigidPrimView(prim_paths_expr="/World/envs/.*/wood/base_link", name="wood_view", reset_xform_properties=False)
        self._woods = RigidPrimView(prim_paths_expr="/World/envs/.*/wood/base_link", name="wood_view", reset_xform_properties=False)        

        scene.add(self._woods)
        #scene.add(self._woods._grasp_pos_l)
        
        self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/ball", name="targets_view", reset_xform_properties=False)
        scene.add(self._targets)

        self._unloading_points = RigidPrimView(prim_paths_expr="/World/envs/.*/ball_0", name="unloading_point_view", reset_xform_properties=False)
        scene.add(self._unloading_points)

        #self._left_grasp_point = 

        self.fwd_default_dof_pos = torch.tensor(
            #[1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self._device
            [0, -1.066, -0.155, -2.239, -1.841, 0.035, 0.035], device=self._device
            #[0.0, -1.0, 0.0, -2.2, 0.8, 0.1, 0.1], device=self._device
        )
        self.active_joints = torch.tensor([0,1,2,3,6,7,8], device=self._device)
        self.action_scale = torch.tensor(self.action_scale, device=self._device)
        self.num_active_dof = len(self.active_joints)
        #self.num_wood_init_pos = len(self._wood_initial_positions)

        self.fwd_dof_targets = self.fwd_default_dof_pos.clone().repeat((self.num_envs,1))
        
        self.grapple_forward_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.wood_forward_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.grapple_down_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.wood_up_axis = torch.tensor([0, 0, -1], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))

        self.unloading_forward_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.grapple_body_vel = torch.zeros((self._num_envs, 3), device=self._device)
        self.max_dist = torch.full((self._num_envs,),20., device=self._device)

        self._fwds.set_gains(
                            kps=self.stf, 
                            kds=self.dam,
                            joint_indices=self.active_joints, 
                            indices=torch.arange(self._fwds.count, dtype=torch.int64, device=self._device)
                            )


        return

    def get_fwd(self):
        fwd = Forwarder(prim_path=self.default_zero_env_path + "/forwarder", name="forwarder", translation=self._forwarder_positions)

        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings("forwarder", get_prim_at_path(fwd.prim_path), self._sim_config.parse_actor_config("forwarder"))

    def get_wood(self):
        wood = Wood(prim_path=self.default_zero_env_path + "/wood", name="wood", translation=self._wood_positions)
        self._sim_config.apply_articulation_settings("wood", get_prim_at_path(wood.prim_path), self._sim_config.parse_actor_config("wood"))
    
    def get_target(self):
        radius = .5
        color = torch.tensor([0, 1, 0])
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/ball", 
            translation=self._ball_position, 
            name="target_0",
            radius=radius,
            color=color,
        )
        self._sim_config.apply_articulation_settings("ball", get_prim_at_path(ball.prim_path), self._sim_config.parse_actor_config("ball"))
        ball.set_collision_enabled(False)

    def get_unloading_point(self):
        radius = .1
        color = torch.tensor([1, 0, 0])
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/ball_0", 
            translation=self._unloading_point_position, 
            name="unloading_point_0",
            radius=radius,
            color=color,
        )
        self._sim_config.apply_articulation_settings("ball", get_prim_at_path(ball.prim_path), self._sim_config.parse_actor_config("ball"))
        ball.set_collision_enabled(False)

    def get_observations(self) -> dict:

        self.grapple_body_pos, self.grapple_body_ori = self._fwds._grapple_body.get_world_poses(clone=False)
        #self.grapple_body_vel = self._fwds._grapple_body.get_linear_velocities(clone=False)
        self._woods.get_angular_velocities(clone=False)

        self.wood_pos, self.wood_ori = self._woods.get_world_poses(clone=False)
        #self.wood_vel = self._woods.get_linear_velocities(clone=False)

        #self.wood_grasp_l_pos = self._woods._grasp_pos_l.get_world_poses(clone=False)
        #print(self.wood_grasp_l_pos[256,:].item())

        self.joints_positions = self._fwds.get_joint_positions(clone=False)
        self.joints_velocities = self._fwds.get_joint_velocities(clone=False)

        #print('Joints vel: ',self.joints_velocities[253,:])
        indices = torch.arange(self._fwds.count, dtype=torch.int32, device=self._device)
        vel = self._fwds._grapple_body.get_linear_velocities(indices=indices)
        #print(torch.sqrt(vel[253,0]**2 + vel[253,1]**2 + vel[253,2]**2))

        self.unloading_pos, self.unloading_ori = self._unloading_points.get_world_poses()
        
        self.target_pos, _ = self._targets.get_world_poses()

        to_wood = torch.zeros((self._num_envs, 3), device=self._device)
        to_wood = self.grapple_body_pos - self.wood_pos

        to_unload = self.wood_pos - self.unloading_pos

        self.obs_buf = torch.cat(
            (
                self.grapple_body_pos,
                to_wood,            # 3
                self.wood_ori,
                self.wood_pos,
                to_unload,
                self.unloading_pos,
                self.target_pos,
                #self.grapple_body_vel,
                #self.wood_vel,
                self.grapple_body_ori,  # 4
                 #4
                self.joints_positions,
                self.joints_velocities
                
                
            ),
            dim=-1,
        )

        #print('OBS BUFFER SHAPE: ', self.obs_buf.shape)
        '''self.obs_buf[:, 0] = fwd_dof_pos
        self.obs_buf[:, 1] = fwd_dof_ori
        self.obs_buf[:, 2] = wood_pos
        self.obs_buf[:, 3] = wood_vel'''

        #print('OBS BUF: ', gripper_body_pos.shape)

        observations = {
            self._fwds.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        #print(self.active_dof_lower_limits[-2:253], self.active_dof_upper_limits[-2:253])

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)

        self.actions[:,0:5] = actions[:,0:5]
        self.actions[:,5] = actions[:,-1]
        self.actions[:, 6] = actions[:,-1]
        
        #self.action_scale = 2.5

        #actions_scaled = self.dt * self.fwd_dof_targets * self.action_scale
        #targets = self.fwd_dof_targets + self.dt * self.actions * self.action_scale
        #self.action_scale = 2
        targets = self.fwd_dof_targets + self.actions * self.action_scale #* self.actions * self.action_scale
        self.fwd_dof_targets = tensor_clamp(targets, self.active_dof_lower_limits, self.active_dof_upper_limits)

        #print(self.fwd_dof_targets[253,1], self.actions[253,1])
        #print(self.actions[253,:])
        #print('Actions gripper: ', self.fwd_dof_targets[0, -2:-1])

        #self.fwd_dof_targets[:, -2] = actions[:,-1] 
        #self.fwd_dof_targets[:, -1] = actions[:,-1] 

        indices = torch.arange(self._fwds.count, dtype=torch.int32, device=self._device)

        #forces = torch.zeros_like(self.fwd_dof_lower_limits)
        #self._fwds.set_joint_efforts(forces, indices=indices)
        #self._fwds.set_joint_position_targets(actions, indices=indices)
        
        self._fwds.set_joint_position_targets(self.fwd_dof_targets, indices=indices, joint_indices=self.active_joints)
        #self._fwds.set_joint_positions(self.fwd_dof_targets, indices=indices, joint_indices=self.active_joints)
        #self.fwd_previous_dof_targets[:] = self.fwd_dof_targets[:]

    def reset_idx(self, env_ids):


        indices = env_ids.to(dtype=torch.int32)
        indices_64 = env_ids.to(dtype=torch.int64)
        num_indices = len(indices)
        num_resets = len(env_ids)
        
        dof_lower_limits = self.dof_limits[0, :, 0]
        dof_upper_limits = self.dof_limits[0, :, 1]

        #print('DOF limits: ', dof_lower_limits)

        '''
        pos = tensor_clamp(
            self.fwd_default_dof_pos.unsqueeze(0)
            + (torch.rand((len(env_ids), self.num_active_dof), device=self._device) - 0.5) ,
            self.active_dof_lower_limits,
            self.active_dof_upper_limits,
        )
        '''

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
        wood_pos[:, 1:2] += hpos[..., 1:2] 

        #sides = torch_random_dir_2((self.num_envs, 1), self._device)
        self.wood_sides = torch.cuda.FloatTensor(num_resets, 1).uniform_() > 0.5
        
        # Swith sides
        wood_pos[:,0] += self.wood_sides.flatten() * -9

        #initial_wood_pos = self.initial_wood_pos.clone()
        #initial_wood_pos.repeat(num_resets,1)

        #
        min_angle = 0.1
        max_angle = 3.14
        #rand_angle = torch_rand_float(min_angle, max_angle, (num_resets, 1), self._device)
        rand_angle = torch_rand_float(89, 90, (num_resets, 1), self._device)

        wood_rot_euler = torch.zeros((num_resets, 3), device=self._device)
        wood_rot_euler[:, 2] = rand_angle.squeeze()
        wood_rot = axisangle2quat(wood_rot_euler)
        
        wood_velocities = torch.zeros((num_resets, 6), device=self._device)

        #self.fwd_dof_targets = pos.detach().clone()
        self.fwd_dof_targets[indices_64] = torch.zeros((num_resets, self.num_active_dof), device=self._device)
        
        #print(self.active_dof_lower_limits[0], self.active_dof_upper_limits[0])

        #self.fwd_dof_targets[indices_64, 0] = torch_rand_float(-self.active_dof_lower_limits[0], 
        #                                                       self.active_dof_upper_limits[0], 
        #                                                       (num_resets, 1), self._device).flatten()
        
        # Randomize joint 0 rotation
        #self.fwd_dof_targets[indices_64, 0] = torch_rand_float(-1, 1, (num_resets, 1), self._device).flatten()
        
        #self.initial_fwd_pos = self.fwd_default_dof_pos.clone().repeat((num_resets, 1))
        #self.fwd_dof_targets = self.fwd_default_dof_pos.clone().repeat((self.num_envs,1))
        self._fwds.set_joint_position_targets(self.fwd_dof_targets[indices_64], indices=indices, joint_indices=self.active_joints)
        self._fwds.set_joint_positions(self.fwd_dof_targets[indices_64], indices=indices, joint_indices=self.active_joints)
        #self._fwds.set_joint_positions(pos, indices=indices, joint_indices=self.active_joints)
        self._fwds.set_joint_velocities(dof_vel, indices=indices)

        #self._fwds.set_world_poses(self.initial_fwd_pos[indices_64].clone(), self.initial_fwd_rot[indices_64].clone(), indices=indices)
        

        self._woods.set_velocities(wood_velocities, indices=indices)
        self._woods.set_world_poses(wood_pos, wood_rot, indices=indices)



        #self.side_reward = torch.ones((512,1), device=self._device).flatten()
        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):

        

        self.active_dof_upper_limits = torch.zeros((self.num_active_dof), device=self._device)
        self.active_dof_lower_limits = torch.zeros((self.num_active_dof), device=self._device)
        self.dof_limits = self._fwds.get_dof_limits()
  
        self.active_dof_lower_limits[:4] = self.dof_limits[0, :4, 0]
        self.active_dof_lower_limits[4] = self.dof_limits[0, 4, 0]
        self.active_dof_lower_limits[-2:] = self.dof_limits[0, -2:, 0]

        self.active_dof_upper_limits[:4] = self.dof_limits[0, :4, 1]
        self.active_dof_upper_limits[4] = self.dof_limits[0, 4, 1]
        self.active_dof_upper_limits[-2:] = self.dof_limits[0, -2:, 1]       

        self.fwd_dof_lower_limits = self. dof_limits[0, :, 0].to(device=self._device)
        self.fwd_dof_upper_limits = self.dof_limits[0, :, 1].to(device=self._device)

                
        self.actions= torch.zeros(
            (self._num_envs, self.num_active_dof), dtype=torch.float, device=self._device
        )

        '''self.fwd_dof_targets = torch.zeros(
            (self._num_envs, self.num_active_dof), dtype=torch.float, device=self._device
        )'''
        

        self.fwd_dof_pos = torch.zeros((self.num_envs, self.num_active_dof), device=self._device)

        #self.initial_wood_pos, self.initial_wood_rots = self._woods.get_world_poses() 

        self.initial_wood_pos, self.initial_wood_rots = self._woods.get_world_poses() 
        #self.initial_wood_pos[:,0] += 6
        #self.initial_wood_pos[:,2] +=2

        #self.unloading_pos, unloading_ori = self._unloading_points.get_world_poses()
        
        self.old_dist_wood_2_target = torch.full((self.num_envs,), 20, device=self._device)

        self.wood_lifted = torch.zeros((self._num_envs))
        self.wood_lift_count = torch.zeros_like(self.wood_lifted, device=self._device)


        self.initial_fwd_pos, self.initial_fwd_rot = self._fwds.get_world_poses()

        
        self.min_dist = torch.zeros_like(self.max_dist)

        # randomize all envs
        indices = torch.arange(self._fwds.count, dtype=torch.int64, device=self._device)
        '''self._fwds.set_gains(
                            kps=torch.tensor(self.stf), 
                            kds=torch.tensor(self.dam), 
                            joint_indices=self.active_joints, 
                            indices=indices
                            )'''
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:


        test_env = 253


        # Grapple body's X-axis vector
        axis1 = tf_vector(self.grapple_body_ori, self.grapple_forward_axis)
        # Wood Y-axis vector 
        axis2 = tf_vector(self.wood_ori, self.wood_forward_axis)

        _,base_link_ori = self._fwds._base_link.get_world_poses(clone=False)
        axis3 = tf_vector(base_link_ori, self.unloading_forward_axis)

        grapple_rel2_target_pos = self.target_pos - self.grapple_body_pos
        wood_rel2_target_pos = self.target_pos - self.wood_pos

        grapple_side = torch.sign(grapple_rel2_target_pos[:,0])
        wood_side = torch.sign(wood_rel2_target_pos[:,0])

        self.side_reward = torch.eq(grapple_side, wood_side, out=None)
        
        grapple_rel_dist = torch.norm((grapple_rel2_target_pos - torch.zeros_like(grapple_rel2_target_pos, device=self._device)), p=2,dim=-1)
        
        n_passed = torch.where(self.progress_buf > 50, 1,0)
        self.far_on_wrong_side = grapple_rel_dist * ~self.side_reward.detach().clone() * n_passed

        # Offset from CM along Y-axis, where grasp allowed
        self.wood_grasp_postion_offset = .8
        self.wood_grasp_positions = self.wood_pos.repeat(3,1,1)
        self.wood_grasp_positions[0,:,:] += axis2*self.wood_grasp_postion_offset
        self.wood_grasp_positions[2,:,:] += -axis2*self.wood_grasp_postion_offset
        
        grapple_body_dist = torch.norm(self.wood_grasp_positions-self.grapple_body_pos.repeat(3,1,1), dim=-1,p=2)
        grab_reward = 1 / (1 + torch.min(grapple_body_dist, dim=0).values**2)
        '''
        grapple_l_pos,_ = self._fwds._grapple_l.get_world_poses(clone=False)
        grapple_r_pos,_ = self._fwds._grapple_r.get_world_poses(clone=False)
        
        # Normalize distance
        grapple_l_dist = torch.norm(self.wood_grasp_positions-grapple_l_pos.repeat(3,1,1),dim=-1,p=2)
        grapple_r_dist = torch.norm(self.wood_grasp_positions-grapple_r_pos.repeat(3,1,1),dim=-1,p=2)
        grapple_body_dist = torch.norm(self.wood_grasp_positions-self.grapple_body_pos.repeat(3,1,1), dim=-1,p=2)

        #self.grapple_l_dist_max = torch.max(grapple_l_dist)
        #self.grapple_r_dist_max = torch.max(grapple_r_dist)

        #self.grapple_l_dist_max = self.grapple_l_dist_max.repeat((512,)).flatten()
        #self.grapple_r_dist_max = self.grapple_r_dist_max.repeat((512,)).flatten()
        
        #grapple_dist_is_close = torch.isclose(torch.min(grapple_l_dist, dim=0).values, torch.min(grapple_r_dist, dim=0).values, 0.1)

        #grapple_l_dist = self.min_max_norm(grapple_l_dist, self.min_dist, self.max_dist)
        #grapple_r_dist = self.min_max_norm(grapple_r_dist, self.min_dist, self.max_dist)
        r_b_close = torch.min(grapple_r_dist, dim=0).values < .4
        l_b_close = torch.min(grapple_l_dist, dim=0).values < .4
        b_close = torch.min(grapple_body_dist, dim=0).values < .4
        self.wood_grasped = r_b_close * l_b_close * b_close
        

        #====
        min_body_2_wood_dist = torch.min(grapple_body_dist,dim=0).values

        grapple_l_rew = 1/(1+(torch.min(grapple_l_dist, dim=0).values)**2)
        grapple_r_rew = 1/(1+(torch.min(grapple_r_dist, dim=0).values)**2)
        grapple_body_rew=1/(1+min_body_2_wood_dist**2)
        
        
        #grab_reward = 1 / (1 + torch.min(grapple_r_dist, dim=0).values**2 + torch.min(grapple_l_dist, dim=0).values**2 + min_body_2_wood_dist**2)
        grab_reward = (grapple_l_rew + grapple_r_rew + grapple_body_rew)
        #grapple_body_rew = torch.exp(-min_body_2_wood_dist)
        ''' 
        #grab_reward = grapple_body_rew 
        #grab_reward = torch.where(torch.min(grapple_r_dist, dim=0).values<.4, grab_reward+grapple_r_rew, grab_reward)
        #grab_reward = torch.where(torch.min(grapple_l_dist, dim=0).values<.4, grab_reward+grapple_l_rew, grab_reward)
        grab_reward *= self.grapple_2_wood_dist_scale

        
        #grab_reward = torch.where(min_body_2_wood_dist<.5, grab_reward**2, grab_reward)
        
        #grab_reward = torch.where(min_body_2_wood_dist<.4, grab_reward*2,grab_reward)
        

        #grab_reward = 1 / (1+dist_grapple_2_wood_cm**2)

        #print(torch.min(grapple_l_dist, dim=0).values[test_env].item(), torch.min(grapple_r_dist, dim=0).values[test_env].item(),torch.min(grapple_body_dist, dim=0).values[test_env].item())

        #print('R: ',torch.isclose(torch.min(grapple_r_dist, dim=0).values, torch.min(grapple_body_dist, dim=0).values, .1)[test_env])
        #print('L: ',torch.isclose(torch.min(grapple_l_dist, dim=0).values, torch.min(grapple_body_dist, dim=0).values, .1)[test_env])


        

        #print(torch.min(grapple_r_dist, dim=0).values[test_env])
        #print(r_b_close[test_env].item(), l_b_close[test_env].item(), b_close[test_env].item(),self.wood_grasped[253].item())
        # Distance from wood CM to Unloading Point
        self.wood_lifted = (self.wood_pos[:, 2] > 0.3) 
        #print(self.wood_lifted[253])
        self.wood_lift_count += self.wood_lifted
        #print(torch.max(self.wood_lift_count), self.wood_pos[:, 2] > 0.3, )

        dist_wood_2_unloading = torch.norm(self.unloading_pos-self.wood_pos, p=2, dim=-1)
        dist_2 = (self.unloading_pos[:,0]-self.wood_pos[:,0])**2 + (self.unloading_pos[:,1]-self.wood_pos[:,1])**2 + (self.unloading_pos[:,2]-self.wood_pos[:,2])**2
        dist = torch.sqrt(dist_2)
        #print('MAX ', torch.max(dist))
        #print('MAx dist wood to target: ', torch.max(dist_wood_2_unloading))
        #dist_wood_2_unloading = self.min_max_norm(dist_wood_2_unloading, self.min_dist, self.max_dist)
        #dist_wood_2_unloading = self.calculate_distance_reward(self.unloading_pos, self.wood_pos)

        v_dist_wood_2_unloading = torch.abs(self.unloading_pos[:,2]-self.wood_pos[:,2]).flatten()
        v_2_unload_reward = 1 / (1 + v_dist_wood_2_unloading ** 2)

        h_dist_wood_2_unloading = torch.norm(self.unloading_pos[:,:2]-self.wood_pos[:,:2], p=2, dim=-1).flatten()
        x_dist_wood_2_unloading = torch.abs(self.unloading_pos[:,0]-self.wood_pos[:,0]).flatten()
        x_2_unload_reward = 1.0 / (1.0 + x_dist_wood_2_unloading ** 2)
        h_2_unload_reward = 1.0 / (1.0 + h_dist_wood_2_unloading ** 2)

        wood_2_unload_reward = 1 / (1.0 + dist_wood_2_unloading  ** 2) 
        #wood_2_unload_reward = torch.exp(-dist_wood_2_unloading)
        lift_n_unloadDist_reward = self.wood_lifted * self.woodLiftScale #* wood_2_unload_reward * self.wood_2_unloading_dist_scale

        
        #self._targets.set_world_poses(positions=self.wood_pos + axis2*-.8, orientations=torch.zeros((self.num_envs, 4), device=self._device))
        # Alignment of wood and grapple axes
        dot1 = torch.bmm(axis1.view(self.num_envs, 1, 3), axis2.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
        dot2 = torch.bmm(axis2.view(self.num_envs, 1, 3), axis3.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper

        rot_reward_scale = 0.01

        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = torch.abs(dot1) * 0.5 * rot_reward_scale
        # Grapple closing reward

        rot_wood_2_unload_reward = torch.abs(dot2) * .5

        # Distance from wood CM to Target
        self.dist_wood_2_target = torch.norm(self.target_pos - self.wood_pos, p=2, dim=-1)

        wood_2_target_reward = 1.0 / (1.0 + self.dist_wood_2_target**2)

        # Reward for vertical distance to target
        v_wood_2_target = torch.abs(self.target_pos[:,-1] - self.wood_pos[:,-1])
        v_wood_2_target_reward = 1 / (1 + v_wood_2_target**2)

        close_2_unload = torch.where(h_dist_wood_2_unloading<1,1,0)
        #print('Close 2 unload: ', close_2_unload[test_env])
        grapple_opened = torch.where(self.fwd_dof_targets[:,5]>0,1,0)

        # First stage reward: grapple to wood distance and rotation 
        reward = grab_reward * rot_reward 
        #reward = torch.where(self.wood_grasped==True, 10*reward, reward)
        #reward += wood_2_unload_reward * self.wood_grasped * 100
        #reward += self.wood_pos[:,2] * self.wood_grasped
        
        #print('Stage 1: ', (grab_reward * rot_reward)[test_env] )

        #print("First stage rew: ", reward[253].item())

        #reward = torch.where(self.wood_lifted==1, reward**2, reward)

        # Reward for bringing the wood to unloading point
        #reward += self.wood_grasped * ( v_2_unload_reward*2 + (1.5- torch.abs(self.joints_positions[:,0]))) * self.wood_2_unloading_dist_scale
        #reward += self.wood_grasped * ( v_2_unload_reward + wood_2_unload_reward * 1.5) * self.wood_2_unloading_dist_scale 
        reward += self.wood_lifted * 1000
        #reward += self.wood_grasped * self.wood_pos[:,2] * 10
        #reward += self.wood_grasped * wood_2_unload_reward * self.wood_2_unloading_dist_scale 
        #* rot_wood_2_unload_reward
        #print('Stage 2: ', (self.wood_lifted * ( v_2_unload_reward*2 + (1.5- torch.abs(self.fwd_dof_targets[:,0]))) * self.wood_2_unloading_dist_scale)[test_env])
        
        '''v_grapple_2_unload = torch.abs(self.grapple_body_pos[:,2]-self.unloading_pos[:,2])
        h_grapple_2_unload = torch.norm(self.grapple_body_pos[:,:2]-self.unloading_pos[:,:2])
        v_grapple_2_unload = self.min_max_norm(v_grapple_2_unload, self.min_dist, self.max_dist)
        h_grapple_2_unload = self.min_max_norm(h_grapple_2_unload, self.min_dist, self.max_dist)

        v_grapple_2_unload_reward = 1./(1.+v_grapple_2_unload**2)
        h_grapple_2_unload_reward = 1./(1.+h_grapple_2_unload**2)
        '''
        #reward += self.wood_lifted * ( v_2_unload_reward + wood_2_unload_reward * 5) * self.wood_2_unloading_dist_scale

        # Reward for opening the grapple during the wood lifting / movement
        grapple_opening_reward =  (1+self.fwd_dof_targets[:,5]) * self.wood_lifted * 0.1

        # 
        #print(reward[253].item(), self.wood_lifted * v_wood_2_target_reward  * close_2_unload)
        reward += v_wood_2_target_reward  * close_2_unload  * grapple_opening_reward * rot_wood_2_unload_reward * 100
        
        #print('Stage 3: ', (v_wood_2_target_reward  * close_2_unload * rot_wood_2_unload_reward * grapple_opening_reward * 100)[test_env])

        #reward = torch.where(self.wood_grasped==True, reward+1000, reward)

        #reward += self.wood_grasped * torch.sum(self.wood_vel, dim=-1) * 0.01

        #reward += close_2_unload * rot_wood_2_unload_reward

         # Instead of this one, maybe bring the reward with square here?
        #reward += torch.where(close_2_unload==True, (wood_2_target_reward + reward)**2,0)
        # ???

        #reward = torch.where(self.wood_grasped == True, reward**2, reward)

        # Reward for bringing the wood to the target in the right orientation
        #reward = torch.where(self.dist_wood_2_target<.5, reward**2,reward)'

        #print("Joint vel: ",self.joints_velocities[253,8])
        #reward += torch.abs(self.joints_velocities[:,8])
        #reward = torch.where(close_2_unload==True, reward**2,reward)

        #reward -= torch.where(self.dist_wood_2_target>self.max_dist, -self.dist_wood_2_target, 0)

        # Action penalty for every step to reduce speed / motion jerkiness
        #action_penalty = torch.sum(self.actions ** 2, dim=-1) * 1e-5
        #reward -= action_penalty
        
        #print(self.fwd_dof_targets[test_env,0])

        #reward -= torch.sum(self.grapple_body_vel, dim=1) * .01
        #reward *= reward*0.01
        # Give a reward if the wood and grapple are at the same side of the bed 
        #reward = -torch.min(grapple_body_dist, dim=0).values
        reward *= self.side_reward 




        # Set rewards 
        self.rew_buf[:] = reward

    def is_done(self) -> None:
        #resets = torch.where(self.dist_wood_2_target >= 25, 1, 0)

        #print()
        ones = torch.ones_like(self.reset_buf, device=self._device)
        resets = torch.zeros_like(self.reset_buf, device=self._device)

        #print(self.far_on_wrong_side[253].item())

        #resets = torch.where((self.actions[:,5]<0.25 )&(self.dist_grapple_2_wood>2),ones,resets)
        resets = torch.where(self.far_on_wrong_side > 5.0, ones, resets)
        
        half_episode = torch.where(self.progress_buf > self._max_episode_length/2, 1,0)
        not_picked = torch.where(self.wood_lift_count<1, 1,0)
        resets = torch.where((not_picked*half_episode)==1, ones, resets)
        
        #resets = torch.where((half_episode*torch.logical_not(self.wood_lifted))==1, ones, resets)
        #resets = torch.where(self.dist_wood_2_target>self.max_dist, ones, resets)
        #resets = torch.where(self.dist_wood_2_target<0.5, ones, resets)
        resets = torch.where(self.progress_buf >= self._max_episode_length, ones, resets)
        
        #print(self.progress_buf[253])
        #

        #resets = torch.where((n_not_closed_passed*torch.logical_not(self.grasp_success))==1, ones, resets)
        self.reset_buf[:] = resets
        


        '''resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        
        '''



    def calculate_distance_reward(self, obj1_pos, obj2_pos):        
        dist = torch.norm(obj1_pos-obj2_pos, p=2, dim=-1).flatten()
        return 1 / (1 + dist**2)

    def min_max_norm(self, value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)
    
    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :7], self._qd[:, :7]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 7:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                            -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))

        return u