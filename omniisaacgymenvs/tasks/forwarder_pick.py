from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.forwarder import Forwarder
from omniisaacgymenvs.robots.articulations.wood import Wood
from omniisaacgymenvs.robots.articulations.views.forwarder_view import ForwarderView
from omniisaacgymenvs.robots.articulations.views.wood_view import WoodView

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.objects import DynamicCapsule, DynamicSphere
from omni.isaac.core.materials import PhysicsMaterial
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
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.action_scale = self._task_cfg["env"]["actionScale"]
        
        self.grapple_2_wood_dist_scale = self._task_cfg["env"]["grappleToWoodDistScale"]
        self.wood_2_unloading_dist_scale = self._task_cfg["env"]["woodToUnloadingPointDistScale"]
        self.woodToTargetDistScale = self._task_cfg["env"]["woodToTargetDistScale"]
        self.woodLiftScale = self._task_cfg["env"]["woodLiftScale"]

        self.kps = self._task_cfg["env"]["kps"]
        self.kds = self._task_cfg["env"]["kds"]
        self.force = self._task_cfg["env"]["force"]
        
        self._forwarder_positions = torch.tensor([0.0, 0.0, 0.1])
        self._wood_positions = torch.tensor([5.0,  -2.0,   4])        

        self._num_observations = 56 #41#35
        self._num_actions = 6
        self.num_dof_fwd = 9 
        self.dt = 1/60 #self._task_cfg["sim"]["dt"] #1/60
       
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
        #self._woods = RigidPrimView(prim_paths_expr="/World/envs/.*/wood", name="wood_view", reset_xform_properties=False) 
        scene.add(self._woods)

        self.fwd_default_dof_pos = torch.tensor(
                        [0, -0.349066, -0.436332, -2.239, .2, 0.035, 0.035], device=self._device
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

        print('======CUDA AVAILABLE',self._device)

        return

    def get_fwd(self):
        fwd = Forwarder(prim_path=self.default_zero_env_path + "/forwarder", name="forwarder", translation=self._forwarder_positions)
        fwd.set_enabled_self_collisions(True)
        # applies articulation settings from the task configuration yaml file

        fwd_config = self._sim_config.parse_actor_config("forwarder")
        fwd_config['enable_self_collisions'] = True
        self._sim_config.apply_articulation_settings("forwarder", get_prim_at_path(fwd.prim_path), fwd_config)

    def get_wood(self):
        
        
        '''wood = DynamicCapsule(    
            prim_path= self.default_zero_env_path +  "/wood", 
            name='wood',
            translation=self._wood_positions,
            orientation=axisangle2quat(torch.tensor([0,.0,.0], device=self._device)),
            radius=0.3,
            height=4.0,
            color=np.array([0.0, 1.0, 0.0]),
            mass=100.0
            )
        material = PhysicsMaterial(
        prim_path="/World/physics_material/wood",  # path to the material prim to create
        dynamic_friction=0.4,
        static_friction=1.1,
        restitution=0.1
        )'''

        wood = Wood(prim_path=self.default_zero_env_path + "/wood", name="wood", translation=self._wood_positions)
        #wood.apply_physics_material(material)
        self._sim_config.apply_articulation_settings("wood", get_prim_at_path(wood.prim_path), self._sim_config.parse_actor_config("wood"))
    


    def get_observations(self) -> dict:

        self.grapple_body_pos, self.grapple_body_ori = self._fwds._grapple_body.get_world_poses(clone=False)
        self.grapple_l_pos, self.grapple_l_ori = self._fwds._grapple_l.get_world_poses(clone=False)
        self.grapple_r_pos, self.grapple_r_ori = self._fwds._grapple_r.get_world_poses(clone=False)
        self.grapple_body_vel = self._fwds._grapple_body.get_velocities(clone=False)
        #self._woods.get_angular_velocities(clone=False)

        # use this instead get_current_dynamic_state()
        self.wood_pos, self.wood_ori = self._woods.get_world_poses(clone=False)
        #self.wood_pos[:,2]+=0.5
        #self.wood_vel = self._woods.get_linear_velocities(clone=False)

        #self.wood_grasp_l_pos = self._woods._grasp_pos_l.get_world_poses(clone=False)
        #self.wood_grasp_r_pos = self._woods._grasp_pos_r.get_world_poses(clone=False)
        #print(self.wood_grasp_l_pos[256,:].item())

        joint_states = self._fwds.get_joints_state()
        joints_positions, joints_velocities = joint_states.positions, joint_states.velocities
        #joints_positions = self._fwds.get_joint_positions(clone=False)
        #joints_velocities = self._fwds.get_joint_velocities(clone=False)

        #print('Joints vel: ',self.joints_velocities[253,:])
        #indices = torch.arange(self._fwds.count, dtype=torch.int32, device=self._device)
        #vel = self._fwds._grapple_body.get_linear_velocities(indices=indices)
        #print(torch.sqrt(vel[253,0]**2 + vel[253,1]**2 + vel[253,2]**2))

        #self.unloading_pos, unloading_ori = self._fwds.unloading_point.get_world_poses(clone=False)
        

        #self.target_pos, _ = self._fwds.targets.get_world_poses()

        #to_wood = torch.zeros((self._num_envs, 3), device=self._device)
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
                self.grapple_l_pos,
                self.grapple_r_pos,
                self.grapple_body_vel,
                #self.wood_vel,
                self.grapple_body_ori,  # 4
                 #4
                joints_positions,
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

        #print(self.active_dof_lower_limits[-2:253], self.active_dof_upper_limits[-2:253])

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)

        self.actions[:,0:4] = actions[:,0:4]
        self.actions[:,4] = actions[:,4]
        self.actions[:,5] = actions[:,5]
        self.actions[:, 6] = -actions[:,5]
        
        #self.action_scale = 2.5

        #actions_scaled = self.dt * self.fwd_dof_targets * self.action_scale
        #targets = self.fwd_dof_targets + self.dt * self.actions * self.action_scale
        #self.action_scale = 2
        targets = self.fwd_dof_targets + self.actions * self.action_scale * self.dt #* self.actions * self.action_scale
        self.fwd_dof_targets = tensor_clamp(targets, self.active_dof_lower_limits, self.active_dof_upper_limits)     
        #print((self.actions * self.action_scale * self.dt)[253,:] *180/torch.pi)
        

        # Target for tuning
        #self.fwd_dof_targets = torch.tensor([ 0.5747, 1.5, -.5708,  0.50000,  2.1169, -0.2854, -0.2854], device=self._device).repeat((self._num_envs,1))
        #print(self.fwd_dof_targets[253,])
        #print(self.actions[253,:])
        #print('Actions gripper: ', self.fwd_dof_targets[0, -2:-1])

        #self.fwd_dof_targets[:, -2] = actions[:,-1] 
        #self.fwd_dof_targets[:, -1] = actions[:,-1] 

        indices = torch.arange(self._fwds.count, dtype=torch.int32, device=self._device)

        #forces = torch.zeros_like(self.fwd_dof_lower_limits)
        #self._fwds.set_joint_efforts(forces, indices=indices)
        #self._fwds.set_joint_position_targets(actions, indices=indices)
        
        self._fwds.set_joint_position_targets(self.fwd_dof_targets, indices=indices, joint_indices=self.active_joints)
        
        #self.fwd_dof_targets[:,:-2]*1e20
        #self._fwds.set_joint_efforts(self.fwd_dof_targets, indices=indices, joint_indices=self.active_joints)
        #self._fwds.set_joint_positions(self.fwd_dof_targets, indices=indices, joint_indices=self.active_joints)
        #self.fwd_previous_dof_targets[:] = self.fwd_dof_targets[:]

    def reset_idx(self, env_ids):


        indices = env_ids.to(dtype=torch.int32)
        indices_64 = env_ids.to(dtype=torch.int64)
        num_indices = len(indices)
        num_resets = len(env_ids)

        '''self._fwds.set_gains(
                            kps=torch.tensor(self.kps, dtype=torch.float).repeat((self._num_envs,1)), 
                            kds=torch.tensor(self.kds, dtype=torch.float).repeat((self._num_envs,1)),
                            joint_indices=self.active_joints, 
                            indices=torch.arange(self._fwds.count, dtype=torch.int64, device=self._device)
                            )
        self._fwds.set_max_efforts(torch.tensor(self.force, dtype=torch.float).repeat((self._num_envs,1)),
                                   joint_indices=self.active_joints, 
                                   indices=torch.arange(self._fwds.count, 
                                   dtype=torch.int64, 
                                   device=self._device)
                            )
        '''
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
        #self.wood_sides = torch.cuda.FloatTensor(num_resets, 1).uniform_() > 0.5
        
        # Swith sides
        #self.wood_sides[indices] = ~self.wood_sides[indices]
        wood_pos[:,0] +=  torch.randint(-10, 0, (num_resets, 1), device=self._device).flatten()

        #print(self.fwd_dof_targets[indices].shape)

        #initial_wood_pos = self.initial_wood_pos.clone()
        #initial_wood_pos.repeat(num_resets,1)

        #
        min_angle = 0.1
        max_angle = 3.14
        rand_angle = torch_rand_float(min_angle, max_angle, (num_resets, 1), self._device)
        #rand_angle = torch_rand_float(89, 90, (num_resets, 1), self._device)

        wood_rot_euler = torch.zeros((num_resets, 3), device=self._device)
        wood_rot_euler[:, 2] = rand_angle.squeeze()
        wood_rot_euler = torch.tensor([0.0,1.58,0.0], device=self._device).clone().repeat((num_resets,1))
        wood_rot = axisangle2quat(wood_rot_euler)
        wood_velocities = torch.zeros((num_resets, 6), device=self._device)

        #self.fwd_dof_targets = pos.detach().clone()
        #self.fwd_dof_targets[indices_64] = torch.zeros((num_resets, self.num_active_dof), device=self._device)
        self.fwd_dof_targets[indices_64] = self.fwd_default_dof_pos.clone().repeat((num_resets,1))

        #torch.rand((num_resets,1)).uniform(self.active_dof_lower_limits,self.active_dof_upper_limits)
        
        #(a - b)*torch.rand(5, 3) + b

        # Randomize pos of all active joints
        #self.fwd_dof_targets[indices_64] = (self.active_dof_lower_limits - self.active_dof_upper_limits) *torch.rand_like(self.fwd_dof_targets[indices_64]) + self.active_dof_upper_limits
        
        #print('Active limits: ', self.active_dof_upper_limits)
        #print('TARGETS ', self.fwd_dof_targets[253,:])
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
        #self.grapple_body_pos, self.grapple_body_ori = self._fwds._grapple_body.get_world_poses(clone=False)
        #self.old_to_wood_dist = torch.norm(wood_pos-self.grapple_body_pos, dim=-1,p=2)     

        self._woods.set_velocities(wood_velocities, indices=indices)
        self._woods.set_world_poses(wood_pos, wood_rot, indices=indices)

        
        self.unloading_pos, self.unloading_ori = self._fwds.unloading_point.get_world_poses(clone=False)
        self.target_pos, _ = self._fwds.targets.get_world_poses(clone=False)


        #self.side_reward = torch.ones((512,1), device=self._device).flatten()
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



        '''self.fwd_dof_targets = torch.zeros(
            (self._num_envs, self.num_active_dof), dtype=torch.float, device=self._device
        )'''
        

        self.initial_wood_pos, self.initial_wood_rots = self._woods.get_world_poses(clone=False) 

        self.wood_lifted = torch.ones((self._num_envs))
        self.wood_lift_count = torch.zeros_like(self.wood_lifted, device=self._device)


        #self.initial_fwd_pos, self.initial_fwd_rot = self._fwds.get_world_poses(clone=False)
        #self.min_dist = torch.zeros_like(self.max_dist)

        # randomize all envs
        indices = torch.arange(self._fwds.count, dtype=torch.int64, device=self._device)
        '''self._fwds.set_gains(
                            kps=torch.tensor(self.stf), 
                            kds=torch.tensor(self.dam), 
                            joint_indices=self.active_joints, 
                            indices=indices
                            )'''
        self.reset_idx(indices)

    def calculate_distance_reward(self, obj1_pos, obj2_pos):        
        dist = torch.norm(obj1_pos-obj2_pos, p=2, dim=-1).flatten()
        return 1 / (1 + dist**2)

    def calculate_metrics(self) -> None:

        target_position = self.target_pos
        # change the z position of the target to be 1.25
        target_position[:, 2] = 1.25 # this is a real height of the target where the log should be dropped

        # Get reward for getting grapple close to wood
        wood_grappler_reward = self.calculate_distance_reward(self.grapple_body_pos, self.wood_pos)

        # Get multiplier for getting grapple close to wood
        close_to_wood = (wood_grappler_reward > 0.9).float()

        reward = wood_grappler_reward

        # Add reward for lifting the wood#
        reward *= 1 + self.wood_pos[:, 2] * close_to_wood * 1e3

        # Set reward for getting close to the target by x-y distance
        wood_drop_zone_reward = self.calculate_distance_reward(self.wood_pos[:, :2], target_position[:, :2])

        # Multiplier for getting close to the target
        close_to_drop_zone = (wood_drop_zone_reward > 0.95).float()


        # Add reward for getting close to the target by z distance
        wood_target_reward =self.calculate_distance_reward(self.wood_pos[:, 2], target_position[:, 2])

        # Add reward for getting close to the drop while having the wood in the grapple
        reward *= 1 + wood_drop_zone_reward * close_to_drop_zone  *  1e5 * close_to_wood * wood_target_reward 

        print('Reward: ', torch.mean(reward), 'Close to wood: ', torch.mean(close_to_wood), 'Close to drop-zone: ', torch.mean(close_to_drop_zone), 'Wood z pos: ', torch.mean(self.wood_pos[:, 2]))


        reward -= 0.001
        # Update reward buffer
        self.rew_buf[:] = reward
        


    def is_done(self) -> None:

        ones = torch.ones_like(self.reset_buf, device=self._device)
        resets = torch.zeros_like(self.reset_buf, device=self._device)

        # Reset if:
        # On wrong side and more than 50 steps passed
        #resets = torch.where((self.side_reward == 0)&(self.progress_buf > 50),ones,resets)
        # If wood was lifted for less 15 steps and half an episode passed
        #resets = torch.where((self.wood_lift_count<15)&(self.progress_buf > self._max_episode_length/2),ones,resets)
        # If max steps reached for episode
        resets = torch.where(self.progress_buf >= self._max_episode_length, ones, resets)
        # Update reset buffer
        self.reset_buf[:] = resets


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