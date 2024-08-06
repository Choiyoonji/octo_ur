import numpy as np
import os
import torch
import jax.numpy as jnp

import gym
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_jit_utils import quat_mul, to_torch, tensor_clamp


Headless = True


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
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class IsaacUR5eBottle(gym.Env):
    def __init__(self):
        im_size = 256

        self.observation_space = gym.spaces.Dict(
            {
                **{
                    "image_primary": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((15,)) * -6.28, high=np.ones((15,)) * 6.28, dtype=np.float32
                ),
            }
        )

        self.action_space = gym.spaces.Box(
            low=np.ones((7,)) * -1/15, high=np.ones((7,)) * 1/15, dtype=np.float32
        )

        self._im_size = im_size
        self._rng = np.random.default_rng(1234)

        self.gym = gymapi.acquire_gym()

        self.sim = None

        self.bowl_asset = None
        self.bottle_asset = None
        self.table_asset = None
        self.ur_asset = None

        self.env = None
        self.bowl_handle = None
        self.bottle_handle = None
        self.table_handle = None
        self.ur5e_handle = None
        self.gripper_handle = None

        self.init_pos = None
        self.init_rot = None
        self.gripper_idx = None

        self.create_sim()
        
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        cam_x = 2
        cam_y = 1
        cam_z = 1
        cam_pos = gymapi.Vec3(cam_x, cam_y, cam_z)
        cam_target = gymapi.Vec3(-cam_x, -cam_y, cam_z)

        if not Headless:

            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                print("*** Failed to create viewer")
                quit() 

        self.create_ground_plane()

        self.load_object_asset()

        self.load_robot_asset()

        print('rigid body : ', self.num_ur5e_bodies)
        print('DOFs : ', self.num_ur5e_dofs)
        print('body dict : ', self.ur5e_link_dict)

        self.create_env()

        if not Headless:
            self.gym.viewer_camera_look_at(self.viewer, self.env, cam_pos, cam_target)

        self.gym.prepare_sim(self.sim)

        camera_props = gymapi.CameraProperties()
        camera_props.width = im_size
        camera_props.height = im_size
        camera_props.enable_tensors = True

        self.camera = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.camera, self.env, gymapi.Vec3(-0.5, -0.7, 1), gymapi.Vec3(0, 1.2, -0.2))

    def create_sim(self):
        sim_type = gymapi.SIM_PHYSX
        sim_params = gymapi.SimParams()

        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.dt = 1 / 60
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = False

        sim_params.physx.num_position_iterations = 22
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.rest_offset = 0.001
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.use_gpu = True

        self.sim = self.gym.create_sim(0, 0, sim_type, sim_params)

    def create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
        plane_params.distance = 0
        plane_params.static_friction = 1
        plane_params.dynamic_friction = 1
        plane_params.restitution = 0

        self.gym.add_ground(self.sim, plane_params)

    def load_object_asset(self):
        asset_root = ''
        bowl_asset_file = '/home/choiyj/isaacgym/assets/urdf/YCB_assets/urdf/024_bowl/model.urdf'
        bottle_asset_file = '/home/choiyj/isaacgym/assets/urdf/YCB_assets/urdf/021_bleach_cleanser/model.urdf'

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.thickness = 0.001
        asset_options.armature = 0.001

        self.spawn_height = gymapi.Vec3(0, 0, 0.3)

        self.bowl_asset = self.gym.load_asset(self.sim, asset_root, bowl_asset_file, asset_options)
        self.bottle_asset = self.gym.load_asset(self.sim, asset_root, bottle_asset_file, asset_options)

        self.table_dims = gymapi.Vec3(2.0, 2.0, 0.4)

        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        asset_options.thickness = 0.001

        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(0, 0, 0.5 * self.table_dims.z + 0.001)

        self.corner = self.table_pose.p - self.table_dims * 0.5

        self.table_asset = self.gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z, asset_options)

    def load_robot_asset(self):
        ur_asset_root = "/home/choiyj/catkin_ws/src/rl_study/src/ur/ur_description/ur5_rg2_ign/urdf"
        ur_asset_file = "ur5_rg2.urdf"

        ur_asset_options = gymapi.AssetOptions()
        ur_asset_options.fix_base_link = True
        ur_asset_options.flip_visual_attachments = True
        ur_asset_options.collapse_fixed_joints = True
        ur_asset_options.disable_gravity = True
        ur_asset_options.override_inertia = True
        ur_asset_options.override_com = True
        ur_asset_options.use_mesh_materials = True

        print("Loading asset '%s' from '%s'" % (ur_asset_file, ur_asset_root))
        self.ur_asset = self.gym.load_asset(self.sim, ur_asset_root, ur_asset_file, ur_asset_options)
        self.num_ur5e_bodies = self.gym.get_asset_rigid_body_count(self.ur_asset)
        self.num_ur5e_dofs = self.gym.get_asset_dof_count(self.ur_asset)
        self.ur5e_link_dict = self.gym.get_asset_rigid_body_dict(self.ur_asset)
        self.rg2_hand_index = self.ur5e_link_dict['rg2_hand']
        self.damping = 0.05

        self.ur5e_dof_props = self.gym.get_asset_dof_properties(self.ur_asset)
        self.ur5e_lower_limits = self.ur5e_dof_props["lower"]
        self.ur5e_upper_limits = self.ur5e_dof_props["upper"]
        self.ur5e_mids = 0.3 * (self.ur5e_upper_limits + self.ur5e_lower_limits)

        self.ur5e_dof_props['driveMode'][:].fill(gymapi.DOF_MODE_POS)
        self.ur5e_dof_props['stiffness'][:6].fill(400.0)
        self.ur5e_dof_props['stiffness'][6:].fill(800.0)
        self.ur5e_dof_props['damping'][:].fill(40.0)

        self.default_dof_pos = np.zeros(self.num_ur5e_dofs, dtype=np.float32)
        self.default_dof_pos = self.ur5e_mids

        self.default_dof_state = np.zeros(self.num_ur5e_dofs, gymapi.DofState.dtype)
        self.default_dof_state['pos'] = self.default_dof_pos

    def create_env(self):
        env_lower = gymapi.Vec3(-2, -2, 0)
        env_upper = gymapi.Vec3(2, 2, 2)

        self.env = self.gym.create_env(self.sim, env_lower, env_upper, 1)

        self.table_handle = self.gym.create_actor(self.env, self.table_asset, self.table_pose, "table", 0, 0)

        x = self.corner.x + self.table_dims.x * 0.5
        y = self.corner.y + self.table_dims.y * 0.2
        z = self.table_dims.z
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(x, y, z)

        self.ur5e_handle = self.gym.create_actor(self.env, self.ur_asset, pose, "ur5e", 0, 0)

        self.gym.set_actor_dof_properties(self.env, self.ur5e_handle, self.ur5e_dof_props)
        self.gym.set_actor_dof_states(self.env, self.ur5e_handle, self.default_dof_state, gymapi.STATE_ALL)
        self.gym.set_actor_dof_position_targets(self.env, self.ur5e_handle, self.default_dof_pos)

        self.gripper_handle = self.gym.find_actor_rigid_body_handle(self.env, self.ur5e_handle, "rg2_hand")

        self.gripper_pose = self.gym.get_rigid_transform(self.env, self.gripper_handle)

        self.init_pos = [self.gripper_pose.p.x, self.gripper_pose.p.y, self.gripper_pose.p.z]
        self.init_rot = [self.gripper_pose.r.x, self.gripper_pose.r.y, self.gripper_pose.r.z, self.gripper_pose.r.w]

        self.gripper_idx = self.gym.find_actor_rigid_body_index(self.env, self.ur5e_handle, "rg2_hand", gymapi.DOMAIN_SIM)

        bowl_pose = gymapi.Transform()
        bowl_pose.p = gymapi.Vec3(x, y, z) + self.spawn_height + gymapi.Vec3(0.4, 0.7, 0)
        bowl_pose.r = gymapi.Quat(0, 0.5, 0.5, 0)
        self.bowl_handle = self.gym.create_actor(self.env, self.bowl_asset, bowl_pose, "bowl", 0, 0)
        # self.gym.set_rigid_body_color(env, bowl_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION)

        bottle_pose = gymapi.Transform()
        bottle_pose.p = gymapi.Vec3(x, y, z) + self.spawn_height - gymapi.Vec3(0.4, -0.7, 0)
        self.bottle_handle = self.gym.create_actor(self.env, self.bottle_asset, bottle_pose, "bottle", 0, 0)
        # self.gym.set_rigid_body_color(env, bottle_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION)

        self.init_data()

    def camera_visulization(self):
        camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.env, self.camera, gymapi.IMAGE_COLOR)
        torch_rgba_tensor = gymtorch.wrap_tensor(camera_rgba_tensor)
        camera_image = torch_rgba_tensor.cpu().numpy()
        # camera_image = Im.fromarray(camera_image)

        return camera_image

    def init_data(self):
        self.init_pos = torch.Tensor(self.init_pos).view(1, 3)
        self.init_rot = torch.Tensor(self.init_rot).view(1, 4)
        self.pos_des = self.init_pos.clone()
        self.rot_des = self.init_rot.clone()

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "ur5e")
        self.jacobian = gymtorch.wrap_tensor(_jacobian)

        self.j_eef = self.jacobian[:, self.rg2_hand_index - 1, :, :6]

        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "ur5e")
        self.mm = gymtorch.wrap_tensor(_massmatrix)[:, :6, :6] 

        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_pos = self.dof_states[:, 0].view(1, self.num_ur5e_dofs, 1)
        self.dof_vel = self.dof_states[:, 1].view(1, self.num_ur5e_dofs, 1)

        self.pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)

        self.pos_cur = self.init_pos
        self.rot_cur = self.init_rot
        self.vel_cur = self.rb_states[self.gripper_idx, 7:]

        self.grip_closed = 1

    def get_obs(self):
        obs = {}

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        camera_rgba_image = self.camera_visulization()
        obs["image_primary"] = jnp.array(camera_rgba_image)

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        self.pos_cur = self.rb_states[self.gripper_idx, :3]
        self.rot_cur = self.rb_states[self.gripper_idx, 3:7]
        self.vel_cur = self.rb_states[self.gripper_idx, 7:]

        qpos = self.dof_pos.wiew(1,-1)[0,:6]
        qpos_numpy = np.array(qpos)

        obs["proprio"] = jnp.array(qpos_numpy)

        return obs

    def get_task(self):
        return {
            "language_instructrion": ["Put the ranch bottle into the pot."]
        }

    def reset(self):
        self.__init__()

        obs = self.get_obs()
        return obs, obs

    def step(self, action):
        grip = action[0]
        pose = action[1:4]
        orientation = axisangle2quat(action[4:])

        dpose = torch.cat([pose, orientation], -1).unsqueeze(-1)

        def control_ik_j(dpose, j_eef, num_envs, device, damping):
            j_eef_T = torch.transpose(j_eef, 1, 2)
            lmbda = torch.eye(6, device=device) * (damping ** 2)
            u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 6)
            print(u)
            return u

        self.pos_action[:, :6] = self.dof_pos.squeeze(-1)[:, :6] + control_ik_j(dpose, self.j_eef, 1, DEVICE, self.damping)
        # print('pos_action : ', pos_action[:, :6])

        if grip == 1:
            grip_closed = 1
            self.pos_action[:, 7] = (self.ur5e_dof_props["lower"][7].item())
            self.pos_action[:, 8] = (self.ur5e_dof_props["lower"][8].item())
        elif grip == -1:
            grip_closed = 0         
            self.pos_action[:, 7] = (self.ur5e_dof_props["upper"][7].item())*0.5
            self.pos_action[:, 8] = (self.ur5e_dof_props["upper"][8].item())*0.5
        else:     
            if grip_closed == 1:
                self.pos_action[:, 7] = (self.ur5e_dof_props["lower"][7].item())
                self.pos_action[:, 8] = (self.ur5e_dof_props["lower"][8].item())
            else:
                self.pos_action[:, 7] = (self.ur5e_dof_props["upper"][7].item())*0.5
                self.pos_action[:, 8] = (self.ur5e_dof_props["upper"][8].item())*0.5

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))

        self.simulate()
        
        if not Headless:
            self.render()

        obs = self.get_obs()

        return obs, 1, False, False, obs

    def simulate(self):
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)

    def render(self):
        # update viewer
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

    def exit(self):
        # close the simulator in a graceful way
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


# register gym environments
gym.register(
    "ur5e-sim-bottle-v0",
    entry_point=lambda: IsaacUR5eBottle(),
)