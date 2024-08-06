from __future__ import print_function, division, absolute_import
import os
import math
import numpy as np

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from copy import copy

from math import *
import numpy as np

import torch
import matplotlib.pyplot as plt
from PIL import Image as Im


DEVICE = 'cuda:0'


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


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def control_ik_j(dpose, j_eef, num_envs, device, damping):
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 6)
    print(u)
    return u


def main():
    gym = gymapi.acquire_gym()

    sim_type = gymapi.SIM_PHYSX
    sim_params = gymapi.SimParams()

    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.dt = 1 / 60
    sim_params.substeps = 2
    sim_params.use_gpu_pipeline = True

    sim_params.physx.num_position_iterations = 22
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.001
    sim_params.physx.contact_offset = 0.02
    sim_params.physx.use_gpu = True

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    if sim is None:
        print("*** Failed to create sim")
        quit()

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
    plane_params.distance = 0
    plane_params.static_friction = 1
    plane_params.dynamic_friction = 1
    plane_params.restitution = 0

    gym.add_ground(sim, plane_params)

    asset_file = 'model.urdf'
    bowl_asset_root = '/home/choiyj/isaacgym/assets/urdf/YCB_assets/urdf/024_bowl'
    bottle_asset_root = '/home/choiyj/isaacgym/assets/urdf/YCB_assets/urdf/021_bleach_cleanser'

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.thickness = 0.002
    asset_options.armature = 0.001

    spawn_height = gymapi.Vec3(0, 0, 0.3)

    bowl_asset = gym.load_asset(sim, bowl_asset_root, asset_file, asset_options)
    bottle_asset = gym.load_asset(sim, bottle_asset_root, asset_file, asset_options)

    cam_x = 2
    cam_y = 1
    cam_z = 1
    cam_pos = gymapi.Vec3(cam_x, cam_y, cam_z)
    cam_target = gymapi.Vec3(-cam_x, -cam_y, cam_z)

    # viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    # if viewer is None:
    #     print("*** Failed to create viewer")
    #     quit()

    table_dims = gymapi.Vec3(2.0, 2.0, 0.4)

    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.002

    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0, 0, 0.5 * table_dims.z + 0.001)

    corner = table_pose.p - table_dims * 0.5

    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

    ur_asset_root = "/home/choiyj/octo_ur/ur_description/ur5_rg2_ign/urdf"
    ur_asset_file = "ur5_rg2.urdf"

    ur_asset_options = gymapi.AssetOptions()
    ur_asset_options.fix_base_link = True
    ur_asset_options.flip_visual_attachments = True
    ur_asset_options.collapse_fixed_joints = True
    ur_asset_options.disable_gravity = True
    ur_asset_options.override_inertia = True
    ur_asset_options.override_com = True
    ur_asset_options.use_mesh_materials = True

    if sim_type == gymapi.SIM_FLEX:
        ur_asset_options.max_angular_velocity = 40.

    print("Loading asset '%s' from '%s'" % (ur_asset_file, ur_asset_root))
    ur_asset = gym.load_asset(sim, ur_asset_root, ur_asset_file, ur_asset_options)
    num_ur5e_bodies = gym.get_asset_rigid_body_count(ur_asset)
    num_ur5e_dofs = gym.get_asset_dof_count(ur_asset)
    ur5e_link_dict = gym.get_asset_rigid_body_dict(ur_asset)

    print('rigid body : ', num_ur5e_bodies)
    print('DOFs : ', num_ur5e_dofs)
    print('body dict : ', ur5e_link_dict)

    rg2_hand_index = ur5e_link_dict['rg2_hand']
    damping = 0.05

    ur5e_dof_props = gym.get_asset_dof_properties(ur_asset)
    ur5e_lower_limits = ur5e_dof_props["lower"]
    ur5e_upper_limits = ur5e_dof_props["upper"]
    ur5e_mids = 0.3 * (ur5e_upper_limits + ur5e_lower_limits)

    ur5e_dof_props['driveMode'][:].fill(gymapi.DOF_MODE_POS)
    ur5e_dof_props['stiffness'][:6].fill(400.0)
    ur5e_dof_props['stiffness'][6:].fill(800.0)
    ur5e_dof_props['damping'][:].fill(40.0)

    default_dof_pos = np.zeros(num_ur5e_dofs, dtype=np.float32)
    default_dof_pos = ur5e_mids

    default_dof_state = np.zeros(num_ur5e_dofs, gymapi.DofState.dtype)
    default_dof_state['pos'] = default_dof_pos

    env_lower = gymapi.Vec3(-2, -2, 0)
    env_upper = gymapi.Vec3(2, 2, 2)

    env = None
    ur5e_handle = None
    gripper_handle = None
    table_handle = None
    init_pos = None
    init_rot = None
    gripper_idx = None

    env = gym.create_env(sim, env_lower, env_upper, 1)

    table_handle = gym.create_actor(env, table_asset, table_pose, "table", 0, 0)

    x = corner.x + table_dims.x * 0.5
    y = corner.y + table_dims.y * 0.2
    z = table_dims.z
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(x, y, z)

    ur5e_handle = gym.create_actor(env, ur_asset, pose, "ur5e", 0, 0)

    gym.set_actor_dof_properties(env, ur5e_handle, ur5e_dof_props)
    gym.set_actor_dof_states(env, ur5e_handle, default_dof_state, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, ur5e_handle, default_dof_pos)

    gripper_handle = gym.find_actor_rigid_body_handle(env, ur5e_handle, "rg2_hand")

    gripper_pose = gym.get_rigid_transform(env, gripper_handle)

    init_pos = [gripper_pose.p.x, gripper_pose.p.y, gripper_pose.p.z]
    init_rot = [gripper_pose.r.x, gripper_pose.r.y, gripper_pose.r.z, gripper_pose.r.w]

    gripper_idx = gym.find_actor_rigid_body_index(env, ur5e_handle, "rg2_hand", gymapi.DOMAIN_SIM)

    bowl_pose = gymapi.Transform()
    bowl_pose.p = gymapi.Vec3(x, y, z) + spawn_height + gymapi.Vec3(0.4, 0.7, 0)
    bowl_pose.r = gymapi.Quat(0, 0.5, 0.5, 0)
    bowl_handle = gym.create_actor(env, bowl_asset, bowl_pose, "bowl", 0, 0)
    # gym.set_rigid_body_color(env, bowl_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION)

    bottle_pose = gymapi.Transform()
    bottle_pose.p = gymapi.Vec3(x, y, z) + spawn_height - gymapi.Vec3(0.4, -0.7, 0)
    bottle_handle = gym.create_actor(env, bottle_asset, bottle_pose, "bottle", 0, 0)
    # gym.set_rigid_body_color(env, bottle_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION)

    # gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

    gym.prepare_sim(sim)

    camera_props = gymapi.CameraProperties()
    camera_props.width = 256
    camera_props.height = 256
    camera_props.enable_tensors = True

    camera = gym.create_camera_sensor(env, camera_props)
    gym.set_camera_location(camera, env, gymapi.Vec3(-0.5, -0.7, 1), gymapi.Vec3(0, 1.2, -0.2))

    def camera_visulization(is_depth_image=False):
        camera_rgba_tensor = gym.get_camera_image_gpu_tensor(sim, env, camera, gymapi.IMAGE_COLOR)
        torch_rgba_tensor = gymtorch.wrap_tensor(camera_rgba_tensor)
        camera_image = torch_rgba_tensor.cpu().numpy()
        # camera_image = Im.fromarray(camera_image)

        return camera_image

    init_pos = torch.Tensor(init_pos).view(1, 3)
    init_rot = torch.Tensor(init_rot).view(1, 4)
    pos_des = init_pos.clone()
    rot_des = init_rot.clone()

    _jacobian = gym.acquire_jacobian_tensor(sim, "ur5e")
    jacobian = gymtorch.wrap_tensor(_jacobian)

    j_eef = jacobian[:, rg2_hand_index - 1, :, :6]

    _massmatrix = gym.acquire_mass_matrix_tensor(sim, "ur5e")
    mm = gymtorch.wrap_tensor(_massmatrix)[:, :6, :6] 

    _rb_states = gym.acquire_rigid_body_state_tensor(sim)
    rb_states = gymtorch.wrap_tensor(_rb_states)

    _dof_states = gym.acquire_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(_dof_states)
    dof_pos = dof_states[:, 0].view(1, num_ur5e_dofs, 1)
    dof_vel = dof_states[:, 1].view(1, num_ur5e_dofs, 1)

    pos_action = torch.zeros_like(dof_pos).squeeze(-1)

    pos_cur = init_pos
    rot_cur = init_rot
    vel_cur = rb_states[gripper_idx, 7:]

    grip_closed = 1

    while True:
        print("=====================================")

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.render_all_camera_sensors(sim)
        gym.start_access_image_tensors(sim)

        camera_rgba_debug_fig = plt.figure("CAMERA_RGB_DEBUG")
        camera_rgba_image = camera_visulization(is_depth_image=False)

        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)
        gym.refresh_mass_matrix_tensors(sim)

        pos_cur = rb_states[gripper_idx, :3]
        rot_cur = rb_states[gripper_idx, 3:7]
        vel_cur = rb_states[gripper_idx, 7:]

        plt.imshow(camera_rgba_image)
        plt.pause(1e-9) 

        if True:
            print("Octo Model is not ready.")
            print('joint: ', dof_pos.view(1,-1)[0,:6])
            print('pos  : ', pos_cur)
            print('rot  : ', rot_cur)

            # update the viewer
            gym.step_graphics(sim)
            print("=====================================")
            # gym.draw_viewer(viewer, sim, True)
            print("=====================================")

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            # gym.sync_frame_time(sim)
            print("=====================================")

            continue

        print('pos_des : ', pos_des)
        cur = [rot_cur[:,0],rot_cur[:,1],rot_cur[:,2],rot_cur[:,3]]

        print('pos_cur : ', pos_cur)

        pos_err = 1 * (pos_des - pos_cur)
        rot_err = orientation_error(rot_des, rot_cur)
        dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)

        print('pos_err : ', pos_err)

        pos_action[:, :6] = dof_pos.squeeze(-1)[:, :6] + control_ik_j(dpose, j_eef, 1, DEVICE, damping)
        print('pos_action : ', pos_action[:, :6])

        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)
            
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

   
if __name__ == "__main__":
    main()     