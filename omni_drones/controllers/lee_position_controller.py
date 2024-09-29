# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
from tensordict import TensorDict

from omni_drones.utils.torch import (
    quat_mul,
    quat_rotate_inverse,
    normalize, 
    quaternion_to_rotation_matrix,
    quaternion_to_euler,
    axis_angle_to_quaternion,
    axis_angle_to_matrix
)
import yaml
import os.path as osp


def compute_parameters(
    rotor_config,
    inertia_matrix,
):
    rotor_angles = torch.as_tensor(rotor_config["rotor_angles"])
    arm_lengths = torch.as_tensor(rotor_config["arm_lengths"])
    force_constants = torch.as_tensor(rotor_config["force_constants"])
    moment_constants = torch.as_tensor(rotor_config["moment_constants"])
    directions = torch.as_tensor(rotor_config["directions"])
    # max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])
    A = torch.stack(
        [
            torch.sin(rotor_angles) * arm_lengths,
            -torch.cos(rotor_angles) * arm_lengths,
            -directions * moment_constants / force_constants,
            torch.ones_like(rotor_angles),
        ]
    ).float()
    mixer = A.T @ (A @ A.T).inverse() @ inertia_matrix

    return mixer

class LeePositionController(nn.Module):
    """
    Computes rotor commands for the given control target using the controller
    described in https://arxiv.org/abs/1003.2005.

    Inputs:
        * root_state: tensor of shape (13,) containing position, rotation (in quaternion),
        linear velocity, and angular velocity.
        * control_target: tensor of shape (7,) contining target position, linear velocity,
        and yaw angle.
    
    Outputs:
        * cmd: tensor of shape (num_rotors,) containing the computed rotor commands.
        * controller_state: empty dict.
    """
    def __init__(
        self, 
        g: float, 
        uav_params,
    ) -> None:
        super().__init__()
        controller_param_path = osp.join(
            osp.dirname(__file__), "cfg", f"lee_controller_{uav_params['name']}.yaml"
        )
        with open(controller_param_path, "r") as f:
            controller_params = yaml.safe_load(f)
        
        self.pos_gain = nn.Parameter(torch.as_tensor(controller_params["position_gain"]).float())
        self.vel_gain = nn.Parameter(torch.as_tensor(controller_params["velocity_gain"]).float())
        self.mass = nn.Parameter(torch.tensor(uav_params["mass"]))
        self.g = nn.Parameter(torch.tensor([0.0, 0.0, g]).abs())

        rotor_config = uav_params["rotor_configuration"]
        inertia = uav_params["inertia"]

        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])

        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)
        
        I = torch.diag_embed(
            torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1])
        )
        self.mixer = nn.Parameter(compute_parameters(rotor_config, I))
        self.attitute_gain = nn.Parameter(
            torch.as_tensor(controller_params["attitude_gain"]).float() @ I[:3, :3].inverse()
        )
        self.ang_rate_gain = nn.Parameter(
            torch.as_tensor(controller_params["angular_rate_gain"]).float() @ I[:3, :3].inverse()
        )
        self.requires_grad_(False)

    def forward(
        self, 
        root_state: torch.Tensor, 
        target_pos: torch.Tensor=None,
        target_vel: torch.Tensor=None,
        target_acc: torch.Tensor=None,
        target_yaw: torch.Tensor=None,
        body_rate: bool=False
    ):
        batch_shape = root_state.shape[:-1]
        device = root_state.device
        if target_pos is None:
            target_pos = root_state[..., :3]
        else:
            target_pos = target_pos.expand(batch_shape+(3,))
        if target_vel is None:
            target_vel = torch.zeros(*batch_shape, 3, device=device)
        else:
            target_vel = target_vel.expand(batch_shape+(3,))
        if target_acc is None:
            target_acc = torch.zeros(*batch_shape, 3, device=device)
        else:
            target_acc = target_acc.expand(batch_shape+(3,))
        if target_yaw is None:
            target_yaw = quaternion_to_euler(root_state[..., 3:7])[..., -1]
        else:
            if not target_yaw.shape[-1] == 1:
                target_yaw = target_yaw.unsqueeze(-1)
            target_yaw = target_yaw.expand(batch_shape+(1,))
        
        cmd = self._compute(
            root_state.reshape(-1, 13),
            target_pos.reshape(-1, 3),
            target_vel.reshape(-1, 3),
            target_acc.reshape(-1, 3),
            target_yaw.reshape(-1, 1),
            body_rate
        )

        return cmd.reshape(*batch_shape, -1)
    
    def _compute(self, root_state, target_pos, target_vel, target_acc, target_yaw, body_rate):
        pos, rot, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        if not body_rate:
            # convert angular velocity from world frame to body frame
            ang_vel = quat_rotate_inverse(rot, ang_vel)
        
        pos_error = pos - target_pos
        vel_error = vel - target_vel

        acc = (
            pos_error * self.pos_gain 
            + vel_error * self.vel_gain 
            - self.g
            - target_acc
        )
        R = quaternion_to_rotation_matrix(rot)
        b1_des = torch.cat([
            torch.cos(target_yaw), 
            torch.sin(target_yaw), 
            torch.zeros_like(target_yaw)
        ],dim=-1)
        b3_des = -normalize(acc)
        b2_des = normalize(torch.cross(b3_des, b1_des, 1))
        R_des = torch.stack([
            b2_des.cross(b3_des, 1), 
            b2_des, 
            b3_des
        ], dim=-1)
        ang_error_matrix = 0.5 * (
            torch.bmm(R_des.transpose(-2, -1), R) 
            - torch.bmm(R.transpose(-2, -1), R_des)
        )
        ang_error = torch.stack([
            ang_error_matrix[:, 2, 1], 
            ang_error_matrix[:, 0, 2], 
            ang_error_matrix[:, 1, 0]
        ],dim=-1)
        ang_rate_err = ang_vel
        ang_acc = (
            - ang_error * self.attitute_gain
            - ang_rate_err * self.ang_rate_gain
            + torch.cross(ang_vel, ang_vel)
        )
        thrust = (-self.mass * (acc * R[:, :, 2]).sum(-1, True))
        ang_acc_thrust = torch.cat([ang_acc, thrust], dim=-1)
        cmd = (self.mixer @ ang_acc_thrust.T).T
        cmd = (cmd / self.max_thrusts) * 2 - 1
        return cmd

    
class AttitudeController(nn.Module):
    r"""
    
    """
    def __init__(self, g, uav_params):
        super().__init__()
        rotor_config = uav_params["rotor_configuration"]
        inertia = uav_params["inertia"]
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])

        self.mass = nn.Parameter(torch.tensor(uav_params["mass"]))
        self.g = nn.Parameter(torch.tensor(g))
        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)
        I = torch.diag_embed(
            torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1])
        )

        self.mixer = nn.Parameter(compute_parameters(rotor_config, I))
        self.gain_attitude = nn.Parameter(
            torch.tensor([3., 3., 0.035]) @ I[:3, :3].inverse()
        )
        self.gain_angular_rate = nn.Parameter(
            torch.tensor([0.52, 0.52, 0.025]) @ I[:3, :3].inverse()
        )


    def forward(
        self, 
        root_state: torch.Tensor, 
        target_thrust: torch.Tensor,
        target_yaw_rate: torch.Tensor=None,
        target_roll: torch.Tensor=None,
        target_pitch: torch.Tensor=None,
    ):
        batch_shape = root_state.shape[:-1]
        device = root_state.device

        if target_yaw_rate is None:
            target_yaw_rate = torch.zeros(*batch_shape, 1, device=device)
        if target_pitch is None:
            target_pitch = torch.zeros(*batch_shape, 1, device=device)
        if target_roll is None:
            target_roll = torch.zeros(*batch_shape, 1, device=device)
        
        cmd = self._compute(
            root_state.reshape(-1, 13),
            target_thrust.reshape(-1, 1),
            target_yaw_rate=target_yaw_rate.reshape(-1, 1),
            target_roll=target_roll.reshape(-1, 1),
            target_pitch=target_pitch.reshape(-1, 1),
        )
        return cmd.reshape(*batch_shape, -1)

    def _compute(
        self, 
        root_state: torch.Tensor,
        target_thrust: torch.Tensor, 
        target_yaw_rate: torch.Tensor, 
        target_roll: torch.Tensor,
        target_pitch: torch.Tensor
    ):
        pos, rot, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        device = pos.device

        R = quaternion_to_rotation_matrix(rot)
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0]).unsqueeze(-1)
        yaw = axis_angle_to_matrix(yaw, torch.tensor([0., 0., 1.], device=device))
        roll = axis_angle_to_matrix(target_roll, torch.tensor([1., 0., 0.], device=device))
        pitch = axis_angle_to_matrix(target_pitch, torch.tensor([0., 1., 0.], device=device))
        R_des = torch.bmm(torch.bmm(yaw,  roll), pitch)
        angle_error_matrix = 0.5 * (
            torch.bmm(R_des.transpose(-2, -1), R) 
            - torch.bmm(R.transpose(-2, -1), R_des)
        )

        angle_error = torch.stack([
            angle_error_matrix[:, 2, 1], 
            angle_error_matrix[:, 0, 2], 
            torch.zeros(yaw.shape[0], device=device)
        ], dim=-1)

        angular_rate_des = torch.zeros_like(ang_vel)
        angular_rate_des[:, 2] = target_yaw_rate.squeeze(1)
        angular_rate_error = ang_vel - torch.bmm(torch.bmm(R_des.transpose(-2, -1), R), angular_rate_des.unsqueeze(2)).squeeze(2)

        angular_acc = (
            - angle_error * self.gain_attitude 
            - angular_rate_error * self.gain_angular_rate 
            + torch.cross(ang_vel, ang_vel)
        )
        angular_acc_thrust = torch.cat([angular_acc, target_thrust], dim=1)
        cmd = (self.mixer @ angular_acc_thrust.T).T
        cmd = (cmd / self.max_thrusts) * 2 - 1
        return cmd


class RateController(nn.Module):
    def __init__(self, g, uav_params) -> None:
        super().__init__()
        rotor_config = uav_params["rotor_configuration"]
        self.rotor_config = rotor_config
        inertia = uav_params["inertia"]
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])
        gain = uav_params['controller_configuration']['gain']

        self.g = nn.Parameter(torch.tensor(g))
        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)
        I = torch.diag_embed(
            torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1])
        )

        self.mixer = nn.Parameter(compute_parameters(rotor_config, I))
        self.gain_angular_rate = nn.Parameter(
            torch.tensor(gain) @ I[:3, :3].inverse()
        )
        self.target_clip = uav_params['target_clip']
        self.max_thrust_ratio = uav_params['max_thrust_ratio']
        self.fixed_yaw = uav_params['fixed_yaw']

    def set_byTunablePara(
        self,
        tunable_parameters: dict = {},
    ):        
        force_constants = torch.as_tensor(self.rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(self.rotor_config["max_rotation_velocities"])
        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)
        inertia_xx = tunable_parameters['inertia_xx']
        inertia_yy = tunable_parameters['inertia_yy']
        inertia_zz = tunable_parameters['inertia_zz']
        gain = tunable_parameters['gain']
        I = torch.diag_embed(
            torch.tensor([inertia_xx, inertia_yy, inertia_zz, 1])
        )

        self.rotor_config['arm_lengths'] = [tunable_parameters['arm_lengths']] * 4
        self.rotor_config['force_constants'] = [tunable_parameters['force_constants']] * 4
        self.rotor_config['max_rotation_velocities'] = [tunable_parameters['max_rotation_velocities']] * 4
        self.rotor_config['moment_constants'] = [tunable_parameters['moment_constants']] * 4
        # self.rotor_config['rotor_angles'] = tunable_parameters['rotor_angles']
        self.rotor_config['time_constant'] = tunable_parameters['time_constant']

        self.mixer = nn.Parameter(compute_parameters(self.rotor_config, I))
        # TODO: jiayu, only crazyflie
        self.gain_angular_rate = nn.Parameter(
            torch.tensor(gain).float() @ I[:3, :3].inverse()
        )
    
    def forward(
        self, 
        root_state: torch.Tensor, 
        target_rate: torch.Tensor,
        target_thrust: torch.Tensor,
    ):
        assert root_state.shape[:-1] == target_rate.shape[:-1]

        batch_shape = root_state.shape[:-1]
        root_state = root_state.reshape(-1, 13)
        target_rate = target_rate.reshape(-1, 3)
        target_thrust = target_thrust.reshape(-1, 1)

        pos, rot, linvel, angvel = root_state.split([3, 4, 3, 3], dim=1)
        body_rate = quat_rotate_inverse(rot, angvel)
        
        # # visual ctbr
        # import numpy as np
        # import os
        # model_path = '/home/jiayu/OmniDrones/scripts'
        # real_path = model_path + '/real.npy'
        # target_path = model_path + '/target.npy'
        # thrust_path = model_path + '/thrust.npy'
        # if not os.path.exists(real_path):
        #     np.save(real_path, body_rate.to('cpu').numpy())
        # else:
        #     existing_data = np.load(real_path)
        #     updated_data = np.concatenate([existing_data, body_rate.to('cpu').numpy()])
        #     np.save(real_path, updated_data)
        # if not os.path.exists(target_path):
        #     np.save(target_path, target_rate.to('cpu').numpy())
        # else:
        #     existing_data = np.load(target_path)
        #     updated_data = np.concatenate([existing_data, target_rate.to('cpu').numpy()])
        #     np.save(target_path, updated_data)
        # if not os.path.exists(thrust_path):
        #     np.save(thrust_path, (target_thrust / self.max_thrusts.sum(-1)).to('cpu').numpy())
        # else:
        #     existing_data = np.load(thrust_path)
        #     updated_data = np.concatenate([existing_data, (target_thrust / self.max_thrusts.sum(-1)).to('cpu').numpy()])
        #     np.save(thrust_path, updated_data)

        rate_error = body_rate - target_rate
        acc_des = (
            - rate_error * self.gain_angular_rate
            + angvel.cross(angvel)
        )
        angacc_thrust = torch.cat([acc_des, target_thrust], dim=1)
        cmd = (self.mixer @ angacc_thrust.T).T
        cmd = (cmd / self.max_thrusts) * 2 - 1
        cmd = cmd.reshape(*batch_shape, -1)
        return cmd

    # only for simopt
    def sim_step(
        self, 
        current_rate: torch.Tensor, 
        target_rate: torch.Tensor,
        target_thrust: torch.Tensor,
    ):

        batch_shape = current_rate.shape[:-1]
        # root_state = root_state.reshape(-1, 13)
        current_rate = current_rate.reshape(-1, 3)
        target_rate = target_rate.reshape(-1, 3)
        target_thrust = target_thrust.reshape(-1, 1)

        # pos, rot, linvel, angvel = root_state.split([3, 4, 3, 3], dim=1)
        # body_rate = quat_rotate_inverse(rot, angvel)

        rate_error = current_rate - target_rate
        acc_des = (
            - rate_error * self.gain_angular_rate
        )
        angacc_thrust = torch.cat([acc_des, target_thrust], dim=1)
        cmd = (self.mixer @ angacc_thrust.T).T
        cmd = (cmd / self.max_thrusts) * 2 - 1
        cmd = cmd.reshape(*batch_shape, -1)
        return cmd

class PIDRateController(nn.Module):
    def __init__(self, dt, g, uav_params) -> None:
        super().__init__()
        rotor_config = uav_params["rotor_configuration"]
        self.rotor_config = rotor_config
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"])

        self.g = nn.Parameter(torch.tensor(g))
        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)

        # PID param
        self.dt = nn.Parameter(torch.tensor(dt))
        self.pid_kp = nn.Parameter(torch.tensor([250.0, 250.0, 120.0]))
        self.pid_ki = nn.Parameter(torch.tensor([500.0, 500.0, 16.7]))
        self.pid_kd = nn.Parameter(torch.tensor([2.5, 2.5, 0.0]))
        self.kff = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        self.count = 0 # if = 0, integ, last_body_rate = 0.0
        self.iLimit = nn.Parameter(torch.tensor([33.3, 33.3, 166.7]))
        self.outLimit = nn.Parameter(torch.tensor((2.0)**15 - 1.0))
        
        self.target_clip = uav_params['target_clip']
        self.max_thrust_ratio = uav_params['max_thrust_ratio']
        self.fixed_yaw = uav_params['fixed_yaw']
        
        # # for action smooth
        # self.use_action_smooth = uav_params['use_action_smooth']
        # self.epsilon = uav_params['epsilon']
        
        self.init_flag = True # init last_body_rate and inte

    def set_byTunablePara(
        self,
        tunable_parameters: dict = {},
    ):        
        # PID param
        self.dt = nn.Parameter(torch.tensor(0.02))
        self.pid_kp = nn.Parameter(torch.tensor(tunable_parameters['pid_kp']))
        self.pid_kd = nn.Parameter(torch.tensor(tunable_parameters['pid_kd'] + [0.0])) # set coeff_yaw = 0.0
        self.pid_ki = nn.Parameter(torch.tensor(tunable_parameters['pid_ki']))
        self.kff = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        self.count = 0 # if = 0, integ, last_body_rate = 0.0
        self.iLimit = nn.Parameter(torch.tensor(tunable_parameters['iLimit']))
        self.outLimit = nn.Parameter(torch.tensor((2.0)**16))
    
    def forward(
        self, 
        root_state: torch.Tensor, 
        target_rate: torch.Tensor,
        target_thrust: torch.Tensor,
        reset_pid: torch.Tensor,
    ):
        assert root_state.shape[:-1] == target_rate.shape[:-1]
        
        # target_rate: degree/s
        # target_thrust: [0, 2**16]
        # body_rate: use degree

        batch_shape = root_state.shape[:-1]
        root_state = root_state.reshape(-1, 13)
        target_rate = target_rate.reshape(-1, 3)
        target_thrust = target_thrust.reshape(-1, 1)
        reset_pid = reset_pid.reshape(-1)
        device = root_state.device
        
        # pid reset
        if self.init_flag:
            self.last_body_rate = torch.zeros(size=(*batch_shape, 3)).to(device).reshape(-1, 3)
            self.integ = torch.zeros(size=(*batch_shape, 3)).to(device).reshape(-1, 3)
            self.init_flag = False
        self.last_body_rate[reset_pid] = torch.zeros(size=(*batch_shape, 3)).to(device).reshape(-1, 3)[reset_pid]
        self.integ[reset_pid] = torch.zeros(size=(*batch_shape, 3)).to(device).reshape(-1, 3).to(device)[reset_pid]

        pos, rot, linvel, angvel = root_state.split([3, 4, 3, 3], dim=1)
        body_rate = quat_rotate_inverse(rot, angvel) * 180.0 / torch.pi

        rate_error = target_rate - body_rate
        
        # P
        outputP = rate_error * self.pid_kp.view(1, -1)
        # D
        deriv = -(body_rate - self.last_body_rate) / self.dt
        deriv[torch.isnan(deriv)] = 0.0
        outputD = deriv * self.pid_kd.view(1, -1)
        # I
        self.integ += rate_error * self.dt
        self.integ = torch.clip(self.integ, -self.iLimit, self.iLimit)
        outputI = self.integ * self.pid_ki.view(1, -1)
        # kff
        outputFF = target_rate * self.kff.view(1, -1)
        
        output = outputP + outputD + outputI + outputFF
        output[torch.isnan(output)] = 0.0
        
        # clip
        output = torch.clip(output, - self.outLimit, self.outLimit)

        # set last error
        self.last_body_rate = body_rate.clone()
        
        # deploy body rate to four rotors
        # output: r, p, y
        r = (output[:, 0] / 2.0).unsqueeze(1)
        p = (output[:, 1] / 2.0).unsqueeze(1)
        # y = - output[:, 2].unsqueeze(1)
        y = output[:, 2].unsqueeze(1)
        
        m1 = target_thrust + r - p + y
        m2 = target_thrust + r + p - y
        m3 = target_thrust - r + p + y
        m4 = target_thrust - r - p - y
        
        ctbr = torch.concat([r, p, y, target_thrust], dim=1).reshape(*batch_shape, -1)

        cmd = torch.concat([m1,m2,m3,m4], dim=1) / 2**16 * 2 - self.max_thrust_ratio
        
        cmd = cmd.reshape(*batch_shape, -1)
        
        return cmd, ctbr

    def debug_step(
        self, 
        real_body_rate: torch.Tensor, 
        target_rate: torch.Tensor,
        target_thrust: torch.Tensor,
    ):
        # assert root_state.shape[:-1] == target_rate.shape[:-1]
        
        # real_body_rate: radian/s
        # target_rate: degree/s
        # target_thrust: [0, 2**16]

        batch_shape = real_body_rate.shape[:-1]
        # root_state = root_state.reshape(-1, 13)
        target_rate = target_rate.reshape(-1, 3)
        target_thrust = target_thrust.reshape(-1, 1)
        device = real_body_rate.device
        if self.count == 0:
            self.last_body_rate = torch.zeros(size=(batch_shape[0], 3)).to(device)
            self.integ = torch.zeros(size=(batch_shape[0], 3)).to(device)
        self.count += 1

        # pos, rot, linvel, angvel = root_state.split([3, 4, 3, 3], dim=1)
        # body_rate = quat_rotate_inverse(rot, angvel)
        
        body_rate = real_body_rate * 180 / torch.pi

        rate_error = target_rate - body_rate
        
        # P
        outputP = rate_error * self.pid_kp.view(1, -1)
        # D
        deriv = -(body_rate - self.last_body_rate) / self.dt
        deriv[torch.isnan(deriv)] = 0.0
        outputD = deriv * self.pid_kd.view(1, -1)
        # I
        self.integ += rate_error * self.dt
        self.integ = torch.clip(self.integ, -self.iLimit, self.iLimit)
        outputI = self.integ * self.pid_ki.view(1, -1)
        # kff
        outputFF = target_rate * self.kff.view(1, -1)
        
        output = outputP + outputD + outputI + outputFF
        output[torch.isnan(output)] = 0.0
        # clip
        output = torch.clip(output, - self.outLimit, self.outLimit)

        # set last error
        self.last_body_rate = body_rate.clone()
        
        # deploy body rate to four rotors
        # output: r, p, y
        r = (output[:, 0] / 2.0).unsqueeze(1)
        p = (output[:, 1] / 2.0).unsqueeze(1)
        y = - output[:, 2].unsqueeze(1) # 固件实现，sim中用的正数
        # y = output[:, 2].unsqueeze(1)
        
        m1 = target_thrust - r + p + y # 固件实现，sim中13对调，24对调
        m2 = target_thrust - r - p - y
        m3 = target_thrust + r - p + y
        m4 = target_thrust + r + p - y

        cmd = torch.concat([m1,m2,m3,m4], dim=1) / 2**16 * 2 - 1
        
        cmd = cmd.reshape(*batch_shape, -1)
        
        return cmd, (r, p, y, target_thrust)