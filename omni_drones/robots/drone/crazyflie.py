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

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.nucleus import get_assets_root_path
from tensordict.nn import make_functional
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from omni_drones.actuators.rotor_group import RotorGroup
from omni_drones.controllers import DSLPIDController
from omni_drones.robots import ASSET_PATH
from omni_drones.robots.drone import MultirotorBase


class Crazyflie(MultirotorBase):

    # NOTE: there are unexpedted behaviors when using the asset from Isaac Sim
    usd_path: str = ASSET_PATH + "/usd/cf2x_pybullet.usd"
    # usd_path: str = ASSET_PATH + "/usd/cf2x_isaac.usd"
    param_path: str = ASSET_PATH + "/usd/crazyflie.yaml"
    DEFAULT_CONTROLLER = DSLPIDController
