from functorch import vmap

import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core import objects
import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils
import torch
import torch.distributions as D
from omni_drones.views import RigidPrimView
from omni.isaac.core.prims import GeometryPrimView
import numpy as np

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv, List, Optional
from omni_drones.utils.torch import cpos, off_diag, others, make_cells, euler_to_quaternion
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from omni.isaac.debug_draw import _debug_draw

REAL_TRIANGLE = [
    [-1, 0, 0],
    [0, -0.6, 0],
    [0, 0.6, 0]
]

REGULAR_HEXAGON = [
    [0, 0, 0],
    [1.7321, -1, 0],
    [0, -2, 0],
    [-1.7321, -1, 0],
    [-1.7321, 1.0, 0],
    [0.0, 2.0, 0.0],
    [1.7321, 1.0, 0.0],
]

REGULAR_TETRAGON = [
    [1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [-1, 1, 0],
    [0, 0, 0],
]

REGULAR_TRIANGLE = [
    [1, 0, 0],
    [-0.5, 0.866, 0],
    [-0.5, -0.866, 0]
]

SINGLE = [
    #[0.618, -1.9021, 0],
    [0, 0, 0],
    [2, 0, 0]
    #[0.618, 1.9021, 0],
]

REGULAR_PENTAGON = [
    [2., 0, 0],
    [0.618, 1.9021, 0],
    [-1.618, 1.1756, 0],
    [-1.618, -1.1756, 0],
    [0.618, -1.9021, 0],
    [0, 0, 0]
]

REGULAR_SQUARE = [
    [1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [-1, 1, 0],
]

DENSE_SQUARE = [
    [1, 1, 0],
    [1, 0, 0],
    [1, -1, 0],
    [0, 1, 0],
    [0, 0, 0],
    [0, -1, 0],
    [-1, -1, 0],
    [-1, 0, 0],
    [-1, 1, 0],
]

FORMATIONS = {
    "hexagon": REGULAR_HEXAGON,
    "tetragon": REGULAR_TETRAGON,
    "square": REGULAR_SQUARE,
    "dense_square": DENSE_SQUARE,
    "regular_pentagon": REGULAR_PENTAGON,
    "single": SINGLE,
    "triangle": REGULAR_TRIANGLE,
    'real': REAL_TRIANGLE,
}

def sample_from_grid(cells: torch.Tensor, n):
    idx = torch.randperm(cells.shape[0], device=cells.device)[:n]
    return cells[idx]

class FormationUnified(IsaacEnv):
    def __init__(self, cfg, headless):
        self.ball_num = cfg.task.ball_num
        self.static_obs_num = cfg.task.static_obs_num if cfg.task.static_obs_type==2 else cfg.task.static_obs_num*10
        self.formation_type = cfg.task.formation_type
        self.ball_hit_distance = cfg.task.ball_hit_distance
        self.col_hit_distance = cfg.task.col_hit_distance
        self.cfg = cfg
        self.frame_counter = 0
        self.real_flight = cfg.task.get("real_flight", 0)
        self.reward_effort_weight = self.cfg.task.reward_effort_weight
        self.time_encoding = self.cfg.task.time_encoding
        self.safe_distance = self.cfg.task.safe_distance
        self.formation_size = self.cfg.task.formation_size
        self.obs_safe_distance = self.cfg.task.obs_safe_distance
        self.soft_obs_safe_distance = self.cfg.task.soft_obs_safe_distance
        self.ball_reward_coeff = self.cfg.task.ball_reward_coeff
        self.ball_speed = self.cfg.task.ball_speed
        self.random_ball_speed = self.cfg.task.random_ball_speed
        self.velocity_coeff = self.cfg.task.velocity_coeff
        self.formation_coeff = self.cfg.task.formation_coeff
        self.height_coeff = self.cfg.task.height_coeff
        self.throw_threshold = self.cfg.task.throw_threshold
        self.ball_hard_coeff = self.cfg.task.ball_hard_reward_coeff / self.cfg.task.ball_reward_coeff

        self.track_pos = self.cfg.task.track_pos
        
        assert self.formation_type in ['h', 'l']
        super().__init__(cfg, headless)

        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.total_frame = self.cfg.total_frames
        self.drone.initialize() 
        self.randomization = cfg.task.get("randomization", {})
        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])
            
        if self.ball_num > 0:
            # create and initialize additional views
            self.ball = RigidPrimView(
                "/World/envs/env_*/ball_*",
                reset_xform_properties=False,
            )

            self.ball.initialize()
            self.ball.set_masses(torch.ones_like(self.ball.get_masses()))

        if self.static_obs_num > 0:
            self.cube = GeometryPrimView(
                "/World/envs/env_*/cube_*",
                reset_xform_properties=False,
            )
            self.cube.initialize()
            if self.cfg.task.static_obs_pos == 'random':
                self.col_rand_x = torch.tensor([-0.3, 0.3], device=self.device).unsqueeze(0).unsqueeze(-1)
                self.col_rand_y = torch.tensor([2, 2], device=self.device).unsqueeze(0).unsqueeze(-1)
                self.set_cube_pos_random()
            elif self.cfg.task.static_obs_pos == 'grid' or self.cfg.task.static_obs_pos == 'fake':
                self.set_cube_pos_grid()
            elif self.cfg.task.static_obs_pos == 'fixed':
                self.set_cube_pos_fix()
            elif self.cfg.task.static_obs_pos == 'real':
                self.set_cube_pos_real()
            else:
                raise NotImplementedError

        # self.init_poses = self.drone.get_world_poses(clone=True)

        # initial state distribution
        self.cells = (
            make_cells([-2, -2, 0.5], [2, 2, 2], [0.5, 0.5, 0.25])
            .flatten(0, -2)
            .to(self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.1, -.1, -.1], device=self.device) * torch.pi,
            torch.tensor([0.1, 0.1, 0.1], device=self.device) * torch.pi
        )
        self.target_heading = torch.zeros(self.num_envs, self.drone.n, 3, device=self.device)
        self.target_heading[..., 0] = 1.

        self.flag = torch.zeros((self.num_envs, self.ball_num), dtype=bool, device=self.device)
        self.t_throw = torch.zeros((self.num_envs, self.ball_num), device=self.device)
        self.mask_observation = torch.tensor([self.cfg.task.obs_range, -1, -1, -1, -1, -1, -1, -1, -1, -1], device=self.device).float()
        # if self.formation_type == 'h':
        #     self.cost_h = torch.ones(self.num_envs, device=self.device)
        # self.cost_v = torch.ones(self.num_envs, device=self.device)
        # self.t_formed_indicator = torch.zeros(self.num_envs, dtype=bool, device=self.device)
        # self.t_formed = torch.full(size=(self.num_envs,1), fill_value=torch.nan, device=self.device).squeeze(1)
        self.t_launched = torch.full(size=(self.num_envs, self.ball_num), fill_value=torch.nan, device=self.device)
        self.ball_reward_flag = torch.zeros((self.num_envs, self.ball_num), dtype=bool, device=self.device)
        self.ball_alarm = torch.ones((self.num_envs, self.ball_num), dtype=bool, device=self.device)
        # self.height_penalty = torch.zeros(self.num_envs, self.drone.n, device=self.device)
        self.separation_penalty = torch.zeros(self.num_envs, self.drone.n, self.drone.n-1, device=self.device)
        self.t_moved = torch.full(size=(self.num_envs,self.ball_num), fill_value=torch.nan, device=self.device)
        self.t_difference = torch.full(size=(self.num_envs,self.ball_num), fill_value=torch.nan, device=self.device)
        self.t_hit = torch.full(size=(self.num_envs, self.ball_num), fill_value=torch.nan, device=self.device)
        self.bad_terminate = torch.zeros(self.num_envs, self.drone.n, device=self.device, dtype=torch.bool)
        
        if self.track_pos:
            self.track_pos_goal = torch.zeros(self.num_envs, self.drone.n, 3, device=self.device)
        # self.morl_smooth_coeff = 1 #self.cfg.task.morl_smooth_coeff
        # self.morl_formation_coeff = 0 #self.cfg.task.morl_formation_coeff
        # self.morl_obstacle_coeff = 0 #self.cfg.task.morl_obstacle_coeff
        # self.morl_forward_coeff = 0 #self.cfg.task.morl_obstacle_coeff

        total_weight = self.cfg.task.morl_smooth_coeff+self.cfg.task.morl_formation_coeff+self.cfg.task.morl_obstacle_coeff+self.cfg.task.morl_forward_coeff

        self.morl_smooth_coeff = self.cfg.task.morl_smooth_coeff / total_weight
        self.morl_formation_coeff = self.cfg.task.morl_formation_coeff / total_weight
        self.morl_obstacle_coeff = self.cfg.task.morl_obstacle_coeff / total_weight
        self.morl_forward_coeff = self.cfg.task.morl_forward_coeff / total_weight

        # self.morl_smooth_coeff = self.cfg.task.ratio1 * self.cfg.task.ratio2
        # self.morl_formation_coeff = self.cfg.task.ratio1 * (1-self.cfg.task.ratio2)
        # self.morl_obstacle_coeff = (1-self.cfg.task.ratio1) * self.cfg.task.ratio3
        # self.morl_forward_coeff = (1-self.cfg.task.ratio1) * (1-self.cfg.task.ratio3)

        self.alpha = 0.
        self.gamma = 0.995
        self.last_cost_h = torch.zeros(self.num_envs, 1, device=self.device)
        # self.last_cost_v = torch.zeros(self.num_envs, 1, device=self.device)

        self.drone_grid = torch.tensor([
            [-0.1, -0.1],
            [-0.1, 0.],
            [-0.1, 0.1],
            [0., -0.1],
            [0., 0.],
            [0., 0.1],
            [0.1, -0.1],
            [0.1, 0.],
            [0.1, 0.1]],device=self.device)

    def _design_scene(self) -> Optional[List[str]]:
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        scene_utils.design_scene()

        formation = self.cfg.task.formation
        if isinstance(formation, str):
            self.formation = torch.as_tensor(
                FORMATIONS[formation], device=self.device
            ).float()
        elif isinstance(formation, list):
            self.formation = torch.as_tensor(
                self.cfg.task.formation, device=self.device
            )
        else:
            raise ValueError(f"Invalid target formation {formation}")

        if self.real_flight==1:
            self.target_pos_single = torch.tensor([1.5, 1.5, 1.], device=self.device)
            self.target_vel = [-0.5, -0.5, 0.]
            self.velocity_coeff *= 2.5
        elif self.real_flight==2:
            self.target_pos_single = torch.tensor([0., 0., 1.], device=self.device)
            self.target_vel = [0., 1., 0.]
            self.velocity_coeff *= 2
        else:
            self.target_pos_single = torch.tensor([0., 0., 1.5], device=self.device)
            self.target_vel = [0., 2., 0.]

        self.target_vel = torch.tensor(self.target_vel)
        self.target_vel = self.target_vel.to(device=self.device)

        self.final_pos_single = self.target_vel * self.max_episode_length * 0.9 * self.cfg.task.sim.dt + self.target_pos_single
        self.final_pos = self.final_pos_single + self.formation

        self.init_pos_single = self.target_pos_single.clone()
        self.target_height = self.target_pos_single[2].item()

        self.formation = self.formation*self.cfg.task.formation_size
        self.formation_L = laplacian(self.formation, True)
        self.formation_L_unnormalized = laplacian(self.formation, False)
        relative_pos = cpos(self.formation, self.formation)
        drone_pdist = off_diag(torch.norm(relative_pos, dim=-1, keepdim=True))
        self.standard_formation_size = drone_pdist.max(dim=-2).values.max(dim=-2).values

        for ball_id in range(self.ball_num):
            ball = objects.DynamicSphere(
                prim_path=f"/World/envs/env_0/ball_{ball_id}",  
                position=torch.tensor([ball_id, 1., 0.]),
                radius = 0.15,
                color=torch.tensor([1., 0., 0.]),
            )

        self.margin = self.cfg.task.static_margin
        self.border = self.cfg.task.grid_border
        self.grid_size = self.cfg.task.grid_size
        if self.cfg.task.static_obs_pos == 'fixed':
            self.fixed_cube_pos = torch.tensor([
                    [0.5, 5,    1.],
                    [0.25, 5.5, 1.],
                    [0., 6, 1.],
                    [-1, 4.5, 1.],
                ], device=self.device)   
            # self.fixed_cube_pos[..., 1] += 1
        elif self.cfg.task.static_obs_pos == 'grid':
            pass
        

        for obs_id in range(self.static_obs_num):
            cube = objects.VisualCylinder(
                prim_path=f"/World/envs/env_0/cube_{obs_id}",  
                position=torch.tensor([obs_id, 0., 1.5]),
                radius=self.col_hit_distance,
                height=self.cfg.task.static_height, 
                color=torch.tensor([0., 0.8, 0.8]),
            )

        self.drone.spawn(translations=self.formation+self.init_pos_single)
        self.target_pos = self.target_pos_single.expand(self.num_envs, self.drone.n, 3) + self.formation
        self.drone_id = torch.Tensor(np.arange(self.drone.n)).to(self.device)
        return ["/World/defaultGroundPlane"]

    def set_track_pos_goal(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.track_pos_goal[env_ids] = self.target_pos_single + self.formation
        
    def set_cube_pos_fix(self, env_ids=None):
        self.cubes_pos = self.fixed_cube_pos.unsqueeze(0).expand(self.num_envs, self.static_obs_num, 3).to(self.device)
        # visualize col position
        if self.cfg.task.eval:
            pos = (
                    self.cubes_pos + self.envs_positions.unsqueeze(1).to(self.device)
            )
            env_ids = torch.arange(self.num_envs, device=self.device)
            env_ids = env_ids.unsqueeze(1).expand(-1, self.static_obs_num).reshape(-1)
            cube_ids = torch.arange(self.static_obs_num, device=self.device).expand(self.num_envs, -1).reshape(-1)
            env_indices = env_ids * self.static_obs_num + cube_ids
            self.cube.set_world_poses(pos.reshape(-1, 3), indices=env_indices)

    def set_cube_pos_random(self, env_ids=None):
        if env_ids is None:
            cube_x_pos = torch.rand((self.num_envs, self.static_obs_num, 1), device=self.device) * 0.5 - 0.25 + self.col_rand_x
            cube_y_pos = torch.rand((self.num_envs, self.static_obs_num, 1), device=self.device) * 0.5 - 0.25 + self.col_rand_y
            cube_z_pos = torch.ones((self.num_envs, self.static_obs_num, 1), device=self.device) * self.target_height
            self.cubes_pos = torch.cat([cube_x_pos, cube_y_pos, cube_z_pos], dim=-1) # [envs, cubes, 3]
            if self.cfg.task.eval:
                pos = (
                    self.cubes_pos + self.envs_positions.unsqueeze(1)
                )
                env_ids = torch.arange(self.num_envs, device=self.device)
                env_ids = env_ids.unsqueeze(1).expand(-1, self.static_obs_num).reshape(-1)
                cube_ids = torch.arange(self.static_obs_num, device=self.device).expand(self.num_envs, -1).reshape(-1)
                env_indices = env_ids * self.static_obs_num + cube_ids
                self.cube.set_world_poses(pos.reshape(-1, 3), indices=env_indices)
        else:
            reset_env_num = env_ids.shape[0]
            cube_x_pos = torch.rand((reset_env_num, self.static_obs_num, 1), device=self.device) * 0.5 - 0.25 + self.col_rand_x
            cube_y_pos = torch.rand((reset_env_num, self.static_obs_num, 1), device=self.device) * 0.5 - 0.25 + self.col_rand_y
            cube_z_pos = torch.ones((reset_env_num, self.static_obs_num, 1), device=self.device) * self.target_height
            cubes_pos = torch.cat([cube_x_pos, cube_y_pos, cube_z_pos], dim=-1) # [envs, cubes, 3]

            self.cubes_pos[env_ids] = cubes_pos
            if self.cfg.task.eval:
                pos = (
                    cubes_pos + self.envs_positions[env_ids].unsqueeze(1)
                )
                env_ids = env_ids.unsqueeze(1).expand(-1, self.static_obs_num).reshape(-1)
                cube_ids = torch.arange(self.static_obs_num, device=self.device).expand(reset_env_num, -1).reshape(-1)
                env_indices = env_ids * self.static_obs_num + cube_ids
                self.cube.set_world_poses(pos.reshape(-1, 3), indices=env_indices)

    def set_cube_pos_grid(self, env_ids=None):
        is_init = False
        if env_ids is None:
            is_init = True
            env_ids = torch.arange(self.num_envs, device=self.device)
        reset_env_num = env_ids.shape[0]
        # only consider static obstacle of type 2
        # self.total_grid_size = (a, b)
        if self.cfg.task.grid_size < 0: # random grid_size
            if not self.cfg.task.eval:
                if self.frame_counter < 1/2 * self.total_frame:
                    self.grid_size = self.cfg.task.grid_size_max
                else:
                    self.grid_size = self.cfg.task.grid_size_min + (1 - 2 * (self.frame_counter / self.total_frame - 1/2)) * (self.cfg.task.grid_size_max - self.cfg.task.grid_size_min)
            else:
                self.grid_size = self.cfg.task.grid_size_min
        
        target_speed = self.target_vel[:2].norm()
        if self.real_flight:
            length = 18
        else:
            length = (target_speed * self.dt * self.max_episode_length).item()- 2 * self.margin
        self.total_grid_size = (int((2*self.border)//self.grid_size), int(length//self.grid_size))
        
        a, b = self.total_grid_size
        assert(a*b >= self.static_obs_num)
        # idx = torch.stack([torch.randperm(a*b)[:self.static_obs_num] for _ in range(reset_env_num)]) # time: 3e-3 s
        random_indices = torch.randint(a*b, size=(reset_env_num, a*b), device=self.device) 
        sorted_indices = torch.argsort(random_indices, dim=-1)
        idx = sorted_indices[:, :self.static_obs_num] # [env_num, cubes] # time: 2e-4s
        # raise NotImplementedError
        grid_a, grid_b = idx // a, idx % a
        if self.cfg.task.zigzag:
            x0 = (grid_a) * self.grid_size + self.margin
            y0 = (grid_b) * self.grid_size - self.border
            mask = (grid_a % 2) == 0
            y0[mask] += self.grid_size/2
        else:
            x0 = (grid_a + 0.5) * self.grid_size + self.margin
            y0 = (grid_b + 0.5) * self.grid_size - self.border
        target_speed = (self.target_vel[:2]).norm()
        sin_theta = self.target_vel[1]/target_speed
        cos_theta = self.target_vel[0]/target_speed
        x = x0*cos_theta - y0*sin_theta # [env_num, cubes]
        y = x0*sin_theta + y0*cos_theta
        z = torch.zeros(reset_env_num, self.static_obs_num, device=self.device)
        cubes_pos = torch.stack([x, y, z], dim=-1)
        if is_init:
            self.cubes_pos = cubes_pos
            pos = (self.cubes_pos + self.envs_positions.unsqueeze(1))    
            env_ids = env_ids.unsqueeze(1).expand(-1, self.static_obs_num).reshape(-1)
        else:
            self.cubes_pos[env_ids] = cubes_pos
            pos = (cubes_pos + self.envs_positions[env_ids].unsqueeze(1))
            env_ids = env_ids.unsqueeze(1).expand(-1, self.static_obs_num).reshape(-1)
        # print(self.cubes_pos)
        if self.cfg.task.eval: # visualize columns
            cube_ids = torch.arange(self.static_obs_num, device=self.device).expand(reset_env_num, -1).reshape(-1)
            env_indices = env_ids * self.static_obs_num + cube_ids
            self.cube.set_world_poses(pos.reshape(-1, 3), indices=env_indices)     
    
    def set_cube_pos_real(self, env_ids=None):
        # x ~ [-6, 0]
        # y = x + offset, where offset ~[-3, 3]
        if env_ids is None:
            cube_x_pos = torch.rand((self.num_envs, self.static_obs_num, 1), device=self.device) * 6 - 6
            cube_y_pos = cube_x_pos + torch.rand((self.num_envs, self.static_obs_num, 1), device=self.device) * 6 - 3
            cube_z_pos = torch.ones((self.num_envs, self.static_obs_num, 1), device=self.device) * self.target_height
            self.cubes_pos = torch.cat([cube_x_pos, cube_y_pos, cube_z_pos], dim=-1) # [envs, cubes, 3]
            if self.cfg.task.eval:
                pos = (
                    self.cubes_pos + self.envs_positions.unsqueeze(1)
                )
                env_ids = torch.arange(self.num_envs, device=self.device)
                env_ids = env_ids.unsqueeze(1).expand(-1, self.static_obs_num).reshape(-1)
                cube_ids = torch.arange(self.static_obs_num, device=self.device).expand(self.num_envs, -1).reshape(-1)
                env_indices = env_ids * self.static_obs_num + cube_ids
                self.cube.set_world_poses(pos.reshape(-1, 3), indices=env_indices)
        else:
            reset_env_num = env_ids.shape[0]
            cube_x_pos = torch.rand((reset_env_num, self.static_obs_num, 1), device=self.device) * 6 - 6
            cube_y_pos = cube_x_pos + torch.rand((reset_env_num, self.static_obs_num, 1), device=self.device) * 6 - 3
            cube_z_pos = torch.ones((reset_env_num, self.static_obs_num, 1), device=self.device) * self.target_height
            cubes_pos = torch.cat([cube_x_pos, cube_y_pos, cube_z_pos], dim=-1) # [envs, cubes, 3]

            self.cubes_pos[env_ids] = cubes_pos
            if self.cfg.task.eval:
                pos = (
                    cubes_pos + self.envs_positions[env_ids].unsqueeze(1)
                )
                env_ids = env_ids.unsqueeze(1).expand(-1, self.static_obs_num).reshape(-1)
                cube_ids = torch.arange(self.static_obs_num, device=self.device).expand(reset_env_num, -1).reshape(-1)
                env_indices = env_ids * self.static_obs_num + cube_ids
                self.cube.set_world_poses(pos.reshape(-1, 3), indices=env_indices)

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[0]
        if self.real_flight:
            # no motor observation, compared to drone_state
            obs_self_dim = 3 + 3 + 4 + 3 + 3 + 3 + 3 # pos, vel, quat, omega, heading, up, rel_vel
        else:
            obs_self_dim = drone_state_dim + 3
        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            obs_self_dim += self.time_encoding_dim
        if self.cfg.algo.share_actor:
            self.id_dim = 3
            obs_self_dim += self.id_dim

        state_dim = drone_state_dim

        if self.cfg.task.use_top_k_obs:
            self.obstacle_num = min(self.cfg.task.top_k, self.ball_num + self.static_obs_num)
        else:
            self.obstacle_num = self.ball_num + self.static_obs_num

        if self.cfg.task.use_world_state:
            obs_others_dim = 3+1+10
        else:
            obs_others_dim = 3+1+3
    
        state_spec = CompositeSpec({
                        "drones": UnboundedContinuousTensorSpec((self.drone.n, state_dim)),
                        "balls": UnboundedContinuousTensorSpec((self.ball_num, 6)),
                        "cols": UnboundedContinuousTensorSpec((self.static_obs_num, 3)),
                    })
            
        self.ball_obs_dim = self.ball_num if self.ball_num>0 else 1
        self.static_obs_dim = self.static_obs_num if self.static_obs_num>0 else 1

        if self.cfg.task.swarm_rl:
            agent_obs_spec = CompositeSpec({
                "obs_self": UnboundedContinuousTensorSpec((1, obs_self_dim)), # 23
                "obs_others": UnboundedContinuousTensorSpec((self.drone.n-1, obs_others_dim)), # 5 * 14 =70
                "attn_obs_ball": UnboundedContinuousTensorSpec((self.ball_obs_dim, 10)), # 7
                "attn_obs_static": UnboundedContinuousTensorSpec((1, 9)), 
                "attn_ball_mask": UnboundedContinuousTensorSpec((self.ball_obs_dim)), 
                "attn_static_mask": UnboundedContinuousTensorSpec((1))
            }).expand(self.drone.n)
        elif not self.cfg.task.use_separate_obs:
            agent_obs_spec = CompositeSpec({
                    "obs_self": UnboundedContinuousTensorSpec((1, obs_self_dim)), # 23
                    "obs_others": UnboundedContinuousTensorSpec((self.drone.n-1, obs_others_dim)), # 5 * 14 =70
                    "attn_obs_obstacles": UnboundedContinuousTensorSpec((self.obstacle_num, 10)), # 7
                }).expand(self.drone.n)
        else:
            
            if self.cfg.task.use_attn_mask:
                agent_obs_spec = CompositeSpec({
                    "obs_self": UnboundedContinuousTensorSpec((1, obs_self_dim)), # 23
                    "obs_others": UnboundedContinuousTensorSpec((self.drone.n-1, obs_others_dim)), # 5 * 14 =70
                    "attn_obs_ball": UnboundedContinuousTensorSpec((self.ball_obs_dim, 10)), # 7
                    "attn_obs_static": UnboundedContinuousTensorSpec((self.static_obs_dim, 10)), 
                    "attn_ball_mask": UnboundedContinuousTensorSpec((self.ball_obs_dim)), 
                    "attn_static_mask": UnboundedContinuousTensorSpec((self.static_obs_dim))
                }).expand(self.drone.n)
            else:
                agent_obs_spec = CompositeSpec({
                    "obs_self": UnboundedContinuousTensorSpec((1, obs_self_dim)), # 23
                    "obs_others": UnboundedContinuousTensorSpec((self.drone.n-1, obs_others_dim)), # 5 * 14 =70
                    "attn_obs_ball": UnboundedContinuousTensorSpec((self.ball_obs_dim, 10)), # 7
                    "attn_obs_static": UnboundedContinuousTensorSpec((self.static_obs_dim, 10)),
                }).expand(self.drone.n)

        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": agent_obs_spec, 
                "state": state_spec
            }
        }).expand(self.num_envs) # .to(self.device)

        self.action_spec = CompositeSpec({
            "agents": {
                "action": torch.stack([self.drone.action_spec]*self.drone.n, dim=0),
            }
        }).expand(self.num_envs).to(self.device)
        reward_dim = 0
        if self.cfg.task.reward_type == 1:
            reward_dim = 1
        elif self.cfg.task.reward_type == 2:
            reward_dim = 3
        self.reward_spec = CompositeSpec({
            "agents": {
                "reward": UnboundedContinuousTensorSpec((self.drone.n, reward_dim))
            }
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", self.drone.n,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "state")
        )

        stats_spec = CompositeSpec({
            "cost_l": UnboundedContinuousTensorSpec(1), 
            "cost_l_unnormalized": UnboundedContinuousTensorSpec(1), 
            "reward_formation": UnboundedContinuousTensorSpec(1),
            "reward_size": UnboundedContinuousTensorSpec(1),
            "separation_reward": UnboundedContinuousTensorSpec(1),
            "morl_formation_reward": UnboundedContinuousTensorSpec(1),
            
            "height_reward": UnboundedContinuousTensorSpec(self.drone.n),
            "pos_reward": UnboundedContinuousTensorSpec(self.drone.n),
            "vel_reward": UnboundedContinuousTensorSpec(self.drone.n),
            "reward_heading": UnboundedContinuousTensorSpec(self.drone.n),
            "morl_forward_reward": UnboundedContinuousTensorSpec(self.drone.n),
            
            "ball_reward": UnboundedContinuousTensorSpec(self.drone.n),
            "cube_reward": UnboundedContinuousTensorSpec(self.drone.n),
            "hit_reward": UnboundedContinuousTensorSpec(self.drone.n),
            "morl_obstacle_reward": UnboundedContinuousTensorSpec(self.drone.n),
            
            "reward_effort": UnboundedContinuousTensorSpec(self.drone.n),
            "reward_action_smoothness": UnboundedContinuousTensorSpec(self.drone.n),
            "throttle_difference_mean": UnboundedContinuousTensorSpec(1),
            "reward_spin": UnboundedContinuousTensorSpec(self.drone.n),
            "reward_throttle_smoothness": UnboundedContinuousTensorSpec(self.drone.n),
            "morl_smooth_reward": UnboundedContinuousTensorSpec(self.drone.n),
            "avg_acc": UnboundedContinuousTensorSpec(1),
            
            "reward": UnboundedContinuousTensorSpec(self.drone.n),
            "return": UnboundedContinuousTensorSpec(1),
            
            "t_launched": UnboundedContinuousTensorSpec(1),
            "t_moved": UnboundedContinuousTensorSpec(1),
            "t_difference": UnboundedContinuousTensorSpec(1),
            "t_hit": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1), 
            # "terminated": UnboundedContinuousTensorSpec(1),
            "forward_success": UnboundedContinuousTensorSpec(1), 
            "crash": UnboundedContinuousTensorSpec(1),
            "not straight": UnboundedContinuousTensorSpec(1), 
            "hit": UnboundedContinuousTensorSpec(1),
            "hit_b": UnboundedContinuousTensorSpec(1),
            "hit_c": UnboundedContinuousTensorSpec(1),
            "too close": UnboundedContinuousTensorSpec(1),
            "done": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "formation_success": UnboundedContinuousTensorSpec(1),
            "action_success": UnboundedContinuousTensorSpec(1),
            "success": UnboundedContinuousTensorSpec(1),
            
            "size": UnboundedContinuousTensorSpec(1),
            "indi_b_dist": UnboundedContinuousTensorSpec(1),
            "curr_formation_dist": UnboundedContinuousTensorSpec(1),

            "center_x": UnboundedContinuousTensorSpec(1),
            "center_y": UnboundedContinuousTensorSpec(1),
            "center_z": UnboundedContinuousTensorSpec(1),

            "morl_obstacle": UnboundedContinuousTensorSpec(self.drone.n),
            "morl_forward": UnboundedContinuousTensorSpec(self.drone.n),
            "morl_smooth": UnboundedContinuousTensorSpec(self.drone.n),
            "morl_formation": UnboundedContinuousTensorSpec(self.drone.n),
        }).expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13)),
            "network_output":  UnboundedContinuousTensorSpec((self.drone.n, 4)),
            "prev_network_output":  UnboundedContinuousTensorSpec((self.drone.n, 4)),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()


    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        self.ball_reward_flag[env_ids] = 0.
        self.bad_terminate[env_ids] = False

        if self.real_flight == 0 or self.cfg.task.eval:
            pos = (
                (self.formation+self.init_pos_single).expand(len(env_ids), *self.formation.shape) # (k, 3) -> (len(env_ids), k, 3)
                + self.envs_positions[env_ids].unsqueeze(1)
            )
            rpy = torch.zeros((*env_ids.shape, self.drone.n, 3)).to(pos.device)
            rot = euler_to_quaternion(rpy)
        else:
            standard_pos = (self.formation+self.init_pos_single).expand(len(env_ids), *self.formation.shape).clone()
            rand_offset = torch.rand_like(standard_pos) * 0.1 - 0.05
            standard_pos += rand_offset
            pos = (standard_pos + self.envs_positions[env_ids].unsqueeze(1))

            rand_rpy = self.init_rpy_dist.sample((*env_ids.shape, self.drone.n))
            rand_rot = euler_to_quaternion(rand_rpy)
            rot = rand_rot

        vel = torch.zeros(len(env_ids), self.drone.n, 6, device=self.device)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(vel, env_ids)
        self.last_cost_h[env_ids] = vmap(cost_formation_hausdorff)(
            pos, desired_p=self.formation
        )
        
        if self.track_pos:
            self.set_track_pos_goal(env_ids)

        if self.ball_num > 0:
            ball_ids = torch.arange(self.ball_num, device=self.device).expand(len(env_ids), -1)
            pos = (
                torch.tensor([[i, 0, 0.15] for i in range(self.ball_num)], device=self.device).expand(len(env_ids), self.ball_num, 3)
                + self.envs_positions[env_ids].unsqueeze(1)
            )
            vel = torch.zeros(len(env_ids)*self.ball_num, 6, device=self.device)
            env_ids = env_ids.unsqueeze(1).expand(-1, self.ball_num).reshape(-1)
            env_indices = env_ids * self.ball_num + ball_ids.reshape(-1)
                
            self.ball.set_world_poses(pos.reshape(-1, 3), env_indices=env_indices)
            self.ball.set_velocities(vel, env_indices)

        if self.static_obs_num > 0:
            if self.cfg.task.static_obs_pos == 'random':
                self.set_cube_pos_random(env_ids)
            elif self.cfg.task.static_obs_pos == 'grid' or self.cfg.task.static_obs_pos == 'fake':
                self.set_cube_pos_grid(env_ids)
            elif self.cfg.task.static_obs_pos == 'fixed':
                self.set_cube_pos_fix(env_ids)
            elif self.cfg.task.static_obs_pos == 'real':
                self.set_cube_pos_real(env_ids)
            else:
                raise NotImplementedError()

        self.flag[env_ids] = False

        self.stats[env_ids] = 0.
        # self.t_formed[env_ids]=torch.nan
        self.t_launched[env_ids]=torch.nan
        self.t_moved[env_ids]=torch.nan
        self.t_difference[env_ids]=torch.nan
        self.t_hit[env_ids] =torch.nan
        self.ball_alarm[env_ids] = 1
        # self.height_penalty[:] = 0.
        self.separation_penalty[env_ids] = 0.
        if not self.cfg.task.throw_together:
            self.t_throw[env_ids] = torch.rand(len(env_ids), self.ball_num, device=self.device) * self.cfg.task.throw_time_range + self.throw_threshold
        else:
            self.t_throw[env_ids] = self.throw_threshold + torch.rand(()) * self.cfg.task.throw_time_range
        if self.cfg.task.random_ball_num:
            # 20%, 30%, 20%, 20%, 10%
            # 60%, 30%, 10%, 0%
            # 50%, 30%, 10%, 10%
            p = torch.rand(size=())
            if p <= 0.5:
                ball_num = 1
            elif p<=0.8:
                ball_num = 2
            elif p<=0.9:
                ball_num = 3
            elif p<=1.0:
                ball_num = 4
            else:
                ball_num = 5
            
            self.t_throw[env_ids, ball_num:] = self.max_episode_length
        
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.last_network = tensordict[('info', 'prev_network_output')]
        self.current_network = tensordict[('info', 'network_output')]
        self.effort = self.drone.apply_action(actions)
        self.throw_multi_ball()

    def throw_multi_ball(self):
        flag = (self.progress_buf.unsqueeze(-1).expand(self.num_envs, self.ball_num) >= self.t_throw)
        should_throw = flag & (~self.flag) # [envs_num, ball_num]
        if self.random_ball_speed:
            ball_speed = torch.rand(size=()) * (self.cfg.task.max_ball_speed - self.cfg.task.min_ball_speed) + self.cfg.task.min_ball_speed
        else:
            ball_speed = self.ball_speed
        if should_throw.any():
            throw_indices = should_throw.nonzero(as_tuple=True)
            throw_ball_num = throw_indices[0].shape[0]
            self.t_launched[throw_indices] = self.progress_buf.unsqueeze(-1).expand(-1, self.ball_num)[throw_indices]
            self.ball_reward_flag[throw_indices] = 1
            # self.mask[should_throw, -1] = False #0.
            # Compute centre of 4 drones for all the environment\
            # The first index represent for the environment
            # 2nd for the Drone ID
            # 3rd for the position state

            # Approximate the maximum distance between drones after forming square
            if self.cfg.task.throw_center:
                centre_D = self.drone.pos[throw_indices[0]][..., :2].mean(1)
                centre_z = self.drone.pos[throw_indices[0]][..., 2].mean(1)
            else:
                random_idx = torch.randint(low=0, high=self.drone.n, size=(len(throw_indices[0]), ))
                centre = self.drone.pos[throw_indices[0], random_idx]
                centre_D = centre[..., :2]
                centre_z = centre[..., 2]
                throw_center_envs = torch.rand(throw_ball_num) < self.cfg.task.throw_center_ratio
                centre_D[throw_center_envs] = self.drone.pos[throw_indices[0]][throw_center_envs][..., :2].mean(1) # [env, 2]
                centre_z[throw_center_envs] = self.drone.pos[throw_indices[0]][throw_center_envs][..., 2].mean(1)

            # target = torch.rand(centre_D.shape, device=self.device)*2
            target_ball_pos = torch.zeros(throw_ball_num,3, device=self.device)
            ball_pos = torch.zeros(throw_ball_num,3, device=self.device)

            throw_ball_pattern = self.cfg.task.throw_ball_pattern
            if throw_ball_pattern < 0:
                if not self.cfg.task.eval:
                    if self.frame_counter < 1/2 * self.total_frame:
                        thres = 1.0
                    else:
                        thres = 1.0 - (self.frame_counter / self.total_frame - 1/2)
                else:
                    thres = 0.5
                p = torch.rand(size=())
                if p < thres:
                    throw_ball_pattern = 2
                else:
                    throw_ball_pattern = 1
            # print("throw_pattern =", throw_ball_pattern)
            # firstly, calculate vel_z
            #============
            if self.cfg.task.throw_ball_pattern == 0: # throw_z might <= 0
                # 注意 ball_vel 和 ball_target_vel 的差别在 vz 上
                # given t_hit, randomize ball init position & final position
                t_hit = torch.rand(throw_ball_num, device=self.device) * 1.5 + 0.5
            
                ball_target_vel = torch.ones(throw_ball_num, 3, device=self.device)
                ball_target_vel[:, 2] = - torch.rand(throw_ball_num, device=self.device) - 1. #[-2, -1]
                ball_target_vel[:, :2] = 2*(torch.rand(throw_ball_num, 2, device=self.device)-0.5) #[-1, 1]
                ball_target_vel = ball_target_vel/torch.norm(ball_target_vel, p=2, dim=1, keepdim=True) * ball_speed
                
                ball_vel = torch.ones(throw_ball_num, 6, device=self.device)
                ball_vel[:, :3] = ball_target_vel.clone()
                ball_vel[:, 2] = ball_target_vel[:, 2] + 9.81*t_hit
                
            elif self.cfg.task.throw_ball_pattern == 1:
                ball_vxy = 2 * (torch.rand(throw_ball_num, 2, device=self.device) - 0.5)
                ball_vxy = ball_vxy/torch.norm(ball_vxy, p=2, dim=1, keepdim=True) * ball_speed
                ball_vel = torch.zeros(throw_ball_num, 6, device=self.device)
                ball_vel[:, :2] = ball_vxy
                t_hit = torch.rand(throw_ball_num, device=self.device) * 0.8 + 0.8 # (0.8, 1.6)
                z = torch.rand(throw_ball_num, device=self.device) * centre_z + 0.5 * centre_z # [0.5h, 1.5h]
                ball_vel[:, 2] = (centre_z - z)/t_hit + 0.5*9.81*t_hit
                ball_vel[:, 3:] = 1.0
            elif self.cfg.task.throw_ball_pattern == 2:
                ball_target_vel = torch.ones(throw_ball_num, 3, device=self.device)
                ball_target_vel[:, 2] = - torch.rand(throw_ball_num, device=self.device) - 0.5 #[-1.5, -0.5]
                ball_target_vel[:, :2] = 2*(torch.rand(throw_ball_num, 2, device=self.device)-0.5) #[-1, 1]
                ball_target_vel = ball_target_vel/torch.norm(ball_target_vel, p=2, dim=1, keepdim=True) * ball_speed
                target_vz = ball_target_vel[:, 2]
                z_max = centre_z + target_vz**2/(2*9.81)
                z = torch.rand(throw_ball_num, device=self.device) * z_max
                # delta_z = v_t*t + 1/2*g*t^2
                t_hit = 1/9.81 * (-target_vz + torch.sqrt(target_vz**2 + 2*9.81*(centre_z - z)))
                ball_vel = torch.ones(throw_ball_num, 6, device=self.device)
                ball_vel[:, :3] = ball_target_vel.clone()
                ball_vel[:, 2] = ball_target_vel[:, 2] + 9.81*t_hit
                
            else:
                raise NotImplementedError()
            
            drone_x_speed = torch.mean(self.root_states[throw_indices[0]][..., 7], 1)
            drone_x_dist = drone_x_speed * t_hit

            drone_y_speed = torch.mean(self.root_states[throw_indices[0]][..., 8], 1)
            drone_y_dist = drone_y_speed * t_hit

            if self.cfg.task.throw_center:
                drone_x_max = torch.max(self.root_states[throw_indices[0]][..., 0], -1)[0]
                drone_x_min = torch.min(self.root_states[throw_indices[0]][..., 0], -1)[0]
                drone_y_max = torch.max(self.root_states[throw_indices[0]][..., 1], -1)[0]
                drone_y_min = torch.min(self.root_states[throw_indices[0]][..., 1], -1)[0]

                target_ball_pos[:, 0] = drone_x_dist + \
                    torch.rand(throw_ball_num, device=self.device) * (drone_x_max - drone_x_min) + drone_x_min
                target_ball_pos[:, 1] = drone_y_dist + \
                    torch.rand(throw_ball_num, device=self.device) * (drone_y_max - drone_y_min) + drone_y_min
                target_ball_pos[:, 2] = centre_z
            else:
                target_ball_pos[:, 0] = drone_x_dist + centre_D[..., 0]
                target_ball_pos[:, 1] = drone_y_dist + centre_D[..., 1]
                target_ball_pos[:, 2] = centre_z

            ball_pos[:, :2] = target_ball_pos[:, :2] - ball_vel[:, :2]*t_hit.view(-1, 1)
            ball_pos[:, 2] = target_ball_pos[:, 2] - ball_vel[:, 2]*t_hit + 0.5*9.81*t_hit**2
            
            #============
            self.t_hit[throw_indices] = t_hit / self.cfg.sim.dt
            assert throw_ball_num == ball_pos.shape[0]

            index_1d = throw_indices[0] * self.ball_num + throw_indices[1]
            self.ball.set_world_poses(positions=ball_pos + self.envs_positions[throw_indices[0]], env_indices=index_1d)
            self.ball.set_velocities(ball_vel, env_indices=index_1d)

            # draw target in red, draw init in green
            # draw_target_coordinates = target_ball_pos + self.envs_positions[should_throw]
            # draw_init_coordinates = ball_pos + self.envs_positions[should_throw]
            # colors = [(1.0, 0.0, 0.0, 1.0) for _ in range(throw_ball_num)] + [(0.0, 1.0, 0.0, 1.0) for _ in range(throw_ball_num)]
            # sizes = [2.0 for _ in range(2*throw_ball_num)]
            
            # self.draw.draw_points(draw_target_coordinates.tolist() + draw_init_coordinates.tolist(), colors, sizes)
        self.flag.bitwise_or_(flag)

    def _compute_state_and_obs(self):
        self.root_states = self.drone.get_state()  # Include pos, rot, vel, ...
        self.info["drone_state"][:] = self.root_states[..., :13]
        pos = self.drone.pos  # Position of all the drones relative to the environment's local coordinate
        vel = self.drone.vel[..., :3] # [env_num, drone_num, 3]
        self.formation_center = self.drone.pos.mean(-2, keepdim=True)
        intra_formation = pos - self.formation_center

        if self.track_pos:
            self.track_pos_goal += self.target_vel * self.cfg.task.sim.dt
            
        self.rheading = self.target_heading - self.root_states[..., 13:16]

        # indi_rel_pos = pos - self.target_pos
        self_rel_vel = self.target_vel - vel # [3, ] - [env_num, drone_num, 3] => [env_num, drone_num, 3]
        if self.real_flight:
            obs_self = [self.root_states[..., :19], self_rel_vel]
        else:
            obs_self = [self.root_states[..., :23], self_rel_vel]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).reshape(-1, 1, 1)
            if self.cfg.task.eval:
                obs_self.append(torch.zeros_like(t.expand(-1, self.drone.n, self.time_encoding_dim)))
            else:
                obs_self.append(t.expand(-1, self.drone.n, self.time_encoding_dim))
        if self.cfg.algo.share_actor:
            obs_self.append(self.drone_id.reshape(1, -1, 1).expand(self.root_states.shape[0], -1, self.id_dim))
        obs_self = torch.cat(obs_self, dim=-1)
        # obs_self[..., 0] -= self.formation_center[..., 0]

        if self.cfg.task.relative_self_pos:
            obs_self[..., 0] -= self.formation_center[..., 0] # target vertical pos
            obs_self[..., 1] -= self.formation_center[..., 1]
            # obs_self[..., 2] -= self.target_height # target height

        relative_pos_ = vmap(cpos)(pos, pos) # [env_num, n, n, 3]
        self.drone_pdist = vmap(off_diag)(torch.norm(relative_pos_, dim=-1, keepdim=True))   # pair wise distance, [env_num, n, n-1, 1]
        relative_pos = vmap(off_diag)(relative_pos_)
        relative_vel = vmap(cpos)(vel, vel)
        relative_vel = vmap(off_diag)(relative_vel)

        # obstacle-observation
        obstacle_vel = []
        relative_obs_pos = []
        relative_obs_dis = []
        ball_state = torch.zeros(self.num_envs, self.ball_obs_dim, 6)

        if self.ball_num > 0:
            # Relative position between the ball and all the drones
            reshape_ball_world_poses = (self.ball.get_world_poses()[0].view(-1, self.ball_num, 3), self.ball.get_world_poses()[1].view(-1, self.ball_num, 4))
            balls_pos, balls_rot = self.get_env_poses(reshape_ball_world_poses) # [env_num, ball_num, 3]
            # balls_pos = self.ball.get_world_poses()[0].view(-1, self.ball_num, 3)
            balls_vel = self.ball.get_linear_velocities().view(-1, self.ball_num, 3) # [env_num, 1, ball_num, 3]
            ball_state = torch.cat([balls_pos, balls_vel], dim=-1)
            relative_b_pos =  balls_pos.unsqueeze(1) - pos[..., :3].unsqueeze(2) # [env_num, drone_num, 1, 3] - [env_num, 1, ball_num, 3]
            relative_b_dis = self.relative_b_dis = torch.norm(relative_b_pos, p=2, dim=-1) # [env_num, drone_num, ball_num, 3] -> [env_num, drone_num, ball_num]

            relative_obs_pos.append(relative_b_pos)
            relative_obs_dis.append(relative_b_dis)
            obstacle_vel.append(balls_vel)

        if self.static_obs_num > 0:
            # reshape_cube_world_poses = (self.cube.get_world_poses()[0].view(-1, self.static_obs_num, 3), self.cube.get_world_poses()[1].view(-1, self.static_obs_num, 4))
            obstacle_vel.append(torch.zeros_like(self.cubes_pos))
            relative_c_pos =  self.cubes_pos.unsqueeze(1) - pos[..., :3].unsqueeze(2) # [env_num, drone_num, 1, 3] - [env_num, 1, ball_num, 3]
            # if self.cfg.task.static_obs_type == 2:
            relative_c_dis = torch.norm(relative_c_pos[..., :2], p=2, dim=-1) # for columns, calculate x & y distance
            # else:
                # relative_c_dis = torch.norm(relative_c_pos[..., :3], p=2, dim=-1)
            self.relative_c_dis = relative_c_dis
            relative_obs_pos.append(relative_c_pos)
            relative_obs_dis.append(relative_c_dis)

        # calculate full obstacle observation
        obstacle_vel = torch.cat(obstacle_vel, dim=1)
        relative_obs_pos = torch.cat(relative_obs_pos, dim=2)
        self.relative_obs_dis = relative_obs_dis = torch.cat(relative_obs_dis, dim=2)
        relative_obs_vel = obstacle_vel.unsqueeze(1) - vel.unsqueeze(2)

        obs_obstacle = torch.cat([
            relative_obs_dis.unsqueeze(-1), 
            relative_obs_pos, # [n, k, m, 3]
            relative_obs_vel,
            obstacle_vel.unsqueeze(1).expand(-1,self.drone.n,-1,-1)
        ], dim=-1).view(self.num_envs, self.drone.n, -1, 10) #[env, agent, obstacle_num, *]
        
        # TODO: add top k mask
        # ball mask
        if self.ball_num == 0:
            ball_mask = torch.ones(self.num_envs, self.drone.n, self.ball_obs_dim, dtype=bool, device=self.device)
        else:
            # not thrown
            ball_mask = (~self.ball_reward_flag).unsqueeze(-2).expand(-1, self.drone.n, -1) #[env_num, drone, ball_num]
            # after landed
            after_landed = (balls_pos[..., 2] < 0.2).unsqueeze(-2).expand(-1, self.drone.n, -1) #[env_num, drone, ball_num]
            ball_mask = ball_mask | after_landed
            if self.cfg.task.use_mask_behind:
                mask_behind = (relative_b_pos[..., 1] < 0) #[env_num, drone, ball_num]
                ball_mask = ball_mask | mask_behind
            if self.cfg.task.use_mask_front:
                mask_front = (relative_b_pos[..., 1] > self.cfg.task.obs_range) #[env_num, drone, ball_num]
                ball_mask = ball_mask | mask_front
            if self.cfg.task.mask_range:
                mask_range = (relative_b_dis > self.cfg.task.obs_range) #[env_num, drone, ball_num]
                ball_mask = ball_mask | mask_range
                
        # static mask
        if self.static_obs_num == 0:
            static_mask = torch.ones(self.num_envs, self.drone.n, self.static_obs_dim, dtype=bool, device=self.device)
        else:
            static_mask = torch.zeros(self.num_envs, self.drone.n, self.static_obs_dim, dtype=bool, device=self.device)
            if self.cfg.task.use_mask_behind:
                mask_behind = (relative_c_pos[..., 1] < 0) #[env_num, drone, ball_num]
                static_mask = static_mask | mask_behind
            if self.cfg.task.use_mask_front:
                mask_front = (relative_c_pos[..., 1] > self.cfg.task.obs_range)
                static_mask = static_mask | mask_front
            if self.cfg.task.mask_range:
                mask_range = (relative_c_dis > self.cfg.task.obs_range)
                static_mask = static_mask | mask_range
        
        # print(static_mask.shape, ball_mask.shape)

        # # choose the closest top k
        # if self.cfg.task.use_top_k_obs:
        #     indices = torch.topk(obs_obstacle[..., 0], k=self.obstacle_num, dim=2, largest=False).indices
        #     obs_obstacle = torch.gather(obs_obstacle, 2, indices.unsqueeze(-1).expand(self.num_envs, self.drone.n, self.obstacle_num, 10))

        # if self.cfg.task.mask_all:
        #     obs_obstacle[:] = self.mask_observation

        if self.ball_num > 0:
            obs_ball = obs_obstacle[:, :, :self.ball_num]
        else:
            obs_ball = torch.zeros(self.num_envs, self.drone.n, self.ball_obs_dim, 10, device=self.device)
        
        if self.static_obs_num > 0:
            obs_static = obs_obstacle[:, :, self.ball_num:]
        else:
            obs_static = torch.zeros(self.num_envs, self.drone.n, self.static_obs_dim, 10, device=self.device)
        
        obs_others = torch.cat([
            relative_pos/2, # [env_num, n, n-1, 3]
            self.drone_pdist/2, 
            relative_vel
        ], dim=-1)

        if self.cfg.task.swarm_rl:
            if self.cfg.task.static_obs_num == 0:
                obs_static = torch.zeros(self.num_envs, self.drone.n, 9, device=self.device)
                static_mask = torch.ones(self.num_envs, self.drone.n, 1, device=self.device)
            else:
                if self.cfg.task.static_obs_pos == "fake":
                    self.cubes_pos[..., :2] = self.root_states[..., 0, :2].unsqueeze(1)
                    self.cubes_pos[..., 1] += 3
                drone_grids = pos[..., :2].unsqueeze(-2) + self.drone_grid.unsqueeze(0).unsqueeze(0) # [envs, drones, 9, 2]
                rel_pos = self.cubes_pos[..., :2].unsqueeze(-2).unsqueeze(1) - drone_grids.unsqueeze(2) # [envs, drones, cubes, 9, 2]
                distance = torch.norm(rel_pos, dim=-1) # [envs, drones, cubes, 9]
                obs_static = torch.min(distance, dim=2)[0]  # [envs, drones, 9]
                static_mask = torch.zeros(self.num_envs, self.drone.n, 1, device=self.device)
            obs = TensorDict({ 
                "obs_self": obs_self.unsqueeze(2),  # [N, K, 1, obs_self_dim]
                "obs_others": obs_others, # [N, K, K-1, obs_others_dim]
                "attn_obs_ball": obs_ball, # [N, K, ball_num, *]
                "attn_obs_static": obs_static.unsqueeze(2) # [envs, drones, 1, 9]
            }, [self.num_envs, self.drone.n]) # [N, K, n_i, m_i] 
            obs["attn_ball_mask"] = ball_mask
            obs["attn_static_mask"] = static_mask
        elif not self.cfg.task.use_separate_obs:
            obs = TensorDict({ 
                "obs_self": obs_self.unsqueeze(2),  # [N, K, 1, obs_self_dim]
                "obs_others": obs_others, # [N, K, K-1, obs_others_dim]
                "attn_obs_obstacles": obs_obstacle, # [N, K, ball_num+static_obs, *]
            }, [self.num_envs, self.drone.n]) # [N, K, n_i, m_i]
        else:
            obs = TensorDict({ 
                "obs_self": obs_self.unsqueeze(2),  # [N, K, 1, obs_self_dim]
                "obs_others": obs_others, # [N, K, K-1, obs_others_dim]
                "attn_obs_ball": obs_ball, # [N, K, ball_num, *]
                "attn_obs_static": obs_static
            }, [self.num_envs, self.drone.n]) # [N, K, n_i, m_i]
            if self.cfg.task.use_attn_mask:
                obs["attn_ball_mask"] = ball_mask
                obs["attn_static_mask"] = static_mask

        state = TensorDict({
            "drones": self.root_states,
            "balls": ball_state,
            "cols": self.cubes_pos if self.cfg.task.static_obs_num else torch.zeros(self.num_envs, self.drone.n, 1, 3, device=self.device),
            }, self.batch_size)

        return TensorDict({
            "agents":{
                "observation": obs,    # input for the network
                "state": state,
            },
            "stats": self.stats,
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        pos, rot = self.get_env_poses(self.drone.get_world_poses())
        center_x = pos[..., 0].mean()
        center_y = pos[..., 1].mean()
        center_z = pos[..., 2].mean()
        self.stats["center_x"].lerp_(center_x, (1-self.alpha))
        self.stats["center_y"].lerp_(center_y, (1-self.alpha))
        self.stats["center_z"].lerp_(center_z, (1-self.alpha))

        # formation objective
        normalized = self.cfg.task.normalize_formation
        cost_l = vmap(cost_formation_laplacian)(pos, desired_L=self.formation_L, normalized=True)
        cost_l_unnormalized = vmap(cost_formation_laplacian)(pos, desired_L=self.formation_L_unnormalized, normalized=False)
        size = self.drone_pdist.max(dim=-2).values.max(dim=-2).values
        
        if self.cfg.task.rescale_formation:
            reward_formation = 1 / (1 + torch.square((cost_l - 0.04)/self.drone.n * 10)) - (0.04*self.drone.n)
            reward_formation = reward_formation/2

            size_delta = size - self.standard_formation_size
            reward_size = 1 / (1 + torch.square(size_delta))
            delta = 1 / (1 + cost_l_unnormalized)
            reward_size += delta
            
            reward_size = (reward_size - 2)/self.drone.n - (0.04*self.drone.n)
            reward_size = reward_size* 3 + 2.36

        else:
            reward_formation = 1 / (1 + torch.square(cost_l * 10))
            reward_size = 1 / (1 + torch.square(size - self.standard_formation_size))
            reward_size += 1 / (1 + cost_l_unnormalized)

        separation = self.drone_pdist.min(dim=-2).values.min(dim=-2).values
        separation_reward = -(separation < self.safe_distance).float() #[env_num, drone_num]
        too_close = separation < self.cfg.task.hard_safe_distance # [env_num, drone_num]
        too_close_reward = too_close.float()

        # flight objective
        head_error = torch.norm(self.rheading, dim=-1)
        reward_heading = torch.clip(1-head_error, min=0)
        
        if not self.track_pos: 
            pos_diff = self.root_states[..., :3] - self.final_pos
        else:
            pos_diff = self.root_states[..., :3] - self.track_pos_goal
        pos_error = torch.norm(pos_diff, dim=-1)
        indi_p_reward = 1 / (1 + pos_error)
        vel_diff = self.root_states[..., 7:10] - self.target_vel #[env_num, drone_num, 3]
        # acceptable_vel = (torch.abs(vel_diff[:, :, 1]) < self.cfg.task.acceptable_v_diff) # [env_num, drone_num]
        # indi_v_reward = 1 / (1 + torch.norm(vel_diff, dim=-1)) * self.velocity_coeff # [env_num, drone_num]
        indi_v_reward = torch.clip(torch.clip(torch.norm(self.target_vel, dim=-1), min=1) - torch.norm(vel_diff, dim=-1), min=0)

        height = pos[..., 2]   # [num_envs, drone.n]
        height_diff = height - self.target_height
        height_reward = torch.clip(1 - height_diff.abs(), min=0)

        # not_straight = (pos[..., 0] < - (self.border + 0.5)) | (pos[..., 0] > (self.border + 0.5))
        # not_straight_reward = not_straight
        if self.real_flight:
            crash = (pos[..., 2] < 0.2) | (pos[..., 2] > 1.8) | (pos[..., 0].abs() > 5) 
        else:
            crash = (pos[..., 2] < 0.2) | (pos[..., 2] > 2.8) # | (pos[..., 0].abs() > 10) | (pos[..., 1] < -5) # [env_num, drone_num]
        # crash_reward = crash # crash_penalty < 0

        # obstacle objective
        if self.cfg.task.ball_num > 0:
            ball_vel = self.ball.get_linear_velocities().view(-1, self.ball_num, 3) # [env_num, ball_num, 3]
            reshape_ball_world_poses = (self.ball.get_world_poses()[0].view(-1, self.ball_num, 3), self.ball.get_world_poses()[1].view(-1, self.ball_num, 4))
            ball_pos, balls_rot = self.get_env_poses(reshape_ball_world_poses) # [env_num, ball_num, 3]
            # ball_pos = self.ball.get_world_poses()[0].view(-1, self.ball_num, 3)

            should_neglect = ((ball_vel[...,2] < 1e-6) & (ball_pos[...,2] < 0.5)) # [env_num, ball_num] 
            self.ball_alarm[should_neglect] = 0 # [env_num, ball_num]
            self.ball_alarm[~should_neglect] = 1
            
            ball_mask = (self.ball_alarm & self.ball_reward_flag) # [env_num, 1, ball_num]
            # compute ball hard reward (< self.obs_safe_distance)
            should_penalise = self.relative_b_dis < self.obs_safe_distance # [env_num, drone_num, ball_num]
            ball_hard_reward = torch.zeros(self.num_envs, self.drone.n, self.ball_num, device=self.device)
            ball_hard_reward[should_penalise] = -self.ball_hard_coeff

            # compute ball soft reward (encourage > self.soft_obs_safe_distance)
            indi_b_dis = self.relative_b_dis #[env_num, drone_num, ball_num]
            # smaller than ball_safe_dist, only consider hard reward
            k = 0.5 * self.ball_hard_coeff / (self.soft_obs_safe_distance-self.obs_safe_distance)
            # between ball_safe_dist and soft_ball_safe_dist, apply linear penalty
            indi_b_reward = (torch.clamp(indi_b_dis, min=self.obs_safe_distance, max=self.soft_obs_safe_distance) - self.soft_obs_safe_distance) * k
            # larger than soft_ball_safe_dist, apply linear
            indi_b_reward += torch.clamp(indi_b_dis-self.soft_obs_safe_distance, min=0)
            
            total_ball_reward = ball_hard_reward + indi_b_reward
            total_ball_reward *= ball_mask.unsqueeze(1)
            ball_reward, _ = torch.min(total_ball_reward, dim=-1) # [env_num, drone_num]

            ball_any_mask = ball_mask.any(dim=-1).unsqueeze(-1) # [env_num, 1]

        else: # self.ball_num == 0
            ball_any_mask = torch.zeros(self.num_envs, 1, dtype=bool, device=self.device)
            ball_reward = torch.zeros(self.num_envs, self.drone.n, device=self.device)
            
        if self.static_obs_num > 0: # 球和静态障碍物最好可以分开算，因为球要考虑 mask 的问题，cube 不太用
            # hit_c = torch.sum(self.relative_c_dis < self.obs_hit_distance, dim=-1)
            cube_hard_reward = (torch.clamp(self.relative_c_dis, min=self.col_hit_distance, max=self.obs_safe_distance) - self.obs_safe_distance)
            cube_reward = torch.mean(cube_hard_reward, dim=-1) # [env, drone_num]
            if self.cfg.task.use_cube_reward_mask:
                cube_any_mask = (self.relative_c_dis < (self.soft_obs_safe_distance + 1.)).any(dim=-1).any(dim=-1).unsqueeze(-1) # [env_num, 1]
            else:
                cube_any_mask = torch.zeros(self.num_envs, 1, dtype=bool, device=self.device)
        else:
            cube_any_mask = torch.zeros(self.num_envs, 1, dtype=bool, device=self.device)
            cube_reward = torch.zeros(self.num_envs, self.drone.n, device=self.device)

        ball_coeff = ((~(ball_any_mask | cube_any_mask)) * self.cfg.task.no_ball_coeff
                                            + (ball_any_mask | cube_any_mask) * self.cfg.task.has_ball_coeff)

        # after throw mask, still only consider ball case
        if self.ball_num > 0:
            after_throw_mask = (~ball_any_mask) & ((~torch.isnan(self.t_launched)).all(dim=-1, keepdim=True))
        else:
            after_throw_mask = torch.zeros_like(ball_any_mask)        
        
        if self.ball_num > 0:
            hit_b = torch.sum(self.relative_b_dis < self.ball_hit_distance, dim=-1)
        else:
            hit_b = torch.zeros(self.num_envs, self.drone.n, device=self.device, dtype=bool)
        if self.static_obs_num > 0:
            hit_c = torch.sum(self.relative_c_dis < self.col_hit_distance, dim=-1)
        else:
            hit_c = torch.zeros(self.num_envs, self.drone.n, device=self.device, dtype=bool)

        # hit = torch.cat([hit_b.unsqueeze(-1), hit_c.unsqueeze(-1)], dim=-1) # self.relative_obs_dis < self.obs_hit_distance # [env_num, drone_num, ball_num]
        hit = hit_b + hit_c
        hit_reward = hit.float()

        acceptable_mean_pos = pos_error.mean(dim=-1).mean(dim=-1) < 10
        if self.real_flight == 2:
            target_y = 20
        else:
            target_y = 40
        forward_sucess = (pos[..., 1] > target_y).all(-1, keepdim=True) # & acceptable_mean_pos

        if self.cfg.task.eval:
            done = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            bad_terminate = crash | too_close | hit
            self.bad_terminate = bad_terminate | self.bad_terminate
            truncated = ~(self.bad_terminate.any(-1, keepdim=True))
            bad_terminate = torch.sum(self.bad_terminate, dim=-1) > 0
            done = done # | (pos[..., 1] > (target_y+2)).all(-1, keepdim=True)
        else:
            bad_terminate = crash | too_close | hit
            bad_terminate = torch.sum(bad_terminate, dim=-1) > 0
            truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            acceptable_mean_vel = (torch.abs(torch.mean(vel_diff[:, :, 1], dim=1)) < self.cfg.task.acceptable_v_diff).unsqueeze(-1) # vel_diff: [env_num, drone.n, 3]
            done = crash | too_close | hit | truncated

        done = torch.sum(done, dim=-1) > 0
        # survival_reward = 1 - bad_terminate.float().unsqueeze(-1)
        bad_terminate_penalty = bad_terminate.float().unsqueeze(-1)

        dynamic_coeff = ((~(ball_any_mask | cube_any_mask)) * self.cfg.task.no_ball_coeff
                          + (ball_any_mask | cube_any_mask) * self.cfg.task.has_ball_coeff)

        morl_obstacle = (ball_reward * self.cfg.task.ball_reward_coeff
            + cube_reward * self.cfg.task.static_hard_coeff
            + hit_reward * self.cfg.task.hit_penalty
            + truncated * self.cfg.task.truncated_reward
            - bad_terminate_penalty * self.cfg.task.truncated_reward)

        self.stats["ball_reward"].lerp_(ball_reward, (1-self.alpha))
        self.stats["cube_reward"].lerp_(cube_reward, (1-self.alpha))
        self.stats["hit_reward"].lerp_(hit_reward, (1-self.alpha))
        self.stats["morl_obstacle_reward"].lerp_(morl_obstacle, (1-self.alpha))

        # action objective

        # reward_effort = torch.exp(-self.effort) # throttle sum
        reward_effort = torch.clip(2.5-self.effort, min=0) # 2.5->1.4
        # reward_throttle_smoothness = torch.exp(-self.drone.throttle_difference)
        reward_throttle_smoothness = torch.clip(.5-self.drone.throttle_difference, min=0) # 0.4->0.1
        output_diff = torch.norm((self.last_network - self.current_network), dim=-1)
        # reward_action_smoothness = torch.exp(-output_diff)
        reward_action_smoothness = torch.clip(2.5-output_diff, min=0) # 2.3->1.8
        y_spin = torch.abs(self.drone.vel[..., -1])
        # reward_spin = 1 / (1 + y_spin)
        reward_spin = torch.clip(1.5-y_spin, min=0) # 1.5->0.25
        avg_acc = torch.norm(self.drone.get_acc()[..., :3], p=2, dim=-1).mean()

        morl_smooth = (reward_effort * self.reward_effort_weight
            + reward_action_smoothness * self.cfg.task.reward_action_smoothness_weight
            + reward_spin * self.cfg.task.spin_reward_coeff
            + reward_throttle_smoothness * self.cfg.task.reward_throttle_smoothness_weight
            + truncated * self.cfg.task.truncated_reward
            - bad_terminate_penalty * self.cfg.task.truncated_reward)

        self.stats["reward_effort"].lerp_(reward_effort, (1-self.alpha))
        self.stats["reward_action_smoothness"].lerp_(reward_action_smoothness, (1-self.alpha))
        avg_throttle_diff = self.drone.throttle_difference.mean(dim=-1, keepdim=True)
        self.stats["throttle_difference_mean"].add_(avg_throttle_diff)
        self.stats['throttle_difference_mean'].div_(
            torch.where(done, self.progress_buf, torch.ones_like(self.progress_buf)).unsqueeze(-1)
        )

        self.stats["reward_spin"].lerp_(reward_spin, (1-self.alpha))
        self.stats["reward_throttle_smoothness"].lerp_(reward_throttle_smoothness, (1-self.alpha))
        self.stats["morl_smooth_reward"].lerp_(morl_smooth, (1-self.alpha))
        self.stats["avg_acc"].lerp_(avg_acc, (1-self.alpha))

        morl_formation = (
            (reward_size * ball_coeff + reward_size * after_throw_mask * self.cfg.task.after_throw_coeff) * self.cfg.task.formation_size_coeff
            + reward_formation * self.formation_coeff * dynamic_coeff
            + separation_reward * self.cfg.task.separation_coeff
            + too_close_reward * self.cfg.task.too_close_penalty
            + truncated * self.cfg.task.truncated_reward
            - bad_terminate_penalty * self.cfg.task.truncated_reward)
        
        self.stats["cost_l"].lerp_(cost_l, (1-self.alpha))
        self.stats["cost_l_unnormalized"].lerp_(cost_l_unnormalized, (1-self.alpha))
        self.stats["reward_formation"].lerp_(reward_formation, (1-self.alpha))
        self.stats["reward_size"].lerp_(reward_size, (1-self.alpha))
        self.stats["separation_reward"].lerp_(separation_reward, (1-self.alpha))
        self.stats["morl_formation_reward"].lerp_(morl_formation, (1-self.alpha))

        morl_forward = (
            height_reward * self.height_coeff * ball_coeff
            + indi_p_reward * self.cfg.task.position_reward_coeff * truncated
            + indi_v_reward * self.velocity_coeff * ball_coeff
            + reward_heading * self.cfg.task.heading_coeff) * dynamic_coeff
        
        morl_forward += (truncated * self.cfg.task.truncated_reward
            - bad_terminate_penalty * self.cfg.task.truncated_reward) 
        
        self.stats["height_reward"].lerp_(height_reward, (1-self.alpha))
        self.stats["pos_reward"].lerp_(indi_p_reward, (1-self.alpha))
        self.stats["vel_reward"].lerp_(indi_v_reward, (1-self.alpha))
        self.stats["reward_heading"].lerp_(reward_heading, (1-self.alpha))
        self.stats["morl_forward_reward"].lerp_(morl_forward, (1-self.alpha))

        # additional constraints regarding all objectives
        size_ratio = size / self.standard_formation_size
        formation_success = (cost_l_unnormalized < 5) # 1
        # action_norm = self.current_network.norm(dim=-1).mean(dim=-1, keepdim=True)
        action_success = avg_throttle_diff < 0.005

        success = truncated & forward_sucess & formation_success & action_success

        self.stats["formation_success"][:] = (formation_success.float())
        self.stats["action_success"][:] = (action_success.float())
        self.stats["success"][:] = (success.float())

        if self.cfg.task.rescale_reward:
            reward = (morl_smooth * self.morl_smooth_coeff / 14
                    + morl_obstacle * self.morl_obstacle_coeff / 40
                    + morl_forward * self.morl_forward_coeff / 25
                    + morl_formation * self.morl_formation_coeff / 4
                    ).reshape(-1, self.drone.n)
        else:
            reward = (morl_smooth * self.morl_smooth_coeff
                    + morl_obstacle * self.morl_obstacle_coeff
                    + morl_forward * self.morl_forward_coeff
                    + morl_formation * self.morl_formation_coeff
                    ).reshape(-1, self.drone.n)
        self.stats["reward"].lerp_(reward, (1-self.alpha))
        self.stats["return"] = self.stats["return"] * self.gamma + torch.mean(reward, dim=1, keepdim=True)

        # self.stats["survival_reward"].lerp_(survival_reward, (1-self.alpha))
        # self.stats["survival_return"].add_(torch.mean(survival_reward))

        # formation_dis = compute_formation_dis(pos, self.formation).expand(-1, self.ball_num) # [env_num, ball_num]
        formation_dis = compute_formation_dis(pos, self.formation) # [env_num, 1]
        # print(formation_dis.shape)
        drone_moved = ((~torch.isnan(self.t_launched)) & (formation_dis > 0.35) & (torch.isnan(self.t_moved))) # [env_num, ball_num]
        self.t_moved[drone_moved] = self.progress_buf.unsqueeze(-1).expand(self.num_envs, self.ball_num)[drone_moved]
        self.t_moved[drone_moved] = self.t_moved[drone_moved] - self.t_launched[drone_moved]
        self.t_difference[drone_moved] = self.t_moved[drone_moved] - self.t_hit[drone_moved]

        self.frame_counter += (torch.sum((done.squeeze()).float() * self.progress_buf)).item()

        self.stats["t_launched"][:] = torch.nanmean(self.t_launched.unsqueeze(1), keepdim=True)
        self.stats["t_moved"][:] = torch.nanmean(self.t_moved.unsqueeze(1), keepdim=True)
        self.stats["t_difference"][:] =  torch.nanmean(self.t_difference.unsqueeze(1), keepdim=True)
        self.stats["t_hit"][:] =  torch.nanmean(self.t_hit.unsqueeze(1), keepdim=True)
        self.stats["truncated"][:] = (truncated.float())
        # self.stats["terminated"][:] = (terminated.float())
        self.stats["forward_success"][:] = (forward_sucess.float())
        self.stats["crash"][:] = torch.any(crash, dim=-1, keepdim=True).float()
        # self.stats["not straight"][:] = torch.any(not_straight, dim=-1, keepdim=True).float()
        self.stats["hit"][:] = torch.any(hit, dim=-1, keepdim=True).float()
        self.stats["hit_b"][:] = torch.any(hit_b, dim=-1, keepdim=True).float()
        self.stats["hit_c"][:] = torch.any(hit_c, dim=-1, keepdim=True).float()
        self.stats["too close"][:] = (too_close.float())
        self.stats["done"][:] = (done.float()).unsqueeze(1)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        # self.stats["morl_smooth"].add_(torch.mean(morl_smooth, dim=1, keepdim=True))
        # self.stats["morl_formation"].add_(torch.mean(morl_formation, dim=1, keepdim=True))
        # self.stats["morl_obstacle"].add_(torch.mean(morl_obstacle, dim=1, keepdim=True))
        # self.stats["morl_forward"].add_(torch.mean(morl_forward, dim=1, keepdim=True))

        self.stats["morl_smooth"] = self.stats["morl_smooth"] * self.gamma + morl_smooth
        self.stats["morl_formation"] = self.stats["morl_formation"] * self.gamma + morl_formation
        self.stats["morl_obstacle"] = self.stats["morl_obstacle"] * self.gamma + morl_obstacle
        self.stats["morl_forward"] = self.stats["morl_forward"] * self.gamma + morl_forward

        self.stats["size"][:] = size
        if self.cfg.task.ball_num > 0:
            self.stats["indi_b_dist"].add_((torch.mean(torch.mean(indi_b_dis, dim=1), dim=1)/self.progress_buf).unsqueeze(-1))
        self.stats["curr_formation_dist"][:] = formation_dis

        assert self.ball_reward_flag.dtype == torch.bool
        assert self.ball_alarm.dtype == torch.bool

        return TensorDict(
            {
                "agents": {
                    "reward": reward
                },
                "done": done,
            },
            self.batch_size
        )


def new_cost(
        d: torch.Tensor
) -> torch.Tensor:
    " Account for the distance between the drone's actual position and targeted position"
    d = torch.clamp(d.square()-0.15**2, min=0) # if the difference is less then 0.1, generating 0 cost  
    return torch.sum(d)     

def huber_cost(
        d: torch.Tensor
) -> torch.Tensor:
    " Account for the distance between the drone's actual position and targeted position"
    d = torch.clamp(d-0.15, min=0) # if the difference is less then 0.1, generating 0 cost  
    return torch.sum(d)    

def cost_formation_laplacian(
    p: torch.Tensor,
    desired_L: torch.Tensor,
    normalized=False,
) -> torch.Tensor:
    """
    A scale and translation invariant formation similarity cost
    """
    L = laplacian(p, normalized)
    cost = torch.linalg.matrix_norm(desired_L - L)
    return cost.unsqueeze(-1)


def laplacian(p: torch.Tensor, normalize=False):
    """
    symmetric normalized laplacian

    p: (n, dim)
    """
    assert p.dim() == 2
    A = torch.cdist(p, p) # A[i, j] = norm_2(p[i], p[j]), A.shape = [n, n]
    D = torch.sum(A, dim=-1) # D[i] = \sum_{j=1}^n norm_2(p[i], p[j]), D.shape = [n, ]
    if normalize:
        DD = D**-0.5
        A = torch.einsum("i,ij->ij", DD, A)
        A = torch.einsum("ij,j->ij", A, DD)
        L = torch.eye(p.shape[0], device=p.device) - A
    else:
        L = D - A
    return L


def cost_formation_var(p: torch.Tensor, target_p: torch.Tensor):
    # assert p.dim() == target_p.dim() and p.dim() == 2
    A = torch.cdist(p, p) # [B, n, n]
    target_A = torch.cdist(target_p, target_p) # [n, n]
    target_A[range(p.shape[0]), range(p.shape[0])] = 1 # avoid zero in frac
    ratio = A/target_A # [B, n, n]
    uptri = torch.triu(ratio) # [B, n, n]
    zero_mask = (uptri == 0)
    means = torch.sum(uptri, dim=(1, 2), keepdim=True) / (~zero_mask).sum(dim=(1, 2), keepdim=True) # [B]
    uptri[zero_mask] = torch.nan
    variance = torch.nansum((uptri - means)**2, dim=(1,2)) / (~zero_mask).sum(dim=(1, 2))
    return variance.unsqueeze(-1) # [B, 1]
    
    
def cost_formation_hausdorff(p: torch.Tensor, desired_p: torch.Tensor) -> torch.Tensor:
    p = p - p.mean(-2, keepdim=True)
    desired_p = desired_p - desired_p.mean(-2, keepdim=True)
    cost = torch.max(directed_hausdorff(p, desired_p), directed_hausdorff(desired_p, p))
    return cost.unsqueeze(-1)


def directed_hausdorff(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    p: (*, n, dim)
    q: (*, m, dim)
    """
    d = torch.cdist(p, q, p=2).min(-1).values.max(-1).values
    return d

def compute_formation_dis(pos: torch.Tensor, formation_p: torch.Tensor):
    rel_pos = pos - pos.mean(-2, keepdim=True) # [env_num, drone_num, 3]
    rel_f = formation_p - formation_p.mean(-2, keepdim=True) # [drone_num, 3]
    # [env_num, drone_num]
    dist = torch.norm(rel_f-rel_pos, p=2, dim=-1)
    dist = torch.mean(dist, dim=-1, keepdim=True)
    return dist