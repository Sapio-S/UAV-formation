import logging
import os
import time

import hydra
import torch
import numpy as np
import wandb

from functorch import vmap
from omegaconf import OmegaConf

from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.utils.torchrl import SyncDataCollector, AgentSpec
from omni_drones.utils.torchrl.transforms import (
    LogOnEpisode, 
    FromMultiDiscreteAction, 
    FromDiscreteAction,
    ravel_composite,
    History
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.learning import (
    MAPPOPolicy, 
    HAPPOPolicy,
    QMIXPolicy,
    DQNPolicy,
    SACPolicy,
    TD3Policy,
    MATD3Policy,
    TDMPCPolicy,
    Policy,
    PPOPolicy,
    PPOAdaptivePolicy, PPORNNPolicy
)

from setproctitle import setproctitle
from torchrl.envs.transforms import (
    TransformedEnv, 
    InitTracker, 
    Compose,
)

from tqdm import tqdm

class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1

from typing import Sequence
from tensordict import TensorDictBase

class EpisodeStats:
    def __init__(self, in_keys: Sequence[str] = None):
        self.in_keys = in_keys
        self._stats = []
        self._episodes = 0

    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        done = tensordict.get(("next", "done"))
        truncated = tensordict.get(("next", "truncated"), None)
        done_or_truncated = (
            (done | truncated) if truncated is not None else done.clone()
        )
        if done_or_truncated.any():
            done_or_truncated = done_or_truncated.squeeze(-1) # [env_num, 1, 1]
            self._episodes += done_or_truncated.sum().item()
            self._stats.extend(
                # [env, n, 1]
                tensordict.select(*self.in_keys)[:, 1:][done_or_truncated[:, :-1]].clone().unbind(0)
            )
    
    def pop(self):
        stats: TensorDictBase = torch.stack(self._stats).to_tensordict()
        self._stats.clear()
        return stats

    def __len__(self):
        return len(self._stats)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    cfg.task.static_height = 1.0 # lower the height of the static obstacles
    cfg.task.eval = True

    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    # print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv
    from omni.isaac.core.utils.viewports import set_camera_view
    from omni.isaac.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
    
    algos = {
        "ppo": PPOPolicy,
        "ppo_adaptive": PPOAdaptivePolicy,
        "ppo_rnn": PPORNNPolicy,
        "mappo": MAPPOPolicy, 
        "happo": HAPPOPolicy,
        "qmix": QMIXPolicy,
        "dqn": DQNPolicy,
        "sac": SACPolicy,
        "td3": TD3Policy,
        "matd3": MATD3Policy,
        "tdmpc": TDMPCPolicy,
        "test": Policy
    }

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]

    # a CompositeSpec is by deafault processed by a entity-based encoder
    # flatten it to use a MLP encoder instead
    if cfg.task.get("flatten_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    if cfg.task.get("flatten_state", False):
        transforms.append(ravel_composite(base_env.observation_spec, "state"))
    if (
        cfg.task.get("flatten_intrinsics", True)
        and ("agents", "intrinsics") in base_env.observation_spec.keys(True)
    ):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "intrinsics"), start_dim=-1))

    if cfg.task.get("history", False):
        transforms.append(History([("agents", "observation")]))
    
    # optionally discretize the action space or use a controller
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform == "velocity":
            from omni_drones.controllers import LeePositionController
            from omni_drones.utils.torchrl.transforms import VelController
            controller = LeePositionController(9.81, base_env.drone.params).to(base_env.device)
            transform = VelController(controller)
            transforms.append(transform)
        elif action_transform == "attitude":
            from omni_drones.controllers import AttitudeController as Controller
            from omni_drones.utils.torchrl.transforms import AttitudeController
            controller = Controller(9.81, base_env.drone.params).to(base_env.device)
            transform = AttitudeController(controller)
            transforms.append(transform)
        elif action_transform == "rate":
            from omni_drones.controllers import RateController as _RateController
            from omni_drones.utils.torchrl.transforms import RateController
            # from torch.distributions.transforms import TanhTransform
            controller = _RateController(9.81, base_env.drone.params).to(base_env.device)
            transform = RateController(controller, rescale_thrust=cfg.task.get("real_drone", False))
            # transforms.append(TanhTransform)
            transforms.append(transform)        
        elif action_transform == "PIDrate":
            from omni_drones.controllers import PIDRateController as _PIDRateController
            from omni_drones.utils.torchrl.transforms import PIDRateController
            controller = _PIDRateController(cfg.sim.dt, 9.81, base_env.drone.params).to(base_env.device)
            transform = PIDRateController(controller, clip_ctbr=cfg.task.get("clip_ctbr", False))
            transforms.append(transform)
        elif not action_transform.lower() == "none":
            raise NotImplementedError(f"Unknown action transform: {action_transform}")
    
    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    agent_spec: AgentSpec = env.agent_spec["drone"]
    policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device="cuda")

    ckpt_name = cfg.get("init_ckpt_path", None)
    if ckpt_name is not None:
        state_dict = torch.load(ckpt_name)
        policy.load_state_dict(state_dict)

    @torch.no_grad()
    def evaluate():
        frames = []

        base_env.enable_render(True)
        base_env.eval()
        env.eval()

        from tqdm import tqdm
        t = tqdm(total=base_env.max_episode_length)
        
        def record_frame(*args, **kwargs):
            pos = env.drone.get_world_poses()[0][cfg.vis_env_id] # [drone.n, 3]
            # print("mean z = ", torch.mean(pos[:, 2]).cpu().item())
            mean_x_offset = torch.mean(pos[:, 0]).cpu().item()
            mean_y_offset = torch.mean(pos[:, 1]).cpu().item()
            
            # draw target in red, draw init in green
            colors = [(1.0, 1.0, 1.0, 1.0) for _ in range(env.drone.n)]
            sizes = [2.0 for _ in range(env.drone.n)]
            draw.draw_points(pos.tolist(), colors, sizes)
            
            set_camera_view(
                eye=np.array([mean_x_offset, 4. + mean_y_offset, 4.5]), 
                target=np.array([mean_x_offset, mean_y_offset, 1.5])
            )
            frame = env.base_env.render(mode="rgb_array")
            frames.append(frame)
            t.update(1)

        trajs = env.rollout(
            max_steps=base_env.max_episode_length,
            policy=lambda x: policy(x, deterministic=True),
            callback=Every(record_frame, 1),
            auto_reset=True,
            break_when_any_done=False,
            return_contiguous=False
        ).clone()

        base_env.enable_render(not cfg.headless)
        env.reset()
        # raise NotImplementedError()
        done = trajs.get(("next", "done")).squeeze(-1) # [env_num, seq_len]
        first_done = torch.argmax(done.long(), dim=1).cpu() # [env_num]
        # print(first_done)
        print("done_info:", first_done[cfg.vis_env_id])
        # raise NotImplementedError
        done_mask = (torch.arange(done.size(1)).expand_as(done) < first_done.unsqueeze(1)).transpose(0, 1).float() # [seq_len, env_num]
        done_mask[done_mask == 0.0] = torch.nan
        cost_l_logs = trajs.get(("next", "stats", "cost_l")).squeeze(-1).transpose(0,1).cpu()
        cost_l_logs *= done_mask
        
        formation_logs = trajs.get(("next", "stats", "morl_formation_reward")).squeeze(-1).transpose(0,1).cpu()
        formation_logs *= done_mask
        size_logs = trajs.get(("next", "stats", "size")).squeeze(-1).transpose(0,1).cpu() 
        size_logs *= done_mask
        state_logs = trajs.get(("next", "agents", "state", "drones")).transpose(0,1).cpu()[..., :10].mean(-2) # [seq_len, env_num, 3]
        state_logs *= done_mask.unsqueeze(-1)
        v_logs = state_logs[..., 7:10]
        p_logs = state_logs[..., :3]
        
        def get_terminate_count(key: str):
            raw = trajs.get(("stats", key)).squeeze(-1).transpose(0,1).cpu().to(dtype=bool) # [seq_len, env_nums]
            first_data = torch.argmax(raw.long(), dim=0) # [env_nums]
            count = torch.sum((first_data!=0)).item()
            return count
        
        count_info = {
            "hit": get_terminate_count("hit"), 
            "hit_b": get_terminate_count("hit_b"), 
            "hit_c": get_terminate_count("hit_c"), 
            "crash": get_terminate_count("crash"), 
            "too close": get_terminate_count("too close"), 
            "not straight": get_terminate_count("not straight")
        }
        
        print(count_info)
        run.log(count_info)
        
        # hit = torch.clamp(hit.cumsum(dim=0), max=1.0) # [seq_len, env_nums]
        # crash = trajs.get(("next", "stats", "crash")).squeeze(-1).transpose(0,1).cpu().sum(dim=-1) # [seq_len, env_num]
        # crash = torch.cumsum(crash, dim=0)
        # not_straight = trajs.get(("next", "stats", "not straight")).squeeze(-1).transpose(0,1).cpu().sum(dim=-1) # [seq_len, env_num]
        # not_straight = torch.cumsum(not_straight, dim=0)
        # too_close = trajs.get(("next", "stats", "too close")).squeeze(-1).transpose(0,1).cpu().sum(dim=-1) # [seq_len, env_num]
        # too_close = torch.cumsum(too_close, dim=0)
        # d_r_logs = trajs.get(("next", "stats", "drone_return")).squeeze(-1).transpose(0,1).cpu()
        # d_r_logs *= done_mask
        #  done mask = [1, 1, .., 1, nan, nan, nan]
        

        info = {}
        # print(formation_logs.shape, d_r_logs.shape)
        for i in range(formation_logs.shape[0]):
            info["curr_cost_l"] = torch.nanmean(cost_l_logs[i])
            info["morl_formation_reward"] = torch.nanmean(formation_logs[i])
            info["curr_size"] = torch.nanmean(size_logs[i])
            info["curr_survive_env"] = (done_mask[i] == 1.0).sum().item()
            info["vx"] = torch.nanmean(v_logs[i, ..., 0])
            info["vy"] = torch.nanmean(v_logs[i, ..., 1])
            info["vz"] = torch.nanmean(v_logs[i, ..., 2])
            info["x"] = torch.nanmean(p_logs[i, ..., 0])
            info["y"] = torch.nanmean(p_logs[i, ..., 1])
            info["z"] = torch.nanmean(p_logs[i, ..., 2])
            # info["crash"] = crash[i]
            # info["not_straight"] = not_straight[i]
            # info["too_close"] = too_close[i]
            # info["d_r"] = torch.nanmean(d_r_logs[i])
            run.log(info)
        
        


        # # 拿第一个 episode 结束时的值
        # def take_first_episode(tensor: torch.Tensor):
        #     indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
        #     return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        # traj_stats = {
        #     k: take_first_episode(v)
        #     for k, v in trajs[("next", "stats")].cpu().items()
        # }
        
        

        # info = {
        #     "eval/stats." + k: torch.nanmean(v.float()).item() 
        #     for k, v in traj_stats.items()
        # }
        # info = {}
        # info["eval/stats.formation_logs"] = formation_logs.nanmean(dim=-1)
        
        # print(info.keys())

        if len(frames):
            video_array = np.stack(frames).transpose(0, 3, 1, 2)
            
            info = {}
            info["recording"] = wandb.Video(
                video_array, fps=1 / cfg.sim.dt, format="mp4"
            )
            log_dir = os.path.join("logs", cfg.wandb.run_name, f"env_id={cfg.vis_env_id}")
            from datetime import datetime
            time_str = datetime.now().strftime("%m%d-%H%M")
            os.makedirs(log_dir, exist_ok=True)
            from torchvision.io import write_video
            video_arr = np.array(frames)
            video_tensor = torch.from_numpy(video_arr)
            print(video_tensor.shape)
            video_name = os.path.join(log_dir, time_str + ".mp4")
            write_video(video_name, video_tensor, fps=1/cfg.sim.dt)
            frames.clear()
        
        run.log(info)

    env.train()
    
    # collector._frames = 0
    # info = {"env_frames": collector._frames}
    evaluate()
    # info.update(evaluate())
    # run.log(info)

    wandb.finish()
    
    simulation_app.close()


if __name__ == "__main__":
    main()
