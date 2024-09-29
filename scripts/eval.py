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

curdir = os.path.dirname(os.path.abspath(__file__))
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
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv
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

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    transforms = [InitTracker()]

    # a CompositeSpec is by deafault processed by a entity-based encoder
    # flatten it to use a MLP encoder instead
    if cfg.task.get("flatten_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    if cfg.task.get("flatten_state", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "state")))
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
            controller = _RateController(9.81, base_env.drone.params).to(base_env.device)
            transform = RateController(controller, rescale_thrust=cfg.task.get("real_drone", False))
            transforms.append(transform)
        elif action_transform == 'tanh':
            from omni_drones.utils.torchrl.transforms import TanhTransform
            transform = TanhTransform()
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

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    
    init_ckpt_path = cfg.get("init_ckpt_path", None)
    if init_ckpt_path:
        state_dict = torch.load(init_ckpt_path)
        policy.load_state_dict(state_dict)

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    @torch.no_grad()
    def evaluate(seed: int=0):
        frames = []
        env.cfg.task.eval=True

        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(seed)

        from tqdm import tqdm
        t = tqdm(total=base_env.max_episode_length)
        
        def record_frame(*args, **kwargs):
            frame = env.base_env.render(mode="rgb_array")
            frames.append(frame)
            t.update(2)

        trajs = env.rollout(
            max_steps=base_env.max_episode_length,
            policy=lambda x: policy(x, deterministic=True),
            callback=Every(record_frame, 2),
            auto_reset=True,
            break_when_any_done=False,
            return_contiguous=False
        ).clone()

        base_env.enable_render(not cfg.headless)
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.nanmean(v.float()).item() 
            for k, v in traj_stats.items()
        }
        env.cfg.task.eval=False

        if cfg.save_traj:
            torch.save(trajs, "trajs/"+ wandb.run.name.split('/')[0]+".pt")

        if len(frames):
            # video_array = torch.stack(frames)
            video_array = np.stack(frames).transpose(0, 3, 1, 2)
            frames.clear()
            info["recording"] = wandb.Video(
                video_array, fps=0.5 / cfg.sim.dt, format="mp4"
            )
        
        return info

    pbar = tqdm(collector)
    env.eval()
    fps = []
    info = {}
    info.update(evaluate())
    run.log(info)
    wandb.finish()
    
    simulation_app.close()


if __name__ == "__main__":
    main()
