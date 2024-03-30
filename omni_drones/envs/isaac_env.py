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


import abc

from typing import Dict, List, Optional, Tuple, Type, Union, Callable

import omni.usd
import torch
import logging
import carb
import numpy as np
import hydra

from omni.isaac.core.utils.extensions import enable_extension

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    CompositeSpec, 
    TensorSpec, 
    UnboundedContinuousTensorSpec, 
    BinaryDiscreteTensorSpec
)
from torchrl.envs import EnvBase

from omni_drones.utils.torchrl import AgentSpec

from collections import OrderedDict

from omni.isaac.orbit.scene import InteractiveScene
from omni.isaac.orbit.sim import SimulationContext
from omni.isaac.orbit.utils.timer import Timer

import omni.isaac.orbit.sim as sim_utils
from dataclasses import dataclass, field

class DebugDraw:
    def __init__(self):
        from omni.isaac.debug_draw import _debug_draw
        self._draw = _debug_draw.acquire_debug_draw_interface()
    
    def clear(self):
        self._draw.clear_lines()
        self._draw.clear_points()

    def plot(self, x: torch.Tensor, size=2.0, color=(1., 1., 1., 1.)):
        if not ((x.ndim == 2) and (x.shape[1] == 3)):
            raise ValueError("x must be a tensor of shape (N, 3).")
        x = x.cpu()
        point_list_0 = x[:-1].tolist()
        point_list_1 = x[1:].tolist()
        sizes = [size] * len(point_list_0)
        colors = [color] * len(point_list_0)
        self._draw.draw_lines(point_list_0, point_list_1, colors, sizes)
        
    def vector(self, x: torch.Tensor, v: torch.Tensor, size=2.0, color=(0., 1., 1., 1.)):
        x = x.cpu().reshape(-1, 3)
        v = v.cpu().reshape(-1, 3)
        if not (x.shape == v.shape):
            raise ValueError("x and v must have the same shape, got {} and {}.".format(x.shape, v.shape))
        point_list_0 = x.tolist()
        point_list_1 = (x + v).tolist()
        sizes = [size] * len(point_list_0)
        colors = [color] * len(point_list_0)
        self._draw.draw_lines(point_list_0, point_list_1, colors, sizes)
    
@dataclass
class TaskCfg:
    num_envs: int = 1
    max_episode_length: int = 1000

    observations: List[Dict[str, Dict]] = field(default_factory=list)
    actions: List[Dict[str, Dict]] = field(default_factory=list)
    rewards: Dict[str, Dict] = field(default_factory=dict)
    termination: Dict[str, Dict] = field(default_factory=dict)
    randomizations: Dict[str, Dict] = field(default_factory=dict)


class IsaacEnv(EnvBase):

    env_ns = "/World/envs"
    template_env_ns = "/World/envs/env_0"

    REGISTRY: Dict[str, Type["IsaacEnv"]] = {}

    def __init__(self, cfg):
        super().__init__(
            device=cfg.sim.device,
            batch_size=[cfg.num_envs],
            run_type_checks=False
        )
        self.cfg = cfg
        task_cfg: TaskCfg = hydra.utils.instantiate(self.cfg.task)

        self.max_episode_length = task_cfg.max_episode_length
        self.substeps = self.cfg.sim.substeps

        if SimulationContext.instance() is None:
            sim_cfg = hydra.utils.instantiate(self.cfg.sim)
            self.sim = SimulationContext(sim_cfg)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")
        
        # set camera view for "/OmniverseKit_Persp" camera
        self.sim.set_camera_view(eye=self.cfg.viewer.eye, target=self.cfg.viewer.lookat)
        # create render product
        import omni.replicator.core as rep
        self._render_product = rep.create.render_product(
            "/OmniverseKit_Persp", tuple(self.cfg.viewer.resolution)
        )
        # create rgb annotator -- used to read data from the render product
        self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._rgb_annotator.attach([self._render_product])

        # print useful information
        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        # print(f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.substeps}")
        # print(f"\tEnvironment step-size : {self.step_dt}")
        print(f"\tPhysics GPU pipeline  : {self.cfg.sim.use_gpu_pipeline}")
        print(f"\tPhysics GPU simulation: {self.cfg.sim.physx.use_gpu}")

        # generate scene
        with Timer("[INFO]: Time taken for scene creation"):
            self.scene = InteractiveScene(self._design_scene())
            for key in self.scene.articulations.keys():
                if self.scene[key] is not None:
                    self.scene[key]._env = self
        print("[INFO]: Scene manager: ", self.scene)

        with Timer("[INFO]: Time taken for simulation reset"):
            self.sim.reset()
        for _ in range(4):
            self.sim.step(render=True)
        self.scene.update(0.)

        self.step_dt = self.physics_dt * self.substeps

        # add flag for checking closing status
        self._is_closed = False

        try:
            self.debug_draw = DebugDraw()
        except ModuleNotFoundError:
            print("To enbale debug_draw, set `headless=false` or `offscreen_render=true`.")

        self._tensordict = TensorDict(
            {
                "progress": torch.zeros(self.num_envs, device=self.device),
            },
            self.batch_size,
        )
        self.progress_buf = self._tensordict["progress"]
        self.done_spec = (
            CompositeSpec(
                {
                    "done": BinaryDiscreteTensorSpec(1, dtype=bool),
                    "terminated": BinaryDiscreteTensorSpec(1, dtype=bool),
                    "truncated": BinaryDiscreteTensorSpec(1, dtype=bool),
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )

        import inspect
        import omni_drones.envs.mdp as mdp
        members = dict(inspect.getmembers(self.__class__, inspect.isclass))

        OBS_FUNCS = mdp.OBS_FUNCS
        OBS_FUNCS.update({k: v for k, v in members.items() if issubclass(v, mdp.ObservationFunc)})
        ACT_FUNCS = mdp.ACT_FUNCS
        ACT_FUNCS.update({k: v for k, v in members.items() if issubclass(v, mdp.ActionFunc)})
        REW_FUNCS = mdp.REW_FUNCS
        REW_FUNCS.update({k: v for k, v in members.items() if issubclass(v, mdp.RewardFunc)})
        TERM_FUNCS = mdp.TERM_FUNCS
        TERM_FUNCS.update({k: v for k, v in members.items() if issubclass(v, mdp.TerminationFunc)})
        RAND_FUNCS = mdp.RAND_FUNCS
        RAND_FUNCS.update({k: v for k, v in members.items() if issubclass(v, mdp.Randomization)})

        self._update_callbacks = [self.update]
        self._debug_vis_callbacks = [self.debug_vis]
        self._reset_callbacks = []

        reward_spec = CompositeSpec({
            "reward": UnboundedContinuousTensorSpec(1),
            "stats": {
                "return": UnboundedContinuousTensorSpec(1),
                "episode_len": UnboundedContinuousTensorSpec(1),
                "success": UnboundedContinuousTensorSpec(1),
            }
        })

        def get_key(key):
            if isinstance(key, str):
                return key
            return tuple(key)

        for key, params in task_cfg.randomizations.items():
            randomization = RAND_FUNCS[key](self, **params)
            self._reset_callbacks.append(randomization.reset)
        # to flush the changes to the physics sim
        self.sim.physics_sim_view.flush()

        self.observation_funcs = OrderedDict()
        for group in task_cfg.observations:
            observation_group = OrderedDict()
            key = get_key(group["key"])
            for obs_name, params in group["items"].items():
                observation_group[obs_name] = OBS_FUNCS[obs_name](self, **params)
            self.observation_funcs[key] = observation_group
        
        obs = self._compute_observation()
        observation_spec = CompositeSpec({}, shape=[self.num_envs])
        for k, v in obs.items(True, True):
            observation_spec[k] = UnboundedContinuousTensorSpec(v.shape)
        self.observation_spec = observation_spec.to(self.device)

        self.action_funcs = OrderedDict()
        action_spec = CompositeSpec({}, shape=[self.num_envs])
        for group in task_cfg.actions:
            key = get_key(group["key"])
            action_group = OrderedDict()
            for act_name, params in group["items"].items():
                act_func = ACT_FUNCS[act_name](self, **params)
                self._debug_vis_callbacks.append(act_func.debug_vis)
                action_group[act_name] = act_func
            action_group = mdp.ActionGroup(action_group)
            self.action_funcs[key] = action_group
            action_spec[key] = UnboundedContinuousTensorSpec(action_group.action_shape)
        self.action_spec = action_spec.to(self.device)
        
        self.reward_funcs = OrderedDict()
        for key, params in task_cfg.rewards.items():
            self.reward_funcs[key] = REW_FUNCS[key](self, **params)
            reward_spec["stats", key] = UnboundedContinuousTensorSpec(1, device=self.device)
        if not len(self.reward_funcs):
            logging.warning("No reward functions specified. Using a default reward function of 1.0.")
            self.reward_funcs["_"] = lambda: torch.ones(self.num_envs, 1, device=self.device)
        self.reward_spec = reward_spec.expand(self.num_envs).to(self.device)
        self.stats = self.reward_spec["stats"].zero()

        self.termination_funcs = OrderedDict()
        for key, params in task_cfg.termination.items():
            self.termination_funcs[key] = TERM_FUNCS[key](self, **params)
        if not len(self.termination_funcs):
            logging.warning("No termination functions specified. Using a default termination function of False")
            self.termination_funcs["_"] = lambda: torch.zeros(self.num_envs, 1, dtype=bool, device=self.device)

        import pprint
        pprint.pprint(self.fake_tensordict().shapes)

    @property
    def num_envs(self) -> int:
        """The number of instances of the environment that are running."""
        return self.scene.num_envs
    
    @property
    def physics_dt(self) -> float:
        return self.sim.get_physics_dt()

    @classmethod
    def __init_subclass__(cls, **kwargs):
        if cls.__name__ in IsaacEnv.REGISTRY:
            raise ValueError
        super().__init_subclass__(**kwargs)
        if not cls.__name__.startswith("_"):
            IsaacEnv.REGISTRY[cls.__name__] = cls
            IsaacEnv.REGISTRY[cls.__name__.lower()] = cls

    @property
    def agent_spec(self):
        if not hasattr(self, "_agent_spec"):
            self._agent_spec = {}
        return _AgentSpecView(self)
    
    @agent_spec.setter
    def agent_spec(self, value):
        raise AttributeError(
            "Do not set agent_spec directly."
            "Use `self.agent_spec[agent_name] = AgentSpec(...)` instead."
        )

    @abc.abstractmethod
    def _design_scene(self):
        raise NotImplementedError

    def close(self):
        return # TODO: fix this
        if not self._is_closed:
            # stop physics simulation (precautionary)
            self.sim.stop()
            # cleanup the scene and callbacks
            self.sim.clear_all_callbacks()
            self.sim.clear()
            # fix warnings at stage close
            omni.usd.get_context().get_stage().GetRootLayer().Clear()
            # update closing status
            self._is_closed = True
            logging.info("IsaacEnv closed.")

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is not None:
            env_mask = tensordict.get("_reset").reshape(self.num_envs)
        else:
            env_mask = torch.ones(self.num_envs, dtype=bool, device=self.device)
        env_ids = env_mask.nonzero().squeeze(-1)
        self._reset_idx(env_ids)
        for callback in self._reset_callbacks:
            callback(env_ids)
        self.sim._physics_sim_view.flush()
        # self.scene.update(self.step_dt)
        self.progress_buf[env_ids] = 0.
        self.stats[env_ids] = 0.
        tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict.update(self._compute_observation())
        return tensordict

    @abc.abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor):
        raise NotImplementedError

    def update(self):
        pass

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        for substep in range(self.substeps):
            for group_name, action_group in self.action_funcs.items():
                action_group.apply_action(tensordict[group_name])
            self.scene.write_data_to_sim()
            self.sim.step(False)
        self.scene.update(self.step_dt)
        for callback in self._update_callbacks:
            callback()

        # perform rendering if gui is enabled
        if self.sim.has_gui():
            self.sim.render()
            self.debug_draw.clear()
            for callback in self._debug_vis_callbacks:
                callback()
        
        self.progress_buf += 1
        tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict.update(self._compute_observation())
        tensordict.update(self._compute_reward())
        terminated = self._compute_termination()
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(1)
        tensordict.set("terminated", terminated)
        tensordict.set("truncated", truncated)
        tensordict.set("done", terminated | truncated)
        return tensordict
    
    def _compute_observation(self):
        observation = TensorDict({}, [self.num_envs])
        for group, funcs in self.observation_funcs.items():
            tensor = torch.cat([func() for func in funcs.values()], dim=-1)
            observation[group] = tensor
        return observation

    def _compute_reward(self):
        reward = []
        for key, func in self.reward_funcs.items():
            r = func()
            self.stats[key].add_(r)
            reward.append(r)
        reward = sum(reward)
        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["success"][:] = (self.progress_buf >= self.max_episode_length * 0.9).unsqueeze(1).float()
        return {"reward": reward, "stats": self.stats.clone()}

    def _compute_termination(self):
        terminated = []
        for key, func in self.termination_funcs.items():
            terminated.append(func())
        terminated = torch.cat(terminated, dim=-1)
        return terminated.any(dim=-1, keepdim=True)

    def _set_seed(self, seed: Optional[int] = -1):
        import omni.replicator.core as rep
        rep.set_global_seed(seed)
        torch.manual_seed(seed)

    def _configure_simulation_flags(self, sim_params: dict = None):
        """Configure simulation flags and extensions at load and run time."""
        # acquire settings interface
        carb_settings_iface = carb.settings.get_settings()
        # enable hydra scene-graph instancing
        # note: this allows rendering of instanceable assets on the GUI
        carb_settings_iface.set_bool("/persistent/omnihydra/useSceneGraphInstancing", True)
        # change dispatcher to use the default dispatcher in PhysX SDK instead of carb tasking
        # note: dispatcher handles how threads are launched for multi-threaded physics
        carb_settings_iface.set_bool("/physics/physxDispatcher", True)
        # disable contact processing in omni.physx if requested
        # note: helpful when creating contact reporting over limited number of objects in the scene
        # if sim_params["disable_contact_processing"]:
        #     carb_settings_iface.set_bool("/physics/disableContactProcessing", True)

        # set flags based on whether rendering is enabled or not
        # note: enabling extensions is order-sensitive. please do not change the order.
        if self.enable_viewport:
            # enable scene querying if rendering is enabled
            # this is needed for some GUI features
            sim_params["enable_scene_query_support"] = True
            # load extra viewport extensions if requested
            if self.enable_viewport:
                # extension to enable UI buttons (otherwise we get attribute errors)
                enable_extension("omni.kit.window.toolbar")
                # extension to make RTX realtime and path-traced renderers
                enable_extension("omni.kit.viewport.rtx")
                # extension to make HydraDelegate renderers
                enable_extension("omni.kit.viewport.pxr")
            # enable viewport extension if not running in headless mode
            enable_extension("omni.kit.viewport.bundle")
            # load extra render extensions if requested
            if self.enable_viewport:
                # extension for window status bar
                enable_extension("omni.kit.window.status_bar")
        # enable isaac replicator extension
        # note: moved here since it requires to have the viewport extension to be enabled first.
        enable_extension("omni.replicator.isaac")

    def to(self, device) -> EnvBase:
        if torch.device(device) != self.device:
            raise RuntimeError(
                f"Cannot move IsaacEnv on {self.device} to a different device {device} once it's initialized."
            )
        return self

    def debug_vis(self):
        pass
    
    def enable_render(self, enable: Union[bool, Callable]=True):
        if isinstance(enable, bool):
            self._should_render = lambda substep: enable
        elif callable(enable):
            self._should_render = enable
        else:
            raise TypeError("enable_render must be a bool or callable.")
    
    def render(self, mode: str="human"):
        if mode == "human":
            return None
        elif mode == "rgb_array":
            # check if viewport is enabled -- if not, then complain because we won't get any data
            if not self.enable_viewport:
                raise RuntimeError(
                    f"Cannot render '{mode}' when enable viewport is False. Please check the provided"
                    "arguments to the environment class at initialization."
                )
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            return rgb_data[:, :, :3]
        else:
            raise NotImplementedError(
                f"Render mode '{mode}' is not supported. Please use: {self.metadata['render.modes']}."
            )

class _AgentSpecView(Dict[str, AgentSpec]):
    def __init__(self, env: IsaacEnv):
        super().__init__(env._agent_spec)
        self.env = env

    def __setitem__(self, k: str, v: AgentSpec) -> None:
        v._env = self.env
        return self.env._agent_spec.__setitem__(k, v)

