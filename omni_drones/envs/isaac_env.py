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
from isaacsim.core.cloner import GridCloner
from isaacsim.core.api.simulation_context import SimulationContext
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import isaacsim.core.utils.extensions as _ext_mod
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.util.debug_draw import _debug_draw

def enable_extension(name):
    return _ext_mod.enable_extension(name)

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Composite, TensorSpec, DiscreteTensorSpec
from torchrl.envs import EnvBase

from omni_drones.robots.robot import RobotBase
from omni_drones.utils.torchrl import AgentSpec

class DebugDraw:
    def __init__(self):
        self._draw = _debug_draw.acquire_debug_draw_interface()

    def clear(self):
        self._draw.clear_lines()

    def plot(self, x: torch.Tensor, size=2.0, color=(1., 1., 1., 1.)):
        if not (x.ndim == 2) and (x.shape[1] == 3):
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


class IsaacEnv(EnvBase):

    env_ns = "/World/envs"
    template_env_ns = "/World/envs/env_0"

    REGISTRY: Dict[str, Type["IsaacEnv"]] = {}

    def __init__(self, cfg, headless):
        super().__init__(
            device=cfg.sim.device, batch_size=[cfg.env.num_envs], run_type_checks=False
        )
        # store inputs to class
        self.cfg = cfg
        self.enable_render(not headless)
        self.enable_viewport = not headless
        # extract commonly used parameters
        self.num_envs = self.cfg.env.num_envs
        self.max_episode_length = self.cfg.env.max_episode_length
        self.substeps = self.cfg.sim.substeps

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # check that simulation is running
        if stage_utils.get_current_stage() is None:
            raise RuntimeError(
                "The stage has not been created. Did you run the simulator?"
            )
        # flatten out the simulation dictionary
        sim_params = self.cfg.sim
        if sim_params is not None:
            if "physx" in sim_params:
                physx_params = sim_params.pop("physx")
                sim_params.update(physx_params)
        # set flags for simulator
        self._configure_simulation_flags(sim_params)
        self.sim = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=self.cfg.sim.dt,
            rendering_dt=self.cfg.sim.dt, # * self.cfg.sim.substeps,
            backend="torch",
            sim_params=sim_params,
            physics_prim_path="/physicsScene",
            device=cfg.sim.device,
        )
        self._create_viewport_render_product()
        self.dt = self.sim.get_physics_dt()
        # add flag for checking closing status
        self._is_closed = False
        # set camera view
        # create cloner for duplicating the scenes
        cloner = GridCloner(spacing=self.cfg.env.env_spacing)
        cloner.define_base_env("/World/envs")
        # create the xform prim to hold the template environment
        if not prim_utils.is_prim_path_valid(self.template_env_ns):
            prim_utils.define_prim(self.template_env_ns)
        # setup single scene
        global_prim_paths = self._design_scene()
        # check if any global prim paths are defined
        if global_prim_paths is None:
            global_prim_paths = list()
        # clone the scenes into the namespace "/World/envs" based on template namespace
        self.envs_prim_paths = cloner.generate_paths(
            self.env_ns + "/env", self.num_envs
        )
        assert len(self.envs_prim_paths) == self.num_envs
        self.envs_positions = cloner.clone(
            source_prim_path=self.template_env_ns,
            prim_paths=self.envs_prim_paths,
            replicate_physics=self.cfg.sim.replicate_physics,
        )
        # convert environment positions to torch tensor
        self.envs_positions = torch.tensor(
            self.envs_positions, dtype=torch.float, device=self.device
        )
        # find the environment closest to the origin for visualization
        self.central_env_idx = self.envs_positions.norm(dim=-1).argmin()
        central_env_pos = self.envs_positions[self.central_env_idx].cpu().numpy()
        if self.enable_viewport:
            set_camera_view(
                eye=central_env_pos + np.asarray(self.cfg.viewer.eye),
                target=central_env_pos + np.asarray(self.cfg.viewer.lookat)
            )

        RobotBase._envs_positions = self.envs_positions.unsqueeze(1)

        # filter collisions within each environment instance
        physics_scene_path = self.sim.get_physics_context().prim_path
        cloner.filter_collisions(
            physics_scene_path,
            "/World/collisions",
            prim_paths=self.envs_prim_paths,
            global_paths=global_prim_paths,
        )
        self.sim.reset()
        self.debug_draw = DebugDraw()

        self._tensordict = TensorDict(
            {
                "progress": torch.zeros(self.num_envs, device=self.device),
            },
            self.batch_size,
        )
        self.progress_buf = self._tensordict["progress"]
        self.done_spec = Composite({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device)
        self._set_specs()
        import pprint
        pprint.pprint(self.fake_tensordict().shapes)


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
    def _set_specs(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _design_scene(self) -> Optional[List[str]]:
        """Creates the template environment scene.

        All prims under the *template namespace* will be duplicated across the
        stage and collisions between the duplicates will be filtered out. In case,
        there are any prims which need to be a common collider across all the
        environments, they should be returned as a list of prim paths. These could
        be prims like the ground plane, walls, etc.

        Returns:
            Optional[List[str]]: List of prim paths which are common across all the
                environments and need to be considered for common collision filtering.
        """
        raise NotImplementedError

    def close(self):
        if not self._is_closed:
            try:
                import omni.timeline
                timeline = omni.timeline.get_timeline_interface()
                if timeline:
                    timeline.stop()
            except Exception as e:
                logging.warning(f"Failed to stop timeline: {e}")
            
            try:
                if hasattr(self, 'sim') and self.sim is not None:
                    self.sim.stop()
                    if hasattr(self.sim, 'clear_all_callbacks'):
                        self.sim.clear_all_callbacks()
                    if hasattr(self.sim, 'clear'):
                        self.sim.clear()
            except Exception as e:
                logging.warning(f"Failed to stop/clear simulation: {e}")
            
            self._is_closed = True
            logging.info("IsaacEnv closed.")

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is not None:
            env_mask = tensordict.get("_reset").reshape(self.num_envs)
        else:
            env_mask = torch.ones(self.num_envs, dtype=bool, device=self.device)
        env_ids = env_mask.nonzero().squeeze(-1)
        
        self._reset_idx(env_ids)
        
        self.progress_buf[env_ids] = 0.
        tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict.update(self._compute_state_and_obs())
        tensordict.set("truncated", (self.progress_buf > self.max_episode_length).unsqueeze(1))
        return tensordict

    @abc.abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor):
        raise NotImplementedError

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        for substep in range(self.substeps):
            self._pre_sim_step(tensordict)
            self.sim.step(self._should_render(substep))

        self._post_sim_step(tensordict)
        self.progress_buf += 1

        tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict.update(self._compute_state_and_obs())
        tensordict.update(self._compute_reward_and_done())
        return tensordict

    def _pre_sim_step(self, tensordict: TensorDictBase):
        pass

    def _post_sim_step(self, tensordict: TensorDictBase):
        pass

    @abc.abstractmethod
    def _compute_state_and_obs(self) -> TensorDictBase:
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_reward_and_done(self) -> TensorDictBase:
        raise NotImplementedError

    def _set_seed(self, seed: Optional[int] = -1):
        # Only set replicator global seed if replicator stack is enabled
        if getattr(self.cfg.sim, "enable_replicator", False):
            try:
                import omni.replicator.core as rep
                rep.set_global_seed(seed)
            except ImportError:
                pass
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
            # extension to enable UI buttons (otherwise we get attribute errors)
            enable_extension("omni.kit.window.toolbar")
            # viewport utility helpers (required by some viewport APIs)
            enable_extension("omni.kit.viewport.utility")
            # extension to make RTX realtime and path-traced renderers
            enable_extension("omni.kit.viewport.rtx")
            # extension to make HydraDelegate renderers
            enable_extension("omni.kit.viewport.pxr")
            # enable viewport bundle when not in headless mode
            enable_extension("omni.kit.viewport.bundle")
            # extension for window status bar
            enable_extension("omni.kit.window.status_bar")
            # enable replicator extensions only if requested
            if getattr(self.cfg.sim, "enable_replicator", False):
                enable_extension("isaacsim.replicator.domain_randomization")
                enable_extension("isaacsim.replicator.examples")
                enable_extension("isaacsim.replicator.writers")

    def to(self, device) -> EnvBase:
        if torch.device(device) != self.device:
            raise RuntimeError(
                f"Cannot move IsaacEnv on {self.device} to a different device {device} once it's initialized."
            )
        return self

    def get_env_poses(self, world_poses: Tuple[torch.Tensor, torch.Tensor]):
        pos, rot = world_poses
        if pos.dim() == 3:
            return pos - self.envs_positions.unsqueeze(1), rot
        else:
            return pos - self.envs_positions, rot

    def get_world_poses(self, env_poses: Tuple[torch.Tensor, torch.Tensor]):
        pos, rot = env_poses
        if pos.dim() == 3:
            return pos + self.envs_positions.unsqueeze(1), rot
        else:
            return pos + self.envs_positions, rot

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
            # require replicator to be enabled to fetch rgb arrays
            if not getattr(self.cfg.sim, "enable_replicator", False) or not hasattr(self, "_rgb_annotator") or self._rgb_annotator is None:
                raise RuntimeError(
                    "RGB rendering requires Replicator. Set cfg.sim.enable_replicator=True to enable it."
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

    def _create_viewport_render_product(self):
        """Create a render product of the viewport for rendering."""
        # set camera view for "/OmniverseKit_Persp" camera
        if self.enable_viewport:
            set_camera_view(eye=self.cfg.viewer.eye, target=self.cfg.viewer.lookat)

        # check if flatcache is enabled
        # this is needed to flush the flatcache data into Hydra manually when calling `env.render()`
        # ref: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_physics.html
        # if  self.sim.get_physics_context().use_flatcache:
        #     from omni.physxflatcache import get_physx_flatcache_interface

        #     # acquire flatcache interface
        #     self._flatcache_iface = get_physx_flatcache_interface()

        # check if viewport is enabled before creating render product
        if self.enable_viewport:
            if getattr(self.cfg.sim, "enable_replicator", False):
                import omni.replicator.core as rep
                # create render product
                self._render_product = rep.create.render_product(
                    "/OmniverseKit_Persp", tuple(self.cfg.viewer.resolution)
                )
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])
            else:
                # leave render product uninitialized to avoid loading synthetic-data stack
                self._render_product = None
                self._rgb_annotator = None
        else:
            carb.log_info("Viewport is disabled. Skipping creation of render product.")


class _AgentSpecView(Dict[str, AgentSpec]):
    def __init__(self, env: IsaacEnv):
        super().__init__(env._agent_spec)
        self.env = env

    def __setitem__(self, k: str, v: AgentSpec) -> None:
        v._env = self.env
        return self.env._agent_spec.__setitem__(k, v)
