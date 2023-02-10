import torch

from typing import Optional, Dict
from torchrl.envs import EnvBase
from torchrl.data import TensorSpec, CompositeSpec
from tensordict.tensordict import TensorDict, TensorDictBase

import omni.usd
import omni.replicator.core as rep
from omni.isaac.cloner import GridCloner
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.carb import set_carb_setting
from omni.isaac.core.utils.extensions import disable_extension
from omni.isaac.core.utils.viewports import set_camera_view

from .utils import stage as stage_utils
from .utils import prims as prim_utils

class IsaacEnv(EnvBase):
    def __init__(self, cfg, headless):

        super().__init__(
            device=...,
            batch_size=...,
            run_type_checks=False
        )

        # store inputs to class
        self.cfg = cfg
        self.enable_render = not headless
        # extract commonly used parameters
        self.num_envs = self.cfg.env.num_envs
        self.device = self.cfg.sim.device
        self.physics_dt = self.cfg.sim.dt
        self.rendering_dt = self.cfg.sim.dt * self.cfg.sim.substeps

        # check that simulation is running
        if stage_utils.get_current_stage() is None:
            raise RuntimeError(
                "The stage has not been created. Did you run the simulator?"
            )
        # flatten out the simulation dictionary
        sim_params = self.cfg.sim.to_dict()
        if sim_params is not None:
            if "physx" in sim_params:
                physx_params = sim_params.pop("physx")
                sim_params.update(physx_params)

        self.sim = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=self.physics_dt,
            rendering_dt=self.rendering_dt,
            backend="torch",
            sim_params=sim_params,
            physics_prim_path="/physicsScene",
            device=self.device,
        )
        # set flags for simulator
        self._configure_simulation_flags(sim_params)
        # add flag for checking closing status
        self._is_closed = False
        # set camera view
        set_camera_view(eye=self.cfg.viewer.eye, target=self.cfg.viewer.lookat)
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
        self.envs_prim_paths = cloner.generate_paths(self.env_ns + "/env", self.num_envs)
        self.envs_positions = cloner.clone(
            source_prim_path=self.template_env_ns,
            prim_paths=self.envs_prim_paths,
            replicate_physics=self.cfg.sim.replicate_physics,
        )
        # convert environment positions to torch tensor
        self.envs_positions = torch.tensor(self.envs_positions, dtype=torch.float, device=self.device)
        # filter collisions within each environment instance
        physics_scene_path = self.sim.get_physics_context().prim_path
        cloner.filter_collisions(
            physics_scene_path, "/World/collisions", prim_paths=self.envs_prim_paths, global_paths=global_prim_paths
        )

        self._tensordict = TensorDict({}, self.batch_size)
        self.observation_spec = CompositeSpec(shape=self.batch_size)
        self.action_spec = CompositeSpec(shape=self.batch_size)
        self.reward_spec = CompositeSpec(shape=self.batch_size)
    
    @property
    def agent_spec(self):
        if not hasattr(self, "_agent_spec"):
            self._agent_spec = {}
        return _AgentSpecView(self)

    def close(self):
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
    
    def _set_seed(self, seed: Optional[int]=-1):
        rep.set_global_seed(seed)
    
    def _configure_simulation_flags(self, sim_params: dict = None):
        """Configure the various flags for performance.

        This function enables flat-cache for speeding up GPU pipeline, enables hydra scene-graph
        instancing for visualizing multiple instances when flatcache is enabled, and disables the
        viewport if running in headless mode.
        """
        # enable flat-cache for speeding up GPU pipeline
        if self.sim.get_physics_context().use_gpu_pipeline:
            self.sim.get_physics_context().enable_flatcache(True)
        # enable hydra scene-graph instancing
        # Think: Create your own carb-settings instance?
        set_carb_setting(self.sim._settings, "/persistent/omnihydra/useSceneGraphInstancing", True)
        # check viewport settings
        if sim_params and "enable_viewport" in sim_params:
            # if viewport is disabled, then don't create a window (minor speedups)
            if not sim_params["enable_viewport"]:
                disable_extension("omni.kit.viewport.window")


from dataclasses import dataclass

@dataclass
class AgentSpec:
    name: str
    n: int
    observation_spec: TensorSpec
    action_spec: TensorSpec
    reward_spec: TensorSpec

class _AgentSpecView(Dict[str, AgentSpec]):
    def __init__(self, env: IsaacEnv):
        super().__init__(env._agent_spec)
        self.env = env
    
    def __setitem__(self, __key, __value) -> None:
        if __key in self:
            raise ValueError(f"Can not set agent_spec with duplicated name {__key}.")
        if isinstance(__value, AgentSpec):
            super().__setitem__(__key, __value)
            name = __value.name
            def expand(spec: TensorSpec) -> TensorSpec:
                return spec.expand(*self.env.batch_size,  *spec.shape)
            self.env.observation_spec[f"{name}.obs"] = expand(__value.observation_spec)
            self.env.action_spec[f"{name}.action"] = expand(__value.action_spec)
            self.env.reward_spec[f"{name}.reward"] = expand(__value.reward_spec)
            self.env._tensordict[f"{name}.return"] = __value.reward_spec()
        else:
            raise TypeError
