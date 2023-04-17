import abc

from typing import Dict, List, Optional, Tuple, Type, Union, Callable

import omni.usd
import torch
import logging
import carb
from omni.isaac.cloner import GridCloner
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils import prims as prim_utils, stage as stage_utils
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.viewports import set_camera_view

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, TensorSpec, DiscreteTensorSpec
from torchrl.envs import EnvBase

from omni_drones.robots.robot import RobotBase
from omni_drones.sensors.camera import PinholeCameraCfg
from omni_drones.utils.torchrl import AgentSpec


class IsaacEnv(EnvBase):

    env_ns = "/World/envs"
    template_env_ns = "/World/envs/env_0"

    REGISTRY: Dict[str, Type["IsaacEnv"]] = {}

    _DEFAULT_CAMERA_CONFIG = {
        "cfg": PinholeCameraCfg(
            sensor_tick=0,
            resolution=(640, 480),
            data_types=["rgb"],
            usd_params=PinholeCameraCfg.UsdCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
        ),
        "parent_prim_path": "/World",
        "translation": (4.0, 2.0, 3.0),
        "target": (0.0, 0.0, 1.0),
    }

    def __init__(self, cfg, headless):
        super().__init__(
            device=cfg.sim.device, batch_size=[cfg.env.num_envs], run_type_checks=False
        )
        # store inputs to class
        self.cfg = cfg
        self.enable_render(not headless)
        # extract commonly used parameters
        self.num_envs = self.cfg.env.num_envs
        self.max_episode_length = self.cfg.env.max_episode_length
        self.min_episode_length = self.cfg.env.min_episode_length

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

        self.sim = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=self.cfg.sim.dt,
            rendering_dt=self.cfg.sim.dt * self.cfg.sim.substeps,
            backend="torch",
            sim_params=sim_params,
            # physics_prim_path="/physicsScene",
            device="cuda:0",
        )
        self.dt = self.sim.get_physics_dt()
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

        self._tensordict = TensorDict(
            {
                "progress": torch.zeros(self.num_envs, device=self.device),
            },
            self.batch_size,
        )
        self.progress_buf = self._tensordict["progress"]
        self.observation_spec = CompositeSpec(shape=self.batch_size)
        self.action_spec = CompositeSpec(shape=self.batch_size)
        self.reward_spec = CompositeSpec(shape=self.batch_size)
        self.done_spec = DiscreteTensorSpec(
            n=2, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device
        )

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

    @property
    def DEFAULT_CAMERA_CONFIG(self):
        import copy

        return copy.deepcopy(self._DEFAULT_CAMERA_CONFIG)

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
            env_mask = tensordict.get("_reset").squeeze()
        else:
            env_mask = torch.ones(self.num_envs, dtype=bool, device=self.device)
        env_ids = env_mask.nonzero().squeeze(-1)
        self._reset_idx(env_ids)
        # self.sim.step(render=False)
        self.sim._physics_sim_view.flush()
        self._tensordict.masked_fill_(env_mask, 0)
        return self._tensordict.update(self._compute_state_and_obs())

    @abc.abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor):
        raise NotImplementedError

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self._pre_sim_step(tensordict)
        for substep in range(1):
            self.sim.step(self._should_render(substep))
        self.progress_buf += 1
        tensordict = TensorDict({"next": {}}, self.batch_size)
        tensordict["next"].update(self._compute_state_and_obs())
        tensordict["next"].update(self._compute_reward_and_done())
        return tensordict

    def _pre_sim_step(self, tensordict: TensorDictBase):
        pass

    @abc.abstractmethod
    def _compute_state_and_obs(self) -> TensorDictBase:
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_reward_and_done(self) -> TensorDictBase:
        raise NotImplementedError

    def _set_seed(self, seed: Optional[int] = -1):
        import omni.replicator.core as rep
        rep.set_global_seed(seed)
        torch.manual_seed(seed)

    def _configure_simulation_flags(self, sim_params: dict = None):
        """Configure various simulation flags for performance improvements at load and run time."""
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

        # enable scene querying if rendering is enabled
        # this is needed for some GUI features
        sim_params["enable_scene_query_support"] = True
        # enable viewport extension if not running in headless mode
        enable_extension("omni.kit.viewport.bundle")
        # enable isaac replicator extension
        # note: moved here since it requires to have the viewport extension to be enabled first.
        enable_extension("omni.replicator.isaac")

    def to(self, device) -> EnvBase:
        if torch.device(device) != self.device:
            raise RuntimeError(
                "Cannot move IsaacEnv to a different device once it's initialized."
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


class _AgentSpecView(Dict[str, AgentSpec]):
    def __init__(self, env: IsaacEnv):
        super().__init__(env._agent_spec)
        self.env = env

    def __setitem__(self, __key, __value) -> None:
        if isinstance(__value, AgentSpec):
            if __key in self:
                raise ValueError(
                    f"Can not set agent_spec with duplicated name {__key}."
                )
            name = __value.name

            def expand(spec: TensorSpec) -> TensorSpec:
                return spec.expand(*self.env.batch_size, __value.n, *spec.shape)

            self.env.observation_spec[f"{name}.obs"] = expand(__value.observation_spec)
            if __value.state_spec is not None:
                shape = (*self.env.batch_size, *__value.state_spec.shape)
                self.env.observation_spec[f"{name}.state"] = __value.state_spec.expand(
                    *shape
                )
            self.env.action_spec[f"{name}.action"] = expand(__value.action_spec)
            self.env.reward_spec[f"{name}.reward"] = expand(__value.reward_spec)

            self.env._tensordict["return"] = self.env.reward_spec[
                f"{name}.reward"
            ].zero()
            super().__setitem__(__key, __value)
            self.env._agent_spec[__key] = __value
        else:
            raise TypeError

