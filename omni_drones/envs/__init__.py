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


from .single import Hover, Track
# from .platform import PlatformHover, PlatformFlyThrough
# from .inv_pendulum import InvPendulumHover, InvPendulumFlyThrough
# from .transport import TransportHover, TransportFlyThrough, TransportTrack
# from .formation import Formation
from .payload import PayloadTrack, PayloadFlyThrough
# from .dragon import DragonHover
# from .rearrange import Rearrange
from .isaac_env import IsaacEnv

try:
    from .pinball import Pinball
    from .forest import Forest
except ModuleNotFoundError:
    print(
        "To run the environments which use `ContactSensor` and `RayCaster`,"
        "please install Isaac lab (https://github.com/NVIDIA-Omniverse/lab)."
    )

from omni.isaac.lab.scene.interactive_scene import *


def _add_entities_from_cfg(self):
    """Add scene entities from the config."""
    # store paths that are in global collision filter
    self._global_prim_paths = list()
    # parse the entire scene config and resolve regex
    for asset_name, asset_cfg in self.cfg.__dict__.items():
        # skip keywords
        # note: easier than writing a list of keywords: [num_envs, env_spacing, lazy_sensor_update]
        if asset_name in InteractiveSceneCfg.__dataclass_fields__ or asset_cfg is None:
            continue
        # resolve regex
        if isinstance(asset_cfg.prim_path, str):
            asset_cfg.prim_path = asset_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
        elif isinstance(asset_cfg.prim_path, list) and all(isinstance(x, str) for x in asset_cfg.prim_path):
            asset_cfg.prim_path = [x.format(ENV_REGEX_NS=self.env_regex_ns) for x in asset_cfg.prim_path]
        # create asset
        if isinstance(asset_cfg, TerrainImporterCfg):
            # terrains are special entities since they define environment origins
            asset_cfg.num_envs = self.cfg.num_envs
            asset_cfg.env_spacing = self.cfg.env_spacing
            self._terrain = asset_cfg.class_type(asset_cfg)
        elif isinstance(asset_cfg, ArticulationCfg):
            self._articulations[asset_name] = asset_cfg.class_type(asset_cfg)
        elif isinstance(asset_cfg, DeformableObjectCfg):
            self._deformable_objects[asset_name] = asset_cfg.class_type(asset_cfg)
        elif isinstance(asset_cfg, RigidObjectCfg):
            self._rigid_objects[asset_name] = asset_cfg.class_type(asset_cfg)
        elif isinstance(asset_cfg, SensorBaseCfg):
            # Update target frame path(s)' regex name space for FrameTransformer
            if isinstance(asset_cfg, FrameTransformerCfg):
                updated_target_frames = []
                for target_frame in asset_cfg.target_frames:
                    target_frame.prim_path = target_frame.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                    updated_target_frames.append(target_frame)
                asset_cfg.target_frames = updated_target_frames
            elif isinstance(asset_cfg, ContactSensorCfg):
                updated_filter_prim_paths_expr = []
                for filter_prim_path in asset_cfg.filter_prim_paths_expr:
                    updated_filter_prim_paths_expr.append(filter_prim_path.format(ENV_REGEX_NS=self.env_regex_ns))
                asset_cfg.filter_prim_paths_expr = updated_filter_prim_paths_expr

            self._sensors[asset_name] = asset_cfg.class_type(asset_cfg)
        elif isinstance(asset_cfg, AssetBaseCfg):
            # manually spawn asset
            if asset_cfg.spawn is not None:
                asset_cfg.spawn.func(
                    asset_cfg.prim_path,
                    asset_cfg.spawn,
                    translation=asset_cfg.init_state.pos,
                    orientation=asset_cfg.init_state.rot,
                )
            # store xform prim view corresponding to this asset
            # all prims in the scene are Xform prims (i.e. have a transform component)
            self._extras[asset_name] = XFormPrimView(asset_cfg.prim_path, reset_xform_properties=False)
        else:
            raise ValueError(f"Unknown asset config type for {asset_name}: {asset_cfg}")
        # store global collision paths
        if hasattr(asset_cfg, "collision_group") and asset_cfg.collision_group == -1:
            asset_paths = sim_utils.find_matching_prim_paths(asset_cfg.prim_path)
            self._global_prim_paths += asset_paths

InteractiveScene._add_entities_from_cfg = _add_entities_from_cfg

