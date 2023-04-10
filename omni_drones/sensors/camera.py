from typing import Optional, Union, Sequence

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.replicator.core as rep

import torch
import warp as wp
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.simulation_context import SimulationContext
from pxr import Gf, Sdf, UsdGeom
from tensordict import TensorDict

from omni_drones.utils.math import quaternion_to_euler
from .config import FisheyeCameraCfg, PinholeCameraCfg

class Camera:
    """
    Viewport camera used for visualization purpose.
    """

    def __init__(
        self,
        cfg: Union[PinholeCameraCfg, FisheyeCameraCfg] = None
    ) -> None:
        if cfg is None:
            cfg = PinholeCameraCfg(
                sensor_tick=0,
                resolution=(640, 480),
                data_types=["rgb"],
                usd_params=PinholeCameraCfg.UsdCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 1.0e5),
                ),
            )
        self.cfg = cfg
        self.resolution = cfg.resolution
        self.shape = (self.resolution[1], self.resolution[0])
        self.sim = SimulationContext.instance()
        self.device = self.sim.device

        self.n = 0

        if isinstance(self.device, str) and "cuda" in self.device:
            self.device = self.device.split(":")[0]
        self.annotators = []

    def spawn(
        self, 
        prim_paths: Sequence[str],
        translations=None,
        targets=None,
    ):
        n = len(prim_paths)

        if translations is None:
            translations = [(0, 0, 0) for _ in range(n)]
        translations = torch.atleast_2d(torch.as_tensor(translations)).expand(n, 3).tolist()

        if targets is None:
            targets = [(1, 0, 0) for _ in range(n)]
        targets = torch.atleast_2d(torch.as_tensor(targets)).expand(n, 3).tolist()

        if not len(translations) == len(prim_paths) == len(targets):
            raise ValueError
        
        for prim_path, translation, target in zip(prim_paths, translations, targets):
            if prim_utils.is_prim_path_valid(prim_path):
                raise RuntimeError(f"Duplicate prim at {prim_path}.")
            prim = prim_utils.create_prim(
                prim_path,
                prim_type="Camera",
                translation=translation,
                orientation=orientation_from_view(translation, target)
            )
            self._define_usd_camera_attributes(prim_path)

        self.n += n

    def initialize(
        self, 
        prim_paths_expr: str = None,
    ):
        if prim_paths_expr is None:
            prim_paths_expr = f"/World/envs/.*/Camera_.*"

        prim_paths = prim_utils.find_matching_prim_paths(prim_paths_expr)
        
        for prim_path in prim_paths:
            render_product = rep.create.render_product(
                prim_path, resolution=self.resolution
            )
            annotators = {}
            for annotator_type in self.cfg.data_types:
                annotator = rep.AnnotatorRegistry.get_annotator(
                    name=annotator_type, device=self.device
                )
                annotator.attach([render_product])
                annotators[annotator_type] = annotator

            self.annotators.append(annotators)
        
        self.count = len(prim_paths)
    
        for _ in range(2):
            self.sim.render()
        
    def get_images(self) -> TensorDict:
        return torch.stack([TensorDict({
            k: wp.to_torch(v.get_data(device=self.device)).permute(2, 0, 1)
            for k, v in annotators.items()
        }, []) for annotators in self.annotators])

    def _define_usd_camera_attributes(self, prim_path):
        """Creates and sets USD camera attributes.

        This function creates additional attributes on the camera prim used by Replicator.
        It also sets the default values for these attributes based on the camera configuration.
        """
        # camera attributes
        # reference: omni.replicator.core.scripts.create.py: camera()
        attribute_types = {
            "cameraProjectionType": "token",
            "fthetaWidth": "float",
            "fthetaHeight": "float",
            "fthetaCx": "float",
            "fthetaCy": "float",
            "fthetaMaxFov": "float",
            "fthetaPolyA": "float",
            "fthetaPolyB": "float",
            "fthetaPolyC": "float",
            "fthetaPolyD": "float",
            "fthetaPolyE": "float",
        }
        # get camera prim
        prim = prim_utils.get_prim_at_path(prim_path)
        # create attributes
        for attr_name, attr_type in attribute_types.items():
            # check if attribute does not exist
            if prim.GetAttribute(attr_name).Get() is None:
                # create attribute based on type
                if attr_type == "token":
                    prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Token)
                elif attr_type == "float":
                    prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
        # set attribute values
        # -- projection type
        projection_type = to_camel_case(self.cfg.projection_type, to="cC")
        prim.GetAttribute("cameraProjectionType").Set(projection_type)
        # -- other attributes
        for param_name, param_value in self.cfg.usd_params.__dict__.items():
            # check if value is valid
            if param_value is None:
                continue
            # convert to camel case (CC)
            param = to_camel_case(param_name, to="cC")
            # get attribute from the class
            prim.GetAttribute(param).Set(param_value)


def orientation_from_view(camera, target):
    camera_position = Gf.Vec3d(camera)
    target_position = Gf.Vec3d(target)
    up_axis = Gf.Vec3d(0, 0, 1)
    matrix_gf = Gf.Matrix4d(1).SetLookAt(camera_position, target_position, up_axis)
    matrix_gf = matrix_gf.GetInverse()
    quat = matrix_gf.ExtractRotationQuat()
    orientation = (quat.real, *quat.imaginary)
    return orientation


def to_camel_case(snake_str: str, to: Optional[str] = "cC") -> str:
    """Converts a string from snake case to camel case.

    Args:
        snake_str (str): A string in snake case (i.e. with '_')
        to (Optional[str], optional): Convention to convert string to. Defaults to "cC".

    Raises:
        ValueError: Invalid input argument `to`, i.e. not "cC" or "CC".

    Returns:
        str: A string in camel-case format.
    """
    # check input is correct
    if to not in ["cC", "CC"]:
        msg = "to_camel_case(): Choose a valid `to` argument (CC or cC)"
        raise ValueError(msg)
    # convert string to lower case and split
    components = snake_str.lower().split("_")
    if to == "cC":
        # We capitalize the first letter of each component except the first one
        # with the 'title' method and join them together.
        return components[0] + "".join(x.title() for x in components[1:])
    else:
        # Capitalize first letter in all the components
        return "".join(x.title() for x in components)
