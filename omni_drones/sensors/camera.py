from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.simulation_context import SimulationContext
from tensordict import TensorDict
from pxr import UsdGeom, Sdf, Gf

import torch
import warp as wp
import omni.replicator.core as rep
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils

from typing import Union, Optional
from omni_drones.utils.math import quaternion_to_euler
from .config import PinholeCameraCfg, FisheyeCameraCfg

class Camera:
    """
    Viewport camera used for visualization purpose.
    """
    def __init__(
        self,
        cfg: Union[PinholeCameraCfg, FisheyeCameraCfg]=None,
        parent_prim_path: str="/World",
        translation = None,
        orientation = None,
        target = None
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
                    clipping_range=(0.1, 1.0e5)
                )
            )
        self.cfg = cfg
        self.resolution = cfg.resolution
        self.shape = (self.resolution[1], self.resolution[0])
        prim_path = stage_utils.get_next_free_path(f"{parent_prim_path}/Camera")
        self.prim = UsdGeom.Camera(prim_utils.define_prim(prim_path, "Camera"))
        self.prim_path = prim_utils.get_prim_path(self.prim)
        self._define_usd_camera_attributes()
        self.xform = XFormPrim(self.prim_path, translation=translation)
        self.xform.initialize()
        self.set_local_pose(translation, orientation, target)

        self.device = SimulationContext.instance().device
        if isinstance(self.device, str) and "cuda" in self.device: 
            self.device = self.device.split(":")[0]

        self.render_product = rep.create.render_product(
            self.prim_path, resolution=self.resolution
        )
        self.annotators = {}
        for annotator_type in cfg.data_types:
            annotator = rep.AnnotatorRegistry.get_annotator(
                name=annotator_type, device=self.device)
            annotator.attach([self.render_product])
            self.annotators[annotator_type] = annotator
        SimulationContext.instance().render()

    def __call__(self) -> TensorDict:
        tensordict = TensorDict({
            k: wp.to_torch(v.get_data(device=self.device)) 
            for k, v in self.annotators.items()
        }, self.shape)
        return tensordict 

    def set_local_pose(self, translation, orientation=None, target=None):
        if target is not None:
            if orientation is not None:
                raise ValueError
            eye_position = Gf.Vec3d(translation)
            target_position = Gf.Vec3d(target)
            up_axis = Gf.Vec3d(0, 0, 1)
            matrix_gf = Gf.Matrix4d(1).SetLookAt(eye_position, target_position, up_axis)
            matrix_gf = matrix_gf.GetInverse()
            quat = matrix_gf.ExtractRotationQuat()
            orientation = [quat.real, *quat.imaginary]
            # orientation = orientation_from_view(target, translation)

        self.xform.set_local_pose(translation, orientation)

    def _define_usd_camera_attributes(self):
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
        prim = prim_utils.get_prim_at_path(self.prim_path)
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
    quat = lookat_to_quatf(
        Gf.Vec3f(camera), Gf.Vec3f(target), Gf.Vec3f(0., 0., 1.)
    )
    return [quat.real, *quat.imaginary]


import math
def lookat_to_quatf(camera: Gf.Vec3f, target: Gf.Vec3f, up: Gf.Vec3f) -> Gf.Quatf:
    """[summary]

    Args:
        camera (Gf.Vec3f): [description]
        target (Gf.Vec3f): [description]
        up (Gf.Vec3f): [description]

    Returns:
        Gf.Quatf: Pxr quaternion object.
    """
    F = (target - camera).GetNormalized()
    R = Gf.Cross(up, F).GetNormalized()
    U = Gf.Cross(F, R)

    q = Gf.Quatf()
    trace = R[0] + U[1] + F[2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q = Gf.Quatf(0.25 / s, Gf.Vec3f((U[2] - F[1]) * s, (F[0] - R[2]) * s, (R[1] - U[0]) * s))
    else:
        if R[0] > U[1] and R[0] > F[2]:
            s = 2.0 * math.sqrt(1.0 + R[0] - U[1] - F[2])
            q = Gf.Quatf((U[2] - F[1]) / s, Gf.Vec3f(0.25 * s, (U[0] + R[1]) / s, (F[0] + R[2]) / s))
        elif U[1] > F[2]:
            s = 2.0 * math.sqrt(1.0 + U[1] - R[0] - F[2])
            q = Gf.Quatf((F[0] - R[2]) / s, Gf.Vec3f((U[0] + R[1]) / s, 0.25 * s, (F[1] + U[2]) / s))
        else:
            s = 2.0 * math.sqrt(1.0 + F[2] - R[0] - U[1])
            q = Gf.Quatf((R[1] - U[0]) / s, Gf.Vec3f((F[0] + R[2]) / s, (F[1] + U[2]) / s, 0.25 * s))
    return q

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

