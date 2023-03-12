from dataclasses import dataclass, field, MISSING
from typing import List, Tuple


@dataclass
class PinholeCameraCfg:
    """Configuration for a pinhole camera sensor."""

    @dataclass
    class UsdCameraCfg:
        """USD related configuration of the sensor.

        The parameter is kept default from USD if it is set to :obj:`None`. This includes the default
        parameters (in case the sensor is created) or the ones set by the user (in case the sensor is
        loaded from existing USD stage).

        Reference:
            * https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator/annotators_details.html
            * https://graphics.pixar.com/usd/docs/api/class_usd_geom_camera.html
        """

        clipping_range: Tuple[float, float] = None
        """Near and far clipping distances (in stage units)."""
        focal_length: float = None
        """Perspective focal length (in mm). Longer lens lengths narrower FOV, shorter lens lengths wider FOV."""
        focus_distance: float = None
        """Distance from the camera to the focus plane (in stage units).

        The distance at which perfect sharpness is achieved.
        """
        f_stop: float = None
        """Lens aperture. Defaults to 0.0, which turns off focusing.

        Controls Distance Blurring. Lower Numbers decrease focus range, larger numbers increase it.
        """
        horizontal_aperture: float = None
        """Horizontal aperture (in mm). Emulates sensor/film width on a camera."""
        horizontal_aperture_offset: float = None
        """Offsets Resolution/Film gate horizontally."""
        vertical_aperture_offset: float = None
        """Offsets Resolution/Film gate vertically."""

    sensor_tick: float = 0.0
    """Simulation seconds between sensor buffers. Defaults to 0.0."""
    data_types: List[str] = field(default_factory=lambda: ["rgb"])
    """List of sensor names/types to enable for the camera. Defaults to ["rgb"]."""
    resolution: Tuple[int, int] = (640, 480)
    """Image resolution."""
    semantic_types: List[str] = field(default_factory=lambda: ["class"])
    """List of allowed semantic types the types. Defaults to ["class"].

    For example, if semantic types is [“class”], only the bounding boxes for prims with semantics of
    type “class” will be retrieved.

    More information available at:
        https://docs.omniverse.nvidia.com/app_code/prod_extensions/ext_replicator/semantic_schema_editor.html
    """
    projection_type: str = "pinhole"
    """Type of projection to use for the camera. Defaults to "pinhole"."""
    usd_params: UsdCameraCfg = UsdCameraCfg()
    """Parameters for setting USD camera settings."""


@dataclass
class FisheyeCameraCfg(PinholeCameraCfg):
    """Configuration for a fisheye camera sensor."""

    @dataclass
    class UsdCameraCfg(PinholeCameraCfg.UsdCameraCfg):

        fisheye_nominal_width: float = None
        """Nominal width of fisheye lens model."""
        fisheye_nominal_height: float = None
        """Nominal height of fisheye lens model."""
        fisheye_optical_centre_x: float = None
        """Horizontal optical centre position of fisheye lens model."""
        fisheye_optical_centre_y: float = None
        """Vertical optical centre position of fisheye lens model."""
        fisheye_max_fov: float = None
        """Maximum field of view of fisheye lens model."""
        fisheye_polynomial_a: float = None
        """First component of fisheye polynomial."""
        fisheye_polynomial_b: float = None
        """Second component of fisheye polynomial."""
        fisheye_polynomial_c: float = None
        """Third component of fisheye polynomial."""
        fisheye_polynomial_d: float = None
        """Fourth component of fisheye polynomial."""
        fisheye_polynomial_e: float = None
        """Fifth component of fisheye polynomial."""

    projection_type: str = "fisheye_polynomial"
    """Type of projection to use for the camera. Defaults to "fisheye_polynomial"."""
    usd_params: UsdCameraCfg = UsdCameraCfg()
    """Parameters for setting USD camera settings."""


def class_to_dict(obj: object):
    """Convert an object into dictionary recursively.

    Note:
        Ignores all names starting with "__" (i.e. built-in methods).

    Args:
        obj (object): An instance of a class to convert.

    Raises:
        ValueError: When input argument is not an object.

    Returns:
        Dict[str, Any]: Converted dictionary mapping.
    """
    # check that input data is class instance
    if not hasattr(obj, "__class__"):
        raise ValueError(f"Expected a class instance. Received: {type(obj)}.")
    # convert object to dictionary
    if isinstance(obj, dict):
        obj_dict = obj
    else:
        obj_dict = obj.__dict__
    # convert to dictionary
    data = dict()
    for key, value in obj_dict.items():
        # disregard builtin attributes
        if key.startswith("__"):
            continue
        # check if attribute is callable -- function
        if callable(value):
            data[key] = f"{value.__module__}:{value.__name__}"
        # check if attribute is a dictionary
        elif hasattr(value, "__dict__") or isinstance(value, dict):
            data[key] = class_to_dict(value)
        else:
            data[key] = value
    return data
