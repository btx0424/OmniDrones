# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# python
import builtins
import typing

# omniverse
import carb
import omni.kit.app

# isaacsim
from omni.isaac.core.utils.constants import AXES_TOKEN
from omni.usd.commands import DeletePrimsCommand
from pxr import Sdf, Usd, UsdGeom


def get_current_stage() -> Usd.Stage:
    """Get the current open USD stage.

    Returns:
        Usd.Stage: The USD stage.
    """
    return omni.usd.get_context().get_stage()


def update_stage() -> None:
    """Update the current USD stage."""
    omni.kit.app.get_app_interface().update()


async def update_stage_async() -> None:
    """Update the current USD stage (asynchronous version)."""
    await omni.kit.app.get_app().next_update_async()


# TODO: make a generic util for setting all layer properties
def set_stage_up_axis(axis: str = "z") -> None:
    """Change the up axis of the current stage

    Args:
        axis (UsdGeom.Tokens, optional): valid values are "x" and "y"
    """
    stage = get_current_stage()
    if stage is None:
        raise Exception("There is no stage currently opened")
    rootLayer = stage.GetRootLayer()
    rootLayer.SetPermissionToEdit(True)
    with Usd.EditContext(stage, rootLayer):
        UsdGeom.SetStageUpAxis(stage, AXES_TOKEN[axis])


def get_stage_up_axis() -> str:
    """Get the current up-axis of USD stage.

    Returns:
        str: The up-axis of the stage.
    """
    stage = get_current_stage()
    return UsdGeom.GetStageUpAxis(stage)


def clear_stage(
    predicate: typing.Optional[typing.Callable[[str], bool]] = None
) -> None:
    """Deletes all prims in the stage without populating the undo command buffer

    Args:
        predicate (typing.Optional[typing.Callable[[str], bool]], optional): user defined function that  takes a prim_path (str) as input and returns True/False if the prim should/shouldn't be deleted. If predicate is None, a default is used that deletes all prims

    Returns:
        [type]: [description]
    """
    # Note: Need to import this here to prevent circular dependencies.
    from omni.isaac.core.utils.prims import (
        get_all_matching_child_prims,
        get_prim_path,
        is_prim_ancestral,
        is_prim_hidden_in_stage,
        is_prim_no_delete,
    )

    def default_predicate(prim_path: str):
        # prim = get_prim_at_path(prim_path)
        # skip prims that we cannot delete
        if is_prim_no_delete(prim_path):
            return False
        if is_prim_hidden_in_stage(prim_path):
            return False
        if is_prim_ancestral(prim_path):
            return False
        if prim_path == "/":
            return False
        # TODO, check if this can be removed
        if prim_path == "/Render/Vars":
            return False
        return True

    if predicate is None:
        prims = get_all_matching_child_prims("/", default_predicate)
        prim_paths_to_delete = [get_prim_path(prim) for prim in prims]
        DeletePrimsCommand(prim_paths_to_delete).do()
    else:
        prims = get_all_matching_child_prims("/", predicate)
        prim_paths_to_delete = [get_prim_path(prim) for prim in prims]
        DeletePrimsCommand(prim_paths_to_delete).do()

    if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
        omni.kit.app.get_app_interface().update()


def print_stage_prim_paths() -> None:
    """Traverses the stage and prints all prim paths."""
    # Note: Need to import this here to prevent circular dependencies.
    from omni.isaac.core.utils.prims import get_prim_path

    for prim in traverse_stage():
        prim_path = get_prim_path(prim)
        print(prim_path)


def add_reference_to_stage(
    usd_path: str, prim_path: str, prim_type: str = "Xform"
) -> Usd.Prim:
    """Add USD reference to the opened stage at specified prim path.

    Args:
        usd_path (str): The path to USD file.
        prim_path (str): The prim path to attach reference.
        prim_type (str, optional): The type of prim. Defaults to "Xform".

    Raises:
        FileNotFoundError: When input USD file is found at specified path.

    Returns:
        Usd.Prim: The USD prim at specified prim path.
    """
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim = stage.DefinePrim(prim_path, prim_type)
    carb.log_info("Loading Asset from path {} ".format(usd_path))
    success_bool = prim.GetReferences().AddReference(usd_path)
    if not success_bool:
        raise FileNotFoundError(
            "The usd file at path {} provided wasn't found".format(usd_path)
        )
    return prim


def create_new_stage() -> Usd.Stage:
    """Create a new stage.

    Returns:
        Usd.Stage: The created USD stage.
    """
    return omni.usd.get_context().new_stage()


async def create_new_stage_async() -> None:
    """Create a new stage (asynchronous version)."""
    await omni.usd.get_context().new_stage_async()
    await omni.kit.app.get_app().next_update_async()


def open_stage(usd_path: str) -> bool:
    """Open the given usd file and replace currently opened stage.

    Args:
        usd_path (str): Path to the USD file to open.

    Raises:
        ValueError: When input path is not a supported file type by USD.

    Returns:
        bool: True if operation is successful, otherwise false.
    """
    if not Usd.Stage.IsSupportedFile(usd_path):
        raise ValueError("Only USD files can be loaded with this method")
    usd_context = omni.usd.get_context()
    usd_context.disable_save_to_recent_files()
    result = omni.usd.get_context().open_stage(usd_path)
    usd_context.enable_save_to_recent_files()
    return result


async def open_stage_async(usd_path: str) -> typing.Tuple[bool, int]:
    """Open the given usd file and replace currently opened stage (asynchronous version).

    Args:
        usd_path (str): Path to the USD file to open.

    Raises:
        ValueError: When input path is not a supported file type by USD.

    Returns:
        bool: True if operation is successful, otherwise false.
    """
    if not Usd.Stage.IsSupportedFile(usd_path):
        raise ValueError("Only USD files can be loaded with this method")
    usd_context = omni.usd.get_context()
    usd_context.disable_save_to_recent_files()
    (result, error) = await omni.usd.get_context().open_stage_async(usd_path)
    usd_context.enable_save_to_recent_files()
    return (result, error)


def save_stage(usd_path: str, save_and_reload_in_place=True) -> bool:

    """Save usd file to path, it will be overwritten with the current stage

    Args:
        usd_path (str): File path to save the current stage to
        save_and_reload_in_place (bool, optional): use save_as_stage to save and reload the root layer in place. Defaults to True.

    Raises:
        ValueError: When input path is not a supported file type by USD.

    Returns:
        bool: True if operation is successful, otherwise false.
    """
    if not Usd.Stage.IsSupportedFile(usd_path):
        raise ValueError("Only USD files can be saved with this method")
    if save_and_reload_in_place:
        result = omni.usd.get_context().save_as_stage(usd_path)
    else:
        layer = Sdf.Layer.CreateNew(usd_path)
        root_layer = omni.usd.get_context().get_stage().GetRootLayer()
        layer.TransferContent(root_layer)
        result = layer.Save()
    return result


def close_stage(callback_fn: typing.Callable = None) -> bool:
    """Closes the current opened USD stage.

    Args:
        callback_fn (typing.Callable, optional): Callback function to call while closing. Defaults to None.

    Returns:
        bool: True if operation is successful, otherwise false.
    """
    if callback_fn is None:
        result = omni.usd.get_context().close_stage()
    else:
        result = omni.usd.get_context().close_stage_with_callback(callback_fn)
    return result


def set_livesync_stage(usd_path: str, enable: bool) -> bool:
    """[summary]

    Args:
        usd_path (str): path to enable live sync for, t will be overwritten with the current stage
        enable (bool): True to enable livesync, false to disable livesync

    Returns:
        bool: True if operation is successful, otherwise false.
    """
    # TODO: Check that the provided usd_path exists
    if save_stage(usd_path):
        if enable:
            omni.usd.get_context().set_stage_live(omni.usd.StageLiveModeType.ALWAYS_ON)
            omni.usd.get_context().set_layer_live(usd_path, enable)
        else:
            omni.usd.get_context().set_stage_live(omni.usd.StageLiveModeType.TOGGLE_OFF)
        return True
    else:
        return False


def traverse_stage() -> typing.Iterable:
    """Traverse through prims in the opened USd stage.

    Returns:
        typing.Iterable: Generator which yields prims from the stage in depth-first-traversal order.
    """
    return get_current_stage().Traverse()


def is_stage_loading() -> bool:
    """Convenience function to see if any files are being loaded.

    Returns:
        bool: True if loading, False otherwise
    """
    context = omni.usd.get_context()
    if context is None:
        return False
    else:
        _, _, loading = context.get_stage_loading_status()
        return loading > 0


def set_stage_units(stage_units_in_meters: float) -> None:
    """Set the stage meters per unit

    Args:
        stage_units_in_meters (float): units for stage, 1.0 means meters, 0.01 mean centimeters
    """
    if get_current_stage() is None:
        raise Exception(
            "There is no stage currently opened, init_stage needed before calling this func"
        )
    UsdGeom.SetStageMetersPerUnit(get_current_stage(), stage_units_in_meters)


def get_stage_units() -> float:
    """Get the stage meters per unit currently set

    Returns:
        float: current stage meters per unit
    """
    return UsdGeom.GetStageMetersPerUnit(get_current_stage())


def get_next_free_path(path: str, parent: str = None) -> str:
    """Returns the next free usd path for the current stage

    Args:
        path (str): path we want to check
        parent (str, optional): Parent prim for the given path. Defaults to None.

    Returns:
        str: a new path that is guaranteed to not exist on the current stage
    """

    if parent is not None:
        # remove trailing slash from parent and leading slash from path
        path = omni.usd.get_stage_next_free_path(
            get_current_stage(), parent.rstrip("/") + "/" + path.lstrip("/"), False
        )
    else:
        path = omni.usd.get_stage_next_free_path(get_current_stage(), path, True)
    return path
