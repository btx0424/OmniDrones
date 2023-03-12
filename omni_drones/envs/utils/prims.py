# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# python
import re
import typing

import omni.kit
import omni.usd
from omni.isaac.core.utils.semantics import add_update_semantics

# isaacsim
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.string import find_root_prim_path_from_regex
from omni.isaac.dynamic_control import _dynamic_control
from omni.usd.commands import DeletePrimsCommand, MovePrimCommand

# omniverse
from pxr import Usd, UsdGeom, UsdPhysics


def get_prim_at_path(prim_path: str) -> Usd.Prim:
    """Get the USD Prim at a given path string

    Args:
        prim_path (str): path of the prim in the stage

    Returns:
        Usd.Prim: USD Prim object at the given path in the current stage
    """
    return get_current_stage().GetPrimAtPath(prim_path)


def is_prim_path_valid(prim_path: str) -> bool:
    """Check if a path has a valid USD Prim at it

    Args:
        prim_path (str): path of the prim in the stage

    Returns:
        bool: True if the path points to a valid prim
    """
    return get_current_stage().GetPrimAtPath(prim_path).IsValid()


def define_prim(prim_path: str, prim_type: str = "Xform") -> Usd.Prim:
    """Create a USD Prim at the given prim_path of type prim_type unless one already exists

    Args:
        prim_path (str): path of the prim in the stage
        prim_type (str, optional): The type of the prim to create. Defaults to "Xform".

    Raises:
        Exception: If there is already a prim at the prim_path

    Returns:
        Usd.Prim: The created USD prim.
    """
    if is_prim_path_valid(prim_path):
        raise Exception("A prim already exists at prim path: {}".format(prim_path))
    return get_current_stage().DefinePrim(prim_path, prim_type)


def get_prim_type_name(prim_path: str) -> str:
    """Get the TypeName of the USD Prim at the path if it is valid

    Args:
        prim_path (str): path of the prim in the stage

    Raises:
        Exception: If there is not a valid prim at the given path

    Returns:
        str: The TypeName of the USD Prim at the path string
    """
    if not is_prim_path_valid(prim_path):
        raise Exception("A prim does not exist at prim path: {}".format(prim_path))
    prim = get_prim_at_path(prim_path)
    return prim.GetPrimTypeInfo().GetTypeName()


def move_prim(path_from: str, path_to: str) -> None:
    """Run the Move command to change a prims USD Path in the stage

    Args:
        path_from (str): Path of the USD Prim you wish to move
        path_to (str): Final destination of the prim
    """
    MovePrimCommand(path_from=path_from, path_to=path_to).do()


def get_first_matching_child_prim(
    prim_path: str, predicate: typing.Callable[[str], bool]
) -> Usd.Prim:
    """Recursively get the first USD Prim at the path string that passes the predicate function

    Args:
        prim_path (str): path of the prim in the stage
        predicate (typing.Callable[[str], bool]): Function to test the prims against

    Returns:
         Usd.Prim: The first prim or child of the prim, as defined by GetChildren, that passes the predicate
    """
    prim = get_current_stage().GetPrimAtPath(prim_path)
    children_stack = [prim]
    out = prim.GetChildren()
    while len(children_stack) > 0:
        prim = children_stack.pop(0)
        if predicate(get_prim_path(prim)):
            return prim
        children = prim.GetChildren()
        children_stack = children_stack + children
        out = out + children
    return None


def get_first_matching_parent_prim(
    prim_path: str, predicate: typing.Callable[[str], bool]
) -> Usd.Prim:
    """Recursively get the first USD Prim at the parent path string that passes the predicate function

    Args:
        prim_path (str): path of the prim in the stage
        predicate (typing.Callable[[str], bool]): Function to test the prims against

    Returns:
        str: The first prim on the parent path, as defined by GetParent, that passes the predicate
    """
    current_prim_path = get_prim_path(get_prim_parent(get_prim_at_path(prim_path)))
    while not is_prim_root_path(current_prim_path):
        if predicate(current_prim_path):
            return get_prim_at_path(current_prim_path)
        current_prim_path = get_prim_path(
            get_prim_parent(get_prim_at_path(current_prim_path))
        )
    return None


def get_all_matching_child_prims(
    prim_path: str,
    predicate: typing.Callable[[str], bool] = lambda x: True,
    depth: typing.Optional[int] = None,
) -> typing.List[Usd.Prim]:
    """Performs a breadth-first search starting from the root and returns all the prims matching the predicate.

    Args:
        prim_path (str): root prim path to start traversal from.
        predicate (typing.Callable[[str], bool]): predicate that checks the prim path of a prim and returns a boolean.
        depth (typing.Optional[int]): maximum depth for traversal, should be bigger than zero if specified.
                                      Defaults to None (i.e: traversal till the end of the tree).

    Returns:
        typing.List[Usd.Prim]: A list containing the root and children prims matching specified predicate.
    """
    prim = get_prim_at_path(prim_path)
    traversal_queue = [(prim, 0)]
    out = []
    while len(traversal_queue) > 0:
        prim, current_depth = traversal_queue.pop(0)
        if is_prim_path_valid(get_prim_path(prim)):
            if predicate(get_prim_path(prim)):
                out.append(prim)
            if depth is None or current_depth < depth:
                children = get_prim_children(prim)
                traversal_queue = traversal_queue + [
                    (child, current_depth + 1) for child in children
                ]
    return out


def find_matching_prim_paths(prim_path_regex: str) -> typing.List[str]:
    """Find all the matching prim paths in the stage based on Regex expression.

    Args:
        prim_path_regex (str): The Regex expression for prim path.

    Returns:
        typing.List[str]: List of prim paths that match input expression.
    """
    expressions_to_match = [prim_path_regex]
    result = []
    while len(expressions_to_match) > 0:
        expression_to_match = expressions_to_match.pop(0)
        root_prim_path, tree_level = find_root_prim_path_from_regex(expression_to_match)
        if root_prim_path is None:
            if is_prim_path_valid(expression_to_match):
                result.append(expression_to_match)
        else:
            immediate_expression_to_match = "/".join(
                expression_to_match.split("/")[: tree_level + 1]
            )
            children_matching = get_all_matching_child_prims(
                prim_path=root_prim_path,
                predicate=lambda a: re.search(immediate_expression_to_match, a)
                is not None,
                depth=1,
            )
            children_matching = [get_prim_path(prim) for prim in children_matching]
            remainder_expression = "/".join(
                expression_to_match.split("/")[tree_level + 1 :]
            )
            if remainder_expression != "":
                remainder_expression = "/" + remainder_expression
            children_expressions = [
                child + remainder_expression for child in children_matching
            ]
            expressions_to_match = expressions_to_match + children_expressions
    return result


def get_prim_children(prim: Usd.Prim) -> typing.List[Usd.Prim]:
    """Return the call of the USD Prim's GetChildren member function

    Args:
        prim (Usd.Prim): The parent USD Prim

    Returns:
        typing.List[Usd.Prim]: A list of the prim's children.
    """
    return prim.GetChildren()


def get_prim_parent(prim: Usd.Prim) -> Usd.Prim:
    """Return the call of the USD Prim's GetChildren member function

    Args:
        prim (Usd.Prim): The USD Prim to call GetParent on

    Returns:
        Usd.Prim: The prim's parent returned from GetParent
    """
    return prim.GetParent()


def query_parent_path(prim_path: str, predicate: typing.Callable[[str], bool]) -> bool:
    """Check if one of the ancestors of the prim at the prim_path can pass the predicate

    Args:
        prim_path (str): path to the USD Prim for which to check the ancestors
        predicate (typing.Callable[[str], bool]): The condition that must be True about the ancestors

    Returns:
        bool: True if there is an ancestor that can pass the predicate, False otherwise
    """
    current_prim_path = get_prim_path(get_prim_parent(get_prim_at_path(prim_path)))
    while not is_prim_root_path(current_prim_path):
        if predicate(current_prim_path):
            return True
        current_prim_path = get_prim_path(
            get_prim_parent(get_prim_at_path(current_prim_path))
        )
    return False


def is_prim_ancestral(prim_path: str) -> bool:
    """Check if any of the prims ancestors were brought in as a reference

    Args:
        prim_path (str): The path to the USD prim.

    Returns:
        True if prim is part of a referenced prim, false otherwise.
    """
    return omni.usd.check_ancestral(get_prim_at_path(prim_path))


def is_prim_root_path(prim_path: str) -> bool:
    """Checks if the input prim path is root or not.

    Args:
        prim_path (str): The path to the USD prim.

    Returns:
        True if the prim path is "/", False otherwise
    """
    if prim_path == "/":
        return True
    else:
        return False


def is_prim_no_delete(prim_path: str) -> bool:
    """Checks whether a prim can be deleted or not from USD stage.

    Args:
        prim_path (str): The path to the USD prim.

    Returns:
        True if prim cannot be deleted, False if it can
    """
    return get_prim_at_path(prim_path).GetMetadata("no_delete")


def is_prim_hidden_in_stage(prim_path: str) -> bool:
    """Checks if the prim is hidden in the USd stage or not.

    Args:
        prim_path (str): The path to the USD prim.

    Note:
        This is not related to the prim visibility.

    Returns:
        True if prim is hidden from stage window, False if not hidden.
    """
    return get_prim_at_path(prim_path).GetMetadata("hide_in_stage_window")


def get_prim_path(prim: Usd.Prim) -> str:
    """Get the path of a given USD prim.

    Args:
        prim (Usd.Prim): The input USD prim.

    Returns:
        str: The path to the input prim.
    """
    return prim.GetPath().pathString


def set_prim_visibility(prim: Usd.Prim, visible: bool) -> None:
    """Sets the visibility of the prim in the opened stage.

    The method does this through the USD API.

    Args:
        prim (Usd.Prim): the USD prim
        visible (bool): flag to set the visibility of the usd prim in stage.
    """
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


def create_prim(
    prim_path: str,
    prim_type: str = "Xform",
    position: typing.Optional[typing.Sequence[float]] = None,
    translation: typing.Optional[typing.Sequence[float]] = None,
    orientation: typing.Optional[typing.Sequence[float]] = None,
    scale: typing.Optional[typing.Sequence[float]] = None,
    usd_path: typing.Optional[str] = None,
    semantic_label: typing.Optional[str] = None,
    semantic_type: str = "class",
    attributes: typing.Optional[dict] = None,
) -> Usd.Prim:
    """Create a prim into current USD stage.

    The method applies specified transforms, the semantic label and set specified attributes.

    Args:
        prim_path (str): The path of the new prim.
        prim_type (str): Prim type name
        position (typing.Sequence[float], optional): prim position (applied last)
        translation (typing.Sequence[float], optional): prim translation (applied last)
        orientation (typing.Sequence[float], optional): prim rotation as quaternion
        scale (np.ndarray (3), optional): scaling factor in x, y, z.
        usd_path (str, optional): Path to the USD that this prim will reference.
        semantic_label (str, optional): Semantic label.
        semantic_type (str, optional): set to "class" unless otherwise specified.
        attributes (dict, optional): Key-value pairs of prim attributes to set.

    Raises:
        Exception: If there is already a prim at the prim_path

    Returns:
        Usd.Prim: The created USD prim.
    """
    # Note: Imported here to prevent cyclic dependency in the module.
    from omni.isaac.core.prims import XFormPrim

    # create prim in stage
    prim = define_prim(prim_path=prim_path, prim_type=prim_type)
    if not prim:
        return None
    # apply attributes into prim
    if attributes is not None:
        for k, v in attributes.items():
            prim.GetAttribute(k).Set(v)
    # add reference to USD file
    if usd_path is not None:
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    # add semantic label to prim
    if semantic_label is not None:
        add_update_semantics(prim, semantic_label, semantic_type)
    # apply the transformations
    XFormPrim(
        prim_path=prim_path,
        position=position,
        translation=translation,
        orientation=orientation,
        scale=scale,
    )

    return prim


def delete_prim(prim_path: str) -> None:
    """Remove the USD Prim and its decendants from the scene if able

    Args:
        prim_path (str): path of the prim in the stage
    """
    DeletePrimsCommand([prim_path]).do()


def get_prim_property(prim_path: str, property_name: str) -> typing.Any:
    """Get the attribute of the USD Prim at the given path

    Args:
        prim_path (str): path of the prim in the stage
        property_name (str): name of the attribute to get

    Returns:
        typing.Any: The attribute if it exists, None otherwise
    """
    prim = get_prim_at_path(prim_path=prim_path)
    return prim.GetAttribute(property_name).Get()


def set_prim_property(
    prim_path: str, property_name: str, property_value: typing.Any
) -> None:
    """Set the attribute of the USD Prim at the path

    Args:
        prim_path (str): path of the prim in the stage
        property_name (str): name of the attribute to set
        property_value (typing.Any): value to set the attribute to
    """
    prim = get_prim_at_path(prim_path=prim_path)
    prim.GetAttribute(property_name).Set(property_value)


def get_prim_object_type(prim_path: str) -> typing.Union[str, None]:
    """Get the dynamic control Ooject type of the USD Prim at the given path.

    If the prim at the path is of Dynamic Control type--i.e. rigid_body, joint, dof, articulation, attractor, d6joint,
    then the correspodning string returned. If is an Xformable prim, then "xform" is returned. Otherwise None
    is returned.

    Args:
        prim_path (str): path of the prim in the stage

    Raises:
        Exception: If the USD Prim is not a suppored type.

    Returns:
        str: String corresponding to the object type.
    """
    dc_interface = _dynamic_control.acquire_dynamic_control_interface()
    object_type = dc_interface.peek_object_type(prim_path)
    if object_type == _dynamic_control.OBJECT_NONE:
        prim = get_prim_at_path(prim_path)
        if prim.IsA(UsdGeom.Xformable):
            return "xform"
        else:
            return None
    elif object_type == _dynamic_control.OBJECT_RIGIDBODY:
        return "rigid_body"
    elif object_type == _dynamic_control.OBJECT_JOINT:
        return "joint"
    elif object_type == _dynamic_control.OBJECT_DOF:
        return "dof"
    elif object_type == _dynamic_control.OBJECT_ARTICULATION:
        return "articulation"
    elif object_type == _dynamic_control.OBJECT_ATTRACTOR:
        return "attractor"
    elif object_type == _dynamic_control.OBJECT_D6JOINT:
        return "d6joint"
    else:
        raise Exception("the object type is not support here yet")


def is_prim_non_root_articulation_link(prim_path: str) -> bool:
    """Used to query if the prim_path corresponds to a link in an articulation which is a non root link.

    Args:
        prim_path (str): prim_path to query

    Returns:
        bool: True if the prim path corresponds to a link in an articulation which is a non root link
              and can't have a transformation applied to it.
    """
    parent_articulation_root = get_first_matching_parent_prim(
        prim_path=prim_path,
        predicate=lambda a: get_prim_at_path(a).HasAPI(UsdPhysics.ArticulationRootAPI),
    )
    if parent_articulation_root is None:
        return False

    has_physics_apis = get_prim_at_path(prim_path).HasAPI(UsdPhysics.RigidBodyAPI)
    if not has_physics_apis:
        return False

    # get all joints under ArticulationRoot
    joint_prims = get_all_matching_child_prims(
        prim_path=get_prim_path(parent_articulation_root),
        predicate=lambda a: "Joint" in get_prim_type_name(a),
    )
    # this assumes if that the first link is a root articulation link
    for joint_prim in joint_prims:
        joint = UsdPhysics.Joint(joint_prim)
        if joint.GetExcludeFromArticulationAttr().Get():
            continue
        body_targets = (
            joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets()
        )
        for target in body_targets:
            if prim_path == str(target):
                return True
    return False


def set_prim_hide_in_stage_window(prim: Usd.Prim, hide: bool):
    """set hide_in_stage_window metadata for prim

    Args:
        prim (Usd.Prim): Prim to set
        hide (bool): True to hide in stage window, false to show
    """
    prim.SetMetadata("hide_in_stage_window", hide)


def set_prim_no_delete(prim: Usd.Prim, no_delete: bool):
    """set no_delete metadata for prim

    Args:
        prim (Usd.Prim): Prim to set
        no_delete (bool):True to make prim undeletable in stage window, false to allow deletion
    """
    prim.SetMetadata("no_delete", no_delete)


def set_targets(prim: Usd.Prim, attribute: str, target_prim_paths: list):
    """Set targets for a prim relationship attribute

    Args:
        prim (Usd.Prim): Prim to create and set attribute on
        attribute (str): Relationship attribute to create
        target_prim_paths (list): list of targets to set
    """
    try:
        input_rel = prim.CreateRelationship(attribute)
        input_rel.SetTargets(target_prim_paths)
    except Exception as e:
        print(e, prim.GetPath())
