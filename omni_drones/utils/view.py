import torch
from typing import Union
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.articulations import ArticulationView

# VIEW_TYPE = Union[RigidPrimView, ArticulationView]

# def reshape(view: VIEW_TYPE, shape):
#     base_indices = torch.arange(view.count, device=view._device).reshape(shape)
#     view.shape = base_indices.shape
#     class BackendUtils:
#         def __init__(self, backend_utils) -> None:
#             self._backend_utils = backend_utils
        
#         def resolve_indices(self, indices, count, device):
#             if indices is None:
#                 indices = torch.arange(base_indices.shape[0], device=device)
#             return base_indices[indices]
        
#         def __getattr__(self, name):
#             return getattr(self._backend_utils, name)
        
#     view._backend_utils = BackendUtils(view._backend_utils)
#     return view
