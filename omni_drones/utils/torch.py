import torch
from typing import Sequence, Union
from contextlib import contextmanager

@contextmanager
def torch_seed(seed: int=0):
    rng_state = torch.get_rng_state()
    rng_state_cuda = torch.cuda.get_rng_state_all()
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state_all(rng_state_cuda)


def off_diag(a: torch.Tensor) -> torch.Tensor:
    assert a.shape[0] == a.shape[1]
    n = a.shape[0]
    return (
        a.flatten(0, 1)[1:]
        .unflatten(0, (n - 1, n + 1))[:, :-1]
        .reshape(n, n - 1, *a.shape[2:])
    )


def cpos(p1: torch.Tensor, p2: torch.Tensor):
    assert p1.shape[1] == p2.shape[1]
    return p1.unsqueeze(1) - p2.unsqueeze(0)


def others(x: torch.Tensor) -> torch.Tensor:
    return off_diag(x.expand(x.shape[0], *x.shape))


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:

    w, x, y, z = torch.unbind(quaternion, dim=-1)
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z

    matrix = torch.stack(
        [
            1 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1 - (txx + tyy),
        ],
        dim=-1,
    )
    matrix = matrix.unflatten(matrix.dim() - 1, (3, 3))
    return matrix


def quaternion_to_euler(quaternion: torch.Tensor) -> torch.Tensor:

    w, x, y, z = torch.unbind(quaternion, dim=quaternion.dim() - 1)

    euler_angles: torch.Tensor = torch.stack(
        (
            torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            torch.asin(2.0 * (w * y - z * x)),
            torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ),
        dim=-1,
    )

    return euler_angles


def euler_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
    euler = torch.as_tensor(euler)
    r, p, y = torch.unbind(euler, dim=-1)
    cy = torch.cos(y * 0.5)
    sy = torch.sin(y * 0.5)
    cp = torch.cos(p * 0.5)
    sp = torch.sin(p * 0.5)
    cr = torch.cos(r * 0.5)
    sr = torch.sin(r * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quaternion = torch.stack([qw, qx, qy, qz], dim=-1)

    return quaternion


def normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def make_cells(
    range_min: Union[Sequence[float], torch.Tensor],
    range_max: Union[Sequence[float], torch.Tensor],
    size: Union[float, Sequence[float], torch.Tensor],
):
    """Compute the cell centers of a n-d grid.

    Examples:
        >>> cells = make_cells([0, 0], [1, 1], 0.1)
        >>> cells[:2, :2]
        tensor([[[0.0500, 0.0500],
                 [0.0500, 0.1500]],

                [[0.1500, 0.0500],
                 [0.1500, 0.1500]]])
    """
    range_min = torch.as_tensor(range_min)
    range_max = torch.as_tensor(range_max)
    size = torch.as_tensor(size)
    shape = ((range_max - range_min) / size).round().int()

    cells = torch.meshgrid(*[torch.linspace(l, r, n+1) for l, r, n in zip(range_min, range_max, shape)], indexing="ij")
    cells = torch.stack(cells, dim=-1)
    for dim in range(cells.dim()-1):
        cells = (cells.narrow(dim, 0, cells.size(dim)-1) + cells.narrow(dim, 1, cells.size(dim)-1)) / 2
    return cells

