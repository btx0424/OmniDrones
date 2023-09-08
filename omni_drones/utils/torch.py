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

import functools
def manual_batch(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        batch_shapes = set(arg.shape[:-1] for arg in args if isinstance(arg, torch.Tensor))
        if not len(batch_shapes) == 1:
            raise ValueError
        batch_shape = batch_shapes.pop()
        args = (
            arg.reshape(-1, arg.shape[-1]) if isinstance(arg, torch.Tensor) else arg 
            for arg in args
        )
        kwargs = {
            k: v.reshape(-1, v.shape[-1]) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        out = func(*args, **kwargs)
        return out.unflatten(0, batch_shape)
    return wrapped


@manual_batch
def quat_rotate(q: torch.Tensor, v: torch.Tensor):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@manual_batch
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

@manual_batch
def euler_rotate(rpy: torch.Tensor, v: torch.Tensor):
    shape = rpy.shape
    r, p, y = torch.unbind(rpy, dim=-1)
    cr = torch.cos(r)
    sr = torch.sin(r)
    cp = torch.cos(p)
    sp = torch.sin(p)
    cy = torch.cos(y)
    sy = torch.sin(y)
    R = torch.stack([
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp, cp * sr, cp * cr
    ], dim=-1).view(*shape[:-1], 3, 3)
    return torch.bmm(R, v.unsqueeze(-1)).squeeze(-1)


@manual_batch
def quat_axis(q: torch.Tensor, axis: int=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def axis_angle_to_quaternion(angle: torch.Tensor, axis: torch.Tensor):
    axis = axis / torch.norm(axis, dim=-1, keepdim=True)
    return torch.cat([torch.cos(angle / 2), torch.sin(angle / 2) * axis], dim=-1)


def axis_angle_to_matrix(angle, axis):
    quat = axis_angle_to_quaternion(angle, axis)
    return quaternion_to_rotation_matrix(quat)


def quat_mul(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat


def symlog(x: torch.Tensor):
    """
    The symlog transformation described in https://arxiv.org/pdf/2301.04104v1.pdf
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x: torch.Tensor):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

