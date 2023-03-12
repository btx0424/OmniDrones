import torch


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


def normalize(x: torch.Tensor, eps: float = 1e-6):
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)
