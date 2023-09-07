import torch
import time

def encode_drone_vel(drone_vel: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        drone_vel (torch.Tensor): (E,4,6)

    Returns:
        torch.Tensor: (E,4,18)
    """
    return torch.stack(
        [
            torch.concat(
                [
                    drone_vel[..., 1, :],
                    drone_vel[..., 2, :],
                    drone_vel[..., 3, :],
                ],
                dim=-1,
            ),
            torch.concat(
                [
                    drone_vel[..., 2, :],
                    drone_vel[..., 3, :],
                    drone_vel[..., 0, :],
                ],
                dim=-1,
            ),
            torch.concat(
                [
                    drone_vel[..., 3, :],
                    drone_vel[..., 0, :],
                    drone_vel[..., 1, :],
                ],
                dim=-1,
            ),
            torch.concat(
                [
                    drone_vel[..., 0, :],
                    drone_vel[..., 1, :],
                    drone_vel[..., 2, :],
                ],
                dim=-1,
            ),
        ],
        dim=1,
    )


# JIT compile the function using torch.jit.script
jit_func = torch.jit.script(encode_drone_vel)


def random_input():
    n_env = 512
    drone_rpos = torch.randn(n_env, 4, 6) 
    return drone_rpos.cuda()


def get_average_time(f, get_input, n=1000):
    input = get_input()
    start_time = time.time()
    for _ in range(n):
        f(input)
    average_time = (time.time() - start_time) / n
    return average_time


t1 = get_average_time(encode_drone_vel, random_input)
t2 = get_average_time(jit_func, random_input)

print(t1, t2, t1 / t2)
