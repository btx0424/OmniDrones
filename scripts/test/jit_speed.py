import torch
import time


# Define a simple function to be compared
def calculate_safety_cost(drone_rpos: torch.Tensor) -> torch.Tensor:
    # r (E,4,3)
    r1, r2, r3 = torch.split(drone_rpos, [3, 3, 3], dim=-1)

    # d (E,4,3)
    d = torch.stack(
        [
            torch.norm(r1, dim=-1),
            torch.norm(r2, dim=-1),
            torch.norm(r3, dim=-1),
        ],
        dim=-1,
    )

    d_m, _ = torch.max(d, dim=-1, keepdim=True)

    safety_cost = 1.0 - d_m / 3
    safety_cost = safety_cost.clip(0.0)
    return safety_cost


# JIT compile the function using torch.jit.script
jit_func = torch.jit.script(calculate_safety_cost)


def random_drone_pos():
    n_env = 128
    drone_rpos = torch.randn(n_env, 4, 9) * 0.5
    return drone_rpos.cuda()


def get_average_time(f, get_input, n=1000):
    start_time = time.time()
    for _ in range(n):
        input = get_input()
        f(input)
    average_time = (time.time() - start_time) / n
    return average_time


t1 = get_average_time(calculate_safety_cost, random_drone_pos)
t2 = get_average_time(jit_func, random_drone_pos)

print(t1 / t2)
