import torch
from omni_drones.utils.torch import euler_to_quaternion, quat_rotate

def lemniscate(t: torch.Tensor, c: torch.Tensor):
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    sin2p1 = torch.square(sin_t) + 1

    x = torch.stack([
        cos_t, sin_t * cos_t, c * sin_t
    ], dim=-1) / sin2p1.unsqueeze(-1)

    return x

def scale_time(t, a: float=1.0):
    return t / (1 + 1/(a*torch.abs(t)))


class LemniscateTrajectory:
    
    T0 = torch.pi / 2

    def __init__(
        self,
        batch_shape: torch.Size, 
        device: str,
        origin = (0., 0., 2.), 
    ):
        self.batch_shape = batch_shape
        self.device = device

        with torch.device(device):
            self.c = torch.zeros(batch_shape)
            self.w = torch.ones(batch_shape) # a time scale factor
            self.scale = torch.ones(batch_shape + (3,))
            self.rot = torch.zeros(batch_shape + (4,))
            self.origin = torch.as_tensor(origin).expand(batch_shape + (3,))
    
    def compute(self, t: torch.Tensor, dt: float, steps: int=1, ids: torch.Tensor=None):
        if ids is None: ids = slice(None)
        
        c = self.c[ids].unsqueeze(-1)
        w = self.w[ids].unsqueeze(-1)
        rot = self.rot[ids].unsqueeze(-2)
        scale = self.scale[ids]
        t_ = dt * torch.arange(steps, device=self.device) # [env, steps]
        t = w * (t.unsqueeze(-1) + t_) # [env, steps]
        x = lemniscate(self.T0 + scale_time(t), c) # [env, steps, 3]

        x = quat_rotate(rot, x) * scale.unsqueeze(1)
        return x + self.origin[ids].unsqueeze(1)