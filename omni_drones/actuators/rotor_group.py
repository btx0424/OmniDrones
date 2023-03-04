import torch
import torch.nn as nn

class RotorGroup(nn.Module):

    def __init__(self, rotor_config, dt: float):
        super().__init__()
        self.num_rotors = rotor_config["num_rotors"]
        self.KF = nn.Parameter(torch.as_tensor(rotor_config["force_constants"]))
        self.KM = nn.Parameter(torch.as_tensor(rotor_config["moment_constants"]))
        self.MAX_ROT_VEL = nn.Parameter(
            torch.as_tensor(rotor_config["max_rotation_velocities"], dtype=torch.float32))
        self.dt = dt
        self.time_up = 0.15
        self.time_down = 0.15
        
        self.max_forces = nn.Parameter(self.MAX_ROT_VEL.square() * self.KF)
        self.max_moments = nn.Parameter(self.MAX_ROT_VEL.square() * self.KM)
        self.throttle = nn.Parameter(torch.zeros(self.num_rotors))
        self.directions = nn.Parameter(torch.as_tensor(rotor_config["directions"]))
        
        self.tau_up = nn.Parameter(4 * dt / self.time_up * torch.ones(self.num_rotors))
        self.tau_down = nn.Parameter(4 * dt / self.time_down * torch.ones(self.num_rotors))

        self.f = torch.square
        self.f_inv = torch.sqrt
        
        for p in self.parameters():
            p.requires_grad_(False)
    
    def forward(self, cmds: torch.Tensor):
        cmds = self.f_inv(torch.clamp((cmds + 1) / 2, 0, 1))

        tau = torch.where(cmds > self.throttle, self.tau_up, self.tau_down)
        tau = torch.clamp(tau, 0, 1)
        self.throttle.add_(tau * (cmds - self.throttle))

        t = self.f(self.throttle)
        thrusts = t * self.max_forces
        moments = (t * self.max_moments) * -self.directions

        return thrusts, moments
