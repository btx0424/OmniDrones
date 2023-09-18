import torch
import torch.nn as nn


class RotorGroup(nn.Module):
    def __init__(self, rotor_config, dt: float):
        super().__init__()
        force_constants = torch.as_tensor(rotor_config["force_constants"])
        moment_constants = torch.as_tensor(rotor_config["moment_constants"])
        max_rot_vels = torch.as_tensor(rotor_config["max_rotation_velocities"]).float()
        self.num_rotors = len(force_constants)

        self.dt = dt
        self.time_up = 0.15
        self.time_down = 0.15
        self.noise_scale = 0.002

        self.KF = nn.Parameter(max_rot_vels.square() * force_constants)
        self.KM = nn.Parameter(max_rot_vels.square() * moment_constants)
        self.throttle = nn.Parameter(torch.zeros(self.num_rotors))
        self.directions = nn.Parameter(torch.as_tensor(rotor_config["directions"]).float())

        self.tau_up = nn.Parameter(0.43 * torch.ones(self.num_rotors))
        self.tau_down = nn.Parameter(0.43 * torch.ones(self.num_rotors))

        self.f = torch.square
        self.f_inv = torch.sqrt

        self.requires_grad_(False)

    def forward(self, cmds: torch.Tensor):
        target_throttle = self.f_inv(torch.clamp((cmds + 1) / 2, 0, 1))

        tau = torch.where(target_throttle > self.throttle, self.tau_up, self.tau_down)
        tau = torch.clamp(tau, 0, 1)
        self.throttle.add_(tau * (target_throttle - self.throttle))

        noise = torch.randn_like(self.throttle) * self.noise_scale * 0.
        t = torch.clamp(self.f(self.throttle) + noise, 0., 1.)
        thrusts = t * self.KF
        moments = (t * self.KM) * -self.directions

        return thrusts, moments
