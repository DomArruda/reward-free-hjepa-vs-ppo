import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGRegLoss(nn.Module):
    def __init__(self, num_slices=128, num_knots=17):
        super().__init__()
        self.num_slices = num_slices
        knots = torch.linspace(-4.0, 4.0, num_knots)
        weights = torch.exp(-0.5 * knots ** 2)
        weights = weights / weights.sum()
        phi_target = torch.exp(-0.5 * knots ** 2)
        self.register_buffer("knots", knots)
        self.register_buffer("weights", weights)
        self.register_buffer("phi_target", phi_target)

    def forward(self, z):
        _, D = z.shape
        proj_dirs = F.normalize(torch.randn(D, self.num_slices, device=z.device), dim=0)
        p = z @ proj_dirs
        angles = p.unsqueeze(-1) * self.knots
        ecf_real = torch.cos(angles).mean(dim=0)
        ecf_imag = torch.sin(angles).mean(dim=0)
        diff_sq = (ecf_real - self.phi_target) ** 2 + ecf_imag ** 2
        return (diff_sq * self.weights).sum(dim=-1).mean()


def inverse_dynamics_loss(model, z_all, z_macro, actions, macro_actions, t1, t2, simplex=False):
    a1_hat = model.inv1(torch.cat([z_all[:, t1], z_all[:, t1 + 1]], dim=-1))
    a2_hat = model.inv2(torch.cat([z_macro[:, t2], z_macro[:, t2 + 1]], dim=-1))
    if simplex:
        a1_hat = F.softmax(a1_hat, dim=-1)
        a2_hat = F.softmax(a2_hat, dim=-1)
    return 0.5 * (F.mse_loss(a1_hat, actions[:, t1]) + F.mse_loss(a2_hat, macro_actions[:, t2]))
