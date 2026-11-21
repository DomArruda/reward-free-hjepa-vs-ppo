import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import SIGRegLoss, inverse_dynamics_loss


class HierarchicalJEPA(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=32, macro_stride=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.k = macro_stride

        self.enc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, latent_dim),
        )
        self.dyn1 = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, latent_dim),
        )
        self.dyn2 = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, latent_dim),
        )
        self.inv1 = nn.Sequential(
            nn.Linear(latent_dim * 2, 64), nn.SiLU(),
            nn.Linear(64, action_dim),
        )
        self.inv2 = nn.Sequential(
            nn.Linear(latent_dim * 2, 64), nn.SiLU(),
            nn.Linear(64, action_dim),
        )

    def pool_macro_actions(self, a_seq):
        return a_seq.mean(dim=-2)

    def step1(self, z, a):
        return z + self.dyn1(torch.cat([z, a], dim=-1))

    def step2(self, z, a_macro):
        return z + self.dyn2(torch.cat([z, a_macro], dim=-1))


class TorchHierarchicalJEPA(HierarchicalJEPA):
    def __init__(self, state_dim=4, action_dim=2, latent_dim=32, macro_stride=6):
        super().__init__(state_dim, action_dim, latent_dim=latent_dim, macro_stride=macro_stride)


class FinHierarchicalJEPA(HierarchicalJEPA):
    def __init__(self, state_dim=17, action_dim=5, latent_dim=32, macro_stride=5):
        super().__init__(state_dim, action_dim, latent_dim=latent_dim, macro_stride=macro_stride)


def train_hjepa(
    states,
    actions,
    use_inv_head=True,
    latent_dim=32,
    macro_stride=6,
    rollout_h=5,
    epochs=280,
    batch_size=512,
    lam_sigreg=0.08,
    simplex_actions=False,
):
    device = states.device
    state_dim = states.shape[-1]
    action_dim = actions.shape[-1]
    model_cls = FinHierarchicalJEPA if simplex_actions else TorchHierarchicalJEPA
    model = model_cls(state_dim, action_dim, latent_dim=latent_dim, macro_stride=macro_stride).to(device)
    sigreg = SIGRegLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)

    N, T_plus_1, _ = states.shape
    T = T_plus_1 - 1
    n_chunks = T // macro_stride

    for _ in range(epochs):
        idx = torch.randint(0, N, (batch_size,), device=device)
        s_b, a_b = states[idx], actions[idx]
        z_all = model.enc(s_b)

        t1 = torch.randint(0, T - rollout_h, (1,)).item()
        z1_pred = z_all[:, t1]
        loss_p1 = 0.0
        for h in range(rollout_h):
            z1_pred = model.step1(z1_pred, a_b[:, t1 + h])
            loss_p1 = loss_p1 + F.mse_loss(z1_pred, z_all[:, t1 + h + 1])
        loss_p1 = loss_p1 / rollout_h

        z_macro = z_all[:, :n_chunks * macro_stride + 1:macro_stride]
        a_chunks = a_b[:, :n_chunks * macro_stride].view(batch_size, n_chunks, macro_stride, action_dim)
        a_macro = model.pool_macro_actions(a_chunks)

        t2 = torch.randint(0, n_chunks - rollout_h, (1,)).item()
        z2_pred = z_macro[:, t2]
        loss_p2 = 0.0
        for h in range(rollout_h):
            z2_pred = model.step2(z2_pred, a_macro[:, t2 + h])
            loss_p2 = loss_p2 + F.mse_loss(z2_pred, z_macro[:, t2 + h + 1])
        loss_p2 = loss_p2 / rollout_h

        if simplex_actions:
            u_rand = F.softmax(torch.randn((batch_size, action_dim), device=device) * 2.0, dim=-1)
        else:
            u_rand = torch.empty((batch_size, action_dim), device=device).uniform_(-1.0, 1.0)
        z_anchor = z_macro[:, t2].detach()
        z_cons = z_anchor
        for _ in range(macro_stride):
            z_cons = model.step1(z_cons, u_rand)
        loss_cons = F.mse_loss(model.step2(z_anchor, u_rand), z_cons.detach())

        loss_reg = sigreg(z_all[:, ::4].reshape(-1, latent_dim))

        inv_term = 0.0
        if use_inv_head:
            inv_term = inverse_dynamics_loss(
                model, z_all, z_macro, a_b, a_macro, t1, t2, simplex=simplex_actions
            )

        loss = (1.0 - lam_sigreg) * (loss_p1 + loss_p2 + loss_cons + inv_term) + lam_sigreg * loss_reg
        opt.zero_grad()
        loss.backward()
        opt.step()

    return model


class ActorCriticPPO(nn.Module):
    def __init__(self, state_dim, action_dim, squash_mean=False):
        super().__init__()
        actor_layers = [
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, action_dim),
        ]
        if squash_mean:
            actor_layers.append(nn.Tanh())
        self.actor_mean = nn.Sequential(*actor_layers)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )

    def get_action_and_value(self, obs, action=None):
        mean = self.actor_mean(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(dim=-1), dist.entropy().sum(dim=-1), self.critic(obs).squeeze(-1)


class GoalPPO(ActorCriticPPO):
    def __init__(self, state_dim=8, action_dim=2):
        super().__init__(state_dim, action_dim, squash_mean=True)


class PortfolioPPO(ActorCriticPPO):
    def __init__(self, state_dim=17, action_dim=5):
        super().__init__(state_dim, action_dim, squash_mean=False)
